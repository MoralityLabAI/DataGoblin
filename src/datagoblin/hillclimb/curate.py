"""Curate candidate training rows for looped modules and adapters under split custody.

Rules, applied in input order: a row needs a root and a split; only train rows are admitted; a root
that appears under more than one split anywhere in the inputs is dropped entirely (leakage guard);
exact duplicate content is dropped; a root keeps at most --max-per-root presentations. Admission never
sets training_ready: promotion of a curated set is a separate ledger decision.

    python -m datagoblin.hillclimb.curate --inputs a.jsonl b.jsonl --out runs/curated --max-per-root 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "datagoblin.hillclimb.curate.v1"
ROOT_KEYS = ("root", "root_id", "lineage_id", "task_id", "request_id")
FAMILY_KEYS = ("family_id", "family")
VOLATILE_KEYS = frozenset({"training_ready", "admission", "ts"})
TRAIN_SPLITS = frozenset({"train"})
MAX_ROWS = 200_000


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _read(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{number}: expected an object")
        rows.append(row)
    return rows


def curate(inputs: list[Path], out: Path, *, max_per_root: int = 4) -> dict[str, Any]:
    if not inputs:
        raise ValueError("at least one input is required")
    if not 1 <= max_per_root <= 16:
        raise ValueError("max_per_root must be between 1 and 16")
    loaded: list[tuple[Path, list[dict[str, Any]]]] = [(path, _read(path)) for path in inputs]
    if sum(len(rows) for _, rows in loaded) > MAX_ROWS:
        raise ValueError(f"more than {MAX_ROWS} input rows")
    splits_by_root: dict[str, set[str]] = defaultdict(set)
    for _, rows in loaded:
        for row in rows:
            root, split = _first(row, ROOT_KEYS), row.get("split")
            if root and isinstance(split, str):
                splits_by_root[root].add(split)
    colliding = {root for root, splits in splits_by_root.items() if len(splits) > 1}

    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_root: Counter[str] = Counter()
    for path, rows in loaded:
        for line, row in enumerate(rows, start=1):
            root, split = _first(row, ROOT_KEYS), row.get("split")
            digest = hashlib.sha256(
                canonical({k: v for k, v in row.items() if k not in VOLATILE_KEYS}).encode()
            ).hexdigest()
            reason = None
            if root is None or not isinstance(split, str):
                reason = "missing_custody_fields"
            elif split not in TRAIN_SPLITS:
                reason = "non_train_split"
            elif root in colliding:
                reason = "root_split_collision"
            elif digest in seen:
                reason = "duplicate"
            elif per_root[root] >= max_per_root:
                reason = "root_cap"
            if reason is not None:
                rejected.append({"input": path.name, "line": line, "root": root, "split": split,
                                 "reason": reason, "digest": digest})
                continue
            assert root is not None
            seen.add(digest)
            per_root[root] += 1
            admitted.append({**row, "training_ready": False,
                             "admission": {"curated_by": SCHEMA, "input": path.name, "digest": digest}})

    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (("admitted.jsonl", admitted), ("rejected.jsonl", rejected)):
        with (out / name).open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(canonical(row) + "\n")
    families = Counter(_first(row, FAMILY_KEYS) or "unknown" for row in admitted)
    manifest = {
        "schema": SCHEMA,
        "inputs": [{"path": str(path), "sha256": sha256_file(path), "rows": len(rows)} for path, rows in loaded],
        "admitted": {"rows": len(admitted), "roots": len(per_root), "families": dict(sorted(families.items()))},
        "rejected": dict(sorted(Counter(row["reason"] for row in rejected).items())),
        "leakage_roots": len(colliding),
        "max_per_root": max_per_root,
        "rights_review_pending": sum(row.get("source_rights_review_required") is True for row in admitted),
        "training_ready": False,
        "outputs": {name: sha256_file(out / name) for name in ("admitted.jsonl", "rejected.jsonl")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-per-root", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        manifest = curate(args.inputs, args.out, max_per_root=args.max_per_root)
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": "curate_error", "detail": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"admitted": manifest["admitted"], "rejected": manifest["rejected"],
                      "leakage_roots": manifest["leakage_roots"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
