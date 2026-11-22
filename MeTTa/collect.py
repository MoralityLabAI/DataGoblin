"""Offline, read-only MeTTa corpus intake from pinned local Git checkouts.

The Git source catalog is an inventory, not a training release. This collector
checks HEAD and Git blob identities, extracts top-level lexical forms without
running MeTTa, and writes candidate data under an ignored output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha(data: bytes) -> str:
    prefix = f"blob {len(data)}\0".encode()
    return hashlib.sha1(prefix + data).hexdigest()


def scan_forms(source: str) -> list[tuple[int, int, str]]:
    """Find top-level atoms, with comments, quoted strings and ! eval prefixes.

    This is a lexical splitter. A balanced expression is not evidence that
    Hyperon accepted, typechecked, or executed it.
    """
    forms: list[tuple[int, int, str]] = []
    depth = 0
    start: int | None = None
    quoted = False
    escaped = False
    comment = False
    for i, char in enumerate(source):
        if comment:
            if char == "\n":
                comment = False
            continue
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == ";":
            comment = True
            continue
        if char == '"':
            if start is None:
                start = i
            quoted = True
            continue
        if char == "(":
            if start is None:
                start = i
            depth += 1
            continue
        if char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unexpected closing parenthesis at byte character {i}")
            if depth == 0 and start is not None:
                forms.append((start, i + 1, source[start : i + 1].strip()))
                start = None
            continue
        if depth == 0 and char.isspace():
            if start is not None:
                atom = source[start:i].strip()
                if atom:
                    forms.append((start, i, atom))
                start = None
            continue
        if start is None:
            start = i
    if quoted or depth:
        raise ValueError("unterminated string or top-level MeTTa expression")
    if start is not None:
        forms.append((start, len(source), source[start:].strip()))
    return [(a, b, atom) for a, b, atom in forms if atom]


def canonical_family(file_row: dict[str, Any]) -> str:
    # Keep source-family modules and their derivative views in one custody unit.
    path = str(file_row["path"])
    parts = path.split("/")
    return f'{file_row["source_id"]}:{parts[0] if len(parts) > 1 else path}'


def proposed_split(file_row: dict[str, Any]) -> str:
    if file_row["proposed_split"] == "ood_reserved":
        return "ood_reserved"
    family = canonical_family(file_row)
    value = int(sha256_bytes(family.encode())[:8], 16) % 10
    return "train_candidate" if value < 8 else "dev_reserved"


def verify_checkout(path: Path, expected_commit: str) -> Path:
    path = path.resolve(strict=True)
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    got = proc.stdout.strip()
    if got != expected_commit:
        raise ValueError(f"checkout HEAD {got} != pinned commit {expected_commit}: {path}")
    return path


def read_pinned_blob(checkout: Path, file_row: dict[str, Any]) -> bytes:
    relative = Path(file_row["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe source path: {relative}")
    file_path = checkout / relative
    if file_path.is_symlink():
        raise ValueError(f"symlink source disallowed: {relative}")
    file_path = file_path.resolve(strict=True)
    if not file_path.is_relative_to(checkout) or not file_path.is_file():
        raise ValueError(f"source escapes checkout: {relative}")
    data = file_path.read_bytes()
    if len(data) != file_row["bytes"] or git_blob_sha(data) != file_row["git_blob_sha"]:
        raise ValueError(f"source blob differs from pinned tree: {relative}")
    return data


def make_rows(file_row: dict[str, Any], data: bytes, commit: str) -> list[dict[str, Any]]:
    text = data.decode("utf-8")
    rows = []
    split = proposed_split(file_row)
    family = canonical_family(file_row)
    for start, _end, atom in scan_forms(text):
        offset_line = text.count("\n", 0, start) + 1
        atom_sha = sha256_bytes(atom.encode("utf-8"))
        row_id = sha256_bytes(f'{file_row["source_id"]}:{file_row["path"]}:{start}:{atom_sha}'.encode())
        rows.append({
            "schema_version": "metta-form-v1",
            "id": row_id,
            "root_id": f'{file_row["source_id"]}:{file_row["path"]}',
            "split_group": family,
            "proposed_split": split,
            "eligible_for_training": False,
            "custody": "QUARANTINE" if file_row["eligibility"] == "quarantine" else "DISCOVERY_ONLY",
            "evidence": "PINNED_GIT_BLOB_STATIC_LEXER",
            "semantics_verified": False,
            "source": {
                "source_id": file_row["source_id"],
                "path": file_row["path"],
                "commit": commit,
                "blob_sha1": file_row["git_blob_sha"],
                "full_file_sha256": sha256_bytes(data),
                "file_bytes": len(data),
                "source_line": offset_line,
                "review_flags": file_row["review_flags"],
            },
            "atom": atom,
            "atom_sha256": atom_sha,
            "kind": "query" if atom.startswith("!") else "expression",
        })
    return rows


def collect(
    catalog_path: Path, checkout_paths: dict[str, Path], output_dir: Path,
) -> dict[str, Any]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != "metta-source-catalog-v1":
        raise ValueError("unsupported source catalog")
    if not checkout_paths:
        raise ValueError("supply at least one pinned checkout")
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to((ROOT / "MeTTa").resolve()):
        raise ValueError("raw corpus output must live outside the committed MeTTa directory")
    if output_dir.is_relative_to((ROOT / "training").resolve()):
        raise ValueError("raw corpus output must live outside training plans")
    sources = {s["source_id"]: s for s in catalog["sources"]}
    unknown = set(checkout_paths) - set(sources)
    if unknown:
        raise ValueError(f"unknown sources: {sorted(unknown)}")
    verified = {s: verify_checkout(path, sources[s]["commit"])
                for s, path in sorted(checkout_paths.items())}
    all_rows = []
    counts: Counter[str] = Counter()
    for entry in catalog["files"]:
        source_id = entry["source_id"]
        if source_id not in verified:
            continue
        blob = read_pinned_blob(verified[source_id], entry)
        try:
            rows = make_rows(entry, blob, sources[source_id]["commit"])
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"cannot scan pinned source {source_id}:{entry['path']}: {exc}") from exc
        all_rows.extend(rows)
        counts[source_id] += len(rows)

    # Exact duplicated code across distinct split groups cannot supply release
    # labels on either side until a steward explicitly resolves the collision.
    owners: dict[str, set[str]] = defaultdict(set)
    for row in all_rows:
        owners[row["atom_sha256"]].add(row["proposed_split"])
    collision_hashes = {digest for digest, splits in owners.items() if len(splits) > 1}
    for row in all_rows:
        if row["atom_sha256"] in collision_hashes:
            row["custody"] = "QUARANTINE"
            row["cross_split_exact_duplicate"] = True
        else:
            row["cross_split_exact_duplicate"] = False
    all_rows.sort(key=lambda r: (r["source"]["source_id"], r["source"]["path"], r["source"]["source_line"], r["id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "forms.jsonl"
    with data_path.open("w", encoding="utf-8") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    digest = sha256_bytes(data_path.read_bytes())
    manifest = {
        "schema_version": "metta-corpus-receipt-v1",
        "source_catalog_sha256": sha256_bytes(catalog_path.read_bytes()),
        "forms_sha256": digest,
        "forms": len(all_rows),
        "sources": dict(sorted(counts.items())),
        "cross_split_exact_duplicate_hashes": len(collision_hashes),
        "semantics_verified": False,
        "training_ready": False,
        "static_only": True,
        "allowed_next_step": "independent_steward_and_runtime_verification",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=ROOT / "MeTTa" / "source_catalog.json")
    parser.add_argument("--checkout", action="append", required=True, metavar="SOURCE_ID=DIR")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "metta" / "intake")
    args = parser.parse_args()
    locations: dict[str, Path] = {}
    for arg in args.checkout:
        if "=" not in arg:
            parser.error("--checkout must be SOURCE_ID=DIR")
        key, raw = arg.split("=", 1)
        if not key or not raw or key in locations:
            parser.error("empty or repeated source checkout")
        locations[key] = Path(raw)
    receipt = collect(args.catalog, locations, args.out)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
