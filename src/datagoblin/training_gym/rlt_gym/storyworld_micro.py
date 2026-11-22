"""Source-native SWMD-MICRO compiler corpus, with encounter-level split custody."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from contextlib import suppress
from pathlib import Path
from typing import Any

from .storyworld_native import _hash, _json, _sha_file, connect, source_encounter_splits

SCHEMA = "datagoblin.swmd_micro.v1"
SOURCE_COMPILER = "codex-skills/small-storyworld-builder/scripts/swmd_build_qlora_examples.py"
OUTPUT_FILES = ("micro_public.jsonl", "micro_sealed.jsonl", "micro_train_sft.jsonl")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, rows: list[dict]) -> dict[str, Any]:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(_json(row) + "\n")
    return {"rows": len(rows), "sha256": _sha_file(path)}


def export_micro(source: Path, out: Path, *, seed: int = 23,
                 examples_per_encounter: int = 4, max_encounters: int = 12,
                 world: str = "storyworlds/diplomacy/forecast_backstab.json",
                 expect_source_commit: str | None = None) -> dict:
    if out.exists():
        raise FileExistsError(f"output exists: {out}")
    if not 1 <= examples_per_encounter <= 32 or not 1 <= max_encounters <= 1000:
        raise ValueError("examples-per-encounter must be 1..32 and max-encounters 1..1000")
    native = connect(source, world, expect_source_commit)
    path = native.root / SOURCE_COMPILER
    if not path.is_file():
        raise ValueError(f"missing native micro compiler: {path}")
    split_map = source_encounter_splits(native, seed)
    if not split_map:
        raise ValueError("source world has no repairable encounters")
    with tempfile.TemporaryDirectory(prefix="dg-sw-micro-") as td:
        temp = Path(td)
        swmd = temp / "source.swmd.min.md"
        swmd.write_text(native.exporter.emit_minified(native.world), encoding="utf-8")
        compiled = temp / "compiled"
        subprocess.run(
            [sys.executable, str(path), "--swmd-glob", str(swmd),
             "--out-dir", str(compiled), "--max-total-encounters",
             str(len(native.world["encounters"])),
             "--examples-per-encounter", str(examples_per_encounter),
             "--val-ratio", "0.05"],
            check=True, capture_output=True, text=True, timeout=45,
        )
        all_rows = _rows(compiled / "train.jsonl") + _rows(compiled / "val.jsonl")
    source_ids = set(sorted(split_map)[:max_encounters])
    public: list[dict] = []
    sealed: list[dict] = []
    train_sft: list[dict] = []
    task_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    seen_roots: set[str] = set()
    for row in all_rows:
        encounter_id = row["meta"]["encounter_id"]
        if encounter_id not in source_ids:
            continue
        split = split_map[encounter_id]
        root = _hash([SCHEMA, native.hashes["world"], encounter_id, row["id"], seed])[:24]
        if root in seen_roots:
            raise ValueError("duplicate source-derived micro root")
        seen_roots.add(root)
        lineage = _hash(["swmd-micro", native.hashes["world"], encounter_id])[:20]
        # A source-relative pointer, not the compiler's temporary SWMD file.
        provenance = {"source_repo": native.descriptor()["repository"],
                      "source_commit": native.revision, "world_path": native.world_path,
                      "encounter_id": encounter_id, "compiler_sha256": _sha_file(path)}
        input_public = {
            "schema": SCHEMA, "root": root, "lineage": lineage, "split": split,
            "mode": "mini_micro", "task_type": row["task_type"],
            "instruction": row["instruction"], "input": row["input"],
            "response_grammar": "one SWMD-MICRO-0.1 JSON object",
            "context_budget_tokens": 8192,
        }
        if len(_json(input_public).encode("utf-8")) > 8192:
            raise ValueError(f"source micro prompt exceeds 8K byte-safe budget: {encounter_id}")
        expected = json.loads(row["output"])
        if expected.get("schema") != "SWMD-MICRO-0.1":
            raise ValueError("source micro compiler generated an unexpected schema")
        public.append(input_public)
        sealed.append({"root": root, "lineage": lineage, "split": split,
                       "expected": expected, "provenance": provenance})
        if split == "train":
            train_sft.append({"root": root, "lineage": lineage, "task_type": row["task_type"],
                              "messages": [
                                  {"role": "system", "content": (
                                      "Compile or edit SWMD-MICRO-0.1 JSON using the supplied task. "
                                      "Preserve the requested schema and source transition IDs."
                                  )},
                                  {"role": "user", "content": row["instruction"] + "\n\n" + row["input"]},
                                  {"role": "assistant", "content": row["output"]},
                              ], "label_source": "source_native_deterministic_compiler"})
        task_counts[row["task_type"]] += 1
        split_counts[split] += 1
    if not public:
        raise ValueError("source micro compiler yielded zero supported rows")
    out.mkdir(parents=True)
    files = {
        "micro_public.jsonl": _write(out / "micro_public.jsonl", public),
        "micro_sealed.jsonl": _write(out / "micro_sealed.jsonl", sealed),
        "micro_train_sft.jsonl": _write(out / "micro_train_sft.jsonl", train_sft),
    }
    manifest = {
        "schema": SCHEMA, "source": native.descriptor(),
        "compiler_sha256": _sha_file(path),
        "evidence": "deterministic_source_compiler_supervision", "trained_model_results": False,
        "seed": seed, "examples_per_encounter": examples_per_encounter,
        "max_encounters": max_encounters,
        "task_types": dict(sorted(task_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "artifacts": files,
    }
    (out / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    return manifest


def replay_micro(source: Path, folder: Path, *, world: str = "storyworlds/diplomacy/forecast_backstab.json",
                 expect_source_commit: str | None = None) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or set(manifest.get("artifacts", {})) != set(OUTPUT_FILES):
        raise ValueError("invalid micro archive manifest")
    for name in OUTPUT_FILES:
        if _sha_file(folder / name) != manifest["artifacts"][name]["sha256"]:
            raise ValueError(f"micro archive hash mismatch: {name}")
    with tempfile.TemporaryDirectory(prefix="dg-sw-micro-replay-") as td:
        fresh_dir = Path(td) / "fresh"
        fresh = export_micro(source, fresh_dir, seed=manifest["seed"],
                             examples_per_encounter=manifest["examples_per_encounter"],
                             max_encounters=manifest["max_encounters"],
                             world=world, expect_source_commit=expect_source_commit)
        if fresh != manifest:
            raise ValueError("micro source or manifest mismatch")
        for name in OUTPUT_FILES:
            if (folder / name).read_bytes() != (fresh_dir / name).read_bytes():
                raise ValueError(f"micro archive content mismatch: {name}")
    return {"status": "verified", "rows": manifest["artifacts"]["micro_public.jsonl"]["rows"]}


def grade_micro(folder: Path, submissions: Path, graded_out: Path) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("invalid micro archive")
    for name in OUTPUT_FILES:
        if _sha_file(folder / name) != manifest["artifacts"][name]["sha256"]:
            raise ValueError(f"micro archive hash mismatch: {name}")
    public = {x["root"]: x for x in _rows(folder / "micro_public.jsonl")}
    sealed = {x["root"]: x for x in _rows(folder / "micro_sealed.jsonl")}
    if set(public) != set(sealed):
        raise ValueError("micro public/sealed root mismatch")
    results, seen = [], set()
    for row in _rows(submissions):
        root = row["root"]
        if root in seen or root not in public:
            raise ValueError("unknown or repeated micro root")
        seen.add(root)
        raw = row.get("output", "")
        parsed: Any = None
        if isinstance(raw, str):
            with suppress(TypeError, ValueError):
                parsed = json.loads(raw)
        accepted = parsed == sealed[root]["expected"]
        schema_valid = isinstance(parsed, dict) and parsed.get("schema") == "SWMD-MICRO-0.1"
        results.append({"root": root, "split": public[root]["split"],
                        "task_type": public[root]["task_type"],
                        "schema_valid": schema_valid, "accepted": accepted})
    graded_out.parent.mkdir(parents=True, exist_ok=True)
    meta = _write(graded_out, results)
    return {"graded": len(results), "accepted": sum(x["accepted"] for x in results), "output": meta}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("../GPTStoryworld"))
    p.add_argument("--world", default="storyworlds/diplomacy/forecast_backstab.json")
    p.add_argument("--expect-source-commit")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--examples-per-encounter", type=int, default=4)
    p.add_argument("--max-encounters", type=int, default=12)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path)
    group.add_argument("--replay", type=Path)
    group.add_argument("--grade", type=Path)
    p.add_argument("--submissions", type=Path)
    p.add_argument("--graded-out", type=Path)
    args = p.parse_args(argv)
    if args.out:
        report = export_micro(args.source, args.out, seed=args.seed,
                              examples_per_encounter=args.examples_per_encounter,
                              max_encounters=args.max_encounters, world=args.world,
                              expect_source_commit=args.expect_source_commit)
    elif args.replay:
        report = replay_micro(args.source, args.replay, world=args.world,
                              expect_source_commit=args.expect_source_commit)
    else:
        if args.submissions is None or args.graded_out is None:
            p.error("--grade requires --submissions and --graded-out")
        report = grade_micro(args.grade, args.submissions, args.graded_out)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
