"""Optional native Hyperon replay of generated MeTTa source-filtering queries.

This checks MeTTa query semantics on a fresh runtime. It deliberately samples
training roots only; development/test/OOD answers never enter the CI receipt.
The curation subset oracle is still a separate Python verifier, and success
here does not verify arbitrary learning targets or authorize corpus release.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from .collect import ROOT, sha256_bytes
from .looped_sft import query_supervision, split_for
from .sft_oracle import FAMILIES, OOD_FAMILY, make_task


def run_native(*, train_roots_per_family: int, queries_per_root: int) -> dict[str, Any]:
    if not 1 <= train_roots_per_family <= 512 or not 1 <= queries_per_root <= 5:
        raise ValueError("invalid bounded native replay workload")
    from hyperon import MeTTa  # optional import: native runner only

    version = importlib.metadata.version("hyperon")
    results: list[dict[str, Any]] = []
    for family in FAMILIES:
        if family == OOD_FAMILY:
            continue
        checked = 0
        for seed in range(4096):
            if split_for(family, seed) != "train_candidate":
                continue
            task = make_task(family, seed)
            for step in range(min(queries_per_root, len(task["goals"]))):
                goal = query_supervision(task, step)
                runner = MeTTa()
                runner.run(task["program"])
                native = runner.run(goal["query"])
                if not native or not isinstance(native[-1], list):
                    raise RuntimeError(f"missing native result: family={family} seed={seed} visit={step}")
                observed = sorted(str(atom) for atom in native[-1])
                expected = goal["expected_record_ids"]
                if observed != expected:
                    raise ValueError(
                        f"MeTTa query mismatch family={family} seed={seed} visit={step}: "
                        f"expected {expected}, observed {observed}"
                    )
                results.append({
                    "root": f"metta-sft-v1:{family}:{seed:06d}",
                    "query_sha256": sha256_bytes(goal["query"].encode()),
                    "source_sha256": sha256_bytes(task["program"].encode()),
                    "native_results": len(observed),
                    "static_expected_results": len(expected),
                    "matches": True,
                })
            checked += 1
            if checked == train_roots_per_family:
                break
        if checked != train_roots_per_family:
            raise ValueError("too few eligible training roots")
    return {
        "schema_version": "metta-native-query-check-v1",
        "runtime": "hyperon",
        "runtime_version": version,
        "train_partition_only": True,
        "source_kind": "synthetic_symbolic_world",
        "verified_query_instances": len(results),
        "roots_per_family": train_roots_per_family,
        "queries_per_root": queries_per_root,
        "runtime_query_compatibility": "passed",
        "curation_subset_oracle_independently_checked": False,
        "native_semantics_for_unchecked_roots": "unknown",
        "training_ready": False,
        "receipts_sha256": hashlib.sha256(
            json.dumps(results, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "rows": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-roots-per-family", type=int, default=8)
    parser.add_argument("--queries-per-root", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "metta" / "native_query_receipt.json")
    args = parser.parse_args()
    receipt = run_native(train_roots_per_family=args.train_roots_per_family,
                         queries_per_root=args.queries_per_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "rows"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
