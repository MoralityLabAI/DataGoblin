"""Join independent native Hyperon query shards into an all-root coverage receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .collect import sha256_bytes
from .looped_sft import query_supervision, root_id, split_for
from .sft_oracle import FAMILIES, OOD_FAMILY, make_task


def join_native(directory: Path, *, shard_count: int, roots_per_family: int = 512,
                queries_per_root: int = 3) -> dict[str, Any]:
    if shard_count < 1 or roots_per_family < 1 or queries_per_root < 1:
        raise ValueError("invalid native join configuration")
    wanted: set[tuple[str, str]] = set()
    for family in FAMILIES:
        if family == OOD_FAMILY:
            continue
        for seed in range(roots_per_family):
            if split_for(family, seed) != "train_candidate":
                continue
            task = make_task(family, seed)
            for step in range(min(queries_per_root, len(task["goals"]))):
                query_sha = sha256_bytes(query_supervision(task, step)["query"].encode())
                wanted.add((root_id(family, seed), query_sha))
    observed: set[tuple[str, str]] = set()
    shards: dict[str, Any] = {}
    families: Counter[str] = Counter()
    for shard_index in range(shard_count):
        path = directory / f"native_query_receipt_{shard_index}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != "metta-native-query-check-v1"
            or payload.get("shard_count") != shard_count
            or payload.get("shard_index") != shard_index
            or payload.get("roots_per_family") != roots_per_family
            or payload.get("queries_per_root") != queries_per_root
            or payload.get("max_train_roots_per_family") is not None
            or payload.get("train_partition_only") is not True
            or payload.get("runtime_query_compatibility") != "passed"
            or payload.get("runtime_version") != "0.2.10"
            or payload.get("training_ready") is not False
        ):
            raise ValueError(f"invalid or incompatible native shard {shard_index}")
        rows = payload["rows"]
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if digest != payload["receipts_sha256"] or len(rows) != payload["verified_query_instances"]:
            raise ValueError(f"corrupt rows in native shard {shard_index}")
        roots_here = {row["root"] for row in rows}
        if len(roots_here) != payload["verified_training_roots"]:
            raise ValueError(f"incomplete roots in native shard {shard_index}")
        for row in rows:
            root = row["root"]
            family, seed_text = root.split(":")[1:]
            seed = int(seed_text)
            if seed % shard_count != shard_index or split_for(family, seed) != "train_candidate":
                raise ValueError(f"wrong partition or shard: {root}")
            if row["matches"] is not True or row["native_results"] != row["static_expected_results"]:
                raise ValueError(f"native mismatch: {root}")
            key = (root, row["query_sha256"])
            if key in observed:
                raise ValueError(f"duplicated native query: {key}")
            observed.add(key)
            families[family] += 1
        shards[str(shard_index)] = {
            "receipt_file_sha256": sha256_bytes(path.read_bytes()),
            "query_instances": len(rows),
            "roots": len(roots_here),
        }
    if wanted != observed:
        raise ValueError(
            f"native query coverage incomplete: missing={len(wanted - observed)} extra={len(observed - wanted)}"
        )
    roots = {root for root, _digest in observed}
    return {
        "schema_version": "metta-native-query-aggregate-v1",
        "runtime": "hyperon==0.2.10",
        "training_roots_verified": len(roots),
        "query_instances_verified": len(observed),
        "train_partition_only": True,
        "unlabeled_reserved_roots_touched": 0,
        "curation_subset_oracle_verified_by_hyperon": False,
        "native_match_query_parity": "pass",
        "training_ready": False,
        "family_query_counts": dict(sorted(families.items())),
        "shards": shards,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=10)
    parser.add_argument("--roots-per-family", type=int, default=512)
    parser.add_argument("--queries-per-root", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = join_native(
        args.directory, shard_count=args.shard_count,
        roots_per_family=args.roots_per_family,
        queries_per_root=args.queries_per_root,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "training_roots_verified": receipt["training_roots_verified"],
        "query_instances_verified": receipt["query_instances_verified"],
        "native_match_query_parity": receipt["native_match_query_parity"],
        "shards": len(receipt["shards"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
