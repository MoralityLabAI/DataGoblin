"""Offline certificate-join controls for sharded native MeTTa query replay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from MeTTa.aggregate_native import join_native
from MeTTa.collect import sha256_bytes
from MeTTa.looped_sft import query_supervision, root_id, split_for
from MeTTa.sft_oracle import FAMILIES, OOD_FAMILY, make_task


def build_fake_shards(directory: Path, *, horizon: int = 12, shards: int = 3) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for idx in range(shards):
        rows = []
        for family in FAMILIES:
            if family == OOD_FAMILY:
                continue
            for seed in range(horizon):
                if seed % shards != idx or split_for(family, seed) != "train_candidate":
                    continue
                task = make_task(family, seed)
                for step in range(3):
                    target = query_supervision(task, step)
                    count = len(target["expected_record_ids"])
                    rows.append({
                        "root": root_id(family, seed),
                        "query_sha256": sha256_bytes(target["query"].encode()),
                        "source_sha256": sha256_bytes(task["program"].encode()),
                        "native_results": count,
                        "static_expected_results": count,
                        "matches": True,
                    })
        data = {
            "schema_version": "metta-native-query-check-v1",
            "shard_count": shards,
            "shard_index": idx,
            "roots_per_family": horizon,
            "queries_per_root": 3,
            "max_train_roots_per_family": None,
            "train_partition_only": True,
            "runtime_query_compatibility": "passed",
            "runtime_version": "0.2.10",
            "training_ready": False,
            "verified_query_instances": len(rows),
            "verified_training_roots": len({r["root"] for r in rows}),
            "receipts_sha256": hashlib.sha256(
                json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "rows": rows,
        }
        (directory / f"native_query_receipt_{idx}.json").write_text(json.dumps(data))


def test_sharded_certificate_join_covers_exact_train_root_horizon(tmp_path: Path) -> None:
    build_fake_shards(tmp_path)
    joined = join_native(tmp_path, shard_count=3, roots_per_family=12, queries_per_root=3)
    expected = sum(split_for(f, seed) == "train_candidate" for f in FAMILIES if f != OOD_FAMILY
                   for seed in range(12))
    assert joined["training_roots_verified"] == expected
    assert joined["query_instances_verified"] == 3 * expected
    assert joined["unlabeled_reserved_roots_touched"] == 0
    assert joined["native_match_query_parity"] == "pass"


def test_certificate_join_rejects_incomplete_shards(tmp_path: Path) -> None:
    build_fake_shards(tmp_path)
    (tmp_path / "native_query_receipt_2.json").unlink()
    with pytest.raises(FileNotFoundError):
        join_native(tmp_path, shard_count=3, roots_per_family=12, queries_per_root=3)


def test_certificate_join_rejects_tampered_native_row_even_if_digest_updated(tmp_path: Path) -> None:
    build_fake_shards(tmp_path)
    path = tmp_path / "native_query_receipt_0.json"
    data = json.loads(path.read_text())
    assert data["rows"]
    data["rows"][0]["query_sha256"] = "0" * 64
    data["receipts_sha256"] = hashlib.sha256(
        json.dumps(data["rows"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="coverage incomplete"):
        join_native(tmp_path, shard_count=3, roots_per_family=12, queries_per_root=3)


def test_certificate_join_rejects_steward_eval_leak(tmp_path: Path) -> None:
    build_fake_shards(tmp_path)
    path = tmp_path / "native_query_receipt_0.json"
    data = json.loads(path.read_text())
    row = data["rows"][0]
    row["root"] = "metta-sft-v1:conflict:000000"
    data["receipts_sha256"] = hashlib.sha256(
        json.dumps(data["rows"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="partition or shard|incomplete roots"):
        join_native(tmp_path, shard_count=3, roots_per_family=12, queries_per_root=3)
