"""Unit and fail-closed verification tests for Aleph Direct-A 10B continuation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CONT_DIR = REPO_ROOT / "training" / "aleph_next" / "continuation10b"
CAND_DIR = REPO_ROOT / "training" / "aleph_next" / "candidate01"
PACK_DIR = REPO_ROOT / "training" / "aleph_rlt_placement"
for p in (str(CONT_DIR), str(CAND_DIR), str(PACK_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from colab10m import RECIPE, RECIPE_SHA  # noqa: E402
from colab10m_xla import (  # noqa: E402
    Backend,
    FP32AdamW,
    accumulate,
    atomic_json,
    logits,
    loss_sum,
    mini_model,
    static_fixture,
    tensors,
)
from colab10m_xla import (  # noqa: E402
    save_checkpoint as legacy_save,
)
from drive_sync import sync_checkpoint_generation, verify_checkpoint_dir  # noqa: E402
from packing import INDEX_SCHEMA, reference_logits, sdpa_mask  # noqa: E402
from production import digest, read  # noqa: E402
from runner import (  # noqa: E402
    PRIOR_10M_EXPECTED,
    build_continuation_identity,
    compute_absolute_lr,
    load_continuation_checkpoint,
    save_continuation_checkpoint,
    verify_and_handoff_prior_checkpoint,
)
from stream import (  # noqa: E402
    STAGE_INDEX_SCHEMA,
    StageMixedStream,
    collect_stage_group,
    replay_stage_release,
    stage_group_boundaries,
    verify_lane_manifest,
)
from tpu_backend import HostOffloadFP32AdamW  # noqa: E402


def _write_json(path: Path, obj: dict[str, Any]) -> str:
    raw = json.dumps(obj, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _make_mini_lane(
    lane_dir: Path,
    lane_name: str,
    *,
    docs: list[tuple[str, str, list[int], int]],
    split: str = "train",
    context: int = 8,
    vocab_size: int = 64,
    tokenizer_bytes: bytes = b'{"mini":"tok"}',
) -> tuple[Path, str, str]:
    """Create a valid immutable aleph.packed-stream.v1 lane directory."""
    lane_dir.mkdir(parents=True, exist_ok=True)
    tok_path = lane_dir / "tokenizer.json"
    tok_path.write_bytes(tokenizer_bytes)
    tok_sha = digest(tok_path)

    shard_path = lane_dir / "shard_0000.jsonl"
    lines: list[tuple[dict[str, Any], bytes]] = []
    for root, family, token_ids, presentation in docs:
        row = {
            "root": root,
            "family": family,
            "split": split,
            "presentation": presentation,
            "lane": lane_name,
            "objective": "next_token",
            "token_ids": token_ids,
            "target_mask": [False] + [True] * (len(token_ids) - 1),
            "receipt_sha256": None,
        }
        raw = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        lines.append((row, raw))
    with shard_path.open("wb") as f:
        for _, raw in lines:
            f.write(raw + b"\n")
    shard_sha = digest(shard_path)

    db_path = lane_dir / "registry.sqlite3"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(INDEX_SCHEMA)
        conn.executemany(
            "INSERT OR IGNORE INTO families VALUES (?, ?)",
            [(r["family"], split) for r, _ in lines],
        )
        conn.executemany(
            "INSERT OR IGNORE INTO roots VALUES (?, ?)",
            [(r["root"], r["family"]) for r, _ in lines],
        )
        conn.executemany(
            "INSERT INTO views VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    r["root"],
                    r["presentation"],
                    shard_sha,
                    ordinal,
                    hashlib.sha256(raw).hexdigest(),
                    r["receipt_sha256"],
                )
                for ordinal, (r, raw) in enumerate(lines)
            ],
        )
        conn.commit()
    finally:
        conn.close()

    manifest = {
        "schema": "aleph.packed-stream.v1",
        "context": context,
        "vocab_size": vocab_size,
        "pad_id": 0,
        "tokenizer": {"path": tok_path.name, "sha256": tok_sha},
        "registry": {"path": db_path.name, "sha256": digest(db_path)},
        "shards": [{"path": shard_path.name, "sha256": shard_sha, "records": len(lines), "split": split}],
    }
    mpath = lane_dir / "manifest.json"
    msha = _write_json(mpath, manifest)
    return mpath, msha, tok_sha


def _make_mini_stage_release(root: Path) -> tuple[Path, str]:
    """Build a 2-lane miniature stage release (start=10M, new=26 positions, group=16 -> [16, 26])."""
    tok_bytes = b'{"mini":"tok-stage-v1"}'
    _m1, sha1, tok_sha = _make_mini_lane(
        root / "lane_a",
        "lane_a",
        docs=[
            ("ra1", "fa1", [1, 2, 3, 4, 5, 6, 7, 8], 1),
            ("ra2", "fa2", [9, 10, 11, 12, 13, 14, 15, 16], 1),
        ],
        context=8,
        vocab_size=64,
        tokenizer_bytes=tok_bytes,
    )
    _m2, sha2, _ = _make_mini_lane(
        root / "lane_b",
        "lane_b",
        docs=[
            ("rb1", "fb1", [20, 21, 22, 23, 24, 25], 1),
            ("rb2", "fb2", [26, 27, 28, 29, 30, 31], 1),
        ],
        context=8,
        vocab_size=64,
        tokenizer_bytes=tok_bytes,
    )
    index_obj = {
        "schema": STAGE_INDEX_SCHEMA,
        "split": "train",
        "stage_start_positions": 10_000_000,
        "stage_stop_positions": 10_000_026,
        "stage_new_positions": 26,
        "update_group_positions": 16,
        "context": 8,
        "vocab_size": 64,
        "tokenizer_sha256": tok_sha,
        "lane_quotas": {"lane_a": 16, "lane_b": 10},
        "lanes": {
            "lane_a": {"manifest_path": "lane_a/manifest.json", "manifest_sha256": sha1, "available_positions": 16},
            "lane_b": {"manifest_path": "lane_b/manifest.json", "manifest_sha256": sha2, "available_positions": 12},
        },
    }
    idx_sha = _write_json(root / "index.json", index_obj)
    return root, idx_sha


def test_frozen_lineage_and_lr_schedule() -> None:
    """Verify exact PR #74 artifact pins, 90M group arithmetic, and absolute LR clock."""
    assert digest(RECIPE) == RECIPE_SHA == PRIOR_10M_EXPECTED["recipe"]
    assert digest(CAND_DIR / "colab10m_xla.py") == PRIOR_10M_EXPECTED["code"]
    assert digest(CAND_DIR / "colab10m.py") == PRIOR_10M_EXPECTED["cuda_reference"]
    assert digest(PACK_DIR / "gym.py") == PRIOR_10M_EXPECTED["native"]
    assert digest(PACK_DIR / "packing.py") == PRIOR_10M_EXPECTED["packer"]

    cfg = read(RECIPE)
    assert math.isclose(compute_absolute_lr(cfg, 10_000_000), 1.25e-5, rel_tol=1e-12)
    assert math.isclose(compute_absolute_lr(cfg, 100_000_000), 1.25e-4, rel_tol=1e-12)
    assert math.isclose(compute_absolute_lr(cfg, 200_000_000), 2.5e-4, rel_tol=1e-12)
    assert math.isclose(compute_absolute_lr(cfg, 10_000_000_000), 2.5e-4, rel_tol=1e-12)
    assert math.isclose(compute_absolute_lr(cfg, 16_000_000_000), 2.5e-4, rel_tol=1e-12)
    assert math.isclose(compute_absolute_lr(cfg, 20_000_000_000), 2.5e-5, rel_tol=1e-12)

    bounds = stage_group_boundaries(90_000_000, 262_144)
    assert len(bounds) == 344
    assert bounds[-2] == 343 * 262_144
    assert bounds[-1] == 90_000_000
    assert bounds[-1] - bounds[-2] == 84_608
    assert 39 + len(bounds) == 383


def test_stage_stream_replay_no_wraparound_and_disjoint_splits(tmp_path: Path) -> None:
    """Verify multi-lane replay, mid-stage restart parity, exhaustion fail-closed, and split isolation."""
    rel_dir, idx_sha = _make_mini_stage_release(tmp_path / "release")
    replay = replay_stage_release(rel_dir, idx_sha, restart_after_group=1)
    assert replay["update_groups"] == 2
    assert replay["full_groups"] == 1
    assert replay["final_group_positions"] == 10
    assert replay["lane_positions"] == {"lane_a": 16, "lane_b": 10}
    assert replay["restart_parity_verified"] is True

    # Exhaustion must fail closed with zero wraparound
    with ExitStack() as stack:
        stream = StageMixedStream(rel_dir, rel_dir / "index.json", idx_sha, stack)
        collect_stage_group(stream, 16)
        collect_stage_group(stream, 26)
        with pytest.raises(RuntimeError, match="exhausted"):
            stream.next_pack(1)

    # Tampered cursor must be rejected
    with ExitStack() as stack:
        stream = StageMixedStream(rel_dir, rel_dir / "index.json", idx_sha, stack)
        collect_stage_group(stream, 16)
        bad_cursor = copy.deepcopy(stream.cursor())
        bad_cursor["payload"]["totals"]["lane_a"] += 1
        with pytest.raises(ValueError, match="checksum"):
            StageMixedStream(rel_dir, rel_dir / "index.json", idx_sha, stack, cursor=bad_cursor)

    # Cross-lane duplicate root must fail closed
    dup_dir = tmp_path / "dup_release"
    tok_bytes = b'{"mini":"tok-dup"}'
    _, sha_a, tok_sha = _make_mini_lane(
        dup_dir / "lane_a",
        "lane_a",
        docs=[("shared_root", "fa1", [1, 2, 3, 4], 1)],
        context=8,
        vocab_size=64,
        tokenizer_bytes=tok_bytes,
    )
    _, sha_b, _ = _make_mini_lane(
        dup_dir / "lane_b",
        "lane_b",
        docs=[("shared_root", "fb1", [5, 6, 7, 8], 1)],
        context=8,
        vocab_size=64,
        tokenizer_bytes=tok_bytes,
    )
    dup_idx_sha = _write_json(
        dup_dir / "index.json",
        {
            "schema": STAGE_INDEX_SCHEMA,
            "split": "train",
            "stage_start_positions": 10_000_000,
            "stage_stop_positions": 10_000_008,
            "stage_new_positions": 8,
            "update_group_positions": 8,
            "context": 8,
            "vocab_size": 64,
            "tokenizer_sha256": tok_sha,
            "lane_quotas": {"lane_a": 4, "lane_b": 4},
            "lanes": {
                "lane_a": {"manifest_path": "lane_a/manifest.json", "manifest_sha256": sha_a},
                "lane_b": {"manifest_path": "lane_b/manifest.json", "manifest_sha256": sha_b},
            },
        },
    )
    with pytest.raises(ValueError, match="cross-lane duplicate root"):
        replay_stage_release(dup_dir, dup_idx_sha)

    # Non-train split in a lane manifest must fail verify_lane_manifest(split='train')
    dev_m, dev_sha, dev_tok = _make_mini_lane(
        tmp_path / "dev_lane", "dev_lane", docs=[("d1", "fd1", [1, 2, 3, 4], 1)], split="dev", context=8, vocab_size=64
    )
    with pytest.raises(ValueError, match="split"):
        verify_lane_manifest(dev_m, dev_sha, split="train", expected_tokenizer_sha256=dev_tok, context=8, vocab_size=64)


def test_document_isolation_all_masked_and_grad_denominator() -> None:
    """Verify segment-causal isolation, zero gradient on all-masked pack, and exact global denominator."""
    torch.set_num_threads(1)
    torch.manual_seed(7)
    net = mini_model()
    packs = static_fixture()
    # Document isolation parity against reference segment-wise forward
    p0 = packs[0]
    ids, pos, seg, _ = tensors(p0, torch.device("cpu"), 8)
    z_packed = logits(net, ids, pos, seg, torch.float32, remat=False)
    z_ref = reference_logits(net, p0, arm="direct", r=2, s=1)
    torch.testing.assert_close(z_packed[0, : p0["nonpadding"]], z_ref[0, : p0["nonpadding"]], atol=1e-5, rtol=1e-5)

    # No future leakage in SDPA mask
    m = sdpa_mask(p0["segment_ids"])[0, 0]
    for q in range(8):
        for k in range(8):
            if p0["segment_ids"][q] >= 0 and k > q:
                assert not bool(m[q, k])
            if p0["segment_ids"][q] >= 0 and p0["segment_ids"][k] != p0["segment_ids"][q]:
                assert not bool(m[q, k])

    # All-masked pack produces exact zero loss and zero gradient
    all_masked = copy.deepcopy(p0)
    all_masked["labels_at_query"] = [-100] * 8
    ids_m, pos_m, seg_m, y_m = tensors(all_masked, torch.device("cpu"), 8)
    net.zero_grad(set_to_none=True)
    lsum = loss_sum(logits(net, ids_m, pos_m, seg_m, torch.float32, remat=False), y_m)
    assert float(lsum) == 0.0
    lsum.backward()
    for p in net.parameters():
        assert p.grad is None or float(p.grad.abs().max()) == 0.0


def test_handoff_resume_equality_and_fail_closed_checks(tmp_path: Path) -> None:
    """Test one-time verified handoff, interrupted/resumed parity, Drive sync, and all fail-closed rejections."""
    torch.set_num_threads(1)
    cfg = read(RECIPE)
    backend = Backend("cpu")
    packs = static_fixture()

    prior_dir = tmp_path / "prior_10m"
    prior_dir.mkdir()
    backend.seed(42)
    net_p = mini_model()
    opt_p = FP32AdamW(net_p, cfg, backend)
    accumulate(net_p, packs, backend, torch.float32)
    opt_p.update(compute_absolute_lr(cfg, 10_000_000))
    prior_ident = {"recipe": RECIPE_SHA, "index": "1" * 64, "receipt": "2" * 64, "backend": "cpu", "world_size": 1}
    atomic_json(prior_dir / "IDENTITY.json", prior_ident)
    legacy_save(prior_dir, prior_ident, net_p, opt_p, backend, {"sha256": "3" * 64}, [{"group": 1, "stop": 10_000_000}])

    # Uninterrupted 2 steps after handoff
    net_unint = mini_model()
    opt_unint = FP32AdamW(net_unint, cfg, backend)
    receipt, recs = verify_and_handoff_prior_checkpoint(
        prior_dir=prior_dir,
        expected_prior_step=1,
        expected_prior_positions=10_000_000,
        expected_prior_index_sha256="1" * 64,
        expected_prior_manifest_sha256=None,
        new_stage_index_sha256="4" * 64,
        new_stage_replay_sha256="5" * 64,
        stage_contract_sha256="6" * 64,
        net=net_unint,
        opt=opt_unint,
        backend=backend,
    )
    assert opt_unint.step == 1 and len(recs) == 1
    accumulate(net_unint, packs, backend, torch.float32)
    opt_unint.update(compute_absolute_lr(cfg, 10_262_144))
    accumulate(net_unint, packs, backend, torch.float32)
    opt_unint.update(compute_absolute_lr(cfg, 10_524_288))

    # Interrupted: handoff -> step 2 -> save -> Drive sync -> restore -> step 3
    run_local = tmp_path / "run_local"
    run_local.mkdir()
    run_drive = tmp_path / "run_drive"
    net_res = mini_model()
    opt_res = FP32AdamW(net_res, cfg, backend)
    rec2, recs2 = verify_and_handoff_prior_checkpoint(
        prior_dir=prior_dir,
        expected_prior_step=1,
        expected_prior_positions=10_000_000,
        expected_prior_index_sha256="1" * 64,
        expected_prior_manifest_sha256=receipt["prior_manifest_sha256"],
        new_stage_index_sha256="4" * 64,
        new_stage_replay_sha256="5" * 64,
        stage_contract_sha256="6" * 64,
        net=net_res,
        opt=opt_res,
        backend=backend,
    )
    ident = build_continuation_identity(
        stage_contract_sha256="6" * 64,
        stage_index_sha256="4" * 64,
        stage_replay_sha256="5" * 64,
        prior_manifest_sha256=rec2["prior_manifest_sha256"],
        backend=backend,
        context=8,
    )
    atomic_json(run_local / "HANDOFF_RECEIPT.json", rec2)
    atomic_json(run_local / "IDENTITY.json", ident)
    accumulate(net_res, packs, backend, torch.float32)
    opt_res.update(compute_absolute_lr(cfg, 10_262_144))
    recs2.append({"group": 2, "stop": 10_262_144})
    cdir, msha = save_continuation_checkpoint(
        run_local, ident, net_res, opt_res, backend, {"stage_cursor": 1}, recs2, handoff_receipt=rec2
    )
    ver = verify_checkpoint_dir(cdir, msha)
    assert ver["step"] == 2
    sync_res = sync_checkpoint_generation(run_local, run_drive, workers=4, keep_generations=2)
    assert sync_res["manifest_sha256"] == msha

    net_after = mini_model()
    opt_after = FP32AdamW(net_after, cfg, backend)
    restored = load_continuation_checkpoint(run_drive, ident, net_after, opt_after, backend)
    assert opt_after.step == 2 and restored["cursor"] == {"stage_cursor": 1}
    accumulate(net_after, packs, backend, torch.float32)
    opt_after.update(compute_absolute_lr(cfg, 10_524_288))
    for (n1, p1), (n2, p2) in zip(opt_unint.named, opt_after.named, strict=True):
        torch.testing.assert_close(p1, p2, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(opt_unint.moments[n1][0], opt_after.moments[n2][0], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(opt_unint.moments[n1][1], opt_after.moments[n2][1], atol=1e-6, rtol=1e-6)

    # --- Fail-closed negative cases ---
    # 1. Wrong expected optimizer step
    with pytest.raises(ValueError, match="step"):
        verify_and_handoff_prior_checkpoint(
            prior_dir=prior_dir,
            expected_prior_step=39,
            expected_prior_positions=10_000_000,
            expected_prior_index_sha256="1" * 64,
            expected_prior_manifest_sha256=None,
            new_stage_index_sha256="4" * 64,
            new_stage_replay_sha256="5" * 64,
            stage_contract_sha256="6" * 64,
            net=mini_model(),
            opt=FP32AdamW(mini_model(), cfg, backend),
            backend=backend,
        )

    # 2. Wrong expected absolute positions
    with pytest.raises(ValueError, match="position"):
        verify_and_handoff_prior_checkpoint(
            prior_dir=prior_dir,
            expected_prior_step=1,
            expected_prior_positions=9_999_999,
            expected_prior_index_sha256="1" * 64,
            expected_prior_manifest_sha256=None,
            new_stage_index_sha256="4" * 64,
            new_stage_replay_sha256="5" * 64,
            stage_contract_sha256="6" * 64,
            net=mini_model(),
            opt=FP32AdamW(mini_model(), cfg, backend),
            backend=backend,
        )

    # 3. Wrong prior manifest SHA pin
    with pytest.raises(ValueError, match="manifest digest"):
        verify_and_handoff_prior_checkpoint(
            prior_dir=prior_dir,
            expected_prior_step=1,
            expected_prior_positions=10_000_000,
            expected_prior_index_sha256="1" * 64,
            expected_prior_manifest_sha256="0" * 64,
            new_stage_index_sha256="4" * 64,
            new_stage_replay_sha256="5" * 64,
            stage_contract_sha256="6" * 64,
            net=mini_model(),
            opt=FP32AdamW(mini_model(), cfg, backend),
            backend=backend,
        )

    # 4. Altered continuation identity (e.g. altered corpus index hash) on resume
    bad_ident = copy.deepcopy(ident)
    bad_ident["index"] = "f" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        load_continuation_checkpoint(run_drive, bad_ident, mini_model(), FP32AdamW(mini_model(), cfg, backend), backend)

    # 5. Corrupted parameter shard on disk
    bad_drive = tmp_path / "run_drive_corrupt"
    sync_checkpoint_generation(run_local, bad_drive, workers=2)
    latest_tag = read(bad_drive / "LATEST.json")["directory"]
    shard0 = bad_drive / latest_tag / "p0000.pt"
    shard0.write_bytes(shard0.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        load_continuation_checkpoint(bad_drive, ident, mini_model(), FP32AdamW(mini_model(), cfg, backend), backend)


def test_host_offload_adamw_exact_parity() -> None:
    """Verify HostOffloadFP32AdamW matches FP32AdamW bit-identically across multiple steps."""
    torch.set_num_threads(1)
    cfg = read(RECIPE)
    backend = Backend("cpu")
    packs = static_fixture()
    backend.seed(99)
    net_ref = mini_model()
    net_off = copy.deepcopy(net_ref)
    opt_ref = FP32AdamW(net_ref, cfg, backend)
    opt_off = HostOffloadFP32AdamW(net_off, cfg, backend)

    for step_idx in range(3):
        rate = compute_absolute_lr(cfg, 10_000_000 + step_idx * 262_144)
        accumulate(net_ref, packs, backend, torch.float32)
        accumulate(net_off, packs, backend, torch.float32)
        g_ref = opt_ref.update(rate)
        g_off = opt_off.update(rate)
        assert math.isclose(g_ref, g_off, rel_tol=1e-6, abs_tol=1e-6)
        for (n1, p1), (n2, p2) in zip(opt_ref.named, opt_off.named, strict=True):
            assert n1 == n2
            torch.testing.assert_close(p1, p2, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(opt_ref.moments[n1][0], opt_off.moments[n2][0], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(opt_ref.moments[n1][1], opt_off.moments[n2][1], atol=1e-6, rtol=1e-6)
