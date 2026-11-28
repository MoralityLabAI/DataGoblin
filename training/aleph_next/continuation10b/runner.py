"""Versioned Direct-A continuation trainer (10M -> 100M -> 10B) with verified checkpoint handoff.

Preserves the frozen 999,352,320-parameter direct-A architecture, explicit BF16 compute,
FP32 master weights, FP32 AdamW moments, and absolute position LR clock (200M warmup to
2.5e-4, stable through 16B, cosine decay to 2.5e-5 at 20B) while leaving the original
10M/39-group ``colab10m_xla.py`` and ``verify_bundle.py`` untouched.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
ALEPH_NEXT = HERE.parent
TRAINING_ROOT = ALEPH_NEXT.parent
for p in (str(HERE), str(TRAINING_ROOT / "aleph_rlt_placement"), str(ALEPH_NEXT / "candidate01")):
    if p not in sys.path:
        sys.path.insert(0, p)

from colab10m import RECIPE, RECIPE_SHA  # noqa: E402
from colab10m_xla import (  # noqa: E402
    CHECKPOINT_SCHEMA as LEGACY_CHECKPOINT_SCHEMA,
)
from colab10m_xla import (  # noqa: E402
    Backend,
    FP32AdamW,
    atomic_json,
    logits,
    loss_sum,
    mini_model,
    static_fixture,
    tensors,
)
from colab10m_xla import (  # noqa: E402
    self_test as legacy_self_test,
)
from production import TOKENIZER_SHA256, digest, read  # noqa: E402
from restart import Aleph, lr, schedule  # noqa: E402
from stream import (  # noqa: E402
    STAGE_REPLAY_SCHEMA,
    StageMixedStream,
    collect_stage_group,
    stage_group_boundaries,
)

CONTINUATION_CHECKPOINT_SCHEMA = "aleph.direct-a.continuation.v1"
HANDOFF_RECEIPT_SCHEMA = "aleph.direct-a.handoff-receipt.v1"

PRIOR_10M_EXPECTED = {
    "schema": LEGACY_CHECKPOINT_SCHEMA,
    "recipe": "58fd8c2805f04ed918b4547d5a177624f839648c6be98dc749426ba39cd32e87",
    "index": "53d27abfcea62a06dccf4e2463c7b83b9b3693abfca4f796d24536e5282730c0",
    "receipt": "e274e031922e48ef40808c058bd20d62233648117771966633620b3a9ad133bc",
    "code": "2e17982ad299a137d454f683f0d085d2362eb381e9a92edce4bb11402dcd35d2",
    "cuda_reference": "f621549d70b9a6a160fcab873bf87879ddad766c518a551e75b18ceeae4b3f57",
    "native": "7491dc5177aedbaf4b9b5d2b73fa5d40c3794e6329d4a7a4085a62a8bddb197b",
    "packer": "6fae20996642ce1bf94a455ad366b7b1db3893761bdb803ed356d5d64795c465",
    "step": 39,
    "absolute_positions": 10_000_000,
}


def compute_absolute_lr(cfg: dict[str, Any], absolute_positions: int) -> float:
    """Evaluate the frozen absolute-position learning-rate schedule."""
    if absolute_positions < 0:
        raise ValueError("absolute_positions must be non-negative")
    return float(lr(cfg, absolute_positions))


def save_continuation_checkpoint(
    out: Path,
    identity: dict[str, Any],
    net: torch.nn.Module,
    opt: FP32AdamW,
    backend: Backend,
    stage_cursor: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    handoff_receipt: dict[str, Any],
) -> tuple[Path, str]:
    """Atomically write a hash-verified FP32 master + moments + RNG + stage-cursor checkpoint."""
    backend.sync()
    nbytes = sum(p.numel() * 12 for _, p in opt.named)
    if shutil.disk_usage(out).free < nbytes + 64 * 1024 * 1024:
        raise RuntimeError("insufficient disk capacity for another full continuation checkpoint")
    tag = f"group_{opt.step:04d}_{uuid.uuid4().hex[:12]}"
    partial = out / (tag + ".partial")
    partial.mkdir()
    entries: list[dict[str, Any]] = []
    for idx, (name, param) in enumerate(opt.named):
        m, v = opt.moments[name]
        shard = partial / f"p{idx:04d}.pt"
        with shard.open("wb") as f:
            torch.save({"weight": param.detach().cpu(), "m": m.cpu(), "v": v.cpu()}, f)
            f.flush()
            os.fsync(f.fileno())
        entries.append({"name": name, "shape": list(param.shape), "file": shard.name, "sha256": digest(shard)})
    meta = partial / "state.pt"
    with meta.open("wb") as f:
        torch.save(
            {
                "rng": backend.get_rng(),
                "cursor": stage_cursor,
                "records": records,
                "step": opt.step,
                "handoff_receipt": handoff_receipt,
            },
            f,
        )
        f.flush()
        os.fsync(f.fileno())
    state_sha = digest(meta)
    manifest = {
        "schema": CONTINUATION_CHECKPOINT_SCHEMA,
        "identity": identity,
        "step": opt.step,
        "absolute_positions": int(records[-1]["stop"]) if records else 0,
        "parameters": entries,
        "state": {"file": meta.name, "sha256": state_sha},
        "state_sha256": state_sha,
        "handoff_sha256": digest(out / "HANDOFF_RECEIPT.json") if (out / "HANDOFF_RECEIPT.json").is_file() else None,
    }
    manifest_path = partial / "manifest.json"
    atomic_json(manifest_path, manifest)
    final_dir = out / tag
    os.replace(partial, final_dir)
    msha = digest(final_dir / "manifest.json")
    atomic_json(out / "LATEST.json", {"directory": tag, "sha256": msha, "step": opt.step})
    return final_dir, msha


def _verify_and_load_shards(
    ckpt_dir: Path,
    manifest: dict[str, Any],
    opt: FP32AdamW,
    backend: Backend,
) -> dict[str, Any]:
    """Fail-closed SHA-256 verification and restore of weights, FP32 AdamW moments, and state.pt."""
    entries = manifest["parameters"]
    if len(entries) != len(opt.named):
        raise ValueError(f"parameter shard count mismatch: {len(entries)} != {len(opt.named)}")
    # Verify every shard hash BEFORE mutating any parameter or moment tensor
    for (name, param), item in zip(opt.named, entries, strict=True):
        if item["name"] != name or list(param.shape) != item["shape"]:
            raise ValueError(f"parameter identity/shape mismatch for {name}")
        fpath = ckpt_dir / item["file"]
        if not fpath.is_file() or digest(fpath) != item["sha256"]:
            raise ValueError(f"corrupt or missing parameter shard {item['file']} ({name})")
    if "state" in manifest and isinstance(manifest["state"], dict):
        state_file = manifest["state"]["file"]
        state_sha_expected = manifest["state"]["sha256"]
    elif "state_sha256" in manifest:
        state_file = "state.pt"
        state_sha_expected = manifest["state_sha256"]
    else:
        raise ValueError("checkpoint manifest lacks state digest")
    state_path = ckpt_dir / state_file
    if not state_path.is_file() or digest(state_path) != state_sha_expected:
        raise ValueError("corrupt or missing state.pt")

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    manifest_step = int(manifest["step"])
    if "step" in state and int(state["step"]) != manifest_step:
        raise ValueError(f"state.pt step {state['step']} != manifest step {manifest_step}")

    # Restore weights and FP32 moments without resetting optimizer state
    with torch.no_grad():
        for (name, param), item in zip(opt.named, entries, strict=True):
            blob = torch.load(ckpt_dir / item["file"], map_location="cpu", weights_only=True)
            if (
                blob["weight"].dtype != torch.float32
                or blob["m"].dtype != torch.float32
                or blob["v"].dtype != torch.float32
            ):
                raise ValueError(f"non-FP32 master or moment tensor in {item['file']}")
            if not (
                torch.isfinite(blob["weight"]).all()
                and torch.isfinite(blob["m"]).all()
                and torch.isfinite(blob["v"]).all()
            ):
                raise ValueError(f"nonfinite weight or moment tensor in {item['file']}")
            param.copy_(blob["weight"].to(backend.device))
            opt.moments[name][0].copy_(blob["m"].to(backend.device))
            opt.moments[name][1].copy_(blob["v"].to(backend.device))
    opt.step = manifest_step
    opt.zero()
    rng_payload = state["rng"]
    if (backend.kind == "cuda" and rng_payload.get("cuda") is None) or (
        backend.xm is not None and rng_payload.get("xla") is None
    ):
        from restart import restore_rng

        restore_rng(rng_payload["host"])
        if backend.kind == "cuda":
            torch.cuda.manual_seed_all(manifest_step)
        if backend.xm is not None:
            backend.xm.set_rng_state(manifest_step, device=str(backend.device))
    else:
        backend.set_rng(rng_payload)
    backend.sync()
    return state


def verify_and_handoff_prior_checkpoint(
    *,
    prior_dir: Path,
    expected_prior_step: int,
    expected_prior_positions: int,
    expected_prior_index_sha256: str,
    expected_prior_manifest_sha256: str | None,
    new_stage_index_sha256: str,
    new_stage_replay_sha256: str,
    stage_contract_sha256: str,
    net: torch.nn.Module,
    opt: FP32AdamW,
    backend: Backend,
    allow_backend_handoff: bool = False,
    expected_prior_schema: str = LEGACY_CHECKPOINT_SCHEMA,
    expected_prior_code_hashes: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Perform a strict, fail-closed one-time verified handoff from a completed prior checkpoint."""
    prior_root = Path(prior_dir).resolve()
    latest_path = prior_root / "LATEST.json"
    if not latest_path.is_file():
        raise ValueError(f"missing prior LATEST.json in {prior_root}")
    pointer = read(latest_path)
    ckpt_dir = (prior_root / pointer["directory"]).resolve()
    if not ckpt_dir.is_relative_to(prior_root) or not ckpt_dir.is_dir():
        raise ValueError("prior checkpoint directory escapes prior_dir or does not exist")
    manifest_path = ckpt_dir / "manifest.json"
    actual_msha = digest(manifest_path)
    if actual_msha != pointer["sha256"]:
        raise ValueError(f"prior LATEST.json sha256 mismatch: {actual_msha} != {pointer['sha256']}")
    if expected_prior_manifest_sha256 is not None and actual_msha != expected_prior_manifest_sha256:
        raise ValueError(
            f"prior manifest digest differs from expected pin: {actual_msha} != {expected_prior_manifest_sha256}"
        )

    manifest = read(manifest_path)
    if manifest.get("schema") != expected_prior_schema:
        raise ValueError(f"unexpected prior checkpoint schema: {manifest.get('schema')} != {expected_prior_schema}")
    if int(manifest["step"]) != expected_prior_step:
        raise ValueError(f"prior checkpoint step {manifest['step']} != expected {expected_prior_step}")

    old_ident = manifest["identity"]
    if (prior_root / "IDENTITY.json").is_file():
        file_ident = read(prior_root / "IDENTITY.json")
        if file_ident != old_ident:
            raise ValueError("prior IDENTITY.json differs from checkpoint manifest identity")

    if old_ident.get("recipe") != RECIPE_SHA:
        raise ValueError(f"prior recipe hash mismatch: {old_ident.get('recipe')} != {RECIPE_SHA}")
    if old_ident.get("index") != expected_prior_index_sha256:
        raise ValueError(f"prior corpus index mismatch: {old_ident.get('index')} != {expected_prior_index_sha256}")
    if expected_prior_code_hashes:
        for k, v in expected_prior_code_hashes.items():
            if old_ident.get(k) != v:
                raise ValueError(f"prior identity field {k!r} mismatch: {old_ident.get(k)} != {v}")

    same_backend = old_ident.get("backend") == backend.identity.get("backend") and old_ident.get(
        "world_size"
    ) == backend.identity.get("world_size")
    if not same_backend and not allow_backend_handoff:
        raise ValueError(
            f"cross-backend handoff from {old_ident.get('backend')} to {backend.identity.get('backend')} "
            "rejected without --allow-backend-handoff"
        )

    state = _verify_and_load_shards(ckpt_dir, manifest, opt, backend)
    old_records = list(state["records"])
    if len(old_records) != expected_prior_step:
        raise ValueError(f"prior record count {len(old_records)} != expected step {expected_prior_step}")
    last_stop = int(old_records[-1]["stop"]) if old_records else 0
    if last_stop != expected_prior_positions:
        raise ValueError(f"prior absolute stop position {last_stop} != expected {expected_prior_positions}")

    # Capture prior cursor digest for lineage audit without mutating prior_dir
    old_cursor = state.get("cursor")
    old_cursor_sha = old_cursor.get("sha256") if isinstance(old_cursor, dict) and "sha256" in old_cursor else None

    handoff_receipt = {
        "schema": HANDOFF_RECEIPT_SCHEMA,
        "prior_checkpoint_directory": pointer["directory"],
        "prior_manifest_sha256": actual_msha,
        "prior_schema": manifest["schema"],
        "prior_identity": old_ident,
        "prior_cursor_sha256": old_cursor_sha,
        "prior_step": expected_prior_step,
        "prior_absolute_positions": expected_prior_positions,
        "recipe_sha256": RECIPE_SHA,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "continuation_runner_sha256": digest(Path(__file__)),
        "continuation_stream_sha256": digest(HERE / "stream.py"),
        "stage_contract_sha256": stage_contract_sha256,
        "new_stage_index_sha256": new_stage_index_sha256,
        "new_stage_replay_sha256": new_stage_replay_sha256,
        "optimizer_moments_preserved": True,
        "rng_state_preserved": True,
        "prior_records_preserved": len(old_records),
        "backend_transition": {
            "from_backend": old_ident.get("backend"),
            "from_hardware": old_ident.get("hardware"),
            "to_backend": backend.identity.get("backend"),
            "to_hardware": backend.identity.get("hardware"),
            "same_backend": same_backend,
        },
    }
    return handoff_receipt, old_records


def load_continuation_checkpoint(
    out: Path,
    expected_identity: dict[str, Any],
    net: torch.nn.Module,
    opt: FP32AdamW,
    backend: Backend,
) -> dict[str, Any]:
    """Strict fail-closed restore inside an active continuation workspace."""
    pointer = read(out / "LATEST.json")
    ckpt_dir = (out / pointer["directory"]).resolve()
    if not ckpt_dir.is_relative_to(out.resolve()):
        raise ValueError("checkpoint pointer escapes workspace")
    mpath = ckpt_dir / "manifest.json"
    if digest(mpath) != pointer["sha256"]:
        raise ValueError("continuation checkpoint manifest SHA-256 mismatch")
    manifest = read(mpath)
    if manifest.get("schema") != CONTINUATION_CHECKPOINT_SCHEMA:
        raise ValueError(f"unexpected continuation schema: {manifest.get('schema')}")
    if manifest.get("identity") != expected_identity:
        raise ValueError("continuation identity mismatch (refusing cross-topology or altered-corpus resume)")
    return _verify_and_load_shards(ckpt_dir, manifest, opt, backend)


def build_continuation_identity(
    *,
    stage_contract_sha256: str,
    stage_index_sha256: str,
    stage_replay_sha256: str,
    prior_manifest_sha256: str,
    backend: Backend,
    context: int = 2048,
) -> dict[str, Any]:
    return {
        "schema": CONTINUATION_CHECKPOINT_SCHEMA,
        "recipe": RECIPE_SHA,
        "tokenizer": TOKENIZER_SHA256,
        "stage_contract": stage_contract_sha256,
        "index": stage_index_sha256,
        "replay": stage_replay_sha256,
        "prior_manifest": prior_manifest_sha256,
        "runner": digest(Path(__file__)),
        "stream": digest(HERE / "stream.py"),
        "native": digest(TRAINING_ROOT / "aleph_rlt_placement" / "gym.py"),
        "packer": digest(TRAINING_ROOT / "aleph_rlt_placement" / "packing.py"),
        "xla_core": digest(ALEPH_NEXT / "candidate01" / "colab10m_xla.py"),
        "torch": str(torch.__version__),
        "microbatch": 1,
        "context": context,
        "precision": "explicit-bf16-dense-fp32-master",
        **backend.identity,
    }


def _train_one_group(
    net: torch.nn.Module,
    opt: FP32AdamW,
    backend: Backend,
    cfg: dict[str, Any],
    stream: StageMixedStream,
    expected_step_meta: dict[str, Any],
    dtype: torch.dtype,
    *,
    remat: bool = True,
    train_deadline: float | None = None,
) -> tuple[bool, dict[str, Any] | None]:
    """Execute exactly one verified update group with global-label-normalized gradients."""
    t0 = time.monotonic()
    packs, denom, meta = collect_stage_group(
        stream,
        int(expected_step_meta["stage_stop"]),
        expected_digest=expected_step_meta["sha256"],
        expected_labels=int(expected_step_meta["labels"]),
    )
    den = torch.tensor(float(denom), device=backend.device, dtype=torch.float32)
    opt.zero()
    lane_loss_tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
    lane_counts: dict[str, int] = defaultdict(int)
    micro_times: list[float] = []
    for lane, pack, k in packs:
        if train_deadline is not None and time.monotonic() >= train_deadline:
            opt.zero()
            return False, None
        ts = time.monotonic()
        ids, pos, seg, y = tensors(pack, backend.device, stream.context)
        numerator = loss_sum(logits(net, ids, pos, seg, dtype, remat), y)
        (numerator / den).backward()
        lane_loss_tensors[lane].append(numerator.detach())
        lane_counts[lane] += k
        backend.sync()
        micro_times.append(time.monotonic() - ts)

    abs_stop = int(meta["absolute_stop"])
    rate = compute_absolute_lr(cfg, abs_stop)
    grad_norm = opt.update(rate)
    lane_loss_sums = {lane: float(torch.stack(vals).sum().cpu()) for lane, vals in sorted(lane_loss_tensors.items())}
    total_loss = sum(lane_loss_sums.values()) / float(denom)
    if not math.isfinite(total_loss):
        raise FloatingPointError("nonfinite group training loss")
    dt = max(1e-6, time.monotonic() - t0)
    micro_sorted = sorted(micro_times)
    record = {
        "group": opt.step,
        "stage_group": meta["stage_group"],
        "stop": abs_stop,
        "stage_stop": meta["stage_stop"],
        "positions": meta["positions"],
        "labels": denom,
        "packs": meta["packs"],
        "padding": meta["padding"],
        "sha256": meta["sha256"],
        "loss": total_loss,
        "gradient_norm": grad_norm,
        "lr": rate,
        "lane_positions": meta["lane_positions"],
        "lane_loss_sums": lane_loss_sums,
        "lane_label_counts": dict(sorted(lane_counts.items())),
        "update_wall_seconds": dt,
        "first_microbatch_seconds": micro_times[0] if micro_times else 0.0,
        "median_microbatch_seconds": micro_sorted[len(micro_sorted) // 2] if micro_sorted else 0.0,
        "useful_positions_per_second": meta["positions"] / dt,
    }
    hbm = backend.memory()
    if hbm:
        record["hbm"] = hbm
    return True, record


def run_continuation_stage(args: argparse.Namespace, backend: Backend) -> dict[str, Any]:
    """Execute bounded continuation updates from a verified handoff or continuation checkpoint."""
    started = time.monotonic()
    if backend.kind not in ("xla", "cuda"):
        raise RuntimeError("full-size continuation requires --device cuda or --device xla")
    min_mem = 28_000_000_000 if backend.kind == "xla" else 38_000_000_000
    if backend.identity["memory_limit"] < min_mem:
        raise RuntimeError("need a 32 GB HBM TPU (v6e-1) or a 40 GB class CUDA accelerator (A100)")
    if backend.kind == "cuda":
        free, _ = torch.cuda.mem_get_info(backend.device)
        if free < 30 * 2**30:
            raise RuntimeError(f"need >= 30 GiB free HBM on A100; free={free}")
    if digest(RECIPE) != RECIPE_SHA:
        raise ValueError("frozen RECIPE.json digest differs")

    cfg = read(RECIPE)
    contract_path = Path(args.stage_contract).resolve()
    contract_sha = digest(contract_path)
    contract = read(contract_path)
    release_dir = Path(args.release_dir).resolve()
    index_path = release_dir / "index.json"
    replay_path = release_dir / "REPLAY.json"
    index_sha = digest(index_path)
    replay_sha = digest(replay_path)
    idx = read(index_path)
    replay = read(replay_path)

    if replay.get("schema") != STAGE_REPLAY_SCHEMA or replay.get("index_sha256") != index_sha:
        raise ValueError("REPLAY.json schema or index_sha256 mismatch")
    if int(idx["stage_start_positions"]) != int(contract["stage"]["start"]):
        raise ValueError("stage start position mismatch between index.json and stage contract")
    if int(idx["stage_stop_positions"]) != int(contract["stage"]["stop"]):
        raise ValueError("stage stop position mismatch between index.json and stage contract")
    if int(idx["stage_new_positions"]) != int(contract["stage"]["new_positions"]):
        raise ValueError("stage new_positions mismatch between index.json and stage contract")
    expected_bounds = stage_group_boundaries(
        int(contract["stage"]["new_positions"]),
        int(contract["stage"]["update_group_positions"]),
    )
    if len(replay["steps"]) != len(expected_bounds):
        raise ValueError(f"replay step count {len(replay['steps'])} != expected {len(expected_bounds)}")

    out = Path(args.out).resolve()
    if args.resume:
        if not (out / "LATEST.json").is_file() or not (out / "HANDOFF_RECEIPT.json").is_file():
            raise ValueError("continuation --resume requires existing LATEST.json and HANDOFF_RECEIPT.json")
    else:
        if not args.handoff_from:
            raise ValueError("fresh continuation workspace requires --handoff-from <prior_run_dir>")
        out.mkdir(parents=True, exist_ok=False)

    if shutil.disk_usage(out).free < 26_000_000_000:
        raise RuntimeError("need >= 26 GB free disk space for two 12 GB checkpoint generations + staging")

    parity = legacy_self_test(backend, torch.bfloat16)
    atomic_json(out / "HARDWARE_FIXTURE.json", parity)

    backend.seed(int(cfg["construction"]["seed"]))
    arch = cfg["construction"]
    net = Aleph(
        arch["width"],
        arch["vocab"],
        heads=arch["query_heads"],
        kv=arch["kv_heads"],
        core_ff=arch["core_ff"],
    )
    if sum(p.numel() for p in net.parameters()) != 999_352_320:
        raise ValueError("model unique parameter count differs from 999,352,320")
    if list(schedule(2, 1)) != [0, *range(1, 11), *range(1, 11), 11, 12, 13]:
        raise ValueError("Direct-A 24-visit schedule changed")
    net = net.to(backend.device)
    opt = FP32AdamW(net, cfg, backend)

    if args.resume:
        handoff_receipt = read(out / "HANDOFF_RECEIPT.json")
        identity = build_continuation_identity(
            stage_contract_sha256=contract_sha,
            stage_index_sha256=index_sha,
            stage_replay_sha256=replay_sha,
            prior_manifest_sha256=handoff_receipt["prior_manifest_sha256"],
            backend=backend,
        )
        state = load_continuation_checkpoint(out, identity, net, opt, backend)
        stage_cursor = state["cursor"]
        records = list(state["records"])
    else:
        prior_expected_step = int(contract["prior"]["optimizer_step"])
        prior_expected_pos = int(contract["prior"]["absolute_positions"])
        prior_code_hashes = (
            {k: PRIOR_10M_EXPECTED[k] for k in ("code", "cuda_reference", "native", "packer", "receipt")}
            if prior_expected_step == 39
            else None
        )
        handoff_receipt, records = verify_and_handoff_prior_checkpoint(
            prior_dir=Path(args.handoff_from),
            expected_prior_step=prior_expected_step,
            expected_prior_positions=prior_expected_pos,
            expected_prior_index_sha256=str(contract["prior"]["corpus_index_sha256"]),
            expected_prior_manifest_sha256=args.prior_manifest_sha256,
            new_stage_index_sha256=index_sha,
            new_stage_replay_sha256=replay_sha,
            stage_contract_sha256=contract_sha,
            net=net,
            opt=opt,
            backend=backend,
            allow_backend_handoff=bool(args.allow_backend_handoff),
            expected_prior_code_hashes=prior_code_hashes,
        )
        identity = build_continuation_identity(
            stage_contract_sha256=contract_sha,
            stage_index_sha256=index_sha,
            stage_replay_sha256=replay_sha,
            prior_manifest_sha256=handoff_receipt["prior_manifest_sha256"],
            backend=backend,
        )
        atomic_json(out / "HANDOFF_RECEIPT.json", handoff_receipt)
        atomic_json(out / "IDENTITY.json", identity)
        stage_cursor = None

    prior_step = int(handoff_receipt["prior_step"])
    completed_stage_groups = opt.step - prior_step
    if completed_stage_groups < 0 or completed_stage_groups > len(replay["steps"]):
        raise ValueError(f"invalid completed_stage_groups={completed_stage_groups}")
    target_stage_groups = min(len(replay["steps"]), completed_stage_groups + int(args.max_new_groups))
    if target_stage_groups <= completed_stage_groups:
        raise ValueError("stage already at or beyond requested group target")

    durable_step = opt.step
    last_cursor = stage_cursor
    train_deadline = started + int(args.wall_seconds) - int(args.checkpoint_reserve_seconds)

    with ExitStack() as stack:
        stream = StageMixedStream(release_dir, index_path, index_sha, stack, cursor=stage_cursor)
        if last_cursor is None:
            last_cursor = stream.cursor()
            if args.save_handoff_checkpoint:
                save_continuation_checkpoint(
                    out, identity, net, opt, backend, last_cursor, records, handoff_receipt=handoff_receipt
                )

        for sg_idx in range(completed_stage_groups, target_stage_groups):
            if time.monotonic() >= train_deadline:
                break
            expected_meta = replay["steps"][sg_idx]
            ok, rec = _train_one_group(
                net,
                opt,
                backend,
                cfg,
                stream,
                expected_meta,
                torch.bfloat16,
                remat=True,
                train_deadline=train_deadline,
            )
            if not ok or rec is None:
                break
            records.append(rec)
            last_cursor = stream.cursor()
            stage_done = sg_idx + 1 == len(replay["steps"])
            should_save = (
                sg_idx == completed_stage_groups
                or stage_done
                or (int(args.save_every_groups) > 0 and (sg_idx + 1) % int(args.save_every_groups) == 0)
            )
            if should_save:
                ts = time.monotonic()
                cdir, msha = save_continuation_checkpoint(
                    out, identity, net, opt, backend, last_cursor, records, handoff_receipt=handoff_receipt
                )
                durable_step = opt.step
                atomic_json(
                    out / f"CHECKPOINT_{opt.step:04d}.json",
                    {
                        "directory": cdir.name,
                        "manifest_sha256": msha,
                        "step": opt.step,
                        "absolute_positions": rec["stop"],
                        "wall_seconds": time.monotonic() - ts,
                    },
                )
            atomic_json(
                out / "PROGRESS.json",
                {
                    "schema": "aleph.direct-a.continuation-progress.v1",
                    "stage_contract_sha256": contract_sha,
                    "stage_index_sha256": index_sha,
                    "prior_step": prior_step,
                    "completed_step": opt.step,
                    "durable_step": durable_step,
                    "absolute_positions": int(records[-1]["stop"]),
                    "stage_complete": (opt.step - prior_step) == len(replay["steps"]),
                    "session_wall_seconds": time.monotonic() - started,
                    "records": records,
                },
            )

        if opt.step > durable_step:
            ts = time.monotonic()
            cdir, msha = save_continuation_checkpoint(
                out, identity, net, opt, backend, last_cursor, records, handoff_receipt=handoff_receipt
            )
            durable_step = opt.step
            atomic_json(
                out / f"CHECKPOINT_{opt.step:04d}.json",
                {
                    "directory": cdir.name,
                    "manifest_sha256": msha,
                    "step": opt.step,
                    "absolute_positions": int(records[-1]["stop"]),
                    "wall_seconds": time.monotonic() - ts,
                },
            )

    summary = {
        "schema": "aleph.direct-a.continuation-session-summary.v1",
        "backend": backend.identity,
        "durable_step": durable_step,
        "absolute_positions": int(records[-1]["stop"]) if records else 0,
        "stage_complete": (durable_step - prior_step) == len(replay["steps"]),
        "session_wall_seconds": time.monotonic() - started,
    }
    atomic_json(out / "SESSION_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_cpu_migration_self_test(*, use_bf16: bool = False) -> dict[str, Any]:
    """Comprehensive zero-credit CPU migration, schedule, handoff, and fail-closed verification."""
    cfg = read(RECIPE)
    # 1. Verify exact absolute LR schedule at 10M, 100M, 200M, 10B, 16B, 20B
    assert math.isclose(compute_absolute_lr(cfg, 10_000_000), 1.25e-5, rel_tol=1e-12)
    assert math.isclose(compute_absolute_lr(cfg, 100_000_000), 1.25e-4, rel_tol=1e-12)
    assert math.isclose(compute_absolute_lr(cfg, 200_000_000), 2.5e-4, rel_tol=1e-12)
    assert math.isclose(compute_absolute_lr(cfg, 10_000_000_000), 2.5e-4, rel_tol=1e-12)
    assert math.isclose(compute_absolute_lr(cfg, 16_000_000_000), 2.5e-4, rel_tol=1e-12)
    assert math.isclose(compute_absolute_lr(cfg, 20_000_000_000), 2.5e-5, rel_tol=1e-12)

    # 2. Verify exact 90M stage boundary arithmetic: 343 * 262144 + 84608 = 90,000,000
    b90 = stage_group_boundaries(90_000_000, 262_144)
    assert len(b90) == 344
    assert b90[-2] == 343 * 262_144
    assert b90[-1] == 90_000_000
    assert b90[-1] - b90[-2] == 84_608

    # 3. Run the frozen legacy BF16/FP32 self-test (document isolation, all-masked, denom, AdamW)
    backend = Backend("cpu")
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    base_res = legacy_self_test(backend, dtype)

    # 4. End-to-end synthetic prior -> handoff -> resume equality on miniature direct-A model
    packs = static_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_dir = root / "prior_10m"
        prior_dir.mkdir()
        cont_unint = root / "cont_uninterrupted"
        cont_unint.mkdir()
        cont_resumed = root / "cont_resumed"
        cont_resumed.mkdir()

        backend.seed(137)
        net_prior = mini_model()
        opt_prior = FP32AdamW(net_prior, cfg, backend)
        # Advance 1 step to build non-zero FP32 AdamW moments & RNG state
        from colab10m_xla import accumulate
        from colab10m_xla import save_checkpoint as legacy_save

        accumulate(net_prior, packs, backend, dtype)
        opt_prior.update(compute_absolute_lr(cfg, 10_000_000))
        prior_ident = {
            "recipe": RECIPE_SHA,
            "index": "a" * 64,
            "receipt": "b" * 64,
            "backend": "cpu",
            "world_size": 1,
        }
        atomic_json(prior_dir / "IDENTITY.json", prior_ident)
        legacy_save(
            prior_dir,
            prior_ident,
            net_prior,
            opt_prior,
            backend,
            {"sha256": "c" * 64},
            [{"group": 1, "stop": 10_000_000}],
        )

        # Uninterrupted 2-step trajectory directly after prior step 1
        net_a = copy.deepcopy(net_prior)
        opt_a = FP32AdamW(net_a, cfg, backend)
        receipt_a, recs_a = verify_and_handoff_prior_checkpoint(
            prior_dir=prior_dir,
            expected_prior_step=1,
            expected_prior_positions=10_000_000,
            expected_prior_index_sha256="a" * 64,
            expected_prior_manifest_sha256=None,
            new_stage_index_sha256="d" * 64,
            new_stage_replay_sha256="e" * 64,
            stage_contract_sha256="f" * 64,
            net=net_a,
            opt=opt_a,
            backend=backend,
        )
        assert opt_a.step == 1 and len(recs_a) == 1
        accumulate(net_a, packs, backend, dtype)
        opt_a.update(compute_absolute_lr(cfg, 10_262_144))
        accumulate(net_a, packs, backend, dtype)
        opt_a.update(compute_absolute_lr(cfg, 10_524_288))

        # Interrupted trajectory: handoff -> step 2 -> save -> restore -> step 3
        net_b = mini_model()
        opt_b = FP32AdamW(net_b, cfg, backend)
        receipt_b, recs_b = verify_and_handoff_prior_checkpoint(
            prior_dir=prior_dir,
            expected_prior_step=1,
            expected_prior_positions=10_000_000,
            expected_prior_index_sha256="a" * 64,
            expected_prior_manifest_sha256=receipt_a["prior_manifest_sha256"],
            new_stage_index_sha256="d" * 64,
            new_stage_replay_sha256="e" * 64,
            stage_contract_sha256="f" * 64,
            net=net_b,
            opt=opt_b,
            backend=backend,
        )
        ident_b = build_continuation_identity(
            stage_contract_sha256="f" * 64,
            stage_index_sha256="d" * 64,
            stage_replay_sha256="e" * 64,
            prior_manifest_sha256=receipt_b["prior_manifest_sha256"],
            backend=backend,
            context=8,
        )
        atomic_json(cont_resumed / "HANDOFF_RECEIPT.json", receipt_b)
        atomic_json(cont_resumed / "IDENTITY.json", ident_b)
        accumulate(net_b, packs, backend, dtype)
        opt_b.update(compute_absolute_lr(cfg, 10_262_144))
        recs_b.append({"group": 2, "stop": 10_262_144})
        save_continuation_checkpoint(
            cont_resumed, ident_b, net_b, opt_b, backend, {"stage_cursor": 1}, recs_b, handoff_receipt=receipt_b
        )

        # Fresh instance restores from cont_resumed and runs step 3
        net_c = mini_model()
        opt_c = FP32AdamW(net_c, cfg, backend)
        restored = load_continuation_checkpoint(cont_resumed, ident_b, net_c, opt_c, backend)
        assert restored["cursor"] == {"stage_cursor": 1} and opt_c.step == 2
        accumulate(net_c, packs, backend, dtype)
        opt_c.update(compute_absolute_lr(cfg, 10_524_288))

        for (n1, p1), (n2, p2) in zip(opt_a.named, opt_c.named, strict=True):
            assert n1 == n2
            torch.testing.assert_close(p1, p2, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(opt_a.moments[n1][0], opt_c.moments[n2][0], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(opt_a.moments[n1][1], opt_c.moments[n2][1], atol=1e-6, rtol=1e-6)

    result = {
        "schema": "aleph.direct-a.continuation-selftest.v1",
        "dtype": str(dtype),
        "legacy_fixture": base_res,
        "schedule_checks_passed": True,
        "stage_90m_boundaries": {"groups": 344, "full": 343, "final_partial": 84_608, "expected_abs_step": 383},
        "handoff_and_resume_parity_passed": True,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "xla", "xla-cpu"), default="cpu")
    parser.add_argument("--stage-contract", type=Path, default=HERE / "stage_100m.json")
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--handoff-from", type=Path)
    parser.add_argument("--prior-manifest-sha256", type=str, default=None)
    parser.add_argument("--allow-backend-handoff", action="store_true")
    parser.add_argument("--save-handoff-checkpoint", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-new-groups", type=int, default=1)
    parser.add_argument("--save-every-groups", type=int, default=20)
    parser.add_argument("--wall-seconds", type=int, default=1800)
    parser.add_argument("--checkpoint-reserve-seconds", type=int, default=240)
    args = parser.parse_args()

    torch.set_num_threads(1)
    if args.self_test:
        if args.device == "cpu":
            run_cpu_migration_self_test(use_bf16=args.bf16)
        else:
            b = Backend(args.device)
            legacy_self_test(b, torch.bfloat16)
        return 0

    if not (
        args.execute
        and args.device in ("cuda", "xla")
        and args.release_dir
        and args.out
        and 1 <= args.max_new_groups <= 38_148
        and 0 <= args.checkpoint_reserve_seconds < args.wall_seconds <= 43_200
    ):
        parser.error("explicit bounded --execute --device cuda|xla --release-dir --out --wall-seconds required")
    run_continuation_stage(args, Backend(args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
