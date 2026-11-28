"""Independent fixed development-panel evaluation comparing 10M (step 39) vs 100M (step 383).

Evaluates strictly on the disjoint held-out ``dev_panel`` (``split="dev"``, 5% reserved root
families never exposed to training) using identical BF16 static-shape Direct-A math on either
Google Cloud TPU (``--device tpu``) or CUDA (``--device cuda``).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(
    (parent for parent in (HERE, *HERE.parents) if (parent / "training" / "aleph_next" / "candidate01").is_dir()),
    HERE.parent.parent,
)
for p in (
    str(HERE),
    str(REPO_ROOT / "training/aleph_next/candidate01"),
    str(REPO_ROOT / "training/aleph_rlt_placement"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from colab10m import RECIPE  # noqa: E402
from colab10m_xla import Backend, FP32AdamW, logits, loss_sum, tensors  # noqa: E402
from packing import PackedStream  # noqa: E402
from production import digest, read  # noqa: E402
from restart import Aleph  # noqa: E402
from runner import _verify_and_load_shards  # noqa: E402
from tpu_backend import (  # noqa: E402
    HostOffloadFP32AdamW,
    TorchTPUBackend,
    make_tpu_loss_fn,
    prepare_tpu_working_model,
)


def eval_checkpoint_on_dev(
    label: str,
    run_dir: Path,
    dev_dir: Path,
    dev_index: dict[str, Any],
    net: Aleph,
    opt: Any,
    backend: Any,
    *,
    tpu_loss_fn: Any = None,
    max_packs_per_lane: int = 128,
) -> dict[str, Any]:
    pointer = read(run_dir / "LATEST.json")
    ckpt_dir = (run_dir / pointer["directory"]).resolve()
    manifest = read(ckpt_dir / "manifest.json")
    _verify_and_load_shards(ckpt_dir, manifest, opt, backend)
    net.eval()

    per_lane: dict[str, Any] = {}
    total_loss_sum = 0.0
    total_labels = 0
    total_positions = 0
    t0 = time.monotonic()

    with torch.no_grad():
        for lane, meta in sorted(dev_index["lanes"].items()):
            mpath = dev_dir / meta["manifest_path"]
            msha = meta["manifest_sha256"]
            lane_loss = 0.0
            lane_labels = 0
            lane_pos = 0
            lane_packs = 0
            stop_cap = min(int(meta["available_positions"]), max_packs_per_lane * 2048)
            with PackedStream(mpath, msha, "dev") as stream:
                while lane_pos < stop_cap:
                    pack = stream.next_pack(stop_cap)
                    if pack is None:
                        break
                    ids, pos, seg, y = tensors(pack, backend.device, 2048)
                    val = (
                        tpu_loss_fn(ids, pos, seg, y)
                        if tpu_loss_fn is not None
                        else loss_sum(logits(net, ids, pos, seg, torch.bfloat16, remat=False), y)
                    )
                    k = sum(1 for v in pack["labels_at_query"] if v != -100)
                    lane_loss += float(val.detach().cpu())
                    lane_labels += k
                    lane_pos += int(pack["nonpadding"])
                    lane_packs += 1
            nll = lane_loss / max(1, lane_labels)
            per_lane[lane] = {
                "packs": lane_packs,
                "positions": lane_pos,
                "labels": lane_labels,
                "nll_nats": round(nll, 6),
                "perplexity": round(math.exp(min(nll, 20.0)), 4),
            }
            total_loss_sum += lane_loss
            total_labels += lane_labels
            total_positions += lane_pos

    overall_nll = total_loss_sum / max(1, total_labels)
    res = {
        "label": label,
        "checkpoint_directory": pointer["directory"],
        "manifest_sha256": pointer["sha256"],
        "step": int(manifest["step"]),
        "overall_nll_nats": round(overall_nll, 6),
        "overall_perplexity": round(math.exp(min(overall_nll, 20.0)), 4),
        "total_positions": total_positions,
        "total_labels": total_labels,
        "per_lane": per_lane,
        "wall_seconds": round(time.monotonic() - t0, 2),
    }
    print(
        f"[eval:{label}] step={res['step']} nll={res['overall_nll_nats']:.6f} ppl={res['overall_perplexity']:.4f}",
        flush=True,
    )
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive"))
    parser.add_argument("--local-root", type=Path, default=Path("/content"))
    parser.add_argument("--device", choices=("tpu", "cuda"), default="tpu")
    parser.add_argument("--run-name", type=str, default="stage_100m_tpu_v5e1")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--prior-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--dev-dir", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    if args.prior_dir and args.run_dir and args.dev_dir:
        local_prior = args.prior_dir.resolve()
        local_run_out = args.run_dir.resolve()
        local_dev = args.dev_dir.resolve()
        out_path = (args.out_json or (local_run_out / "DEV_EVAL_10M_VS_100M.json")).resolve()
    else:
        drive_prior = args.drive_root / "aleph-v2-run"
        drive_dev = args.drive_root / "aleph-10b-continuation-v1" / "dev_panel"
        drive_run_out = args.drive_root / "aleph-10b-continuation-v1" / "runs" / args.run_name

        local_dev = args.local_root / "stage_100m_dev_panel"
        local_prior = args.local_root / "stage_prior_10m"
        local_run_out = args.local_root / "tpu_stage100m_run"
        sync_py = HERE / "drive_sync.py"
        out_path = (args.out_json or (drive_run_out / "DEV_EVAL_10M_VS_100M.json")).resolve()

        if not (local_dev / "dev_index.json").is_file():
            shutil.copytree(drive_dev, local_dev, dirs_exist_ok=True)
        if not (local_prior / "LATEST.json").is_file():
            subprocess.run(
                [
                    sys.executable,
                    str(sync_py),
                    "sync",
                    str(drive_prior),
                    str(local_prior),
                    "--workers",
                    str(args.workers),
                ],
                check=True,
            )
        if not (local_run_out / "LATEST.json").is_file():
            subprocess.run(
                [
                    sys.executable,
                    str(sync_py),
                    "sync",
                    str(drive_run_out),
                    str(local_run_out),
                    "--workers",
                    str(args.workers),
                ],
                check=True,
            )

    dev_idx_sha = digest(local_dev / "dev_index.json")
    dev_index = read(local_dev / "dev_index.json")

    cfg = read(RECIPE)
    arch = cfg["construction"]
    net = Aleph(arch["width"], arch["vocab"], heads=arch["query_heads"], kv=arch["kv_heads"], core_ff=arch["core_ff"])
    if args.device == "tpu":
        backend: Any = TorchTPUBackend()
        prepare_tpu_working_model(net, backend.device)
        opt: Any = HostOffloadFP32AdamW(net, cfg, backend)
        tpu_loss_fn: Any = make_tpu_loss_fn(net, compile_tpu=True, remat=False)
    else:
        backend = Backend("cuda")
        net = net.to(backend.device)
        opt = FP32AdamW(net, cfg, backend)
        tpu_loss_fn = None

    eval_10m = eval_checkpoint_on_dev(
        "aleph_10m_step039", local_prior, local_dev, dev_index, net, opt, backend, tpu_loss_fn=tpu_loss_fn
    )
    eval_100m = eval_checkpoint_on_dev(
        "aleph_100m_step383", local_run_out, local_dev, dev_index, net, opt, backend, tpu_loss_fn=tpu_loss_fn
    )

    delta_nll = round(eval_100m["overall_nll_nats"] - eval_10m["overall_nll_nats"], 6)
    per_lane_delta = {
        lane: round(eval_100m["per_lane"][lane]["nll_nats"] - eval_10m["per_lane"][lane]["nll_nats"], 6)
        for lane in sorted(eval_10m["per_lane"])
    }
    report = {
        "schema": "aleph.direct-a.stage100m-dev-eval.v1",
        "dev_panel_index_sha256": dev_idx_sha,
        "hardware": backend.identity,
        "baseline_10m": eval_10m,
        "continued_100m": eval_100m,
        "overall_nll_delta_nats": delta_nll,
        "per_lane_nll_delta_nats": per_lane_delta,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== DEV_EVAL_10M_VS_100M ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
