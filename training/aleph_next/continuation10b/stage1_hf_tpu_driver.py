"""End-to-end 1x Google TPU v5e-1 Stage-100M driver backed by private Hugging Face Hub repos.

Eliminates Colab's broken ``drive.mount`` (HTTP 400 EnableCredentialPropagation on TPU v5e-1):
  1. Downloads immutable ``stage_100m`` + disjoint ``dev_panel`` from private HF dataset repo
     (``AlephFunk/aleph-10b-stage-100m-v1``) and verifies exact SHA-256 digests.
  2. Downloads immutable prior 10M native checkpoint (``prior_10m/group_039_7fbdb1f91cf8``) or
     resumes the latest verified continuation checkpoint from private HF model repo
     (``AlephFunk/aleph-10b-stage-100m-checkpoints``).
  3. Qualifies groups 40-41 uninterrupted -> uploads verified group 40 to HF -> restores group 40
     FROM HF into a fresh directory -> runs group 41 -> asserts matched restart parity and publishes
     ``QUALIFICATION_RECEIPT.json`` + group 41 (2 durable generations).
  4. Continues in resumable chunks (default 35 groups, ~25 min/chunk) to absolute step 383
     (100,000,000 cumulative positions), uploading every verified checkpoint generation to HF,
     publishing ``LATEST.json`` strictly last, and pruning generations beyond the newest 2.
  5. Evaluates 10M (step 39) vs 100M (step 383) on strictly disjoint ``dev_panel`` and publishes
     ``DEV_EVAL_10M_VS_100M.json`` to HF.

Authenticates exclusively via the ``HF_TOKEN`` environment variable (never logged or written to disk).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, snapshot_download

PRIOR_MANIFEST_SHA = "aca6d5c46301b6bedda375ad3f597dcff29aa4bd7fd1ed0c97677e678a9858ab"
STAGE_INDEX_SHA = "194a8b0febce37de7497c34b9d20c8f1e10879a1065fe297a8da5d4c607ae232"
STAGE_REPLAY_SHA = "41a18f99e0287057f253fc2b1e599fe538cebb396d35ce59d4e23f4e07fba7cd"
DEV_INDEX_SHA = "131cb6c80fae1a7063c58fa72a693fdbb92dc3b4dbd5c3084cbfd3bab3bd84bf"

HERE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(1 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def run(cmd: list[str]) -> None:
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def current_step(run_dir: Path) -> int:
    return int(json.loads((run_dir / "LATEST.json").read_text(encoding="utf-8"))["step"])


def hf_run_has_latest(api: HfApi, ckpt_repo: str, run_prefix: str) -> bool:
    files = set(api.list_repo_files(ckpt_repo, repo_type="model"))
    return f"{run_prefix}/LATEST.json" in files


def hf_download_subtree(
    repo_id: str,
    repo_type: str,
    subpath: str,
    local_root: Path,
    token: str,
) -> Path:
    t0 = time.monotonic()
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        allow_patterns=[f"{subpath}/**", subpath],
        local_dir=str(local_root),
        token=token,
    )
    target = local_root / subpath
    print(f"[hf-dl] downloaded {repo_id}/{subpath} -> {target} in {time.monotonic() - t0:.1f}s", flush=True)
    return target


def hf_download_latest_run_ckpt(
    api: HfApi,
    ckpt_repo: str,
    run_prefix: str,
    dst_run_dir: Path,
    token: str,
) -> dict[str, Any]:
    """Download top-level run metadata + ONLY the latest verified checkpoint directory from HF."""
    t0 = time.monotonic()
    dst_run_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = dst_run_dir.parent / f".hf_dl_{dst_run_dir.name}"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)
    try:
        snapshot_download(
            repo_id=ckpt_repo,
            repo_type="model",
            allow_patterns=[f"{run_prefix}/*.json"],
            local_dir=str(tmp_root),
            token=token,
        )
        run_src = tmp_root / run_prefix
        latest = json.loads((run_src / "LATEST.json").read_text(encoding="utf-8"))
        tag = latest["directory"]
        snapshot_download(
            repo_id=ckpt_repo,
            repo_type="model",
            allow_patterns=[f"{run_prefix}/{tag}/**"],
            local_dir=str(tmp_root),
            token=token,
        )
        # Verify every shard hash via drive_sync before moving into dst_run_dir
        sync_py = HERE / "drive_sync.py"
        run(
            [sys.executable, str(sync_py), "verify", str(run_src / tag), "--expected-manifest-sha256", latest["sha256"]]
        )
        shutil.copytree(run_src, dst_run_dir, dirs_exist_ok=True)
        print(
            f"[hf-restore] restored & verified {tag} (step {latest['step']}) in {time.monotonic() - t0:.1f}s",
            flush=True,
        )
        return latest
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def hf_publish_latest_generation(
    api: HfApi,
    ckpt_repo: str,
    run_prefix: str,
    local_run_dir: Path,
    *,
    keep_generations: int = 2,
    prune_older: bool = False,
) -> dict[str, Any]:
    """Verify every local shard hash, upload the latest checkpoint generation atomically, and optionally prune older."""
    t0 = time.monotonic()
    latest = json.loads((local_run_dir / "LATEST.json").read_text(encoding="utf-8"))
    tag = latest["directory"]
    step = int(latest["step"])
    msha = latest["sha256"]

    sync_py = HERE / "drive_sync.py"
    run([sys.executable, str(sync_py), "verify", str(local_run_dir / tag), "--expected-manifest-sha256", msha])

    # 1. Upload checkpoint shard directory first (LATEST.json is NOT touched until shards are committed!)
    api.upload_folder(
        repo_id=ckpt_repo,
        repo_type="model",
        folder_path=str(local_run_dir / tag),
        path_in_repo=f"{run_prefix}/{tag}",
        commit_message=f"Upload verified checkpoint {tag} (step {step})",
    )

    # 2. Atomically publish top-level metadata + LATEST.json strictly LAST in a single commit
    ops: list[Any] = []
    for meta_name in (
        "IDENTITY.json",
        "HANDOFF_RECEIPT.json",
        "HARDWARE_FIXTURE.json",
        "PROGRESS.json",
        "SESSION_SUMMARY.json",
        "QUALIFICATION_RECEIPT.json",
        "DEV_EVAL_10M_VS_100M.json",
        "LATEST.json",
    ):
        p = local_run_dir / meta_name
        if p.is_file():
            ops.append(CommitOperationAdd(path_in_repo=f"{run_prefix}/{meta_name}", path_or_fileobj=str(p)))
    for ckpt_meta in sorted(local_run_dir.glob("CHECKPOINT_*.json")):
        ops.append(CommitOperationAdd(path_in_repo=f"{run_prefix}/{ckpt_meta.name}", path_or_fileobj=str(ckpt_meta)))

    pruned_dirs: list[str] = []
    if prune_older and keep_generations >= 2:
        remote_files = api.list_repo_files(ckpt_repo, repo_type="model")
        prefix_slash = f"{run_prefix}/"
        gen_dirs: set[str] = set()
        for rf in remote_files:
            if rf.startswith(prefix_slash):
                rest = rf[len(prefix_slash) :]
                if "/" in rest:
                    top = rest.split("/", 1)[0]
                    if top.startswith("group_"):
                        gen_dirs.add(top)

        # Sort by parsed step descending (group_0383_...)
        def _parse_step(dname: str) -> tuple[int, str]:
            parts = dname.split("_")
            try:
                return (int(parts[1]), dname)
            except Exception:
                return (-1, dname)

        ordered = sorted(gen_dirs, key=_parse_step, reverse=True)
        for old_dir in ordered[keep_generations:]:
            if old_dir != tag:
                old_prefix = f"{run_prefix}/{old_dir}/"
                for rf in remote_files:
                    if rf.startswith(old_prefix):
                        ops.append(CommitOperationDelete(path_in_repo=rf))
                pruned_dirs.append(old_dir)

    api.create_commit(
        repo_id=ckpt_repo,
        repo_type="model",
        operations=ops,
        commit_message=f"Publish LATEST.json -> {tag} (step {step})",
    )
    dt = time.monotonic() - t0
    print(
        f"[hf-pub] published {tag} (step {step}, msha={msha[:12]}) in {dt:.1f}s pruned={pruned_dirs}",
        flush=True,
    )
    return {
        "directory": tag,
        "step": step,
        "manifest_sha256": msha,
        "wall_seconds": round(dt, 2),
        "pruned": pruned_dirs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo", type=str, default="AlephFunk/aleph-10b-stage-100m-v1")
    parser.add_argument("--ckpt-repo", type=str, default="AlephFunk/aleph-10b-stage-100m-checkpoints")
    parser.add_argument("--run-name", type=str, default="stage_100m_tpu_v5e1")
    parser.add_argument("--local-root", type=Path, default=Path("/content"))
    parser.add_argument("--device", choices=("tpu", "cuda", "xla"), default="tpu")
    parser.add_argument("--chunk-groups", type=int, default=35)
    parser.add_argument("--target-abs-step", type=int, default=383)
    parser.add_argument("--session-wall-seconds", type=int, default=21_600)
    parser.add_argument("--skip-dev-eval", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is required")
    api = HfApi(token=token)
    who = api.whoami()
    print(f"[hf-auth] authenticated as {who.get('name')}", flush=True)

    run_prefix = f"runs/{args.run_name}"
    local_hf_ds = args.local_root / "hf_stage_100m_ds"
    local_hf_prior = args.local_root / "hf_prior_10m"
    local_unint = args.local_root / "tpu_qual_unint"
    local_run = args.local_root / "tpu_stage100m_run"

    runner_py = HERE / "runner.py"
    eval_py = HERE / "eval_dev_panel.py"
    contract_json = HERE / "stage_100m.json"

    t_sess = time.monotonic()
    du = shutil.disk_usage(args.local_root)
    print(f"[disk] local_nvme free={du.free / 1e9:.2f} GB / total={du.total / 1e9:.2f} GB", flush=True)

    repo_head = "unknown"
    with contextlib.suppress(Exception):
        repo_head = subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip()
    print(f"[repo] HEAD={repo_head}", flush=True)

    # 1. Stage immutable stage_100m + dev_panel from private HF dataset repo
    local_release = local_hf_ds / "stage_100m"
    if not (local_release / "index.json").is_file():
        hf_download_subtree(args.dataset_repo, "dataset", "stage_100m", local_hf_ds, token)
    if sha256_file(local_release / "index.json") != STAGE_INDEX_SHA:
        raise ValueError("stage_100m/index.json SHA-256 mismatch")
    if sha256_file(local_release / "REPLAY.json") != STAGE_REPLAY_SHA:
        raise ValueError("stage_100m/REPLAY.json SHA-256 mismatch")
    print("[corpus] verified stage_100m index & REPLAY SHA-256", flush=True)

    local_prior = local_hf_prior / "prior_10m"

    # 2. Phase A: If no qualified checkpoint exists on HF yet, run groups 40-41 + HF-restored restart parity
    if not hf_run_has_latest(api, args.ckpt_repo, run_prefix):
        print("\n=== PHASE A: TPU v5e-1 Qualification (groups 40-41 + HF Restart Parity) ===", flush=True)
        if not (local_prior / "LATEST.json").is_file():
            hf_download_subtree(args.ckpt_repo, "model", "prior_10m", local_hf_prior, token)
        if json.loads((local_prior / "LATEST.json").read_text(encoding="utf-8"))["sha256"] != PRIOR_MANIFEST_SHA:
            raise ValueError("Prior 10M manifest SHA-256 mismatch")

        for d in (local_unint, local_run):
            if d.exists():
                shutil.rmtree(d)

        cmd_unint = [
            sys.executable,
            str(runner_py),
            "--execute",
            "--device",
            args.device,
            "--stage-contract",
            str(contract_json),
            "--release-dir",
            str(local_release),
            "--handoff-from",
            str(local_prior),
            "--prior-manifest-sha256",
            PRIOR_MANIFEST_SHA,
            "--out",
            str(local_unint),
            "--max-new-groups",
            "2",
            "--save-every-groups",
            "1",
            "--wall-seconds",
            "1800",
            "--checkpoint-reserve-seconds",
            "180",
        ]
        if args.device != "cuda":
            cmd_unint.append("--allow-backend-handoff")
        run(cmd_unint)

        unint_prog = json.loads((local_unint / "PROGRESS.json").read_text(encoding="utf-8"))
        rec40 = unint_prog["records"][-2]
        rec41_unint = unint_prog["records"][-1]
        ckpt40_meta = json.loads((local_unint / "CHECKPOINT_0040.json").read_text(encoding="utf-8"))

        # Stage group 40 into clean dir & publish to HF
        stage40 = args.local_root / "tpu_stage40_ptr"
        if stage40.exists():
            shutil.rmtree(stage40)
        stage40.mkdir()
        for fname in ("IDENTITY.json", "HANDOFF_RECEIPT.json", "HARDWARE_FIXTURE.json", "CHECKPOINT_0040.json"):
            shutil.copyfile(local_unint / fname, stage40 / fname)
        shutil.move(str(local_unint / ckpt40_meta["directory"]), str(stage40 / ckpt40_meta["directory"]))
        (stage40 / "LATEST.json").write_text(
            json.dumps(
                {"directory": ckpt40_meta["directory"], "sha256": ckpt40_meta["manifest_sha256"], "step": 40}, indent=2
            ),
            encoding="utf-8",
        )
        hf_publish_latest_generation(api, args.ckpt_repo, run_prefix, stage40, keep_generations=2, prune_older=False)

        # Restore group 40 FROM durable HF into fresh local_run and run group 41
        hf_download_latest_run_ckpt(api, args.ckpt_repo, run_prefix, local_run, token)
        run(
            [
                sys.executable,
                str(runner_py),
                "--execute",
                "--device",
                args.device,
                "--stage-contract",
                str(contract_json),
                "--release-dir",
                str(local_release),
                "--resume",
                "--out",
                str(local_run),
                "--max-new-groups",
                "1",
                "--save-every-groups",
                "1",
                "--wall-seconds",
                "1800",
                "--checkpoint-reserve-seconds",
                "180",
            ]
        )
        res_prog = json.loads((local_run / "PROGRESS.json").read_text(encoding="utf-8"))
        rec41_res = res_prog["records"][-1]

        loss_diff = abs(rec41_unint["loss"] - rec41_res["loss"])
        grad_diff = abs(rec41_unint["gradient_norm"] - rec41_res["gradient_norm"])
        parity_ok = (
            rec41_unint["stop"] == rec41_res["stop"]
            and rec41_unint["positions"] == rec41_res["positions"]
            and rec41_unint["labels"] == rec41_res["labels"]
            and rec41_unint["sha256"] == rec41_res["sha256"]
            and rec41_unint["lr"] == rec41_res["lr"]
            and rec41_unint["lane_label_counts"] == rec41_res["lane_label_counts"]
            and loss_diff <= 1e-4
            and grad_diff <= 1e-3
        )
        print(
            f"[parity] unint_loss={rec41_unint['loss']:.8f} res_loss={rec41_res['loss']:.8f} diff={loss_diff:.2e} | "
            f"unint_gnorm={rec41_unint['gradient_norm']:.6f} res_gnorm={rec41_res['gradient_norm']:.6f} "
            f"diff={grad_diff:.2e} | PARITY_OK={parity_ok}",
            flush=True,
        )
        if not parity_ok:
            raise RuntimeError("HF-restored TPU v5e-1 matched restart parity failed!")

        qual_receipt = {
            "schema": "aleph.direct-a.tpu-v5e1-stage100m-qualification.v1",
            "repo_commit": repo_head,
            "device": args.device,
            "durable_store": f"hf://models/{args.ckpt_repo}/{run_prefix}",
            "prior_manifest_sha256": PRIOR_MANIFEST_SHA,
            "stage_index_sha256": STAGE_INDEX_SHA,
            "stage_replay_sha256": STAGE_REPLAY_SHA,
            "group_40": rec40,
            "group_41_uninterrupted": rec41_unint,
            "group_41_restored": rec41_res,
            "parity": {"passed": parity_ok, "loss_abs_diff": loss_diff, "grad_norm_abs_diff": grad_diff},
        }
        (local_run / "QUALIFICATION_RECEIPT.json").write_text(json.dumps(qual_receipt, indent=2), encoding="utf-8")
        hf_publish_latest_generation(api, args.ckpt_repo, run_prefix, local_run, keep_generations=2, prune_older=False)
        shutil.rmtree(local_unint, ignore_errors=True)
        shutil.rmtree(stage40, ignore_errors=True)
        print("[qual-go] TPU v5e-1 qualified! Proceeding immediately to Stage-100M continuation...", flush=True)
    elif not (local_run / "LATEST.json").is_file():
        hf_download_latest_run_ckpt(api, args.ckpt_repo, run_prefix, local_run, token)

    # 3. Phase B: Capped resumable continuation to step 383 (100,000,000 positions)
    while True:
        step = current_step(local_run)
        if step >= args.target_abs_step:
            print(f"[STAGE-100M COMPLETE] step={step} (100,000,000 cumulative positions on TPU v5e-1)!", flush=True)
            break
        elapsed = time.monotonic() - t_sess
        rem_wall = int(args.session_wall_seconds - elapsed)
        if rem_wall < 600:
            print(f"[CAP] stopping cleanly before session wall cap at step={step}", flush=True)
            break
        chunk = min(args.chunk_groups, args.target_abs_step - step)
        chunk_wall = min(rem_wall, max(1200, chunk * 80 + 480))
        print(f"\n=== TPU v5e-1 Chunk: step {step} -> {step + chunk} (elapsed={elapsed / 60:.1f}m) ===", flush=True)
        run(
            [
                sys.executable,
                str(runner_py),
                "--execute",
                "--device",
                args.device,
                "--stage-contract",
                str(contract_json),
                "--release-dir",
                str(local_release),
                "--resume",
                "--out",
                str(local_run),
                "--max-new-groups",
                str(chunk),
                "--save-every-groups",
                str(chunk),
                "--wall-seconds",
                str(chunk_wall),
                "--checkpoint-reserve-seconds",
                "240",
            ]
        )
        hf_publish_latest_generation(api, args.ckpt_repo, run_prefix, local_run, keep_generations=2, prune_older=True)
        gens = []
        for child in local_run.iterdir():
            if child.is_dir() and child.name.startswith("group_") and (child / "manifest.json").is_file():
                m = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
                gens.append((int(m["step"]), child))
        gens.sort(key=lambda it: it[0], reverse=True)
        for _, old_dir in gens[2:]:
            shutil.rmtree(old_dir, ignore_errors=True)
        prog = json.loads((local_run / "PROGRESS.json").read_text(encoding="utf-8"))
        lr_rec = prog["records"][-1]
        print(
            f"[tpu-telemetry] step={lr_rec['group']} stop={lr_rec['stop']:,} loss={lr_rec['loss']:.6f} "
            f"gnorm={lr_rec['gradient_norm']:.4f} lr={lr_rec['lr']:.6e} "
            f"pos/s={lr_rec['useful_positions_per_second']:.1f}",
            flush=True,
        )

    # 4. Disjoint dev_panel evaluation (10M step 39 vs 100M step 383) when complete
    if not args.skip_dev_eval and current_step(local_run) >= args.target_abs_step:
        print("\n=== PHASE C: Disjoint dev_panel Evaluation (10M step 39 vs 100M step 383) ===", flush=True)
        local_dev = local_hf_ds / "dev_panel"
        if not (local_dev / "dev_index.json").is_file():
            hf_download_subtree(args.dataset_repo, "dataset", "dev_panel", local_hf_ds, token)
        if sha256_file(local_dev / "dev_index.json") != DEV_INDEX_SHA:
            raise ValueError("dev_panel/dev_index.json SHA-256 mismatch")
        if not (local_prior / "LATEST.json").is_file():
            hf_download_subtree(args.ckpt_repo, "model", "prior_10m", local_hf_prior, token)
        eval_out = local_run / "DEV_EVAL_10M_VS_100M.json"
        run(
            [
                sys.executable,
                str(eval_py),
                "--device",
                args.device,
                "--prior-dir",
                str(local_prior),
                "--run-dir",
                str(local_run),
                "--dev-dir",
                str(local_dev),
                "--out-json",
                str(eval_out),
            ]
        )
        api.upload_file(
            path_or_fileobj=str(eval_out),
            path_in_repo=f"{run_prefix}/DEV_EVAL_10M_VS_100M.json",
            repo_id=args.ckpt_repo,
            repo_type="model",
            commit_message="Upload disjoint dev_panel 10M vs 100M evaluation receipt",
        )
        print("[eval-pub] published DEV_EVAL_10M_VS_100M.json to HF!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
