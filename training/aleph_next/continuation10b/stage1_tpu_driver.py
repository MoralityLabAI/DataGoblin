"""End-to-end 1x Google TPU v5e-1 Qualification, Matched Restart Parity & Stage-100M Continuation driver.

Executes ``train/aleph_continuation10b/runner.py`` on Colab TPU v5e-1 (3.26 CU/hr, ~3,275 pos/s,
~24.9 CU for 90M tokens vs 45.1 CU on A100):
  1. Verifies Drive read/write + exact Stage-100M immutable release SHA-256 hashes.
  2. Stages read-only prior 10M A100 checkpoint (group_039_7fbdb1f91cf8) and Stage-100M release to local NVMe.
  3. Runs group 40 + group 41 uninterrupted via ``--device tpu --allow-backend-handoff``, saving group 40 mid-run.
  4. Syncs verified group 40 checkpoint to durable Drive (``.../runs/stage_100m_tpu_v5e1``).
  5. Restores group 40 from durable Drive into a fresh directory and runs group 41 in a fresh process.
  6. Asserts matched restart parity (loss, grad norm, lane labels, LR, stop, sha256) -> writes
     ``QUALIFICATION_RECEIPT.json`` to Drive and syncs verified group 41 (2 durable generations).
  7. Continues in capped resumable chunks to absolute step 383 (100,000,000 positions), syncing
     verified checkpoints to Drive every chunk and retaining 2 durable generations.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PRIOR_MANIFEST_SHA = "aca6d5c46301b6bedda375ad3f597dcff29aa4bd7fd1ed0c97677e678a9858ab"
STAGE_INDEX_SHA = "194a8b0febce37de7497c34b9d20c8f1e10879a1065fe297a8da5d4c607ae232"
STAGE_REPLAY_SHA = "41a18f99e0287057f253fc2b1e599fe538cebb396d35ce59d4e23f4e07fba7cd"

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive"))
    parser.add_argument("--local-root", type=Path, default=Path("/content"))
    parser.add_argument("--device", choices=("tpu", "cuda", "xla"), default="tpu")
    parser.add_argument("--run-name", type=str, default="stage_100m_tpu_v5e1")
    parser.add_argument("--chunk-groups", type=int, default=25)
    parser.add_argument("--target-abs-step", type=int, default=383)
    parser.add_argument("--session-wall-seconds", type=int, default=32_400)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    drive_prior = args.drive_root / "aleph-v2-run"
    drive_cont_base = args.drive_root / "aleph-10b-continuation-v1"
    drive_release = drive_cont_base / "stage_100m"
    drive_run_out = drive_cont_base / "runs" / args.run_name

    local_prior = args.local_root / "stage_prior_10m"
    local_release = args.local_root / "stage_100m_release"
    local_unint = args.local_root / "tpu_qual_unint"
    local_run = args.local_root / "tpu_stage100m_run"

    runner_py = HERE / "runner.py"
    sync_py = HERE / "drive_sync.py"
    contract_json = HERE / "stage_100m.json"

    t_sess = time.monotonic()
    for label, p in [("local_nvme", args.local_root), ("google_drive", args.drive_root)]:
        du = shutil.disk_usage(p)
        print(f"[disk] {label}: free={du.free / 1e9:.2f} GB / total={du.total / 1e9:.2f} GB", flush=True)

    if not (drive_prior / "LATEST.json").is_file():
        raise FileNotFoundError(f"Missing {drive_prior / 'LATEST.json'}")
    if not (drive_release / "index.json").is_file():
        raise FileNotFoundError(f"Missing {drive_release / 'index.json'}")

    drive_run_out.mkdir(parents=True, exist_ok=True)
    probe = drive_run_out / ".rw_probe.tmp"
    probe.write_text("aleph-tpu-v5e1-ok", encoding="utf-8")
    if probe.read_text(encoding="utf-8") != "aleph-tpu-v5e1-ok":
        raise RuntimeError("Drive read/write probe mismatch")
    probe.unlink()
    print(f"[drive-rw] verified durable read/write at {drive_run_out}", flush=True)

    repo_head = "unknown"
    with contextlib.suppress(Exception):
        repo_head = subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip()
    print(f"[repo] HEAD={repo_head}", flush=True)

    if not (local_release / "index.json").is_file():
        ts = time.monotonic()
        shutil.copytree(drive_release, local_release, dirs_exist_ok=True)
        print(f"[release] copied 1.03 GB release to NVMe in {time.monotonic() - ts:.1f}s", flush=True)
    if sha256_file(local_release / "index.json") != STAGE_INDEX_SHA:
        raise ValueError("Stage-100M index.json SHA-256 mismatch")
    if sha256_file(local_release / "REPLAY.json") != STAGE_REPLAY_SHA:
        raise ValueError("Stage-100M REPLAY.json SHA-256 mismatch")

    # Phase A: If no qualified Drive checkpoint exists yet, run 10M -> 40/41 qualification + Drive restart parity
    if not (drive_run_out / "LATEST.json").is_file():
        print("\n=== PHASE A: Qualification (groups 40-41 + Drive Restart Parity) ===", flush=True)
        run([sys.executable, str(sync_py), "sync", str(drive_prior), str(local_prior), "--workers", str(args.workers)])
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

        # Stage group 40 cleanly inside its own directory so drive_sync path confinement passes
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
        run([sys.executable, str(sync_py), "sync", str(stage40), str(drive_run_out), "--workers", str(args.workers)])

        # Restore group 40 FROM durable Drive into fresh local_run and run group 41
        run([sys.executable, str(sync_py), "sync", str(drive_run_out), str(local_run), "--workers", str(args.workers)])
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
            raise RuntimeError("Matched restart parity failed!")

        run(
            [
                sys.executable,
                str(sync_py),
                "sync",
                str(local_run),
                str(drive_run_out),
                "--workers",
                str(args.workers),
                "--keep-generations",
                "2",
            ]
        )
        qual_receipt = {
            "schema": "aleph.direct-a.tpu-v5e1-stage100m-qualification.v1",
            "repo_commit": repo_head,
            "device": args.device,
            "prior_manifest_sha256": PRIOR_MANIFEST_SHA,
            "stage_index_sha256": STAGE_INDEX_SHA,
            "stage_replay_sha256": STAGE_REPLAY_SHA,
            "group_40": rec40,
            "group_41_uninterrupted": rec41_unint,
            "group_41_drive_restored": rec41_res,
            "parity": {"passed": parity_ok, "loss_abs_diff": loss_diff, "grad_norm_abs_diff": grad_diff},
        }
        (drive_run_out / "QUALIFICATION_RECEIPT.json").write_text(json.dumps(qual_receipt, indent=2), encoding="utf-8")
        shutil.rmtree(local_unint, ignore_errors=True)
        shutil.rmtree(stage40, ignore_errors=True)
        print("[qual-go] Qualification PASSED! Proceeding to Stage-100M continuation...", flush=True)
    elif not (local_run / "LATEST.json").is_file():
        run([sys.executable, str(sync_py), "sync", str(drive_run_out), str(local_run), "--workers", str(args.workers)])

    # Phase B: Capped resumable continuation to step 383 (100,000,000 positions)
    while True:
        step = current_step(local_run)
        if step >= args.target_abs_step:
            print(f"[STAGE-100M COMPLETE] step={step} (100,000,000 cumulative positions)!", flush=True)
            break
        elapsed = time.monotonic() - t_sess
        rem_wall = int(args.session_wall_seconds - elapsed)
        if rem_wall < 600:
            print(f"[CAP] stopping cleanly before session wall cap at step={step}", flush=True)
            break
        chunk = min(args.chunk_groups, args.target_abs_step - step)
        chunk_wall = min(rem_wall, max(1200, chunk * 120 + 480))
        print(f"\n=== Chunk: step {step} -> {step + chunk} (elapsed={elapsed / 60:.1f}m) ===", flush=True)
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
        run(
            [
                sys.executable,
                str(sync_py),
                "sync",
                str(local_run),
                str(drive_run_out),
                "--workers",
                str(args.workers),
                "--keep-generations",
                "2",
                "--prune-older",
            ]
        )
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
            f"[telemetry] step={lr_rec['group']} stop={lr_rec['stop']:,} loss={lr_rec['loss']:.6f} "
            f"gnorm={lr_rec['gradient_norm']:.4f} lr={lr_rec['lr']:.6e} "
            f"pos/s={lr_rec['useful_positions_per_second']:.1f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
