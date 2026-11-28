"""Hash-verified durable Drive/local staging and atomic checkpoint publication.

Supports parallel copying of 128-shard FP32 checkpoints (12 GB each) between fast
ephemeral Colab local NVMe and mounted Google Drive, verifying every shard SHA-256
before atomically publishing ``LATEST.json`` last and retaining at least two good
durable checkpoint generations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _copy_and_verify_one(src: Path, dst: Path, expected_sha256: str) -> tuple[str, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and sha256_file(dst) == expected_sha256:
        return dst.name, dst.stat().st_size
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copyfile(src, tmp)
    actual = sha256_file(tmp)
    if actual != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 mismatch after copying {src.name}: {actual} != {expected_sha256}")
    os.replace(tmp, dst)
    return dst.name, dst.stat().st_size


def _state_entry(manifest: dict[str, Any]) -> tuple[str, str]:
    if "state" in manifest and isinstance(manifest["state"], dict):
        return manifest["state"]["file"], manifest["state"]["sha256"]
    if "state_sha256" in manifest:
        return "state.pt", manifest["state_sha256"]
    raise ValueError("checkpoint manifest lacks state digest")


def verify_checkpoint_dir(ckpt_dir: Path, expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    """Verify every shard, state.pt, and manifest.json inside a native checkpoint directory."""
    cdir = Path(ckpt_dir).resolve()
    mpath = cdir / "manifest.json"
    if not mpath.is_file():
        raise FileNotFoundError(f"missing manifest.json in {cdir}")
    msha = sha256_file(mpath)
    if expected_manifest_sha256 is not None and msha != expected_manifest_sha256:
        raise ValueError(f"manifest.json SHA-256 mismatch: {msha} != {expected_manifest_sha256}")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    files = [(item["file"], item["sha256"]) for item in manifest["parameters"]]
    files.append(_state_entry(manifest))
    total_bytes = 0
    for fname, expected in files:
        fpath = (cdir / fname).resolve()
        if not fpath.is_relative_to(cdir) or not fpath.is_file():
            raise FileNotFoundError(f"missing checkpoint member {fname}")
        actual = sha256_file(fpath)
        if actual != expected:
            raise ValueError(f"corrupt checkpoint member {fname}: {actual} != {expected}")
        total_bytes += fpath.stat().st_size
    return {
        "directory": cdir.name,
        "manifest_sha256": msha,
        "step": int(manifest["step"]),
        "schema": manifest.get("schema"),
        "file_count": len(files) + 1,
        "total_bytes": total_bytes + mpath.stat().st_size,
    }


def sync_checkpoint_generation(
    src_run_dir: Path,
    dst_run_dir: Path,
    *,
    workers: int = 16,
    keep_generations: int = 2,
    prune_older: bool = False,
) -> dict[str, Any]:
    """Copy the latest checkpoint generation from src_run_dir to dst_run_dir and publish LATEST.json last."""
    t0 = time.monotonic()
    src_root = Path(src_run_dir).resolve()
    dst_root = Path(dst_run_dir).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    latest = json.loads((src_root / "LATEST.json").read_text(encoding="utf-8"))
    tag = latest["directory"]
    msha_expected = latest["sha256"]
    src_ckpt = (src_root / tag).resolve()
    if not src_ckpt.is_relative_to(src_root) or not src_ckpt.is_dir():
        raise ValueError(f"invalid LATEST.json directory: {tag}")
    if sha256_file(src_ckpt / "manifest.json") != msha_expected:
        raise ValueError("source manifest.json does not match LATEST.json sha256")
    manifest = json.loads((src_ckpt / "manifest.json").read_text(encoding="utf-8"))

    # Copy top-level immutable metadata first (never LATEST.json until shards verified)
    for meta_name in (
        "IDENTITY.json",
        "HANDOFF_RECEIPT.json",
        "HARDWARE_FIXTURE.json",
        "PROGRESS.json",
        "SESSION_SUMMARY.json",
    ):
        sp = src_root / meta_name
        if sp.is_file():
            _copy_and_verify_one(sp, dst_root / meta_name, sha256_file(sp))
    for ckpt_meta in sorted(src_root.glob("CHECKPOINT_*.json")):
        _copy_and_verify_one(ckpt_meta, dst_root / ckpt_meta.name, sha256_file(ckpt_meta))

    dst_ckpt = dst_root / tag
    dst_partial = dst_root / (tag + ".partial")
    target_dir = dst_ckpt if dst_ckpt.is_dir() else dst_partial
    target_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[Path, Path, str]] = [
        (src_ckpt / item["file"], target_dir / item["file"], item["sha256"]) for item in manifest["parameters"]
    ]
    st_file, st_sha = _state_entry(manifest)
    jobs.append((src_ckpt / st_file, target_dir / st_file, st_sha))
    jobs.append((src_ckpt / "manifest.json", target_dir / "manifest.json", msha_expected))

    copied_bytes = 0
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 32))) as pool:
        futs = [pool.submit(_copy_and_verify_one, s, d, h) for s, d, h in jobs]
        for fut in as_completed(futs):
            _, nbytes = fut.result()
            copied_bytes += nbytes

    if target_dir is dst_partial:
        os.replace(dst_partial, dst_ckpt)

    # Publish LATEST.json strictly LAST after every shard & manifest has been verified on durable storage
    atomic_write_json(dst_root / "LATEST.json", latest)

    pruned: list[str] = []
    if prune_older and keep_generations >= 2:
        # Never prune group_039 (or any directory without a valid manifest), keep newest keep_generations
        gens: list[tuple[int, Path]] = []
        for child in dst_root.iterdir():
            if child.is_dir() and child.name.startswith("group_") and (child / "manifest.json").is_file():
                try:
                    m = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
                    gens.append((int(m["step"]), child))
                except Exception:
                    continue
        gens.sort(key=lambda item: (item[0], item[1].name), reverse=True)
        for _, old_dir in gens[keep_generations:]:
            if old_dir.name != tag:
                shutil.rmtree(old_dir)
                pruned.append(old_dir.name)

    return {
        "schema": "aleph.direct-a.drive-sync-receipt.v1",
        "directory": tag,
        "step": int(manifest["step"]),
        "manifest_sha256": msha_expected,
        "files_verified": len(jobs),
        "bytes_verified": copied_bytes,
        "wall_seconds": round(time.monotonic() - t0, 3),
        "pruned_generations": pruned,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ver = sub.add_parser("verify", help="Verify all SHA-256 digests in a checkpoint directory")
    p_ver.add_argument("ckpt_dir", type=Path)
    p_ver.add_argument("--expected-manifest-sha256", type=str, default=None)

    p_sync = sub.add_parser("sync", help="Parallel hash-verified sync of LATEST checkpoint generation")
    p_sync.add_argument("src_run_dir", type=Path)
    p_sync.add_argument("dst_run_dir", type=Path)
    p_sync.add_argument("--workers", type=int, default=16)
    p_sync.add_argument("--keep-generations", type=int, default=2)
    p_sync.add_argument("--prune-older", action="store_true")

    args = parser.parse_args()
    if args.cmd == "verify":
        res = verify_checkpoint_dir(args.ckpt_dir, args.expected_manifest_sha256)
    else:
        res = sync_checkpoint_generation(
            args.src_run_dir,
            args.dst_run_dir,
            workers=args.workers,
            keep_generations=args.keep_generations,
            prune_older=args.prune_older,
        )
    print(json.dumps(res, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
