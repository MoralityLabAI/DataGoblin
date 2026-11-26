"""Run the existing TPU trainer against the separately frozen replacement 10M release."""

import argparse
import sys
from pathlib import Path

from verify_bundle import load_release, verify_extracted

CANDIDATE = Path(__file__).resolve().parent.parent / "candidate01"
sys.path.insert(0, str(CANDIDATE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-groups", type=int, default=1)
    parser.add_argument("--wall-seconds", required=True, type=int)
    parser.add_argument("--checkpoint-reserve-seconds", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("xla", "cuda"), default="xla")
    args = parser.parse_args()
    if not (args.execute and 1 <= args.max_groups <= 39
            and 0 <= args.checkpoint_reserve_seconds < args.wall_seconds <= 5400):
        parser.error("explicit bounded pilot execution required")
    release = load_release(args.release, args.release_sha256)
    # Full detached replay should already have run on CPU before provisioning.
    # Here the externally pinned index/replay hashes and packer's per-shard
    # digests protect every group consumed by the unchanged native trainer.
    verify_extracted(args.base, release, full=False)
    import torch
    from colab10m_xla import Backend, run

    torch.set_num_threads(1)
    run(args, Backend(args.device), snapshot_pin=release["index_sha256"],
        replay_path=args.base / "REPLAY.json", replacement=True)


if __name__ == "__main__":
    main()
