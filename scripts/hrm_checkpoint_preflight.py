from __future__ import annotations

import argparse
from pathlib import Path

from datagoblin.hrm_text.preflight import HardCaps, run_checkpoint_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed HRM checkpoint footprint preflight")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ram-mb", type=int, default=2048)
    parser.add_argument("--cpu-percent", type=int, default=50)
    parser.add_argument("--io-mb-s", type=int, default=50)
    parser.add_argument("--wall-clock-seconds", type=int, default=1800)
    args = parser.parse_args()
    assessment = run_checkpoint_preflight(
        args.model_path,
        args.out,
        run_id=args.run_id,
        caps=HardCaps(args.ram_mb, args.cpu_percent, args.io_mb_s, args.wall_clock_seconds),
    )
    print(args.out / "summary.json")
    return 0 if assessment.load_permitted else 3


if __name__ == "__main__":
    raise SystemExit(main())

