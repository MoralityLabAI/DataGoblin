from __future__ import annotations

import argparse
from pathlib import Path

from datagoblin.experiments.ldt_set_metric_robustness_v10 import (
    emit_development_observations,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit deterministic v10 development metric probes."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = emit_development_observations(args.spec, args.out)
    print(f"wrote {len(rows)} observations to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
