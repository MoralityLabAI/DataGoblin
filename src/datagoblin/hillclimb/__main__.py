"""Print the hill-climb ledger state: `python -m datagoblin.hillclimb [--state-dir DIR] [--lane ID]`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .lanes import HillclimbError
from .service import Service


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--lane")
    parser.add_argument("--last", type=int, default=10)
    args = parser.parse_args(argv)
    service = Service(root=args.root, state_dir=args.state_dir, read_only=True)
    try:
        status = service.ledger_status(lane=args.lane, last=args.last)
    except HillclimbError as exc:
        print(json.dumps({"error": exc.code, "detail": exc.detail}))
        return 2
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
