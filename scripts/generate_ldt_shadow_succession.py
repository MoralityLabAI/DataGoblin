#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datagoblin.experiments.ldt_shadow_succession import (
    generate_shadow_succession,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate append-only content-free LDT shadow rounds"
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    if arguments.out.resolve().is_relative_to(repository_root):
        raise SystemExit("generated shadow fixtures must remain outside the repository")
    report = generate_shadow_succession(arguments.contract, arguments.out)
    print(json.dumps(report, sort_keys=True))
