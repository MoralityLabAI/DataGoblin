#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datagoblin.experiments.ldt_shadow import generate_shadow_fixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the frozen content-free LDT shadow fixture"
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    report = generate_shadow_fixture(arguments.contract, arguments.out)
    print(json.dumps(report, sort_keys=True))
