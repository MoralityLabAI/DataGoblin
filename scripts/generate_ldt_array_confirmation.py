#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datagoblin.experiments.ldt_array_confirmation import (
    generate_confirmation_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the frozen variable-array LDT confirmation surface"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    report = generate_confirmation_dataset(arguments.config, arguments.out)
    print(json.dumps(report, sort_keys=True))
