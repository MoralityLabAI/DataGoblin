#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from datagoblin.experiments.ldt_target_independent_v7 import (
    generate_target_independent_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the frozen v7 development and sealed confirmation data"
        )
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    result = generate_target_independent_dataset(
        arguments.preregistration.resolve(),
        arguments.out.resolve(),
    )
    print(result["generation_passed"])
