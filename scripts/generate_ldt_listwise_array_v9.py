#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from datagoblin.experiments.ldt_listwise_array_v9 import (
    generate_listwise_array_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate v9 listwise development and sealed array data"
        )
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    result = generate_listwise_array_dataset(
        arguments.preregistration.resolve(),
        arguments.out.resolve(),
    )
    print(result["generation_passed"])
