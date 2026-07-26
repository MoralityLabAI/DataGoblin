#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from datagoblin.experiments.ldt_structured_array_v8 import (
    generate_structured_array_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate v8 mixed-mechanism development and sealed data"
        )
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    result = generate_structured_array_dataset(
        arguments.preregistration.resolve(),
        arguments.out.resolve(),
    )
    print(result["generation_passed"])
