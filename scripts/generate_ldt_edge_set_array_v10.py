#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from datagoblin.experiments.ldt_edge_set_array_v10 import (
    generate_edge_set_array_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate v10 edge/set development and sealed array data"
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    result = generate_edge_set_array_dataset(
        arguments.preregistration.resolve(),
        arguments.out.resolve(),
    )
    print(result["generation_passed"])
