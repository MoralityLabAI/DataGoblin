"""Count a proposed 1B layout and its repeated work without allocating the model."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

from datagoblin.serialization import pretty_json


@dataclass(frozen=True)
class Layout:
    vocabulary: int = 49_152
    width: int = 2_048
    heads: int = 16
    kv_heads: int = 4
    feedforward: int = 8_192
    outer_blocks: int = 2
    reasoning_blocks: int = 10
    answer_blocks: int = 2
    bridges: int = 8
    low_rank_pairs: int = 12
    low_rank_width: int = 256
    router_width: int = 256

    def block_weights(self):
        if (
            self.width <= 0 or self.heads <= 0 or self.kv_heads <= 0 or self.feedforward <= 0
            or self.width % self.heads or self.heads % self.kv_heads
        ):
            raise ValueError("invalid grouped-query dimensions")
        kv = self.kv_heads * (self.width // self.heads)
        attention = 2 * self.width**2 + 2 * self.width * kv
        swiglu = 3 * self.width * self.feedforward
        return attention + swiglu + 2 * self.width


def report(layout, reasoning_visits, answer_visits):
    if min(reasoning_visits, answer_visits, *asdict(layout).values()) < 0:
        raise ValueError("counts and visits must be nonnegative")
    block = layout.block_weights()
    widths = {
        "tied_embedding_readout": layout.vocabulary * layout.width,
        "outer_blocks": layout.outer_blocks * block,
        "reasoning_stack": layout.reasoning_blocks * block,
        "answer_stack": layout.answer_blocks * block,
        "dense_bridge_bank": layout.bridges * layout.width**2,
        "low_rank_pairs": 2 * layout.low_rank_pairs * layout.width * layout.low_rank_width,
        "router": 2 * layout.width * layout.router_width,
        "final_rms_norm": layout.width,
    }
    unique = sum(widths.values())
    visits = layout.outer_blocks + reasoning_visits * layout.reasoning_blocks + answer_visits * layout.answer_blocks
    return {
        "schema": "datagoblin.1b-layout-ledger.v1",
        "layout": asdict(layout),
        "scope": "Planning arithmetic for a bias-free grouped-query, SwiGLU and RMS-normalized layout.",
        "role_weights": widths,
        "unique_parameters": unique,
        "headroom_under_one_billion": 10**9 - unique,
        "bf16_parameter_bytes": 2 * unique,
        "illustrative_full_adam_persistent_bytes": unique * (2 + 2 + 4 + 8),
        "adam_inventory": {"parameters": 2, "gradients": 2, "master_parameters": 4, "two_moments": 8},
        "reasoning_visits": reasoning_visits,
        "answer_visits": answer_visits,
        "executed_transformer_blocks": visits,
        "block_parameter_touches_per_token": visits * block,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reasoning-visits", type=int, default=6)
    parser.add_argument("--answer-visits", type=int, default=6)
    args = parser.parse_args()
    print(pretty_json(report(Layout(), args.reasoning_visits, args.answer_visits)), end="")


if __name__ == "__main__":
    main()
