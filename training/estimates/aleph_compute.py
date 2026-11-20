#!/usr/bin/env python3
"""Analytical Aleph 1B GPU-hour envelope. No provider quote or measured throughput.

Parameter accounting matches the research dossier's proposed two-shell,
four-operator architecture. Training FLOPs are estimates including a causal
attention term. External probe and agent costs must be added separately.
"""
from __future__ import annotations

import argparse
import json


def estimate(rounds: int, *, tokens: int, context: int, prefix_fraction: float,
             mfu: float, overhead: float, peak_bf16_tflops: float,
             gpus: int, hourly_rate: float, total_cap: float,
             training_cap: float) -> dict:
    d, ff, vocab = 1536, 4096, 65536
    operators, blocks_per_operator, fixed_blocks = 4, 7, 4
    matrix_params_per_block = 4 * d * d + 3 * d * ff
    block_params = matrix_params_per_block + 2 * d
    physical_blocks = fixed_blocks + operators * blocks_per_operator
    embedding = vocab * d
    base = embedding + physical_blocks * block_params + d
    router = d * 64 + 64 * operators**2 + 64 + operators**2
    mixing = d * operators + operators
    halt = d + 1
    initial_states = d * operators
    steering_maps = 2 * operators * d * 16
    steering_coefficients = d * operators * 16 + operators * 16
    params = (base + router + mixing + halt + initial_states
              + steering_maps + steering_coefficients)
    assert params == 1_007_141_013, params

    executed_blocks = fixed_blocks + operators * blocks_per_operator * rounds
    effective_matrix_params = embedding + executed_blocks * matrix_params_per_block
    linear_flops = 6 * tokens * effective_matrix_params
    attention_density = 0.5 * (1 + prefix_fraction**2)
    attention_flops = 12 * attention_density * executed_blocks * context * d * tokens
    total_flops = linear_flops + attention_flops
    active_gpu_hours = total_flops / (peak_bf16_tflops * 1e12 * mfu * 3600)
    allocated_hours = active_gpu_hours * (1 + overhead)
    per_gpu_hour_tokens = tokens / allocated_hours
    return {
        "status": "analytic_estimate_not_measured",
        "unique_neural_parameters": params,
        "trained_tokens": tokens,
        "rounds": rounds,
        "physical_transformer_blocks": physical_blocks,
        "executed_block_visits_per_token": executed_blocks,
        "model_training_flops": total_flops,
        "context_tokens": context,
        "active_training_mfu_assumption": mfu,
        "overhead_fraction": overhead,
        "gpu_count": gpus,
        "allocated_gpu_hours": allocated_hours,
        "cluster_wall_hours": allocated_hours / gpus,
        "processed_tokens_per_gpu_hour": per_gpu_hour_tokens,
        "estimated_cost_usd": allocated_hours * hourly_rate,
        "training_cap_gpu_hours": training_cap,
        "total_cap_gpu_hours": total_cap,
        "remaining_inside_training_cap": training_cap - allocated_hours,
        "remaining_inside_total_cap_before_probes": total_cap - allocated_hours,
        "gpu_hours_are_accounting_units": True,
        "silico_ru_conversion": "unverified",
        "billing_effects_not_in_model": ["agent tokens", "provider reservations",
                                         "probe GPU time", "storage", "data I/O",
                                         "engine startup and failed runs beyond allowance"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=10_000_000_000)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--mfu", type=float, default=0.30)
    p.add_argument("--overhead", type=float, default=0.25)
    p.add_argument("--prefix-fraction", type=float, default=0.5)
    p.add_argument("--peak-bf16-tflops", type=float, default=989.4)
    p.add_argument("--hourly-rate", type=float, default=4.0)
    p.add_argument("--training-cap", type=float, default=400)
    p.add_argument("--total-cap", type=float, default=700)
    args = p.parse_args()
    assert 0 < args.mfu <= 1 and args.gpus > 0 and args.tokens > 0
    assert args.overhead >= 0 and 0 <= args.prefix_fraction <= 1
    results = {
        "assumptions": vars(args),
        "reference": "Proposed Data Goblin Aleph 1B architecture, 2026-11-19",
        "cases": [
            estimate(rounds, tokens=args.tokens, context=args.context,
                     prefix_fraction=args.prefix_fraction, mfu=args.mfu,
                     overhead=args.overhead, peak_bf16_tflops=args.peak_bf16_tflops,
                     gpus=args.gpus, hourly_rate=args.hourly_rate,
                     total_cap=args.total_cap, training_cap=args.training_cap)
            for rounds in (2, 3)
        ],
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
