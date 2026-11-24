"""Count the proposed strict-1B pair and estimate its cost, without allocating model weights."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import torch
from gym import ARMS, Aleph, coefficients, schedule

CONFIG = Path(__file__).with_name("fullscale.json")


def block_parameters(width, kv_width, ffn):
    return 2 * width**2 + 2 * width * kv_width + 3 * width * ffn + 2 * width


def work_ledger(cfg, reasoning_passes):
    arch = cfg["architecture"]
    d, h, kv = arch["width"], arch["query_heads"], arch["kv_heads"]
    plain = block_parameters(d, kv * (d // h), arch["shell_and_answer_ff"])
    core = block_parameters(d, kv * (d // h), arch["core_ff"])
    s = arch["answer_passes"]
    word = schedule(reasoning_passes, s)
    tied = d * arch["vocab"]
    total = arch["outer_blocks"] * plain + arch["physical_answer_blocks"] * plain
    total += arch["physical_core_blocks"] * core + tied + d
    touches = sum(core if 1 <= physical <= 10 else plain for physical in word)
    # Dense GEMMs: one forward multiply-add and two backward GEMMs per weight.
    # The tied vocabulary head runs once per position; lookup and RMSNorm are excluded.
    linear_flops = 6 * (touches - 2 * d * len(word) + tied)
    # Query-key and value aggregation each take ~2d operations per attended pair.
    # The average ideal triangular attention has (L+1)/2 previous positions including itself.
    causal_flops = 6 * d * len(word) * (arch["max_context"] + 1)
    cost = cfg["user_cost_case"]
    rate = cost["tokens_per_leg"] / (cost["hours_per_leg"] * 3600)
    return {
        "r": reasoning_passes,
        "s": s,
        "visits": len(word),
        "plain_block": plain,
        "core_block": core,
        "tied_embedding_head": tied,
        "final_norm": d,
        "unique": total,
        "bf16_weights_bytes": 2 * total,
        "illustrative_16byte_adam_inventory": 16 * total,
        "block_parameter_touches_per_token": touches,
        "linear_training_flops_per_position": linear_flops,
        "ideal_causal_attention_training_flops_per_position": causal_flops,
        "estimated_training_flops_per_10B": (linear_flops + causal_flops) * cost["tokens_per_leg"],
        "aggregate_useful_tflop_s_at_user_token_rate": (linear_flops + causal_flops) * rate / 1e12,
    }


def linear_three_pass_check():
    """Exact polynomial check in m=J-I. Products commute for one stationary linear core."""

    def expanded(slopes):
        poly = [Fraction(1)]
        for slope in slopes:
            poly = [*poly, Fraction(0)]
            for index in reversed(range(1, len(poly))):
                poly[index] += slope * poly[index - 1]
        return poly

    rlt = expanded((Fraction(1, 3), Fraction(1, 2), Fraction(1)))
    damp = expanded((Fraction(5, 12), Fraction(5, 12), Fraction(1)))
    diff = [a - b for a, b in zip(rlt, damp, strict=True)]
    reversed_rlt = expanded((Fraction(1, 2), Fraction(1, 3), Fraction(1)))
    if diff != [0, 0, Fraction(-1, 144), Fraction(-1, 144)] or reversed_rlt != rlt:
        raise ValueError("stationary transfer polynomial mismatch")
    return [str(x) for x in diff]


def check_unequal_visit_maps():
    i = torch.eye(2, dtype=torch.int64)
    m1 = torch.tensor([[1, 2], [0, -1]], dtype=torch.int64)
    m2 = torch.tensor([[0, 1], [2, 1]], dtype=torch.int64)
    j3 = torch.tensor([[2, 0], [-1, 1]], dtype=torch.int64)
    scaled_rlt = 24 * (2 * i + m2) @ (3 * i + m1)
    scaled_damp = (12 * i + 5 * m2) @ (12 * i + 5 * m1)
    expected = j3 @ (12 * (m2 - m1) - m2 @ m1)
    if not torch.equal(j3 @ (scaled_rlt - scaled_damp), expected):
        raise ValueError("unequal visit transfer identity failed")
    return expected.tolist()


def compare(cfg, r, candidate, measured_rate=None, hourly_price=None):
    arch = cfg["architecture"]
    if candidate in ("direct", "terminal"):
        raise ValueError("candidate would duplicate the direct arm")
    baseline = work_ledger(cfg, r)
    reference = work_ledger(cfg, cfg["tournament"]["engineering_reference_passes"])
    if baseline["unique"] != arch["fixture_expected_params"] or baseline["unique"] > 10**9:
        raise ValueError("strict budget mismatch")
    with torch.device("meta"):
        model = Aleph(
            arch["width"], arch["vocab"], heads=arch["query_heads"], kv=arch["kv_heads"], core_ff=arch["core_ff"]
        )
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if count != baseline["unique"] or len(model.blocks) != 14:
            raise ValueError("meta model count differs from the arithmetic")
        tokens = torch.empty((1, 4), dtype=torch.long)
        for arm in ("direct", candidate):
            shape = model(tokens, arm, r, arch["answer_passes"]).shape
            if shape != (1, 4, arch["vocab"]):
                raise ValueError("meta forward shape differs")
    w = cfg["user_cost_case"]
    leg = cfg["corpus"]["processed_per_first_leg"]
    ext = cfg["tournament"]["extension_tokens"]
    legs = (leg, leg, ext)
    billed_toks = sum(legs)
    implied_rate = w["tokens_per_leg"] / (w["hours_per_leg"] * 3600)
    price = w["credits_per_leg"] / w["hours_per_leg"] if hourly_price is None else hourly_price
    token_rate = implied_rate if measured_rate is None else measured_rate
    hours = [n / (3600 * token_rate) for n in legs]
    credits = sum(hours) * price
    whole_updates, final_positions = divmod(leg, cfg["tournament"]["default_global_nonpadding_positions"])
    schedule_coeffs = coefficients(candidate, r, arch["answer_passes"])
    return {
        "meta_trainable_parameters_per_arm": count,
        "meta_forward": "direct and candidate have shape [1,4,49152]; meta tensors only",
        "linear_transfer_difference_ascending_powers_of_J_minus_I": linear_three_pass_check(),
        "unequal_linear_visit_fixture_difference_times_144": check_unequal_visit_maps(),
        "requested_candidate": candidate,
        "coefficient_distinct_from_direct": any(a != 1 for a, _ in schedule_coeffs.values()),
        "distinct_runtime_coefficients": sorted({a for a, _ in schedule_coeffs.values()}),
        "candidate_is_distinct_from_local_constant_damp": (
            schedule_coeffs
            != coefficients("pass_damp" if candidate == "pass_rlt" else "core_edges_damp", r, arch["answer_passes"])
            if candidate in ("pass_rlt", "core_edges")
            else None
        ),
        "pair_geometry": baseline,
        "engineering_reference": reference,
        "ideal_flop_ratio_to_engineering_reference": (
            baseline["estimated_training_flops_per_10B"] / reference["estimated_training_flops_per_10B"]
        ),
        "full_global_updates_per_10B_at_proposed_batch": whole_updates,
        "partial_final_update_nonpadding_positions": final_positions,
        "estimated_total_flops_three_legs": baseline["estimated_training_flops_per_10B"] * billed_toks / leg,
        "three_leg_nonpadding_positions": billed_toks,
        "maximum_distinct_stream_positions_if_extension_fresh": leg + ext,
        "total_block_positions_before_padding": billed_toks * baseline["visits"],
        "estimate_source": "operator case" if measured_rate is None else "caller supplied sustained measurement",
        "nonpadding_positions_per_second": token_rate,
        "estimated_hours_by_leg": hours,
        "estimated_credits_by_leg": [v * price for v in hours],
        "estimated_total_credits": credits,
        "estimated_credit_shortfall": max(0.0, credits - w["credits_available"]),
        "unpopulated_input_pins": [k for k, v in cfg["corpus"].items() if v is None],
    }


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r", type=int, choices=(2, 3), default=cfg["tournament"]["reasoning_passes"])
    p.add_argument("--candidate", choices=ARMS, default=cfg["tournament"]["provisional_arm_b"])
    p.add_argument("--tokens-per-second", type=float, help="inclusive of evaluation, I/O, retries and checkpointing")
    p.add_argument("--credits-per-hour", type=float)
    args = p.parse_args()
    if (args.tokens_per_second is None) != (args.credits_per_hour is None):
        p.error("supply both measured rate and price")
    if args.tokens_per_second is not None and (
        not all(0 < v < float("inf") for v in (args.tokens_per_second, args.credits_per_hour))
    ):
        p.error("rate and price must be finite and positive")
    print(json.dumps(compare(cfg, args.r, args.candidate, args.tokens_per_second, args.credits_per_hour), indent=2))


if __name__ == "__main__":
    main()
