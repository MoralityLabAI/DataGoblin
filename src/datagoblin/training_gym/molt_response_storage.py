"""MOLT-10: audit exact response archives and compare resident-storage choices.

This consumes engineering receipts. It does not verify model weights, enforce
a RAM cap, or certify training equivalence independently of the input report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from datagoblin.training_gym.molt_native_observer_accounting import count, digest, number, require

ARMS = ("dense", "decode", "gram", "recompute")


def expected_payload(records: int, visits: int, hidden: int, vocab: int, rank: int, itemsize: int) -> dict:
    require(all(type(x) is int and x > 0 for x in (records, visits, hidden, vocab, rank, itemsize)),
            "invalid positive payload dimensions")
    require(rank <= hidden and itemsize in (4, 8), "invalid coordinate chart or dtype")
    d = records * visits * vocab * itemsize
    f = records * visits * hidden * itemsize
    coords = records * rank * itemsize
    chart = hidden * rank * itemsize
    head = hidden * vocab * itemsize
    gram = hidden * hidden * itemsize
    common = coords + chart
    return {
        "dense_response_bytes": d,
        "factor_response_bytes": f,
        "native_coordinate_bytes": coords,
        "native_chart_bytes": chart,
        "frozen_centered_head_bytes": head,
        "frozen_teacher_gram_bytes": gram,
        "dense_plus_native_bytes": d + common,
        "factor_plus_native_if_head_resident_bytes": f + common,
        "factor_plus_native_with_head_bytes": f + common + head,
        "gram_plus_native_with_head_and_gram_bytes": f + common + head + gram,
        "recompute_target_archive_bytes": 0,
        "recompute_plus_native_bytes_excluding_teacher": common,
        "response_payload_reduction_fraction": 1 - f / d,
    }


def frontier(options: list[dict]) -> list[dict]:
    return [r for r in options if not any(
        s["incremental_payload_bytes"] <= r["incremental_payload_bytes"]
        and s["measured_wall_seconds"] <= r["measured_wall_seconds"]
        and (s["incremental_payload_bytes"] < r["incremental_payload_bytes"]
             or s["measured_wall_seconds"] < r["measured_wall_seconds"])
        for s in options
    )]


def audit(report: dict, *, cap_bytes: int, teacher_resident: bool, centered_head_resident: bool) -> dict:
    require(report.get("schema") == "molt10-response-factor-v1"
            and report.get("status") == "engineering_development_only", "unexpected report schema")
    require(report.get("confirmation_ready") is False and report.get("promotion_authorized") is False,
            "unexpected authority")
    require(report.get("real_hardware_memory_reduction_verified") is False
            and report.get("all_representations_resident_for_comparison") is True,
            "unexpected resource claims")
    require(type(teacher_resident) is bool and type(centered_head_resident) is bool,
            "residency must be explicit")
    require(type(cap_bytes) is int and cap_bytes >= 0, "invalid payload budget")
    for key in ("teacher_checkpoint_sha256", "native_factor_sha256", "source_sha256"):
        digest(report[key])
    p = report["payload"]
    n, visits, h, vocab, rank, size = (count(p[k]) for k in
                                       ("records", "visits", "hidden", "vocab", "native_rank", "itemsize"))
    require(report["dtype"] == "float64" and size == 8 and 1 <= report["visit"] <= 8
            and visits == 9 - report["visit"], "unexpected fixture dtype or visit count")
    expected = expected_payload(n, visits, h, vocab, rank, size)
    require(all(p.get(k) == v for k, v in expected.items()), "payload accounting mismatch")
    anchor_ids = set()
    donors = 0
    for receipt in report["anchor_receipts"]:
        a = receipt["anchor"]
        require(isinstance(a, str) and a and a not in anchor_ids, "invalid anchor roster")
        anchor_ids.add(a)
        donors += count(receipt["distinct_donors"])
        require(number(receipt["factor_decode_max_abs"]) <= 1e-9, "teacher decode parity failed")
    require(donors == n and anchor_ids, "incomplete anchor receipts")
    arms = {}
    for row in report["continuation"]:
        arm = row["arm"]
        require(arm in ARMS and arm not in arms, "unexpected continuation arm")
        digest(row["checkpoint"])
        updates, batch = count(row["updates"]), count(row["batch"])
        require(updates > 0 and batch > 0, "invalid continuation work")
        require(row["student_forward_example_visits"] == 2 * updates * batch * 8,
                "student work mismatch")
        require(row["teacher_recompute_example_visits"] == (
            2 * updates * batch * 8 if arm == "recompute" else 0), "teacher work mismatch")
        for key, tolerance in (("first_gradient_max_abs_error", 1e-9),
                               ("last_gradient_max_abs_error", 1e-7),
                               ("final_parameter_max_abs_error", 1e-7),
                               ("max_training_loss_error", 1e-7), ("max_aux_loss_error", 1e-9)):
            require(number(row[key]) <= tolerance, "failed continuation parity gate")
        require(type(row["mean_aux_loss"]) in (float, int) and math.isfinite(row["mean_aux_loss"]),
                "nonfinite response loss")
        number(row["wall_seconds"])
        arms[arm] = row
    require(set(arms) == set(ARMS), "incomplete continuation roster")
    baseline = arms["dense"]
    require(all((r["updates"], r["batch"], r["predictions"]) ==
                (baseline["updates"], baseline["batch"], baseline["predictions"])
                for r in arms.values()), "continuation schedule or prediction mismatch")
    common = expected["native_coordinate_bytes"] + expected["native_chart_bytes"]
    head = 0 if centered_head_resident else expected["frozen_centered_head_bytes"]
    base_sizes = {
        "dense": common + expected["dense_response_bytes"],
        "decode": common + expected["factor_response_bytes"] + head,
        "gram": common + expected["factor_response_bytes"] + head + expected["frozen_teacher_gram_bytes"],
        "recompute": common,
    }
    options = [{"arm": arm, "incremental_payload_bytes": base_sizes[arm],
                "measured_wall_seconds": float(arms[arm]["wall_seconds"]),
                "within_payload_cap": base_sizes[arm] <= cap_bytes,
                "available": arm != "recompute" or teacher_resident}
               for arm in ARMS]
    feasible = [r for r in options if r["available"] and r["within_payload_cap"]]
    per_record_saved = visits * (vocab - h) * size
    def minimum_records(overhead: int) -> int | None:
        return overhead // per_record_saved + 1 if per_record_saved > 0 else None
    return {
        "schema": "molt10-response-storage-v1",
        "status": "offline_engineering_accounting_only",
        "cap_bytes": cap_bytes,
        "teacher_resident": teacher_resident,
        "centered_head_resident": centered_head_resident,
        "archive_break_even_strict_records": {
            "decode_with_head": minimum_records(expected["frozen_centered_head_bytes"]),
            "gram_with_head": minimum_records(expected["frozen_centered_head_bytes"]
                                              + expected["frozen_teacher_gram_bytes"]),
        },
        "options": options,
        "feasible_pareto_options": frontier(feasible),
        "source_identity_verified": False,
        "actual_process_tree_cap_verified": False,
        "promotion_authorized": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cap-bytes", type=int, required=True)
    p.add_argument("--teacher-resident", action="store_true")
    p.add_argument("--centered-head-resident", action="store_true")
    args = p.parse_args()
    if args.report.stat().st_size > 32 * 1024 * 1024:
        p.error("report exceeds fixture cap")
    raw = args.report.read_bytes()
    result = audit(json.loads(raw), cap_bytes=args.cap_bytes, teacher_resident=args.teacher_resident,
                   centered_head_resident=args.centered_head_resident)
    result["source_report_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
