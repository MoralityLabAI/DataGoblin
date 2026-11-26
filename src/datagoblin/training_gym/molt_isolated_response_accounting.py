"""MOLT-11 isolated-run receipt accounting, not a hard RAM certificate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from datagoblin.training_gym.molt_native_observer_accounting import count, digest, number, require
from datagoblin.training_gym.molt_response_storage import ARMS


def measured_peak(phase: dict, worker_peak: int | None = None) -> int:
    require(phase["status"] == "success" and phase["returncode"] == 0, "worker failed")
    require(phase["single_process_only"] is True and phase["hard_address_space_cap"] is True
            and phase["hard_process_tree_rss_cap"] is False, "unexpected resource contract")
    guard = count(phase["limits"]["sampled_rss_guard_bytes"])
    virtual = count(phase["limits"]["address_space_bytes"])
    require(0 < guard <= virtual and count(phase["sample_count"]) > 0, "missing cap or samples")
    require(count(phase["sampled_peak_virtual_bytes"]) <= virtual, "virtual cap violation")
    observed = max(count(phase[k]) for k in ("sampled_peak_rss_bytes", "sampled_kernel_hwm_bytes"))
    if worker_peak is not None:
        require(phase.get("worker_kernel_peak_rss_bytes") == worker_peak, "worker peak receipt mismatch")
        observed = max(observed, count(worker_peak))
    require(0 < observed <= guard, "sampled or worker RSS guard violation")
    number(phase["elapsed_seconds"])
    return observed


def audit(report: dict, *, payload_cap: int, rss_cap: int, reuse_runs: int) -> dict:
    require(report.get("schema") == "molt11-isolation-v1" and report.get("status") == "engineering_pass",
            "complete isolated engineering run required")
    require(report.get("promotion_authorized") is False
            and report.get("training_effect_claim_authorized") is False
            and report.get("hard_process_tree_rss_cap") is False, "unexpected authority")
    require(type(payload_cap) is int and payload_cap >= 0
            and type(rss_cap) is int and rss_cap > 0
            and type(reuse_runs) is int and 1 <= reuse_runs <= 10000, "invalid planning limits")
    common = report["common"]
    require(common["schema"] == "molt11-common-v1", "unexpected common archive")
    common_hash = digest(common["archive"]["sha256"])
    digest(common["teacher_checkpoint_sha256"])
    digest(common["factor_sha256"])
    common_bytes = count(common["archive"]["tensor_payload_bytes"])
    n, h, v, c = (count(common[k]) for k in ("records", "hidden", "vocab", "native_rank"))
    visit = count(common["visit"])
    require(0 < n <= 256 and 0 < c <= h and v > h and 1 <= visit <= 8, "invalid dimensions")
    roster = common["roster"]
    require(isinstance(roster, list) and len(roster) == n, "invalid common roster")
    require(all(isinstance(r, list) and len(r) == 2 and r[0] != r[1]
                and all(isinstance(x, str) and x for x in r) for r in roster), "invalid pair identity")
    require(len({tuple(r) for r in roster}) == n, "duplicate common pair")
    require(number(common["native_answer_parity_max_abs"]) <= 1e-10, "answer path parity failed")
    quality = common["teacher_validation"]
    exact, constant = number(quality["exact"]), number(quality["constant_exact"])
    require(exact <= 1 and constant <= 1, "invalid teacher score")
    require(common["competence_gate"] == {"min_exact": 0.25, "min_lift_over_constant": 0.1},
            "unexpected competence gate")
    competent = exact >= 0.25 and exact - constant >= 0.10
    require(type(common["competence_gate_passed"]) is bool
            and common["competence_gate_passed"] == competent
            and report["competence_gate_passed"] == competent, "competence receipt mismatch")
    phases = report["phases"]
    require(set(phases) == {"common", "prepare-dense", "prepare-factor", "audit",
                            *(f"train-{a}" for a in ARMS)}, "incomplete phase roster")
    common_peak = measured_peak(phases["common"], count(common["peak_rss_bytes"]))
    audit_peak = measured_peak(phases["audit"])
    targets = report["target_archives"]
    require(set(targets) == {"dense", "factor"}, "incomplete target archives")
    target_info = {}
    for kind, width in (("dense", v), ("factor", h)):
        t = targets[kind]
        require(t["schema"] == "molt11-target-v1" and t["arm"] == kind
                and t["common_sha256"] == common_hash
                and t["teacher_checkpoint_sha256"] == common["teacher_checkpoint_sha256"],
                "target archive identity mismatch")
        response = n * (9 - visit) * width * 8
        head = 8 * v * h if kind == "factor" else 0
        require((t["response_bytes"], t["head_bytes"], t["archive"]["tensor_payload_bytes"])
                == (response, head, response + head), "target payload mismatch")
        target_info[kind] = {
            "hash": digest(t["archive"]["sha256"]), "bytes": response + head,
            "file_bytes": count(t["archive"]["file_bytes"]),
            "peak": measured_peak(phases[f"prepare-{kind}"], count(t["peak_rss_bytes"])),
            "seconds": number(phases[f"prepare-{kind}"]["elapsed_seconds"]),
        }
        require(target_info[kind]["file_bytes"] >= response + head, "unexpected archive file size")
    arm_rows = {}
    for a in report["arms"]:
        arm = a["arm"]
        require(arm in ARMS and arm not in arm_rows and a["schema"] == "molt11-arm-v1"
                and a["common_sha256"] == common_hash and a["visit"] == visit
                and a["competence_gate_passed"] == competent, "arm identity mismatch")
        require(a["training_effect_claim_authorized"] is False and a["promotion_authorized"] is False,
                "arm authority mismatch")
        require(a["common_tensor_bytes"] == common_bytes, "common payload mismatch")
        updates, batch = count(a["updates"]), count(a["batch"])
        require(updates > 0 and batch > 0 and a["student_example_visits"] == 16 * updates * batch,
                "student work mismatch")
        if arm == "recompute":
            require(a["target_sha256"] is None and a["target_tensor_bytes"] == 0
                    and a["extra_gram_bytes"] == 0 and count(a["recompute_teacher_parameter_bytes"]) > 0
                    and a["recompute_centered_head_bytes"] == 8 * v * h
                    and a["teacher_example_visits"] == 16 * updates * batch, "recompute contract mismatch")
        else:
            t = target_info["dense" if arm == "dense" else "factor"]
            require(a["target_sha256"] == t["hash"] and a["target_tensor_bytes"] == t["bytes"]
                    and a["extra_gram_bytes"] == (8 * h * h if arm == "gram" else 0)
                    and a["recompute_teacher_parameter_bytes"] == 0
                    and a["recompute_centered_head_bytes"] == 0
                    and a["teacher_example_visits"] == 0, "stored target contract mismatch")
        peak = count(a["lifetime_peak_at_receipt_bytes"])
        require(count(a["rss_after_load_bytes"]) <= count(a["lifetime_peak_before_training_bytes"])
                <= count(a["lifetime_peak_after_training_bytes"]) <= peak, "inconsistent lifetime RSS")
        measured_peak(phases[f"train-{arm}"], peak)
        digest(a["checkpoint_sha256"])
        number(a["training_wall_seconds"])
        number(a["training_cpu_seconds"])
        require(all(isinstance(x, list) and len(x) == updates
                    and all(type(z) in (int, float) and math.isfinite(z) for z in x)
                    for x in (a["losses"], a["response_losses"])), "invalid loss history")
        arm_rows[arm] = a
    require(set(arm_rows) == set(ARMS), "incomplete arm roster")
    ref = arm_rows["dense"]
    require(all(all(r[k] == ref[k] for k in ("updates", "batch", "seed", "aux_weight", "predictions"))
                for r in arm_rows.values()), "schedule or prediction parity failed")
    parity = report["audit"]
    require(parity["schema"] == "molt11-audit-v1" and parity["competence_gate_passed"] == competent
            and parity["promotion_authorized"] is False
            and parity["training_effect_claim_authorized"] is False, "invalid audit contract")
    require(len(parity["arms"]) == 4 and {r["arm"] for r in parity["arms"]} == set(ARMS),
            "incomplete parity audit")
    for r in parity["arms"]:
        require(number(r["parameter_max_abs_error"]) <= 1e-7
                and number(r["task_loss_max_abs_error"]) <= 1e-7
                and number(r["response_loss_max_abs_error"]) <= 1e-9, "failed parity audit")
    options = []
    for arm in ARMS:
        r = arm_rows[arm]
        t = target_info.get("dense" if arm == "dense" else "factor") if arm != "recompute" else None
        # Charge a separately resident complete teacher when recomputing.
        payload = common_bytes + sum(count(r[k]) for k in (
            "target_tensor_bytes", "extra_gram_bytes", "recompute_teacher_parameter_bytes",
            "recompute_centered_head_bytes"))
        train_peak = measured_peak(phases[f"train-{arm}"], r["lifetime_peak_at_receipt_bytes"])
        phase_peak = max(common_peak, train_peak, t["peak"] if t else 0)
        # Conditional repeat forecast for this exact workload. Setup costs
        # common to all choices are excluded, as is the parity-only audit.
        prep = t["seconds"] if t else 0.
        seconds = prep + reuse_runs * number(phases[f"train-{arm}"]["elapsed_seconds"])
        options.append({"arm": arm, "accounted_tensor_payload_bytes": payload,
                        "observed_training_process_peak_rss_bytes": train_peak,
                        "observed_setup_and_training_peak_rss_bytes": phase_peak,
                        "target_archive_file_bytes": t["file_bytes"] if t else 0,
                        "measured_target_preparation_phase_seconds": prep,
                        "measured_training_phase_seconds": phases[f"train-{arm}"]["elapsed_seconds"],
                        "training_wall_seconds": r["training_wall_seconds"],
                        "conditional_repeat_seconds": seconds,
                        "within_observed_rss_and_payload_caps": payload <= payload_cap and phase_peak <= rss_cap})
    feasible = [o for o in options if o["within_observed_rss_and_payload_caps"]]
    feasible.sort(key=lambda o: (o["conditional_repeat_seconds"], o["accounted_tensor_payload_bytes"], o["arm"]))
    return {"schema": "molt11-isolated-accounting-v1", "status": "offline_measured_receipt_accounting",
            "payload_cap_bytes": payload_cap, "rss_cap_bytes": rss_cap, "reuse_runs": reuse_runs,
            "teacher_competence_gate_passed": competent, "options": options,
            "fastest_feasible_repeat_forecast": feasible[0]["arm"] if feasible else None,
            "common_preparation_peak_rss_bytes": common_peak, "parity_audit_peak_rss_bytes": audit_peak,
            "source_identity_verified": False, "hard_process_tree_rss_cap_verified": False,
            "repeat_forecast_validated": False, "training_effect_claim_authorized": False,
            "promotion_authorized": False}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--payload-cap-bytes", type=int, required=True)
    p.add_argument("--rss-cap-bytes", type=int, required=True)
    p.add_argument("--reuse-runs", type=int, default=1)
    args = p.parse_args()
    if args.report.stat().st_size > 32 * 1024 * 1024:
        p.error("report exceeds fixture cap")
    raw = args.report.read_bytes()
    result = audit(json.loads(raw), payload_cap=args.payload_cap_bytes,
                   rss_cap=args.rss_cap_bytes, reuse_runs=args.reuse_runs)
    result["source_report_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
