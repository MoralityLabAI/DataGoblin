import copy

import pytest

from datagoblin.training_gym.molt_isolated_response_accounting import audit, measured_peak

MIB = 1024**2
ARMS = ("dense", "decode", "gram", "recompute")


def phase(peak, elapsed=2.0):
    return {
        "status": "success", "returncode": 0, "single_process_only": True,
        "hard_address_space_cap": True, "hard_process_tree_rss_cap": False,
        "limits": {"sampled_rss_guard_bytes": 2048 * MIB, "address_space_bytes": 8192 * MIB},
        "sample_count": 10, "sampled_peak_virtual_bytes": 1000 * MIB,
        "sampled_peak_rss_bytes": peak - MIB, "sampled_kernel_hwm_bytes": peak,
        "worker_kernel_peak_rss_bytes": peak, "elapsed_seconds": elapsed,
    }


def fixture():
    n, h, v, c, visit = 256, 128, 2048, 8, 4
    common_hash = "a" * 64
    ck = "b" * 64
    common = {
        "schema": "molt11-common-v1", "records": n, "hidden": h, "vocab": v, "native_rank": c,
        "visit": visit, "teacher_checkpoint_sha256": ck, "factor_sha256": "c" * 64,
        "archive": {"sha256": common_hash, "tensor_payload_bytes": 100000},
        "roster": [[f"a{i}", f"d{j}"] for i in range(8) for j in range(32)],
        "native_answer_parity_max_abs": 0., "teacher_validation": {"exact": 0.10, "constant_exact": 0.06},
        "competence_gate": {"min_exact": 0.25, "min_lift_over_constant": 0.10},
        "competence_gate_passed": False, "peak_rss_bytes": 500 * MIB,
    }
    phases = {"common": phase(500 * MIB), "audit": phase(300 * MIB)}
    targets = {}
    for kind, width, head, key, peak, elapsed in (
        ("dense", v, 0, "d", 550, 3.), ("factor", h, 8 * h * v, "e", 510, 2.)
    ):
        response = n * 5 * width * 8
        targets[kind] = {
            "schema": "molt11-target-v1", "arm": kind, "common_sha256": common_hash,
            "teacher_checkpoint_sha256": ck, "response_bytes": response, "head_bytes": head,
            "archive": {"sha256": key * 64, "tensor_payload_bytes": response + head,
                        "file_bytes": response + head + 100},
            "peak_rss_bytes": peak * MIB,
        }
        phases[f"prepare-{kind}"] = phase(peak * MIB, elapsed)
    arms = []
    for arm, peak, elapsed in (("dense", 640, 4.), ("decode", 600, 3.8),
                               ("gram", 595, 4.1), ("recompute", 620, 7.)):
        t = targets["dense" if arm == "dense" else "factor"] if arm != "recompute" else None
        arms.append({
            "schema": "molt11-arm-v1", "arm": arm, "common_sha256": common_hash, "visit": visit,
            "competence_gate_passed": False, "training_effect_claim_authorized": False,
            "promotion_authorized": False, "common_tensor_bytes": 100000,
            "updates": 2, "batch": 4, "seed": 11, "aux_weight": 1., "student_example_visits": 128,
            "teacher_example_visits": 128 if arm == "recompute" else 0,
            "target_sha256": t["archive"]["sha256"] if t else None,
            "target_tensor_bytes": t["archive"]["tensor_payload_bytes"] if t else 0,
            "extra_gram_bytes": 8 * h * h if arm == "gram" else 0,
            "recompute_teacher_parameter_bytes": 3500000 if arm == "recompute" else 0,
            "recompute_centered_head_bytes": 8 * v * h if arm == "recompute" else 0,
            "rss_after_load_bytes": (peak - 60) * MIB,
            "lifetime_peak_before_training_bytes": (peak - 40) * MIB,
            "lifetime_peak_after_training_bytes": peak * MIB,
            "lifetime_peak_at_receipt_bytes": peak * MIB, "checkpoint_sha256": "f" * 64,
            "training_wall_seconds": elapsed - 1, "training_cpu_seconds": elapsed - 1,
            "losses": [1., 2.], "response_losses": [0., 0.1], "predictions": [1, 2],
        })
        phases[f"train-{arm}"] = phase(peak * MIB, elapsed)
    return {
        "schema": "molt11-isolation-v1", "status": "engineering_pass",
        "promotion_authorized": False, "training_effect_claim_authorized": False,
        "hard_process_tree_rss_cap": False, "competence_gate_passed": False,
        "common": common, "phases": phases, "target_archives": targets, "arms": arms,
        "audit": {"schema": "molt11-audit-v1", "competence_gate_passed": False,
                  "promotion_authorized": False, "training_effect_claim_authorized": False,
                  "arms": [{"arm": a, "parameter_max_abs_error": 0., "task_loss_max_abs_error": 0.,
                            "response_loss_max_abs_error": 0.} for a in ARMS]},
    }


def test_observed_caps_and_repeat_accounting():
    r = audit(fixture(), payload_cap=5 * MIB, rss_cap=610 * MIB, reuse_runs=3)
    assert r["fastest_feasible_repeat_forecast"] == "decode"
    assert {a["arm"] for a in r["options"] if a["within_observed_rss_and_payload_caps"]} == {"decode", "gram"}
    decode = next(a for a in r["options"] if a["arm"] == "decode")
    assert decode["conditional_repeat_seconds"] == pytest.approx(2 + 3 * 3.8)
    assert not r["hard_process_tree_rss_cap_verified"]
    assert not r["teacher_competence_gate_passed"]
    assert not r["training_effect_claim_authorized"]


def test_recompute_includes_resident_teacher():
    r = audit(fixture(), payload_cap=20 * MIB, rss_cap=2048 * MIB, reuse_runs=1)
    rec = next(a for a in r["options"] if a["arm"] == "recompute")
    assert rec["accounted_tensor_payload_bytes"] == 100000 + 3500000 + 8 * 128 * 2048
    assert rec["target_archive_file_bytes"] == 0


def test_incomplete_or_bad_receipts_rejected():
    for bad in ("phase", "identity", "payload", "parity", "competence"):
        r = copy.deepcopy(fixture())
        if bad == "phase":
            r["phases"]["train-dense"]["status"] = "wall_timeout"
        elif bad == "identity":
            r["arms"][0]["target_sha256"] = "0" * 64
        elif bad == "payload":
            r["target_archives"]["dense"]["response_bytes"] -= 1
        elif bad == "parity":
            r["audit"]["arms"][1]["parameter_max_abs_error"] = 1.
        else:
            r["competence_gate_passed"] = True
        with pytest.raises(ValueError):
            audit(r, payload_cap=100 * MIB, rss_cap=2048 * MIB, reuse_runs=1)


def test_guard_uses_worker_high_water():
    r = phase(2000 * MIB)
    r["worker_kernel_peak_rss_bytes"] = 2050 * MIB
    with pytest.raises(ValueError):
        measured_peak(r, 2050 * MIB)
