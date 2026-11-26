import copy

import pytest

from datagoblin.training_gym.molt_response_storage import audit, expected_payload


def fixture():
    dimensions = {"records": 32, "visits": 5, "hidden": 128, "vocab": 2048, "native_rank": 8, "itemsize": 8}
    payload = {**dimensions, **expected_payload(32, 5, 128, 2048, 8, 8)}
    arms = []
    for arm, elapsed in (("dense", 2.), ("decode", 1.8), ("gram", 1.5), ("recompute", 3.)):
        arms.append({"arm": arm, "checkpoint": "d" * 64, "updates": 4, "batch": 2,
                     "student_forward_example_visits": 128,
                     "teacher_recompute_example_visits": 128 if arm == "recompute" else 0,
                     "first_gradient_max_abs_error": 0., "last_gradient_max_abs_error": 0.,
                     "final_parameter_max_abs_error": 0., "max_training_loss_error": 0.,
                     "max_aux_loss_error": 0., "mean_aux_loss": 0., "wall_seconds": elapsed,
                     "predictions": [1, 2]})
    return {
        "schema": "molt10-response-factor-v1", "status": "engineering_development_only",
        "confirmation_ready": False, "promotion_authorized": False,
        "real_hardware_memory_reduction_verified": False, "all_representations_resident_for_comparison": True,
        "teacher_checkpoint_sha256": "a" * 64, "native_factor_sha256": "b" * 64,
        "source_sha256": "c" * 64, "dtype": "float64", "visit": 4, "payload": payload,
        "anchor_receipts": [{"anchor": a, "distinct_donors": 16, "factor_decode_max_abs": 1e-12}
                            for a in ("a", "b")],
        "continuation": arms,
    }


def test_head_and_teacher_residency():
    r = fixture()
    cold = audit(r, cap_bytes=2_300_000, teacher_resident=False, centered_head_resident=False)
    assert cold["archive_break_even_strict_records"] == {"decode_with_head": 28, "gram_with_head": 30}
    assert [p["arm"] for p in cold["feasible_pareto_options"]] == ["decode"]
    warm = audit(r, cap_bytes=400_000, teacher_resident=True, centered_head_resident=True)
    assert {p["arm"] for p in warm["feasible_pareto_options"]} == {"decode", "gram", "recompute"}
    assert not warm["actual_process_tree_cap_verified"]


def test_parity_and_accounting_fail_closed():
    r = fixture()
    for field in ("first_gradient_max_abs_error", "last_gradient_max_abs_error", "max_aux_loss_error"):
        invalid = copy.deepcopy(r)
        invalid["continuation"][1][field] = 1.
        with pytest.raises(ValueError):
            audit(invalid, cap_bytes=10**7, teacher_resident=False, centered_head_resident=False)
    r["payload"]["frozen_centered_head_bytes"] -= 1
    with pytest.raises(ValueError):
        audit(r, cap_bytes=10**7, teacher_resident=False, centered_head_resident=False)
