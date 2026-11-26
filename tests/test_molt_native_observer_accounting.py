import copy

import pytest

from datagoblin.training_gym.molt_native_observer_accounting import audit


def fixture(n=16, accepted=16, duplicate=False):
    donor = []
    methods = {}
    for method in ("stack", "terminal"):
        a = accepted if method == "stack" else 0
        for i in range(n):
            donor.append({"anchor": "anchor", "donor": f"donor{i}", "method": method,
                          "delta_digest": f"{0 if duplicate else i:064x}",
                          "accepted": i < a, "argmax_parity": i < a,
                          "relative_error": 0.05 if i < a else 1., "absolute_error": 1.})
        candidate = 364 + n + 32 * a + 64 * (n - a)
        use = not duplicate and candidate < n * 64
        methods[method] = {"accepted": a, "fallback": n - a, "map_bytes": 364,
                           "candidate_payload_bytes": candidate,
                           "planned_payload_bytes": candidate if use else n * 64,
                           "amortized_map_selected": use,
                           "relative_error_max": 0.05 if a == n else 1.}
    return {
        "schema": "molt09-native-observer-v1", "status": "engineering_development_only",
        "promotion_authorized": False, "formal_certificate_authorized": False,
        "teacher_checkpoint_sha256": "a" * 64, "native_factor_sha256": "b" * 64,
        "native_span_rank": 8, "compact_rank": 4, "common_basis_bytes": 8192,
        "teacher_targets_shared_across_arms": True, "bank_records": n,
        "donor_receipts": donor,
        "map_receipts": [{
            "anchor": "anchor", "binding_digest": "c" * 64,
            "donor_count": n, "distinct_displacement_count": 1 if duplicate else n,
            "native_components": 8, "rank": 4, "common_basis_bytes": 8192,
            "native_api_parity_max_abs": 0., "clean_parity_max_abs": 0.,
            "fd_relative_error": 0., "observer_rows": 5500,
            "native_coordinate_bytes": 64 * n, "ambient_write_bytes": 1024 * n,
            "common_teacher_target_bytes": 22000 * n, "methods": methods,
        }],
    }


def test_map_overhead_and_common_payload():
    result = audit(fixture())
    assert result["native_coordinate_bytes"] == 1024
    assert result["methods"]["stack"]["planned_bytes"] == 892
    assert result["methods"]["stack"]["saved_vs_native"] == 132
    assert result["methods"]["terminal"]["saved_vs_native"] == 0
    assert result["native_total_with_common_payload"] == 8192 + 352000 + 1024
    assert result["best_mixed_saved_vs_native"] == 132
    assert not result["certificate_authorized"]


def test_fallback_and_duplicate():
    assert audit(fixture(accepted=11))["best_mixed_saved_vs_native"] == 0
    assert audit(fixture(duplicate=True))["best_mixed_saved_vs_native"] == 0


def test_bad_receipts_fail_closed():
    report = fixture()
    changed = copy.deepcopy(report)
    changed["donor_receipts"].pop()
    with pytest.raises(ValueError):
        audit(changed)
    changed = copy.deepcopy(report)
    changed["map_receipts"][0]["methods"]["stack"]["planned_payload_bytes"] -= 1
    with pytest.raises(ValueError):
        audit(changed)
    changed = copy.deepcopy(report)
    changed["donor_receipts"][0]["relative_error"] = 0.5
    with pytest.raises(ValueError):
        audit(changed)
