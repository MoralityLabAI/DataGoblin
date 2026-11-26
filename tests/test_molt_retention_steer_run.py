from datagoblin.training_gym.molt_retention_steer_run import run


def test_fixed_panel_denominator_and_parity():
    report = run(16, 2, 4, (53, 59))
    assert len(report["rows"]) == 4 * 2 * 2 * (8 + 6)
    assert len(report["audits"]) == 8
    assert all(v < 1e-9 for v in report["parity"].values())
    assert not report["promotion_authorized"]
    assert report["strong_sham_audit"]["unmatched"] == 0
    for contrast in report["contrasts"]:
        assert contrast["pairs"] == 4
        assert -1 <= contrast["balanced_success_delta"] <= 1
        assert contrast["matched_pairs"] == contrast["pairs"]
