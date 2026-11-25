"""Fixed-budget, physical/logical count and paired-slope checks."""

import math

import pytest
from scale_release import SEEDS, complete, geometry, ledger, slopes


def test_strict_budget_arithmetic_and_local_denominator():
    direct = [{"width": 256, "arm": "direct", "seed": s, "flops": 20e9, "train_seconds": 2} for s in SEEDS]
    receipt = ledger(direct, {"256": {"status": "completed"}})
    assert receipt["depths"]["2"]["unique"] == receipt["depths"]["3"]["unique"] == 999352320
    assert receipt["depths"]["2"]["visits"] == 24
    assert receipt["depths"]["3"]["visits"] == 34
    for n in (10**8, 10**9, 10**10, 2 * 10**10):
        row = receipt["depths"]["2"]["targets"][str(n)]
        assert row["assumed_tpu_credits_if_quote_at_this_depth"] == pytest.approx(n / 10**10 * 350)
        assert row["one_core_equivalent_years"]["median"] == pytest.approx(
            row["estimated_training_flops"] / 10e9 / 31557600
        )
    affordable = receipt["paired_10b_then_affordable_extension"]
    assert affordable["winner_extra_positions"] == 5_700_000_000
    assert affordable["total_credits"] == pytest.approx(899.5)
    assert affordable["credits_remaining"] >= receipt["chosen_reserve_credits"]
    assert receipt["three_leg_over_balance"] == pytest.approx(50)


def test_competence_controls_terminal_rung_and_shape_counts():
    assert geometry(80)["params"] == 1_414_480
    assert geometry(256)["params"] == 14_179_584
    assert not complete({"80": {"controls_competent": False}})
    assert not complete({"256": {"controls_competent": True}})
    assert complete({"256": {"controls_competent": False}})
    assert complete({"680": {"controls_competent": True}})


def test_scale_secants_reject_unmatched_streams():
    rows = []
    for width, params, loss in ((80, 1e6, 4.0), (256, 1e7, 3.0)):
        for s in SEEDS:
            steps = [{"tokens": step * s, "metrics": {"text_nll": loss}, "checked_macro": 0.5} for step in (16, 32)]
            rows.append(
                {"width": width, "arm": "direct", "seed": s, "params": params, "trace": steps, "stream": str(s)}
            )
    rungs = {"80": {"status": "completed"}, "256": {"status": "completed"}}
    slope = slopes(rows, rungs)["direct-80-256-step32"]["seeds"]["137"]
    assert slope["text_nll_per_ln_params"] == pytest.approx(-1 / math.log(10))
    rows[-1]["stream"] = "different"
    with pytest.raises(ValueError, match="unmatched"):
        slopes(rows, rungs)


def test_shard_publication_preserves_first_completed_bytes(tmp_path):
    from scale_shard import publish

    stage = tmp_path / "stage"
    stage.mkdir()
    source = stage / "fit.pt"
    dest = tmp_path / "fit.pt"
    source.write_bytes(b"original fit")
    publish(source, dest)
    publish(source, dest)
    # Replace the staging name; the POSIX branch may hard-link the original inode.
    second = stage / "other.pt"
    second.write_bytes(b"conflict")
    with pytest.raises(ValueError, match="collision"):
        publish(second, dest)
    assert dest.read_bytes() == b"original fit"
