import pytest

from datagoblin.hashing import sha256_json
from datagoblin.training_gym.rlt_hermes_training import collect_row


def test_duplicate_repair_inputs_preserve_both_sources_under_one_root():
    rows, receipts = {}, {}
    puzzle = {"tree_grid": [["T", "X"]], "row_tent_totals": [1], "col_tent_totals": [0, 1]}
    proposal, target = [["T", "X"]], [["T", "C"]]
    collect_row(rows, receipts, puzzle, proposal, target, {"proposer": "a"}, False)
    collect_row(rows, receipts, puzzle, proposal, target, {"proposer": "b"}, False)
    collect_row(rows, receipts, puzzle, target, target, {"proposer": "c"}, True)

    assert len(rows) == 2
    assert len({row["root"] for row in rows.values()}) == 1
    first = next(iter(rows.values()))
    assert first["split"] == "train"
    assert first["input"]["proposed_grid"] == proposal
    assert first["target"]["repaired_grid"] == target
    receipt = receipts[first["id"]]
    assert receipt["input_sha256"] == sha256_json(first["input"])
    assert receipt["target_sha256"] == sha256_json(first["target"])
    assert [item["proposer"] for item in receipt["sources"]] == ["a", "b"]


def test_conflicting_targets_for_one_input_are_rejected():
    rows, receipts = {}, {}
    puzzle = {"tree_grid": [["T", "X"]], "row_tent_totals": [1], "col_tent_totals": [0, 1]}
    proposal = [["T", "M"]]
    collect_row(rows, receipts, puzzle, proposal, [["T", "C"]], {"proposer": "a"}, False)
    with pytest.raises(ValueError, match="conflicting verified targets"):
        collect_row(rows, receipts, puzzle, proposal, [["T", "X"]], {"proposer": "b"}, False)
