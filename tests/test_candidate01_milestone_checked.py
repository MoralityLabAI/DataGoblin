"""Small deterministic tests for previous-corpus exclusion and whole-record selection."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/aleph_next/candidate01"))
import milestone_checked as milestone


@pytest.mark.parametrize("lane", list(milestone.QUOTAS))
@pytest.mark.parametrize("collision", ("semantic", "prompt"))
def test_fresh_prefix_excludes_previous_semantics_and_keeps_whole_rows(lane, collision, monkeypatch):
    families = {
        "state/register-shear",
        "state/register-shear-swap",
        "operator/bilinear-grid",
        "operator/quadratic-line",
        "composition/tree-top-binary",
        "composition/tree-top-unary",
        "repair/acceptance-fault",
        "repair/equivalent-relabel",
    }
    source = {"code_sha256_lf": milestone.code(), "approved_train_families": list(families)}
    skill = milestone.SKILLS[lane][0]
    row = milestone.generated(skill, 0)
    previous, _ = milestone.checked_row(row, 0, families, source["code_sha256_lf"])
    previous_roots = {previous["semantic_key"]} if collision == "semantic" else set()
    previous_prompts = {milestone.c.tasks.digest(previous["prompt"])} if collision == "prompt" else set()
    prior = (previous_roots, previous_prompts, source)
    monkeypatch.setitem(milestone.QUOTAS, lane, 12)
    monkeypatch.setattr(milestone.c, "tokens", lambda *args: ([1, 2, 3, 4, 5], [False, False, False, True, True], 3))
    results = list(milestone.admissions(lane, prior, None))
    assert len(results) == 3 and len({v[0]["semantic_key"] for v in results}) == 3
    assert all(v[0]["semantic_key"] not in prior[0] and v[2]["presentation"] == 1 for v in results)
    assert results[-1][-1]["prior_" + collision + "_duplicate"] >= 1
    assert all(v[0]["prompt"] != previous["prompt"] for v in results)
    assert all(v[2]["receipt_sha256"] == milestone.c.tasks.digest(v[1]) for v in results)
    assert 2 * 5 < milestone.QUOTAS[lane] <= 3 * 5


def test_corrupted_public_or_unregistered_family_cannot_receive_receipt():
    row = milestone.generated("operator", 0)
    family = "operator/" + row["family"]
    with pytest.raises(ValueError, match="outside the starter train grammar"):
        milestone.checked_row(row, 0, {"operator/missing"}, milestone.code())
    broken = copy.deepcopy(row)
    broken["answer"]["value"] = -1
    with pytest.raises(ValueError, match="mathematical verifier"):
        milestone.checked_row(broken, 0, {family}, milestone.code())
