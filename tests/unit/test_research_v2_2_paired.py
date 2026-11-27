"""Regression checks for paired evidence, budget gates and native Distill custody."""

import numpy as np
import pytest

from datagoblin.research_v2_2.distill_native import canonical_digest, reference_record
from datagoblin.research_v2_2.paired import paired_effects, selective_gate


def _row(root, family, score):
    return {"root_id": root, "family": family, "primary_score": score,
            "capability_score": score}


def test_paired_family_macro_uses_root_not_row_weight():
    old = [_row("a", "A", 0), _row("b", "A", 0), _row("c", "B", 1)]
    new = [_row("a", "A", 1), _row("b", "A", 1), _row("c", "B", 0)]
    report = paired_effects(old, new, families=("A", "B"))
    assert report["macro_primary_delta"] == 0
    assert report["paired_primary_bootstrap_95"] == [0, 0]
    assert report["family_denominators"] == {"A": 2, "B": 1}
    assert report["interval_status"] == "descriptive_only"


def test_bootstrap_reproducible_and_pair_rejections():
    old = [_row(str(i), "A", .5) for i in range(8)]
    new = [_row(str(i), "A", float(i % 2)) for i in range(8)]
    report = paired_effects(old, new, families=("A",))
    assert report == paired_effects(old, new, families=("A",))
    assert np.allclose(report["paired_primary_bootstrap_95"],
                       report["paired_capability_bootstrap_95"])
    with pytest.raises(ValueError):
        paired_effects(old, new[::-1], families=("A",))
    with pytest.raises(ValueError):
        paired_effects(old + old, new + new, families=("A",))
    with pytest.raises(ValueError):
        paired_effects(old, new, families=("A", "B"))


def test_selective_gate_requires_full_panel_and_compute():
    old = [_row(str(i), "A", .5) for i in range(8)]
    report = paired_effects(old, old, families=("A",))
    assert selective_gate(report, {"rlt_passes": 100}, {"rlt_passes": 75})["passes"]
    assert not selective_gate(report, {"rlt_passes": 100}, {"rlt_passes": 76})["passes"]
    short = paired_effects(old[:1], old[:1], families=("A",))
    assert not selective_gate(short, {"rlt_passes": 100}, {"rlt_passes": 0})["passes"]
    with pytest.raises(ValueError):
        selective_gate(report, {}, {"rlt_passes": 1})


def test_distill_canonical_unicode_and_nonfinite():
    import hashlib
    assert canonical_digest({"x": "\u03b1"}) == hashlib.sha256(b'{"x":"\\u03b1"}').hexdigest()
    with pytest.raises(ValueError):
        canonical_digest({"x": float("nan")})


def test_native_distill_replay_if_available():
    pytest.importorskip("Distill")
    from Distill.core import digest
    from Distill.research_cases import compile_research
    from Distill.research_oracle import reference

    packet = compile_research("adaptive_evidence", seed=9001, ordinal=0, difficulty="standard")
    assert canonical_digest(packet) == digest(packet)
    answers = []
    for question in packet["questions"]:
        if question["id"] == "case.solve":
            fields = {f: "fixture rationale" for f in question["fields"]}
            fields["answer"] = reference(packet["public_case"])
            answers.append({"question_id": question["id"], "status": "answered", "fields": fields})
        else:
            answers.append({"question_id": question["id"], "status": "abstain",
                            "fields": {"reason": "fixture certificate only"}})
    response = {"schema_version": "distill.response.v2", "packet_id": packet["packet_id"],
                "model": {"provider": "fixture", "revision": "v1", "run_id": "fixture-1"},
                "answers": answers}
    record = reference_record(packet, response)
    assert record["training_admission"] is False
    assert record["custody"] == "reference_only"
    assert record["decision"] == reference(packet["public_case"])
    # Wrong schema must be rejected without any teacher transport.
    with pytest.raises(ValueError, match="native JSON"):
        reference_record(packet, {"schema_version": "distill.response.v1"})
