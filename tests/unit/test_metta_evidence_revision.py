from itertools import pairwise

from MeTTa.evidence_revision import FAMILIES, episode, summary


def test_duplicate_proof_and_independent_source_are_distinct_events():
    before, *changes = map(summary, episode("alias_arrival", 1))
    assert all(snapshot["proofs"] for snapshot in (before, *changes))
    assert [len(s["proofs"]) for s in (before, *changes)] == [1, 2, 3, 1]
    assert any(a["proofs"] != b["proofs"] and a["minimal_sources"] == b["minimal_sources"]
               for a, b in pairwise((before, *changes)))
    assert not before["disjoint_source_pairs"] and changes[1]["disjoint_source_pairs"]
    assert not changes[2]["disjoint_source_pairs"]


def test_common_premise_and_supersession_change_what_can_be_claimed():
    for seed in range(12):
        common = [summary(snapshot) for snapshot in episode("shared_bottleneck", seed)]
        assert not any(row["disjoint_source_pairs"] for row in common)
        assert not common[2]["proofs"]
        assert len(common[3]["minimal_sources"]) == 2
        lifecycle = [summary(snapshot) for snapshot in episode("superseded_shortcut", seed)]
        assert len(lifecycle[0]["proofs"]) == 2
        assert len(lifecycle[0]["minimal_sources"]) == 1
        assert len(lifecycle[-1]["proofs"]) == 1
        assert len(lifecycle[-1]["minimal_sources"]) == 1
        assert all(len(episode(family, seed)) == 4 for family in FAMILIES)
