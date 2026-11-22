import pytest


def test_one_rule_ablations_and_identical_rule_aliases():
    pytest.importorskip("verifiers.v1")
    pytest.importorskip("hyperon")
    from metta_research_bench.looped.ablation_data import MUTATIONS, ablation_rules
    from metta_research_bench.looped.pool import curate_root, source_pool

    rules = ablation_rules()
    for name, (before, after) in MUTATIONS.items():
        assert rules[f"without_{name}"] == rules["good"].replace(before, after, 1)
    root, rollouts, verifier = source_pool("lineage", 61000, 24)
    selection, _ = curate_root(verifier.world, root, rollouts, rules)
    assert all(index is not None for index in selection.values())
    assert selection["without_closure"] == selection["without_conflict"] == selection["good"]


def test_holm_adjustment_is_monotone_in_ordered_p_values():
    pytest.importorskip("verifiers.v1")
    from metta_research_bench.looped.ablation_stats import holm

    assert holm({"first": 0.01, "second": 0.04, "third": 0.041}) == {
        "first": 0.03, "second": 0.08, "third": 0.08,
    }
