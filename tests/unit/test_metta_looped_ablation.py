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


def test_factorial_rules_remove_only_lineage_and_revision():
    pytest.importorskip("verifiers.v1")
    pytest.importorskip("torch")
    from metta_research_bench.looped.ablation_data import MUTATIONS, ablation_rules
    from metta_research_bench.looped.interaction import ARMS, JOINT, factorial_rules

    rules = factorial_rules()
    singles = ablation_rules()
    assert tuple(rules) == ARMS
    assert all(rules[arm] == singles[arm] for arm in ARMS[:-1])
    assert rules[JOINT] == singles["without_revision"].replace(*MUTATIONS["lineage"], 1)


def test_factorial_contrast_is_paired_and_recovers_joint_effect():
    pytest.importorskip("verifiers.v1")
    pytest.importorskip("torch")
    from metta_research_bench.looped.interaction import ARMS, factorial_metrics
    from metta_research_bench.looped.lineage_replication import lineage_metrics

    rows = []
    for root, rewards in (("r1", (1, 0, 1, 1)), ("r2", (1, 1, 1, 1))):
        for seed in (1, 2):
            for arm, reward in zip(ARMS, rewards, strict=True):
                rows.append({
                    "curator": arm, "root": root, "seed": seed, "family": "lineage", "reward": reward,
                    "grade": {key: 0 for key in ("unsafe_additions", "premature_stop", "exact_first_edit",
                                                  "remaining_edits", "grading_constraint_checks")},
                    "policy": {"policy_verifier_calls": 0, "passes": 0, "vetoes": 0},
                })
    interaction = factorial_metrics(rows, [1, 2])["interaction"]
    assert interaction["estimate"] == 0.5
    assert interaction["additive_joint_prediction"] == 0.5
    assert interaction["paired_roots"] == 2
    effect = lineage_metrics(rows, [1, 2])["lineage_guard_effect"]
    assert effect["mean_over_revision"]["estimate"] == 0.25
    assert effect["revision_present"]["estimate"] == 0.5
    assert effect["revision_absent"]["estimate"] == 0
    with pytest.raises(ValueError, match="four arms"):
        factorial_metrics(rows[:-1], [1, 2])


@pytest.mark.parametrize("family", ["revision", "lineage"])
def test_family_replication_has_a_larger_single_family_reserve(family):
    pytest.importorskip("verifiers.v1")
    from pydantic import ValidationError

    from metta_research_bench.taskset import FAMILY_REPLICATION_ROOTS, SPLITS, BenchConfig

    split = f"looped_{family}_replication"
    config = BenchConfig(split=split, roots_per_family=FAMILY_REPLICATION_ROOTS)
    assert config.roots_per_family == 400
    assert SPLITS[config.split][0] == (family,)
    with pytest.raises(ValidationError, match="at most 48"):
        BenchConfig(split="looped_interaction_frozen", roots_per_family=49)
    with pytest.raises(ValidationError):
        BenchConfig(split=split, roots_per_family=401)
