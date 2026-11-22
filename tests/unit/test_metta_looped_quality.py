import pytest


def test_native_rules_change_selected_quality_without_changing_budget():
    pytest.importorskip("verifiers.v1")
    pytest.importorskip("hyperon")
    from metta_research_bench.looped.pool import RULES, build_pool, rule_hashes

    pool, cards, datasets, summary = build_pool(4, 24)
    assert len(pool) == len(cards) == 20
    assert all(len({tuple(tuple(action.items()) for action in roll) for roll in row["rollouts"]})
               == row["candidate_count"] for row in pool)
    assert summary["matched_roots"] == 20
    rules = {name: path.read_text(encoding="utf-8") for name, path in RULES.items()}
    assert summary["rules_sha256"] == rule_hashes(rules)
    good, sloppy = summary["audits"]["good"], summary["audits"]["sloppy"]
    assert good["supervised_tokens_per_pass"] == sloppy["supervised_tokens_per_pass"]
    assert good["safe_completion"] > sloppy["safe_completion"]
    assert good["unsafe_additions"] == 0
    assert len(datasets["good"]) == len(datasets["sloppy"]) == 20
    for root, audit in zip(pool, cards, strict=True):
        for arm, index in audit["selected"].items():
            assert len(root["rollouts"][index]) == root["action_tokens_per_rollout"], arm


def test_paired_sources_are_disjoint_from_old_splits():
    pytest.importorskip("verifiers.v1")
    from metta_research_bench.looped.pool import TRAIN_START
    from metta_research_bench.taskset import SPLITS

    earlier = {(family, seed) for name, (families, start) in SPLITS.items() if not name.startswith("looped_")
               for family in families for seed in range(start, start + 48)}
    fresh = {(family, seed) for name, (families, start) in SPLITS.items() if name.startswith("looped_")
             for family in families for seed in range(start, start + 48)}
    training = {(family, seed) for family in SPLITS["looped_dev"][0] for seed in range(TRAIN_START, TRAIN_START + 48)}
    assert earlier.isdisjoint(fresh | training)
    assert fresh.isdisjoint(training)


def test_gold_validation_rebuilds_a_state_with_missing_prerequisites():
    import asyncio

    pytest.importorskip("verifiers.v1")
    from metta_research_bench.taskset import BenchConfig, MeTTaResearchBench

    tasks = MeTTaResearchBench(BenchConfig(split="looped_dev", roots_per_family=12))
    dependency_cases = [task for task in tasks if task.data.family == "dependency"]
    assert len(dependency_cases) == 12
    assert all(asyncio.run(task.validate()) for task in dependency_cases)
