import pytest

from datagoblin.training_gym.rlt_gym import tasks as rlt

from Agonist.tinker_curriculum import rlt_tasks, task_pool
from Agonist.tinker_gym import ROLES, Journal
from Agonist.tinker_live import source_pool


def test_mixed_pool_is_balanced_and_split_disjoint(tmp_path):
    train = task_pool(17, 9, "train", "mixed")
    dev = task_pool(17, 9, "dev", "mixed")
    assert len(train) == len(dev) == 36
    assert {t.root for t in train}.isdisjoint(t.root for t in dev)
    assert all(sum(t.role == role for t in train) == 9 for role in ROLES)
    assert {t.public.get("source", "fixture") for t in train} == {"fixture", "rlt", "research"}
    for task in train:
        assert task.grade(task.gold)["success"]
        assert task.demonstration()["provenance"] == "independent_original_fixture_oracle"
    for task in dev:
        with pytest.raises(ValueError):
            task.demonstration()
    journal = Journal(tmp_path, "clean")
    assert source_pool(journal, 17, 9, "train", "mixed") == train
    assert source_pool(journal, 17, 9, "dev", "mixed") == dev


def test_rlt_candidates_rechecked_against_original_environment():
    rows = rlt.generate_tasks(seed=11, per_family=1)
    for raw in rows:
        if raw["split"] != "train":
            continue
        views = {t.role: t for t in rlt_tasks(raw, "train", 11)}
        assert rlt.verify(raw, views["repair"].public["candidates"][views["repair"].gold])["passed"]
        assert rlt.verify(raw, views["curriculum"].public["candidates"][views["curriculum"].gold])["passed"]
        assert rlt.verify(raw, views["diagnose"].public["proposal"])["code"] == views["diagnose"].gold
    sealed = next(row for row in rows if row["split"] == "sealed_poison")
    with pytest.raises(ValueError):
        rlt_tasks(sealed, "train", 11)


def test_research_scores_and_public_projection():
    pool = task_pool(23, 7, "train", "research")
    assert {t.public.get("source") for t in pool} == {"research"}
    assert all("truth" not in t.public and "experiment_tape" not in t.public for t in pool)
    for task in pool:
        if task.role in ("curriculum", "repair"):
            assert "truth" not in task.public["world"]
            assert "experiment_tape" not in task.public["world"]
            assert max(task.scores.values()) == task.scores[task.gold]
            assert "abstain" in task.scores


def test_all_seven_existing_rlt_families():
    pool = task_pool(2, 7, "train", "rlt")
    assert {t.public["family"] for t in pool} == set(rlt.FAMILIES)
    assert len({t.root for t in pool}) == 7
