import json

import pytest

from datagoblin.training_gym.molt_goblin import FAMILIES, generate, make_task, score, verify


def test_prefix_bound_and_invalid_edge():
    task = {"family": "prefix_dag", "problem": {"positions": 4, "edges": [[0, 1], [1, 3]]}}
    assert verify(task) == {"invalid_edges": [], "stabilization_bound": 3}
    task["problem"]["edges"].append([3, 3])
    assert verify(task) == {"invalid_edges": [[3, 3]], "stabilization_bound": None}


def test_exact_parallel_capacity_and_tie():
    def candidate(cid, utility, qualified=True):
        return {"id": cid, "qualified": qualified, "prefix_ok": True,
                "ram_mb": 2, "cpu_slots": 1, "visits": 1, "utility": utility}
    p = {"caps": {"ram_mb": 4, "cpu_slots": 2, "visits": 2},
         "candidates": [candidate("c", 3), candidate("b", 4), candidate("a", 3),
                        candidate("unsafe", 99, False)]}
    assert verify({"family": "parallel_array", "problem": p}) == {"selected": ["a", "b"], "utility": 7}


def test_pareto_uncertainty_and_safety():
    p = {"candidates": [
        {"id": "a", "qualified": True, "intervals": [[4, 5], [4, 5]]},
        {"id": "b", "qualified": True, "intervals": [[1, 2], [1, 2]]},
        {"id": "c", "qualified": True, "intervals": [[4, 6], [4, 6]]},
        {"id": "d", "qualified": False, "intervals": [[99, 99], [99, 99]]},
    ]}
    assert verify({"family": "pareto_array", "problem": p}) == {
        "statuses": {"a": "UNRESOLVED", "b": "DOMINATED", "c": "UNRESOLVED", "d": "INELIGIBLE"}}


def test_split_firewall_and_score(tmp_path):
    pub, steward = tmp_path / "public", tmp_path / "private"
    generate("ordinary_test", 19, 2, pub, steward)
    assert not (pub / "train_sft.jsonl").exists()
    labels = [json.loads(x) for x in (steward / "labels.jsonl").read_text().splitlines()]
    pred = tmp_path / "predictions.jsonl"
    pred.write_text("\n".join(json.dumps({"task_id": a["task_id"], "answer": a["answer"]})
                              for a in labels), encoding="utf-8")
    report = score(pub, steward, pred)
    assert report["missing"] == 0
    assert all(v == {"correct": 2, "total": 2} for v in report["families"].values())
    assert not report["promotion_authorized"]
    with pytest.raises(FileExistsError):
        generate("ordinary_test", 19, 2, pub, steward)
    for family in FAMILIES:
        assert make_task("train", 19, family, 0)["root_id"] != make_task(
            "ordinary_test", 19, family, 0)["root_id"]


def test_train_only_export(tmp_path):
    generate("train", 9, 1, tmp_path / "train", tmp_path / "private")
    assert len((tmp_path / "train" / "train_sft.jsonl").read_text().splitlines()) == len(FAMILIES)
