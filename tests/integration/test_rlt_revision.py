from __future__ import annotations

import json
from pathlib import Path

import pytest

from datagoblin.training_gym.rlt_revision import SPLITS, build, score


def test_revision_pack_and_stale_control(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    manifest = build(pack, seed=17, train=8, per_split=8)
    assert manifest == build(tmp_path / "repeat", seed=17, train=8, per_split=8)
    seen = set()
    for split in SPLITS:
        tasks = [json.loads(line) for line in (pack / f"{split}.tasks.jsonl").read_text().splitlines()]
        targets = {
            row["id"]: row for row in map(json.loads, (pack / f"{split}.targets.jsonl").read_text().splitlines())
        }
        lineages = {task["lineage"] for task in tasks}
        assert len(lineages) == 8 and not (seen & lineages)
        seen.update(lineages)
        stale = []
        oracle = []
        for task in tasks:
            links = dict(task["links"])
            patch = task["patch"]
            # Walk both rule sets together, independently of the generator's solver.
            initial = revised = task["start"]
            path = []
            for _ in range(task["steps"]):
                path.append(initial)
                initial = links[initial]
                revised = patch["destination"] if revised == patch["source"] else links[revised]
            assert patch["source"] in (path[0], path[-1]) if split == "revision_ood" else (
                patch["source"] in path[1:-1]
            )
            assert task["steps"] in (range(7, 10) if split == "length_ood" else range(3, 6))
            assert targets[task["id"]]["before"] == initial
            assert targets[task["id"]]["after"] == revised
            stale.append({"id": task["id"], "before": initial, "after": initial})
            oracle.append({"id": task["id"], "before": initial, "after": revised})
        path = tmp_path / "predictions.jsonl"
        for rows, expected in ((oracle, [8, 8]), (stale, [0, 8])):
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = score(pack, split, path)
            assert result["changed_after_exact"] == expected
            assert result["no_op_after_exact"] == [8, 8]
            assert result["both_variants_exact"] == expected
            assert result["stale_after_when_before_correct"] == [8 - expected[0], 8]
        path.write_text(json.dumps(stale[0]) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="every item"):
            score(pack, split, path)
