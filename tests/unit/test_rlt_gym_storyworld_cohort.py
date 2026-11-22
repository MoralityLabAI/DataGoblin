"""World-level split isolation and exact multi-world Storyworld corpus replay."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from datagoblin.training_gym.rlt_gym.storyworld_cohort import (
    DEFAULT_REGISTRY,
    _registry,
    collect_cohort,
    replay_cohort,
)
from datagoblin.training_gym.rlt_gym.storyworld_local import replay_run, run
from datagoblin.training_gym.rlt_gym.storyworld_native import connect


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


class StoryworldCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = os.environ.get("GPTSTORYWORLD_SOURCE")
        if not source:
            raise unittest.SkipTest("set GPTSTORYWORLD_SOURCE to enable four-world source tests")
        cls.source = Path(source)
        cls.registry = DEFAULT_REGISTRY

    def test_four_world_cohort_isolation_and_replay(self):
        with tempfile.TemporaryDirectory(prefix="dg-sw-cohort-") as td:
            root = Path(td)
            manifest = collect_cohort(self.source, self.registry, root / "cohort",
                                      seed=23, per_mode=2, micro_examples=2,
                                      micro_max_encounters=4)
            self.assertEqual(len(manifest["worlds"]), 4)
            self.assertEqual(manifest["totals"]["native_tasks"], 32)
            self.assertEqual(manifest["totals"]["native_controls"], 112)
            self.assertEqual(
                manifest["totals"]["micro_tasks"] + manifest["totals"]["micro_overbudget_rows"], 32
            )
            self.assertGreater(manifest["totals"]["micro_tasks"], 0)
            self.assertGreater(manifest["totals"]["train_sft_rows"], 0)
            names = set()
            for w in manifest["worlds"]:
                self.assertNotIn(w["id"], names)
                names.add(w["id"])
                self.assertEqual(w["source"]["commit"], manifest["source_commit"])
                folder = root / "cohort" / w["id"]
                native = _rows(folder / "native/public_tasks.jsonl")
                mini = _rows(folder / "micro/micro_public.jsonl")
                oversize = _rows(folder / "micro/micro_oversize.jsonl")
                self.assertEqual(len(mini), w["micro_tasks"])
                self.assertEqual(len(oversize), w["micro_overbudget_rows"])
                self.assertTrue(all(x["split"] == w["split"] for x in [*native, *mini, *oversize]))
                if w["split"] != "train":
                    self.assertEqual(_rows(folder / "native/train_targets.jsonl"), [])
                    self.assertEqual(_rows(folder / "micro/micro_train_sft.jsonl"), [])
            self.assertEqual(replay_cohort(self.source, self.registry, root / "cohort")["status"], "verified")
            with self.assertRaises(FileExistsError):
                collect_cohort(self.source, self.registry, root / "cohort", per_mode=1)
            target = root / "cohort" / "fifth_testimony" / "micro" / "micro_public.jsonl"
            target.write_bytes(target.read_bytes() + b'{"tampered":true}\n')
            with self.assertRaisesRegex(ValueError, "archive hash mismatch"):
                replay_cohort(self.source, self.registry, root / "cohort")

    def test_out_of_domain_world_can_feed_native_model_runner(self):
        with tempfile.TemporaryDirectory(prefix="dg-sw-cohort-player-") as td:
            root = Path(td)
            manifest = collect_cohort(self.source, self.registry, root / "cohort",
                                      seed=37, per_mode=1, micro_examples=1, micro_max_encounters=2)
            world = next(w for w in manifest["worlds"] if w["split"] == "ood_test")
            native = connect(self.source, world["path"], expect_commit=manifest["source_commit"])
            archive = root / "cohort" / world["id"] / "native"

            def stub(_payload: dict) -> dict:
                return {"choices": [{"message": {"content": "abstain"}}]}

            report = run(native, archive, root / "model", model="stub", mode="hypergame",
                         split="ood_test", max_roots=2, transport=stub)
            self.assertEqual(report["metrics"]["graded"], 2)
            self.assertEqual(replay_run(native, archive, root / "model")["status"], "verified")

    def test_registry_validation_rejects_world_reuse_and_split_poisoning(self):
        original = _registry(self.registry)
        self.assertEqual(len(original["worlds"]), 4)
        with tempfile.TemporaryDirectory(prefix="dg-sw-cohort-reg-") as td:
            path = Path(td) / "registry.json"
            corrupt = json.loads(json.dumps(original))
            corrupt["worlds"][1]["path"] = corrupt["worlds"][0]["path"]
            path.write_text(json.dumps(corrupt), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                _registry(path)
            corrupt = json.loads(json.dumps(original))
            corrupt["worlds"][1]["path"] = "../secret/other.json"
            path.write_text(json.dumps(corrupt), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "invalid source-world"):
                _registry(path)
            corrupt = json.loads(json.dumps(original))
            for row in corrupt["worlds"]:
                row["split"] = "train"
            path.write_text(json.dumps(corrupt), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "held-out"):
                _registry(path)


if __name__ == "__main__":
    unittest.main()
