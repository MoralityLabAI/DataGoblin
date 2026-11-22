"""Source-native three-mode Storyworld gym regression and custody tests."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from datagoblin.training_gym.rlt_gym.storyworld_native import (
    collect,
    connect,
    generate,
    grade_submissions,
    replay_archive,
    score_builder,
    score_hyper_actions,
)


class StoryworldNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidate = os.environ.get("GPTSTORYWORLD_SOURCE")
        if not candidate:
            raise unittest.SkipTest("set GPTSTORYWORLD_SOURCE to enable native cross-repo tests")
        cls.native = connect(Path(candidate), expect_commit=os.environ.get("GPTSTORYWORLD_EXPECT_SHA"))

    def test_builder_and_minified_controls_share_split_and_native_validator(self):
        tasks = generate(self.native, seed=23, per_mode=6)
        builders = [(t, s) for t, s in tasks if t["mode"] in ("builder", "mini")]
        self.assertEqual(len(builders), 12)
        by_lineage = {}
        for task, sealed in builders:
            by_lineage.setdefault(task["lineage"], set()).add(task["split"])
            self.assertNotIn("gold_target", json.dumps(task))
            self.assertNotIn("private_bit", json.dumps(task))
            candidates = task["public"]["candidates"]
            self.assertEqual(len({c["id"] for c in candidates}), 3)
            gold = score_builder(self.native, task, sealed, "PICK " + sealed["gold_target"])
            self.assertTrue(gold["accepted"])
            self.assertTrue(gold["native_valid"])
            distractor = next(c["id"] for c in candidates if c["id"] != sealed["gold_target"])
            wrong = score_builder(self.native, task, sealed, "PICK " + distractor)
            self.assertTrue(wrong["native_valid"])
            self.assertFalse(wrong["accepted"])  # A valid graph edge can still be the wrong restoration.
            self.assertFalse(score_builder(self.native, task, sealed, "PICK __UNKNOWN__")["accepted"])
            if task["mode"] == "mini":
                self.assertIn("__BROKEN_TARGET__", task["public"]["swmd_card"])
                self.assertNotIn("PICK " + sealed["gold_target"], task["public"]["swmd_card"])
        self.assertEqual(len(by_lineage), 6)
        self.assertTrue(all(len(splits) == 1 for splits in by_lineage.values()))

    def test_hypergame_uses_real_source_transitions_and_sealed_bit(self):
        tasks = generate(self.native, seed=44, per_mode=1)
        hyper = [(t, s) for t, s in tasks if t["mode"] == "hypergame"]
        self.assertEqual(len(hyper), 2)
        first, second = hyper
        self.assertEqual(first[0]["public"], second[0]["public"])
        self.assertEqual({first[1]["private_bit"], second[1]["private_bit"]}, {0, 1})
        self.assertNotIn("private_bit", json.dumps(first[0]))
        for task, sealed in hyper:
            self.assertIsNone(task["public"]["visible_state"]["vector"][
                task["public"]["public_spec"]["hidden_index"]])
            hg = self.native.hypergame
            s = hg.make_scenario(sealed["scenario_seed"], sealed["dimension"])
            oracle = hg.rollout(s, sealed["private_bit"], hg.oracle_action, sealed["horizon"])
            grade = score_hyper_actions(self.native, sealed, [r["chosen_action"] for r in oracle["records"]])
            self.assertEqual(grade["true_utility"], oracle["grade"]["true_utility"])
            self.assertTrue(grade["accepted"])
            self.assertFalse(grade["false_accept"])
            self.assertFalse(score_hyper_actions(self.native, sealed, ["shortcut"])["accepted"])
            self.assertTrue(score_hyper_actions(self.native, sealed, ["shortcut"])["false_accept"])
            self.assertFalse(score_hyper_actions(self.native, sealed, ["route_999"])["accepted"])

    def test_collection_archive_replay_and_tamper(self):
        with tempfile.TemporaryDirectory(prefix="dg-sw-test-") as td:
            root = Path(td)
            a = collect(self.native, root / "a", seed=31, per_mode=2)
            b = collect(self.native, root / "b", seed=31, per_mode=2)
            self.assertEqual(a, b)
            self.assertEqual(a["profile"], "public_seed_smoke")
            self.assertEqual(a["artifacts"]["public_tasks.jsonl"]["rows"], 8)
            self.assertEqual(a["artifacts"]["baseline_rollouts.jsonl"]["rows"], 28)
            self.assertEqual(replay_archive(self.native, root / "a")["status"], "verified")
            public = [json.loads(line) for line in (root / "a/public_tasks.jsonl").read_text().splitlines()]
            private = [json.loads(line) for line in (root / "a/private_sealed.jsonl").read_text().splitlines()]
            training = [json.loads(line) for line in (root / "a/train_targets.jsonl").read_text().splitlines()]
            self.assertTrue(all(x["split"] == "train" for x in training))
            groups = {}
            for t in public:
                groups.setdefault(t["lineage"], []).append(t["split"])
            self.assertTrue(all(len(set(splits)) == 1 for splits in groups.values()))
            submitted = root / "submissions.jsonl"
            proposals = []
            for t, s in zip(public, private, strict=True):
                if t["mode"] == "hypergame":
                    hg = self.native.hypergame
                    scenario = hg.make_scenario(s["scenario_seed"], s["dimension"])
                    trace = hg.rollout(scenario, s["private_bit"], hg.oracle_action, s["horizon"])
                    proposals.append({"root": t["root"], "actions": [x["chosen_action"] for x in trace["records"]]})
                else:
                    proposals.append({"root": t["root"], "action": "PICK " + s["gold_target"]})
            submitted.write_text("\n".join(json.dumps(p) for p in proposals) + "\n", encoding="utf8")
            report = grade_submissions(self.native, root / "a", submitted, root / "graded.jsonl")
            self.assertEqual(report["graded"], len(public))
            grades = [json.loads(line)["grade"] for line in (root / "graded.jsonl").read_text().splitlines()]
            self.assertTrue(all(g["accepted"] for g in grades))
            path = root / "a/public_tasks.jsonl"
            path.write_bytes(path.read_bytes().replace(b'"schema":', b'"tamper":', 1))
            with self.assertRaisesRegex(ValueError, "archive hash mismatch"):
                replay_archive(self.native, root / "a")

    def test_private_seed_never_persisted_and_source_pin_checked(self):
        with tempfile.TemporaryDirectory(prefix="dg-sw-private-") as td:
            root = Path(td)
            seed = root / "key.bin"
            secret = b"private reproducibility key for synthetic smoke 01"
            seed.write_bytes(secret)
            archive = root / "private"
            manifest = collect(self.native, archive, seed=0, per_mode=1, private_seed_file=seed)
            self.assertEqual(manifest["profile"], "private_seeded")
            self.assertIsNone(manifest["seed"])
            self.assertNotIn(secret.decode(), (archive / "manifest.json").read_text())
            self.assertEqual(replay_archive(self.native, archive, private_seed_file=seed)["status"], "verified")
            with self.assertRaises(ValueError):
                replay_archive(self.native, archive)
            with self.assertRaisesRegex(ValueError, "source commit mismatch"):
                connect(self.native.root, expect_commit="0" * 40)


if __name__ == "__main__":
    unittest.main()
