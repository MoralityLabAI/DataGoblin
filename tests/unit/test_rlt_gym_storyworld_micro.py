"""Source-locked minified Storyworld micro-corpus and split custody checks."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from datagoblin.training_gym.rlt_gym.storyworld_micro import export_micro, grade_micro, replay_micro
from datagoblin.training_gym.rlt_gym.storyworld_native import connect, generate


class StoryworldMicroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = os.environ.get("GPTSTORYWORLD_SOURCE")
        if not source:
            raise unittest.SkipTest("set GPTSTORYWORLD_SOURCE to enable native micro-compiler tests")
        cls.source = Path(source)
        cls.native = connect(cls.source, expect_commit=os.environ.get("GPTSTORYWORLD_EXPECT_SHA"))

    def test_source_compiler_tasks_and_encounter_splits(self):
        with tempfile.TemporaryDirectory(prefix="dg-sw-micro-test-") as td:
            root = Path(td)
            manifest = export_micro(self.source, root / "a", seed=23, examples_per_encounter=4)
            twin = export_micro(self.source, root / "b", seed=23, examples_per_encounter=4)
            self.assertEqual(manifest, twin)
            self.assertEqual(manifest["artifacts"]["micro_public.jsonl"]["rows"], 28)
            self.assertEqual(replay_micro(self.source, root / "a")["status"], "verified")
            public = [json.loads(line) for line in (root / "a/micro_public.jsonl").read_text().splitlines()]
            secret = [json.loads(line) for line in (root / "a/micro_sealed.jsonl").read_text().splitlines()]
            sft = [json.loads(line) for line in (root / "a/micro_train_sft.jsonl").read_text().splitlines()]
            self.assertEqual({x["task_type"] for x in public},
                             {"compile", "compression", "repair", "targeted_edit"})
            self.assertEqual({x["root"] for x in public}, {x["root"] for x in secret})
            self.assertEqual({x["root"] for x in sft}, {x["root"] for x in public if x["split"] == "train"})
            self.assertTrue(all("expected" not in json.dumps(x) for x in public))
            by_lineage = {}
            for p in public:
                by_lineage.setdefault(p["lineage"], set()).add(p["split"])
            self.assertEqual(len(by_lineage), 7)
            self.assertTrue(all(len(splits) == 1 for splits in by_lineage.values()))
            source_tasks = generate(self.native, seed=23, per_mode=7)
            source_split = {}
            for task, _ in source_tasks:
                if task["mode"] in {"builder", "mini"}:
                    source_split[task["public"]["encounter_id"]] = task["split"]
            for item in secret:
                self.assertEqual(item["split"], source_split[item["provenance"]["encounter_id"]])
            submissions = root / "submit.jsonl"
            submissions.write_text(
                "\n".join(json.dumps({"root": x["root"], "output": json.dumps(x["expected"])})
                          for x in secret) + "\n", encoding="utf-8",
            )
            score = grade_micro(root / "a", submissions, root / "score.jsonl")
            self.assertEqual(score["graded"], len(public))
            self.assertEqual(score["accepted"], len(public))
            path = root / "a/micro_train_sft.jsonl"
            path.write_bytes(path.read_bytes() + b"tamper\n")
            with self.assertRaisesRegex(ValueError, "micro archive hash mismatch"):
                replay_micro(self.source, root / "a")

    def test_guard_against_empty_and_revision_change(self):
        with tempfile.TemporaryDirectory(prefix="dg-sw-micro-guard-") as td:
            root = Path(td)
            with self.assertRaises(ValueError):
                export_micro(self.source, root / "none", examples_per_encounter=0)
            with self.assertRaisesRegex(ValueError, "source commit mismatch"):
                export_micro(self.source, root / "mismatch", expect_source_commit="0" * 40)


if __name__ == "__main__":
    unittest.main()
