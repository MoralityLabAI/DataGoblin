"""Open MeTTa diagnostics must partition entire roots and retain quarantine."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from corpus import validate_corpus  # noqa: E402
from public_root_splits import build_plan, sha256  # noqa: E402


class PublicRootSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sources, _, issues = validate_corpus(ROOT)
        if issues:
            raise AssertionError(issues)
        cls.source_map = {source["id"]: source for source in sources}
        source = ROOT / "training_candidates/hyperon_source_query_pairs.jsonl"
        cls.rows = [json.loads(line) for line in source.read_text().splitlines() if line]
        cls.packets_path = source
        cls.exceptions_path = (
            ROOT / "training_candidates/hyperon_query_exceptions.jsonl"
        )

    def _run_rows(self, rows: list[dict], exceptions: list[dict] | None = None) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            packets = base / "packets.jsonl"
            bad = base / "exceptions.jsonl"
            packets.write_text("".join(json.dumps(row) + "\n" for row in rows))
            bad.write_text(
                "".join(json.dumps(row) + "\n" for row in (exceptions or []))
            )
            return build_plan(
                packets, bad, base / "plan.json",
                source_map=self.source_map,
            )

    def test_real_public_snapshot_is_root_disjoint_and_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "plan.json"
            plan = build_plan(
                self.packets_path, self.exceptions_path, out,
                source_map=self.source_map,
            )
            self.assertEqual(plan["native_query_packets"], 187)
            self.assertEqual(plan["exception_observations"], 5)
            self.assertEqual(plan["candidate_source_roots"], 19)
            self.assertEqual(plan["exception_only_roots"], 1)
            self.assertEqual(sum(plan["packets_by_partition"].values()), 187)
            self.assertEqual(plan["root_components"], 19)
            self.assertEqual(
                plan["shared_exact_query_keys_across_roots"], 0
            )
            self.assertTrue(all(plan["packets_by_partition"].values()))
            self.assertFalse(plan["training_ready"])
            self.assertFalse(plan["sealed_eval_eligible"])
            self.assertFalse(plan["independent_semantic_oracle"])
            self.assertTrue(plan["candidate_source_labels_already_public"])
            self.assertIn("exception_quarantine", plan["root_assignments"].values())
            self.assertEqual(plan, json.loads(out.read_text()))
            self.assertNotIn('"output":', out.read_text())
            self.assertNotIn('"source_prefix":', out.read_text())

    def test_deterministic_plan_same_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            a = build_plan(
                self.packets_path, self.exceptions_path, base / "a.json",
                source_map=self.source_map,
            )
            b = build_plan(
                self.packets_path, self.exceptions_path, base / "b.json",
                source_map=self.source_map,
            )
            self.assertEqual(a, b)
            self.assertEqual((base / "a.json").read_bytes(), (base / "b.json").read_bytes())

    def test_shared_query_keeps_distinct_roots_in_one_partition(self) -> None:
        first: dict = self.rows[0]
        sample = [copy.deepcopy(first)]
        seen = {first["metadata"]["source_id"]}
        for row in self.rows:
            sid = row["metadata"]["source_id"]
            if sid not in seen:
                sample.append(copy.deepcopy(row))
                seen.add(sid)
            if len(sample) == 3:
                break
        self.assertEqual(len(sample), 3)
        query = json.loads(sample[0]["input"])["query"]
        modified = json.loads(sample[1]["input"])
        modified["query"] = query
        sample[1]["input"] = json.dumps(modified, ensure_ascii=False, sort_keys=True)
        meta = sample[1]["metadata"]
        meta["answer_text_sha256"] = sha256(sample[1]["output"].encode("utf-8"))
        sid = meta["source_id"]
        meta["source_context_sha256"] = sha256(
            modified["source_prefix"].encode("utf-8")
        )
        meta["prefix_bytes"] = len(modified["source_prefix"].encode("utf-8"))
        sample[1]["id"] = sha256(
            f"{sid}:{meta['evaluation_ordinal']}:{sha256(sample[1]['input'].encode())}".encode()
        )
        plan = self._run_rows(sample)
        partitions = plan["root_assignments"]
        self.assertEqual(
            partitions[sample[0]["metadata"]["source_id"]],
            partitions[sample[1]["metadata"]["source_id"]],
        )
        self.assertEqual(plan["root_components"], 2)
        self.assertEqual(plan["shared_exact_query_keys_across_roots"], 1)

    def test_rejects_forged_prefix(self) -> None:
        sample = [copy.deepcopy(self.rows[0])]
        parsed = json.loads(sample[0]["input"])
        parsed["source_prefix"] += "unexpected content"
        sample[0]["input"] = json.dumps(parsed, sort_keys=True)
        with self.assertRaisesRegex(ValueError, "prefix or answer digest mismatch"):
            self._run_rows(sample)

    def test_rejects_early_training_release(self) -> None:
        sample = [copy.deepcopy(self.rows[0])]
        sample[0]["metadata"]["training_ready"] = True
        with self.assertRaisesRegex(ValueError, "promoted before the audit"):
            self._run_rows(sample)

    def test_rejects_duplicate_native_observations(self) -> None:
        sample = [copy.deepcopy(self.rows[0]), copy.deepcopy(self.rows[1])]
        sample[1]["metadata"]["assertion_observation_id"] = (
            sample[0]["metadata"]["assertion_observation_id"]
        )
        with self.assertRaisesRegex(ValueError, "duplicated assertion observation"):
            self._run_rows(sample)


if __name__ == "__main__":
    unittest.main()
