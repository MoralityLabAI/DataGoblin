"""Check source-to-native-result alignment before treating observations as evidence."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from native_assertions import scan_eval_sites, verify_receipts  # noqa: E402


class NativeAssertionAlignmentTests(unittest.TestCase):
    def test_top_level_eval_with_comments_strings_and_spaced_prefix(self) -> None:
        program = (
            '; !(assertEqual ignored 4)\n'
            '(= (quote-test) "! (test ignored 4)")\n'
            '! (new-goal-status! lunch inactive)\n'
            '!(assertEqual (status (Goal lunch)) inactive)\n'
            '!(assertAlphaEqualToResult (f $x) ((g $x)))\n'
        )
        sites = scan_eval_sites(program)
        self.assertEqual(len(sites), 3)
        self.assertEqual([x["operator"] for x in sites], [
            "new-goal-status!", "assertEqual", "assertAlphaEqualToResult"
        ])
        self.assertFalse(sites[0]["is_assertion"])
        self.assertEqual(sites[1]["query_expression"], "(status (Goal lunch))")
        self.assertEqual(sites[1]["author_expected_expression"], "inactive")
        self.assertTrue(sites[2]["is_assertion"])

    def test_unbalanced_top_level_query_rejected(self) -> None:
        with self.assertRaises(ValueError):
            scan_eval_sites("!(assertEqual (f 1) 2")

    def test_existing_hyperon_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = verify_receipts(
                ROOT,
                ROOT / "runtime_receipts/hyperon_0_2_10.jsonl",
                Path(tmp) / "observations.jsonl",
                Path(tmp) / "summary.json",
            )
            self.assertEqual(summary["source_execution_receipts"], 20)
            self.assertGreater(summary["native_assertions"], 150)
            self.assertGreater(summary["outcome_counts"].get("native_assertion_pass", 0), 150)
            self.assertGreater(summary["outcome_counts"].get("native_assertion_error", 0), 0)
            self.assertEqual(summary["verified_training_rollouts"], 0)
            self.assertFalse(summary["training_ready"])
            rows = [json.loads(line) for line in (Path(tmp) / "observations.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["training_ready"] is False for row in rows))
            self.assertEqual(len(rows), summary["native_assertions"])
            failed = [x for x in rows if x["native_result"] == "native_assertion_error"]
            self.assertTrue(any("g1_docs.metta" in x["source_id"] for x in failed))


if __name__ == "__main__":
    unittest.main()
