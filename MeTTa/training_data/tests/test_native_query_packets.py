"""Native source-query export must retain context, source lineage and quarantine."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from native_query_packets import build_packets  # noqa: E402


class NativeQueryPacketTests(unittest.TestCase):
    def test_packets_replay_source_prefix_and_are_not_released(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp)
            report = build_packets(
                ROOT,
                ROOT / "native_assertion_verdicts/hyperon_0_2_10.jsonl",
                dst / "packets.jsonl",
                dst / "exceptions.jsonl",
                dst / "report.json",
            )
            self.assertGreater(report["packets"], 150)
            self.assertEqual(report["source_assertion_observations"], 192)
            self.assertGreaterEqual(report["exceptions"], 5)
            self.assertFalse(report["training_ready"])
            rows = [json.loads(line) for line in (dst / "packets.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), report["packets"])
            self.assertTrue(all(x["metadata"]["training_ready"] is False for x in rows))
            self.assertTrue(all(x["metadata"]["native_agreement_observed"] is True for x in rows))
            self.assertTrue(all(x["metadata"]["split"] == "unassigned" for x in rows))
            self.assertTrue(all("source_prefix" in json.loads(x["input"]) for x in rows))
            self.assertTrue(all(x["output"] for x in rows))

    def test_context_budget_filters_records_without_manufacturing_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp)
            report = build_packets(
                ROOT,
                ROOT / "native_assertion_verdicts/hyperon_0_2_10.jsonl",
                dst / "packets.jsonl",
                dst / "exceptions.jsonl",
                dst / "report.json",
                max_context_bytes=128,
            )
            self.assertGreater(report["exceptions"], 10)
            self.assertEqual(report["heldout_labels_written"], 0)
            self.assertEqual(report["root_splits_assigned"], 0)


if __name__ == "__main__":
    unittest.main()
