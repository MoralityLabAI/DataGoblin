"""Frozen-catalog bridge checks: exact dedup and quarantined additions only."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from catalog_bridge import FROZEN_CATALOG, build_bridge  # noqa: E402


class SourceCatalogBridgeTests(unittest.TestCase):
    def test_reconciles_source_roots_without_training_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "catalog.json"
            report_path = Path(tmp) / "report.json"
            catalog, report = build_bridge(ROOT, FROZEN_CATALOG, out, report_path)
            frozen = json.loads(FROZEN_CATALOG.read_text(encoding="utf-8"))
            self.assertGreaterEqual(report["annotated_source_files"], 602)
            self.assertEqual(report["frozen_files"], len(frozen["files"]))
            self.assertEqual(
                report["merged_file_roots"],
                report["frozen_files"] + report["new_quarantined_file_roots"],
            )
            self.assertGreater(report["new_quarantined_file_roots"], 450)
            self.assertGreater(report["overlapping_exact_file_roots"], 0)
            self.assertFalse(report["training_ready"])
            self.assertFalse(catalog["training_ready"])
            self.assertTrue(out.is_file())
            self.assertTrue(report_path.is_file())
            new = [row for row in catalog["files"] if row.get("source_intake")]
            self.assertTrue(all(row["eligibility"] == "quarantine" for row in new))
            self.assertTrue(all(row["training_ready"] is False for row in new))
            self.assertEqual(catalog["files"], sorted(
                catalog["files"], key=lambda row: (row["source_id"], row["path"])
            ))


if __name__ == "__main__":
    unittest.main()
