"""Public Git-tree corpus inventory integrity and rights custody tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_discovery import audit_index  # noqa: E402


class DiscoveryIndexTests(unittest.TestCase):
    def test_large_metadata_inventory_remains_discovery_only(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            report = audit_index(ROOT / "discovery_index", Path(folder) / "audit.json")
            self.assertGreaterEqual(report["indexed_repositories"], 28)
            self.assertGreater(report["indexed_metta_files"], 10000)
            self.assertGreater(report["gross_git_source_bytes"], 650_000_000)
            self.assertGreater(report["bulk_or_generated_bytes"], 500_000_000)
            self.assertGreater(report["distinct_git_blobs"], 5000)
            self.assertTrue(report["discovery_only"])
            self.assertFalse(report["training_ready"])
            self.assertIsNone(report["token_count"])
            self.assertIn("UNKNOWN", report["license_buckets"])
            self.assertTrue((Path(folder) / "audit.json").exists())


if __name__ == "__main__":
    unittest.main()
