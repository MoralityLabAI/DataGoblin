"""A source import batch must not commit duplicate Git blobs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from crawl_repo import choose_sources  # noqa: E402


class SourceBatchDedupTests(unittest.TestCase):
    def test_prevents_duplicates_inside_new_batch(self) -> None:
        candidates = [
            ("examples/he_math.metta", 10, "a" * 40),
            ("examples/math.metta", 10, "a" * 40),
            ("examples/other.metta", 15, "b" * 40),
        ]
        chosen = choose_sources(candidates, set(), set(), "trueagi-io--PeTTa", 10, 100)
        self.assertEqual([x[0] for x in chosen], ["examples/he_math.metta", "examples/other.metta"])

    def test_rejects_existing_blob_even_at_new_path(self) -> None:
        candidates = [("examples/x.metta", 20, "f" * 40), ("examples/y.metta", 12, "e" * 40)]
        chosen = choose_sources(candidates, set(), {"f" * 40}, "repo--test", 4, 50)
        self.assertEqual(chosen, [("examples/y.metta", 12, "e" * 40)])

    def test_respects_byte_and_file_caps(self) -> None:
        candidates = [("a.metta", 20, "a" * 40), ("b.metta", 30, "b" * 40), ("c.metta", 10, "c" * 40)]
        chosen = choose_sources(candidates, set(), set(), "repo--test", 2, 35)
        self.assertEqual([x[0] for x in chosen], ["a.metta", "c.metta"])


if __name__ == "__main__":
    unittest.main()
