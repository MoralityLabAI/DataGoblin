"""Tests for source-author assertion extraction and strict unverified custody."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from assertions import extract_corpus, find_assertions, mask_comments_and_strings, split_top_level  # noqa: E402


class AssertionExtractionTests(unittest.TestCase):
    def test_mask_preserves_positions(self) -> None:
        code = '; !(test (not-code) x)\n"(also !(test ignored y))"\n!(test (+ 1 2) 3)'
        masked = mask_comments_and_strings(code)
        self.assertEqual(len(masked), len(code))
        self.assertEqual(len(find_assertions(code)), 1)

    def test_extract_nested_expected_output(self) -> None:
        code = '!(assertEqualToResult\n  (match &self (Frog $x) $x)\n  (Sam))\n'
        items = find_assertions(code)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["query_expression"], "(match &self (Frog $x) $x)")
        self.assertEqual(items[0]["author_expected_expression"], "(Sam)")
        self.assertEqual(items[0]["parse_status"], "parsed")

    def test_inline_comment_and_quoted_parentheses(self) -> None:
        code = '!(assertEqual (repr "a; b (x)") ; author note\n  "a; b (x)")'
        item = find_assertions(code)[0]
        self.assertEqual(item["query_expression"], '(repr "a; b (x)")')
        self.assertEqual(item["author_expected_expression"], '"a; b (x)"')

    def test_single_or_unbalanced_arguments(self) -> None:
        self.assertEqual(split_top_level("assertEqual (add 2 3) 5"), (["assertEqual", "(add 2 3)", "5"], True))
        self.assertFalse(split_top_level("assertEqual (unbalanced x")[1])
        self.assertEqual(find_assertions("!(test (unfinished"), [{
            "start_offset": 0,
            "end_offset": 18,
            "assertion_kind": "test",
            "parse_status": "unbalanced",
            "query_expression": None,
            "author_expected_expression": None,
            "assertion_source": "!(test (unfinished",
        }])

    def test_corpus_candidates_are_not_verified(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            details = extract_corpus(ROOT, Path(folder) / "assertions.jsonl", Path(folder) / "report.json")
            self.assertGreaterEqual(details["candidate_assertions"], 900)
            self.assertGreaterEqual(details["parsed_query_expected_pairs"], 600)
            self.assertEqual(details["verified_rollouts"], 0)
            lines = (Path(folder) / "assertions.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), details["candidate_assertions"])
            self.assertGreater(details["distinct_assertion_texts"], 0)


if __name__ == "__main__":
    unittest.main()
