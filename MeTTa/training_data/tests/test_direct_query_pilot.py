"""Check MeTTa direct-query pilot selection and conservative native observations."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from direct_query_pilot import (  # noqa: E402
    classify_native_output,
    make_script,
    select_public_cases,
)


class DirectQueryPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = ROOT / "training_candidates/hyperon_source_query_pairs.jsonl"
        cls.rows = [json.loads(line) for line in source.read_text().splitlines() if line]

    def test_round_robin_preserves_source_file_coverage(self) -> None:
        examples = [
            {"id": "a0", "metadata": {"lineage_root": "a", "evaluation_ordinal": 0}},
            {"id": "a1", "metadata": {"lineage_root": "a", "evaluation_ordinal": 1}},
            {"id": "b0", "metadata": {"lineage_root": "b", "evaluation_ordinal": 0}},
            {"id": "b1", "metadata": {"lineage_root": "b", "evaluation_ordinal": 1}},
            {"id": "c0", "metadata": {"lineage_root": "c", "evaluation_ordinal": 0}},
        ]
        assignments = {
            "a": "train_candidate",
            "b": "dev_public",
            "c": "test_public",
        }
        selected = select_public_cases(examples, assignments, 4)
        self.assertEqual([x["id"] for x in selected], ["a0", "b0", "c0", "a1"])
        with self.assertRaisesRegex(ValueError, "missing.*partition"):
            select_public_cases(examples, {"a": "train_candidate"}, 4)

    def test_real_first_packet_reconstructs_a_direct_query(self) -> None:
        packet = self.rows[0]
        script, prior = make_script(packet)
        payload = json.loads(packet["input"])
        self.assertEqual(prior, packet["metadata"]["evaluation_ordinal"])
        self.assertIn(payload["source_prefix"].encode(), script)
        self.assertTrue(script.endswith(("\n!" + payload["query"] + "\n").encode()))
        self.assertLessEqual(len(script), 32_768)

    def test_corrupted_ordinal_is_rejected(self) -> None:
        packet = copy.deepcopy(self.rows[0])
        packet["metadata"]["evaluation_ordinal"] += 1
        with self.assertRaisesRegex(ValueError, "ordinal"):
            make_script(packet)

    def test_native_output_is_strictly_line_aligned(self) -> None:
        self.assertEqual(
            classify_native_output(0, False, b"[()]\n[(value)]\n", b"", 1),
            ("direct_query_observed", "[(value)]"),
        )
        self.assertEqual(
            classify_native_output(0, False, b"[(value)]\n", b"", 1),
            ("ambiguous_output_alignment", None),
        )
        self.assertEqual(
            classify_native_output(0, False, b"[(Error broken)]\n", b"", 0)[0],
            "native_error_atom",
        )
        self.assertEqual(
            classify_native_output(0, True, b"", b"", 0),
            ("timeout", None),
        )
        self.assertEqual(
            classify_native_output(2, False, b"", b"", 0),
            ("exit_nonzero", None),
        )
        self.assertEqual(
            classify_native_output(0, False, b"[(value)]\n", b"warn", 0),
            ("stderr_nonempty", None),
        )

    def test_existing_packets_remain_public_and_unapproved(self) -> None:
        self.assertEqual(len(self.rows), 187)
        self.assertTrue(all(row["metadata"]["training_ready"] is False for row in self.rows))
        self.assertTrue(
            all(row["metadata"]["independent_semantic_oracle"] is False for row in self.rows)
        )


if __name__ == "__main__":
    unittest.main()
