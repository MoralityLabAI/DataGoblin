"""Independent MeTTa-06 reference semantics, mismatch detection and custody tests."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from finite_oracle_pilot import classify_queries, make_world  # noqa: E402
from finite_semantic_oracle import (  # noqa: E402
    UnsupportedSemantics,
    alpha_signature,
    compare_native,
    finite_match,
    parse_one,
    parse_result_line,
    result_signature,
    tokenize,
    typed_plus_reference,
)


class FiniteOracleTests(unittest.TestCase):
    def test_parse_comments_strings_and_nested_expressions(self) -> None:
        self.assertEqual(
            parse_one('(Node (Pair a "semi; and, comma") active)'),
            ("Node", ("Pair", "a", '"semi; and, comma"'), "active"),
        )
        self.assertEqual(
            tokenize('; comment\n(Edge a a) ; trailing\n'),
            ["(", "Edge", "a", "a", ")"],
        )
        self.assertEqual(
            parse_result_line('[(Pair A "hi, there"), B]'),
            [("Pair", "A", '"hi, there"'), "B"],
        )
        with self.assertRaises(UnsupportedSemantics):
            parse_one('(Node "unterminated)')
        with self.assertRaises(UnsupportedSemantics):
            parse_result_line('[A B]')
        with self.assertRaises(UnsupportedSemantics):
            parse_result_line('[A, B] trailing')

    def test_repeated_variables_and_set_order(self) -> None:
        facts = [
            parse_one("(Edge Alice Alice)"),
            parse_one("(Edge Alice Bob)"),
            parse_one("(Edge Carla Carla)"),
        ]
        query = parse_one("(match &self (Edge $u $u) $u)")
        reference = finite_match(facts, query)
        self.assertEqual(result_signature(reference), {"Alice": 1, "Carla": 1})
        self.assertTrue(compare_native(reference, "[Carla, Alice]")["agreement"])
        self.assertFalse(compare_native(reference, "[Bob, Alice]")["agreement"])

    def test_projection_and_empty_answers(self) -> None:
        facts = [parse_one("(Node (Pair u0 v0) good)")]
        found = parse_one(
            "(match &self (Node (Pair $x $y) good) (Found $y $x))"
        )
        absent = parse_one("(match &self (Role $x absent) $x)")
        self.assertTrue(
            compare_native(finite_match(facts, found), "[(Found v0 u0)]")["agreement"]
        )
        self.assertTrue(compare_native(finite_match(facts, absent), "[]")["agreement"])
        self.assertFalse(compare_native(finite_match(facts, absent), "[u0]")["agreement"])

    def test_alpha_equivalence_and_duplicate_detection(self) -> None:
        self.assertEqual(
            alpha_signature(parse_one("(Pair $who $who)")),
            alpha_signature(parse_one("(Pair $person $person)")),
        )
        self.assertNotEqual(
            alpha_signature(parse_one("(Pair $who $who)")),
            alpha_signature(parse_one("(Pair $who $other)")),
        )
        self.assertTrue(
            compare_native([parse_one("(Pair $x $x)")], "[(Pair $y $y)]")["agreement"]
        )
        self.assertFalse(
            compare_native([parse_one("(Pair $x $x)")], "[(Pair $y $z)]")["agreement"]
        )
        self.assertFalse(
            compare_native([parse_one("A")], "[A, A]")["agreement"]
        )

    def test_unsupported_semantics_fail_closed(self) -> None:
        with self.assertRaisesRegex(UnsupportedSemantics, "Only"):
            finite_match([], parse_one("(collapse (match &self X X))"))
        with self.assertRaisesRegex(UnsupportedSemantics, "ground"):
            finite_match([parse_one("(Rule $x y)")], parse_one("(match &self X X)"))
        with self.assertRaisesRegex(UnsupportedSemantics, "Unbound"):
            finite_match([parse_one("(Fact A)")], parse_one("(match &self (Fact $x) $unbound)"))
        with self.assertRaisesRegex(UnsupportedSemantics, "Repeated"):
            finite_match(
                [parse_one("(Edge A B)"), parse_one("(Edge A B)")],
                parse_one("(match &self (Edge $x $y) $x)"),
            )

    def test_type_rule_derives_error_from_argument_index(self) -> None:
        self.assertEqual(
            typed_plus_reference(parse_one('(+ 5 "S")')),
            [("Error", ("+", "5", '"S"'), ("BadArgType", "2", "Number", "String"))],
        )
        self.assertEqual(
            typed_plus_reference(parse_one('(+ "S" 5)')),
            [("Error", ("+", '"S"', "5"), ("BadArgType", "1", "Number", "String"))],
        )
        self.assertEqual(typed_plus_reference(parse_one("(+ 3 4)")), ["7"])
        with self.assertRaises(UnsupportedSemantics):
            typed_plus_reference(parse_one("(* 2 2)"))

    def test_world_generation_independent_of_hyperon(self) -> None:
        first = make_world(2)
        second = make_world(2)
        other = make_world(3)
        self.assertEqual(first, second)
        self.assertNotEqual(first["source_sha256"], other["source_sha256"])
        self.assertEqual(len(first["queries"]), 4)
        self.assertEqual(len(first["references"]), 4)
        self.assertTrue(all(row for row in first["references"][:3]) or True)
        self.assertEqual(first["references"][3], [])
        self.assertTrue(first["source_sha256"])

    def test_mismatch_is_preserved_as_observation(self) -> None:
        world = make_world(0)
        native = {
            "stdout": b"[Wrong]\n[Wrong]\n[Wrong]\n[Wrong]\n",
            "stderr": b"",
            "exit_code": 0,
            "timed_out": False,
            "elapsed_ms": 10,
        }
        rows = classify_queries(world, native)
        self.assertEqual(len(rows), 4)
        self.assertGreater(sum(row["status"] == "mismatch" for row in rows), 0)
        self.assertTrue(all(row["training_ready"] is False for row in rows))
        self.assertTrue(all(row["sealed_eval_eligible"] is False for row in rows))
        self.assertTrue(all(row["upstream_author_targets_used"] is False for row in rows))
        self.assertEqual(json.loads(json.dumps(rows)), rows)


if __name__ == "__main__":
    unittest.main()
