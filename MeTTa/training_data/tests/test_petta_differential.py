"""MeTTa-08 pinned PeTTa invocation and differential output contracts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from finite_oracle_pilot import make_world  # noqa: E402
from finite_semantic_oracle import (  # noqa: E402
    UnsupportedSemantics,
    render,
)
from petta_differential import (  # noqa: E402
    IMAGE_RE,
    PETTA_COMMIT,
    assess_hyperon,
    assess_petta,
    container_command,
    parse_petta_output,
    query_program,
)


class PettaDifferentialTests(unittest.TestCase):
    def test_petta_prints_one_result_per_line(self) -> None:
        self.assertEqual(parse_petta_output(""), [])
        self.assertEqual(parse_petta_output("Alice\nCarla\n"), ["Alice", "Carla"])
        self.assertEqual(
            parse_petta_output("(Found v0 u0)\n(Found v1 u1)\n"),
            [("Found", "v0", "u0"), ("Found", "v1", "u1")],
        )
        self.assertEqual(parse_petta_output("()\n"), [()])
        with self.assertRaises(UnsupportedSemantics):
            parse_petta_output("[Alice, Carla]")
        with self.assertRaises(UnsupportedSemantics):
            parse_petta_output("(Unclosed\n")

    def test_query_runs_are_separate_and_have_no_hidden_prior_queries(self) -> None:
        world = make_world(0)
        for index, query in enumerate(world["queries"]):
            script = query_program(world, index)
            self.assertTrue(script.endswith(f"!{query}\n"))
            self.assertEqual(script.count("!(match &self"), 1)
            self.assertIn("(Role u00_000", script)
            self.assertNotIn(f"!{world['queries'][(index + 1) % 4]}", script)
        with self.assertRaises(ValueError):
            query_program(world, 4)

    def test_reference_agreement_and_negative_control(self) -> None:
        world = make_world(0)
        for reference in world["references"]:
            rows = [render(value) for value in reference]
            run = {
                "stdout": ("\n".join(rows) + ("\n" if rows else "")).encode(),
                "stderr": b"",
                "exit_code": 0,
                "timed_out": False,
            }
            self.assertEqual(assess_petta(reference, run)["status"], "agree")
            changed = dict(run)
            changed["stdout"] = b"UnexpectedValue\n"
            self.assertEqual(assess_petta(reference, changed)["status"], "mismatch")

    def test_hyperon_observation_uses_its_own_output_syntax(self) -> None:
        world = make_world(0)
        rows = [
            f"[{', '.join(render(x) for x in values)}]"
            for values in world["references"]
        ]
        run = {
            "stdout": ("\n".join(rows) + "\n").encode(),
            "stderr": b"",
            "exit_code": 0,
            "timed_out": False,
        }
        cases = assess_hyperon(world, run)
        self.assertTrue(all(x["status"] == "agree" for x in cases))

    def test_process_errors_are_not_reported_as_agreement(self) -> None:
        world = make_world(0)
        run = {
            "stdout": b"boom",
            "stderr": b"error",
            "exit_code": 3,
            "timed_out": False,
        }
        self.assertEqual(
            assess_petta(world["references"][0], run)["status"],
            "runtime_or_alignment_failure",
        )
        self.assertTrue(
            all(
                x["status"] == "runtime_or_alignment_failure"
                for x in assess_hyperon(world, run)
            )
        )

    def test_immutable_image_and_networkless_resource_caps(self) -> None:
        digest = "ghcr.io/patham9/petta-ci@sha256:" + "f" * 64
        self.assertTrue(IMAGE_RE.fullmatch(digest))
        self.assertFalse(IMAGE_RE.fullmatch("ghcr.io/patham9/petta-ci:latest"))
        cmd = container_command(Path("/repo"), digest, Path("/data/world.metta"))
        self.assertIn("--network", cmd)
        self.assertIn("none", cmd)
        self.assertIn("--read-only", cmd)
        self.assertIn("--pids-limit", cmd)
        self.assertIn("1024m", cmd)
        self.assertIn(digest, cmd)
        self.assertIn("ee5cacb6", PETTA_COMMIT)
        self.assertTrue(any("/petta:ro" in item for item in cmd))
        self.assertTrue(any("/input/world.metta:ro" in item for item in cmd))


if __name__ == "__main__":
    unittest.main()
