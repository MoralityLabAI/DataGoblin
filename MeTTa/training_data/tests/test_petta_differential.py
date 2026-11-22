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
)


class PettaDifferentialTests(unittest.TestCase):
    def test_petta_prints_a_metta_list_of_results(self) -> None:
        self.assertEqual(parse_petta_output("()"), [])
        self.assertEqual(
            parse_petta_output("(Alice Carla)"), ["Alice", "Carla"]
        )
        self.assertEqual(
            parse_petta_output("((Found v0 u0) (Found v1 u1))"),
            [("Found", "v0", "u0"), ("Found", "v1", "u1")],
        )
        with self.assertRaises(UnsupportedSemantics):
            parse_petta_output("[Alice, Carla]")
        with self.assertRaises(UnsupportedSemantics):
            parse_petta_output("Alice")

    def test_reference_agreement_and_negative_control(self) -> None:
        world = make_world(0)
        pe_lines = [
            render(tuple(values)) for values in world["references"]
        ]
        run = {
            "stdout": ("\n".join(pe_lines) + "\n").encode(),
            "stderr": b"",
            "exit_code": 0,
            "timed_out": False,
        }
        evaluations = assess_petta(world, run)
        self.assertEqual([x["status"] for x in evaluations], ["agree"] * 4)
        modified = dict(run)
        modified["stdout"] = b"(Unexpected)\n" + run["stdout"].split(b"\n", 1)[1]
        failures = assess_petta(world, modified)
        self.assertEqual(failures[0]["status"], "mismatch")
        self.assertTrue(all(x["status"] == "agree" for x in failures[1:]))

    def test_hyperon_observation_uses_its_own_output_syntax(self) -> None:
        world = make_world(0)
        reference = world["references"]
        rows = [f"[{', '.join(render(x) for x in values)}]" for values in reference]
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
        self.assertTrue(
            all(x["status"] == "runtime_or_alignment_failure"
                for x in assess_petta(world, run))
        )
        self.assertTrue(
            all(x["status"] == "runtime_or_alignment_failure"
                for x in assess_hyperon(world, run))
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
