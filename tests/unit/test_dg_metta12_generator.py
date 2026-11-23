"""All registered reference roots must be reachable under a five-call ceiling."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "metta_research_bench" / "courier_transfer"
sys.path.insert(0, str(ROOT))
# The standalone reference gate runs without the optional benchmark package dependencies.
from generator import reference_root  # noqa: E402
from manifest_check import roots_for  # noqa: E402
from reference import FiniteOracle  # noqa: E402


class AllReferenceRootsTests(unittest.TestCase):
    def test_exact_oracle_five_call_feasibility_and_ties(self):
        manifest = json.loads((ROOT / "roots_manifest.json").read_text())
        roots = roots_for(manifest)
        self.assertEqual(len(roots), 520)
        tied = 0
        for root in roots:
            world, state = reference_root(root, manifest)
            oracle = FiniteOracle(world)
            self.assertTrue(oracle.can_finish(state, 5), root)
            options = oracle.action_set(state, 5)
            self.assertGreater(len(options), 0, root)
            tied += len(options) > 1
        self.assertGreater(tied, 0)

    def test_invalid_root_is_rejected(self):
        manifest = json.loads((ROOT / "roots_manifest.json").read_text())
        with self.assertRaises(ValueError):
            reference_root("DG12/frozen/route_ties/9999", manifest)


if __name__ == "__main__":
    unittest.main()
