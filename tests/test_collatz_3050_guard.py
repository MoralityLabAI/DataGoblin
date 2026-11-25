import json
import unittest
from pathlib import Path

from datagoblin.training_gym.collatz_3050_guard import build_guard, check_corpus, digest
from datagoblin.training_gym.collatz_corpus import admitted_record
from datagoblin.training_gym.collatz_verifier import parameter_split


def manifest():
    return json.loads((Path(__file__).parent/"fixtures"/"col3050_development.json").read_text())


class GuardTests(unittest.TestCase):
    def test_pinned_roles(self):
        m = manifest()
        g = build_guard(m)
        self.assertEqual(g["verified_role_count"], 12)
        self.assertEqual(g["exposed_raw_parameters"], [13, 17, 25, 73])
        self.assertEqual(g["training_rows_admitted"], 0)
        self.assertGreater(g["checked_full_map_steps"], 12)

    def test_rehashed_false_trace(self):
        m = manifest()
        m["cases"][0]["recipient"]["trace"][-1] = 3
        m["manifest_sha256"] = digest(m)
        with self.assertRaises(ValueError):
            build_guard(m)

    def test_raw_and_scaled_quarantine(self):
        g = build_guard(manifest())
        root = next(d for d in g["exposed_raw_parameters"] if parameter_split(d) == "train")
        candidates = [admitted_record(root, 1)]
        # Exact scaled duplicate of a protected root; choose a legal training parameter.
        scale = next(s for s in range(5, 100, 4) if s % 3 and parameter_split(root * s) == "train")
        candidates.append(admitted_record(root * scale, scale))
        result = check_corpus(g, candidates)
        self.assertFalse(result["passed"])
        self.assertEqual(result["training_overlap_count"], 2)
