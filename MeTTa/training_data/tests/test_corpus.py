"""Offline tests for pinned source provenance, export and training custody."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from corpus import safe_file, validate_corpus  # noqa: E402
from export import export  # noqa: E402


class MeTTaCorpusTests(unittest.TestCase):
    def test_seed_is_valid_and_unverified(self) -> None:
        rows, summary, errors = validate_corpus(ROOT)
        self.assertFalse(errors, errors)
        self.assertGreaterEqual(len(rows), 54)
        self.assertGreaterEqual(summary["repos"], 12)
        self.assertGreater(summary["embedded_assertions_unverified"], 0)
        self.assertEqual(summary["runtime_passed"], 0)
        self.assertEqual(summary["verified_rollouts"], 0)

    def test_export_keeps_source_status(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            output = Path(dirname) / "sources.jsonl"
            amount = export(ROOT, output, "source")
            entries = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(amount, len(entries))
            self.assertTrue(all(entry["metadata"]["verification_status"] == "not_run" for entry in entries))
            self.assertTrue(all("text" in entry and "metadata" in entry for entry in entries))
            self.assertTrue(all("trajectory" not in entry for entry in entries))

    def test_rejects_unsafe_path(self) -> None:
        for candidate in ("../outside.metta", "/etc/passwd", "raw\\x.metta", ""):
            with self.assertRaises(ValueError):
                safe_file(ROOT, candidate)

    def test_detects_mutated_upstream(self) -> None:
        rows, _, _ = validate_corpus(ROOT)
        source = rows[0]
        with tempfile.TemporaryDirectory() as dirname:
            trial = Path(dirname)
            raw = trial / source["raw_path"]
            raw.parent.mkdir(parents=True)
            raw.write_bytes((ROOT / source["raw_path"]).read_bytes() + b"\n; mutation\n")
            annotation_path = trial / "annotations" / source["source_repo"].replace("/", "--") / (
                source["source_path"] + ".json"
            )
            annotation_path.parent.mkdir(parents=True)
            annotation_path.write_text(json.dumps(source), encoding="utf-8")
            license_path = trial / "licenses" / source["source_repo"].replace("/", "--") / "LICENSE"
            license_path.parent.mkdir(parents=True)
            shutil.copyfile(ROOT / "licenses" / source["source_repo"].replace("/", "--") / "LICENSE", license_path)
            _, _, errors = validate_corpus(trial)
            self.assertTrue(any("Git blob SHA mismatch" in issue for issue in errors), errors)


if __name__ == "__main__":
    unittest.main()
