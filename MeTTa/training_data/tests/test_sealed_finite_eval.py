"""MeTTa-07 private-key evaluation commitments and tamper resistance."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from finite_semantic_oracle import render  # noqa: E402
from sealed_finite_eval import (  # noqa: E402
    challenge_bundle,
    prepare,
    private_world,
    read_key,
    score,
)


class SealedFiniteEvalTests(unittest.TestCase):
    KEY = b"PUBLIC-DUMMY-KEY-FOR-UNIT-TESTS-ONLY:" + b"x" * 48

    @staticmethod
    def _rows(raw: bytes) -> list[dict]:
        return [json.loads(s) for s in raw.splitlines()]

    @staticmethod
    def _write_predictions(path: Path, rows: list[dict], answers: dict) -> None:
        path.write_text(
            "".join(
                json.dumps({
                    "challenge_id": row["challenge_id"],
                    "answers": [render(x) for x in answers[row["challenge_id"]]],
                }, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    def test_keyed_worlds_vary_and_publish_no_seed(self) -> None:
        a = private_world(self.KEY, 0)
        b = private_world(self.KEY, 1)
        c = private_world(self.KEY + b"x", 0)
        self.assertNotEqual(a["root_id"], b["root_id"])
        self.assertNotEqual(a["root_id"], c["root_id"])
        self.assertNotEqual(a["source_prefix"], b["source_prefix"])
        self.assertEqual(len(a["reference"]), 4)
        self.assertIn("(Bridge", a["source_prefix"])
        self.assertNotIn(self.KEY.decode(), a["source_prefix"])
        with self.assertRaises(ValueError):
            private_world(b"x", 0)
        with self.assertRaises(ValueError):
            private_world(self.KEY, 256)

    def test_prepare_scoring_and_no_reference_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            secret = base / "secret.bin"
            challenge = base / "challenge.jsonl"
            commitment = base / "commitment.json"
            preds = base / "predictions.jsonl"
            summary = base / "aggregate.json"
            secret.write_bytes(self.KEY)
            manifest = prepare(secret, challenge, commitment, roots=4)
            raw, expected_manifest, refs = challenge_bundle(self.KEY, roots=4)
            self.assertEqual(manifest, expected_manifest)
            self.assertEqual(challenge.read_bytes(), raw)
            rows = self._rows(raw)
            self.assertEqual(len(rows), 16)
            self.assertEqual(len({row["root_id"] for row in rows}), 4)
            self.assertTrue(all("reference" not in row for row in rows))
            self.assertTrue(all("expected" not in row for row in rows))
            self.assertTrue(all(row["training_ready"] is False for row in rows))
            self.assertNotIn(self.KEY, raw)
            self._write_predictions(preds, rows, refs)
            report = score(secret, challenge, commitment, preds, summary)
            self.assertEqual(report["correct"], 16)
            self.assertEqual(report["accuracy"], 1.0)
            self.assertEqual(report["missing"], 0)
            self.assertFalse(report["answers_written_to_report"])
            self.assertFalse(report["external_custody_attested"])
            self.assertNotIn("reference", summary.read_text())
            self.assertNotIn(self.KEY, summary.read_bytes())
            self.assertEqual(report, json.loads(summary.read_text()))

    def test_wrong_answers_are_not_silently_approved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            key_path = base / "secret.bin"
            key_path.write_bytes(self.KEY)
            c, m, p, report_path = [
                base / name for name in
                ("challenges.jsonl", "manifest.json", "pred.jsonl", "score.json")
            ]
            prepare(key_path, c, m, 2)
            rows = self._rows(c.read_bytes())
            _, _, references = challenge_bundle(self.KEY, 2)
            self._write_predictions(p, rows, references)
            predictions = p.read_text().splitlines()
            first = json.loads(predictions[0])
            first["answers"] = ["IncorrectAnswer"]
            predictions[0] = json.dumps(first, sort_keys=True)
            p.write_text("\n".join(predictions) + "\n")
            report = score(key_path, c, m, p, report_path)
            self.assertEqual(report["challenges"], 8)
            self.assertEqual(report["correct"], 7)
            self.assertEqual(report["incorrect_or_missing"], 1)

    def test_partial_invalid_and_duplicated_submissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            key = base / "key.bin"
            c, m, p, s = [base / n for n in ("a.jsonl", "a.json", "p.jsonl", "s.json")]
            key.write_bytes(self.KEY)
            prepare(key, c, m, 2)
            rows = self._rows(c.read_bytes())
            p.write_text(
                json.dumps({"challenge_id": rows[0]["challenge_id"], "answers": ["(unbalanced"]}) + "\n"
            )
            report = score(key, c, m, p, s)
            self.assertEqual(report["missing"], 7)
            self.assertEqual(report["invalid_syntax"], 1)
            self.assertEqual(report["correct"], 0)
            p.write_text(
                json.dumps({"challenge_id": rows[0]["challenge_id"], "answers": []}) + "\n"
                + json.dumps({"challenge_id": rows[0]["challenge_id"], "answers": []}) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "duplicated"):
                score(key, c, m, p, s)

    def test_changed_commitment_or_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            secret = base / "key.bin"
            challenge, commitment, preds, summary = [
                base / n for n in ("challenge.jsonl", "commitment.json", "preds", "report")
            ]
            secret.write_bytes(self.KEY)
            prepare(secret, challenge, commitment, 2)
            preds.write_text("")
            secret.write_bytes(self.KEY + b"x")
            with self.assertRaisesRegex(ValueError, "identity"):
                score(secret, challenge, commitment, preds, summary)
            secret.write_bytes(self.KEY)
            challenge.write_bytes(challenge.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "identity"):
                score(secret, challenge, commitment, preds, summary)
            with self.assertRaises(ValueError):
                (base / "short.bin").write_bytes(b"weak")
                read_key(base / "short.bin")


if __name__ == "__main__":
    unittest.main()
