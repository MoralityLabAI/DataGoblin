"""MeTTa-09: model submission contract, mock inference and private-key custody."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from finite_semantic_oracle import render  # noqa: E402
from local_model_submit import (  # noqa: E402
    RejectRedirects,
    parse_completion,
    prompt_payload,
    read_challenges,
    submit,
    validate_endpoint,
)
from sealed_finite_eval import challenge_bundle, score  # noqa: E402


class LocalModelSubmitTests(unittest.TestCase):
    KEY = b"PUBLIC-METTA09-MOCK-KEY-ONLY:" + b"k" * 48
    ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"

    def test_requires_loopback_server(self) -> None:
        self.assertEqual(validate_endpoint(self.ENDPOINT), self.ENDPOINT)
        self.assertIsNone(RejectRedirects().redirect_request(None, None, 302, "moved", {}, "https://remote.example/v1/chat/completions"))
        self.assertTrue(
            validate_endpoint("http://[::1]:8080/v1/chat/completions")
        )
        for target in (
            "http://localhost:8000/v1/chat/completions",
            "https://127.0.0.1:8000/v1/chat/completions",
            "http://192.168.1.20:8000/v1/chat/completions",
            "http://localhost.attacker.com:8000/v1/chat/completions",
            "http://127.0.0.1@attacker.com:8000/v1/chat/completions",
            "http://127.0.0.1:8000/v1/chat/completions?proxy=x",
            "http://127.0.0.1:8000/anything",
        ):
            with self.subTest(endpoint=target), self.assertRaises(ValueError):
                validate_endpoint(target)

    def test_completion_parsing_distinguishes_invalid_and_empty(self) -> None:
        def response(value: str) -> dict:
            return {"choices": [{"message": {"content": value}}]}
        self.assertEqual(parse_completion(response('{"answers":[]}')), ([], "parsed"))
        self.assertEqual(
            parse_completion(response('{"answers":["A","(Selected n01)"]}')),
            (["A", "(Selected n01)"], "parsed"),
        )
        self.assertEqual(parse_completion(response("plain text")), (None, "malformed_json"))
        self.assertEqual(
            parse_completion(response('{"answers":{} }')), (None, "invalid_answer_list")
        )
        self.assertEqual(
            parse_completion(response('{"answers":[],"secret":"no"}')),
            (None, "invalid_answer_object"),
        )
        self.assertEqual(parse_completion({}), (None, "missing_choices"))

    def test_mock_submissions_score_only_inside_private_scorer(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            key = directory / "key.bin"
            challenges = directory / "challenges.jsonl"
            commitment = directory / "commitment.json"
            predictions = directory / "predictions.jsonl"
            receipt = directory / "receipt.json"
            aggregate = directory / "aggregate.json"
            key.write_bytes(self.KEY)
            raw, manifest, references = challenge_bundle(self.KEY, 2)
            challenges.write_bytes(raw)
            commitment.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            cases = read_challenges(challenges)
            self.assertEqual(len(cases), 8)
            expected_ids = [case["challenge_id"] for case in cases]
            issued = iter(expected_ids)

            def public_mock(_endpoint: str, payload: dict, _timeout: int) -> dict:
                challenge_id = next(issued)
                query = json.loads(payload["messages"][1]["content"])["query"]
                self.assertTrue(query.startswith("(match &self "))
                self.assertNotIn(self.KEY.decode(), json.dumps(payload))
                return {
                    "model": "bonsai-8b-local",
                    "choices": [{
                        "message": {"content": json.dumps({
                            "answers": [render(x) for x in references[challenge_id]],
                        })}
                    }],
                    "usage": {"prompt_tokens": 320, "completion_tokens": 48},
                }

            report = submit(
                challenges, predictions, receipt,
                endpoint=self.ENDPOINT, model_id="bonsai-8b-local",
                model_revision="public-mock", max_cases=8,
                model_manifest_sha256="a" * 64, transport=public_mock,
            )
            self.assertEqual(report["submitted"], 8)
            self.assertEqual(report["statuses"], {"parsed": 8})
            self.assertEqual(report["known_prompt_tokens"], 8 * 320)
            self.assertEqual(report["unknown_token_usage_cases"], 0)
            self.assertFalse(report["model_weights_independently_verified"])
            self.assertFalse(report["sealed_model_result_attested"])
            results = [json.loads(x) for x in predictions.read_text().splitlines()]
            self.assertEqual([x["challenge_id"] for x in results], expected_ids)
            self.assertTrue(all(isinstance(x["answers"], list) for x in results))
            self.assertNotIn("source_prefix", receipt.read_text())
            self.assertNotIn(self.KEY, receipt.read_bytes())
            scored = score(key, challenges, commitment, predictions, aggregate)
            self.assertEqual(scored["correct"], 8)
            self.assertFalse(scored["external_custody_attested"])

    def test_malformed_completion_is_null_and_cannot_score_empty_query(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            cases, _, _ = challenge_bundle(self.KEY, 1)
            challenges = directory / "challenges.jsonl"
            predictions = directory / "predictions.jsonl"
            receipt = directory / "receipt.json"
            challenges.write_bytes(cases)
            def invalid(_endpoint: str, _payload: dict, _timeout: int) -> dict:
                return {"choices": [{"message": {"content": "invalid response"}}]}
            report = submit(
                challenges, predictions, receipt,
                endpoint=self.ENDPOINT, model_id="bonsai-27b-local",
                model_revision="dev-test", max_cases=4, transport=invalid,
            )
            self.assertEqual(report["statuses"], {"malformed_json": 4})
            self.assertEqual(report["unknown_token_usage_cases"], 4)
            self.assertEqual(report["submitted"], 4)
            rows = [json.loads(line) for line in predictions.read_text().splitlines()]
            self.assertTrue(all(row["answers"] is None for row in rows))

    def test_context_over_budget_and_bad_source_guard(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            raw, _, _ = challenge_bundle(self.KEY, 1)
            challenges = directory / "a.jsonl"
            challenges.write_bytes(raw)
            def over(_endpoint: str, _payload: dict, _timeout: int) -> dict:
                return {
                    "choices": [{"message": {"content": '{"answers":[]}'}}],
                    "usage": {"prompt_tokens": 12500, "completion_tokens": 1},
                }
            report = submit(
                challenges, directory / "out.jsonl", directory / "receipt.json",
                endpoint=self.ENDPOINT, model_id="model",
                model_revision="test", max_cases=1,
                context_window_tokens=12000, transport=over,
            )
            self.assertEqual(report["statuses"], {"context_over_budget": 1})
            self.assertIsNone(json.loads((directory / "out.jsonl").read_text())["answers"])
            row = json.loads(raw.splitlines()[0])
            row["reference"] = "leaked"
            challenges.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "Invalid"):
                read_challenges(challenges)

    def test_prompt_contains_only_challenge_and_runtime_contract(self) -> None:
        raw, _, _ = challenge_bundle(self.KEY, 1)
        sample = json.loads(raw.splitlines()[0])
        payload = prompt_payload(sample, "bonsai-8b", 256)
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["seed"], 0)
        message = json.loads(payload["messages"][1]["content"])
        self.assertEqual(set(message), {"query", "source_prefix"})
        self.assertTrue(message["query"].startswith("(match &self "))


if __name__ == "__main__":
    unittest.main()
