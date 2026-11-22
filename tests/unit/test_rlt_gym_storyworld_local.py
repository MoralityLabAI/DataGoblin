"""Offline contract tests for the opt-in, turn-by-turn local Storyworld player."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from datagoblin.training_gym.rlt_gym.storyworld_local import (
    _read_lines,
    parse_output,
    prompt_for,
    replay_run,
    run,
    validated_endpoint,
)
from datagoblin.training_gym.rlt_gym.storyworld_native import collect, connect


def _fake_response(action: str) -> dict:
    return {"model": "inky-mini-stub", "choices": [{"message": {"content": action}}],
            "usage": {"prompt_tokens": 160, "completion_tokens": 4, "total_tokens": 164}}


def _public_oracle(payload: dict) -> dict:
    """Choose HG-02's native legal plan from the observation alone, without sealed state."""
    body = json.loads(payload["messages"][1]["content"])
    mode, obs = body["mode"], body["observation"]
    if mode != "hypergame":
        return _fake_response("PICK " + obs["candidates"][0]["id"])
    state, spec = obs["visible_state"], obs["public_spec"]
    if state["phase"] == "route":
        if not state["audited"]:
            return _fake_response("audit")
        vector = state["vector"]
        parity = sum(vector[i] for i in spec["parity_indices"]) % 2
        hidden = vector[spec["hidden_index"]]
        return _fake_response(f"route_{parity ^ hidden}")
    if state["phase"] == "verify":
        n = sum((i % 3 + 1) * state["vector"][i] for i in spec["verification_indices"]) % 3
        return _fake_response(f"verify_{n}")
    if state["phase"] == "authorize":
        n = sum((i % 5 + 1) * state["vector"][i] for i in spec["authorization_indices"]) % 5
        return _fake_response(f"authorize_{n}")
    return _fake_response("publish")


class StoryworldLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = os.environ.get("GPTSTORYWORLD_SOURCE")
        if not source:
            raise unittest.SkipTest("set GPTSTORYWORLD_SOURCE to enable local player tests")
        cls.native = connect(Path(source), expect_commit=os.environ.get("GPTSTORYWORLD_EXPECT_SHA"))

    def test_endpoint_and_parse_contract(self):
        self.assertEqual(
            validated_endpoint("http://127.0.0.1:8080/v1/chat/completions"),
            "http://127.0.0.1:8080/v1/chat/completions",
        )
        for url in (
            "https://127.0.0.1:8080/v1/chat/completions",
            "http://api.example.com:8080/v1/chat/completions",
            "http://user:password@localhost:8080/v1/chat/completions",
            "http://localhost:8080/v1/chat/completions?token=x",
            "http://localhost:8080/not-local",
            "http://localhost/v1/chat/completions",
        ):
            with self.assertRaises(ValueError):
                validated_endpoint(url)
        self.assertEqual(parse_output("hypergame", "ACTION audit", ["audit"]), "audit")
        self.assertEqual(parse_output("hypergame", "audit\nexplanation", ["audit"]), "")
        self.assertEqual(parse_output("mini", "PICK enc_ending_1", ["enc_ending_1"]), "PICK enc_ending_1")
        self.assertEqual(parse_output("mini", "enc_ending_1", ["enc_ending_1"]), "")
        with self.assertRaises(ValueError):
            prompt_for("mini", {"text": "word" * 5000}, context_budget=8192, output_tokens=96)
        with self.assertRaises(ValueError):
            prompt_for("mini", {"candidates": []}, context_budget=16384, output_tokens=96)

    def test_interactive_native_hypergame_twin_rollout_and_replay(self):
        with tempfile.TemporaryDirectory(prefix="dg-sw-local-") as tmp:
            root = Path(tmp)
            collect(self.native, root / "archive", seed=23, per_mode=2)
            report = run(self.native, root / "archive", root / "model",
                         model="inky-mini-stub", mode="hypergame", split="train",
                         max_roots=4, transport=_public_oracle)
            self.assertEqual(report["metrics"]["graded"], 4)
            self.assertEqual(report["metrics"]["accepted"], 4)
            self.assertEqual(report["metrics"]["requests"], 20)
            self.assertEqual(report["metrics"].get("unsafe_releases", 0), 0)
            events = _read_lines(root / "model/events.jsonl")
            self.assertEqual(len(events), 20)
            self.assertNotIn('"private_bit"', (root / "model/events.jsonl").read_text())
            for root_id in {r["root"] for r in events}:
                steps = [e for e in events if e["root"] == root_id]
                spec = steps[0]["observation"]["public_spec"]
                hidden = spec["hidden_index"]
                self.assertIsNone(steps[0]["observation"]["visible_state"]["vector"][hidden])
                self.assertIn(steps[1]["observation"]["visible_state"]["vector"][hidden], (0, 1))
                self.assertEqual([e["action"] for e in steps][0], "audit")
            self.assertEqual(replay_run(self.native, root / "archive", root / "model")["status"], "verified")
            self.assertEqual({r["grade"]["true_utility"] for r in
                              _read_lines(root / "model/results.jsonl")}, {"9"})
            with self.assertRaises(FileExistsError):
                run(self.native, root / "archive", root / "model",
                    model="inky-mini-stub", mode="hypergame", transport=_public_oracle)

    def test_builder_and_mini_model_submissions_replay(self):
        with tempfile.TemporaryDirectory(prefix="dg-sw-local-") as tmp:
            root = Path(tmp)
            collect(self.native, root / "archive", seed=23, per_mode=2)
            for mode in ("builder", "mini"):
                report = run(self.native, root / "archive", root / mode,
                             model="inky-mini-stub", mode=mode, split="train",
                             max_roots=2, transport=_public_oracle)
                self.assertEqual(report["metrics"]["graded"], 2)
                self.assertEqual(report["metrics"]["requests"], 2)
                self.assertEqual(replay_run(self.native, root / "archive", root / mode)["status"], "verified")
                for event in _read_lines(root / mode / "events.jsonl"):
                    self.assertNotIn("gold_target", json.dumps(event))
                    self.assertTrue(event["parse_ok"])

    def test_interrupted_local_rollout_resumes_from_public_observation(self):
        with tempfile.TemporaryDirectory(prefix="dg-sw-resume-") as tmp:
            root = Path(tmp)
            collect(self.native, root / "archive", seed=23, per_mode=1)
            calls = 0

            def fail_after_audit(payload: dict) -> dict:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise ConnectionError("test injection of interrupted local endpoint")
                return _public_oracle(payload)

            with self.assertRaisesRegex(ConnectionError, "interrupted"):
                run(self.native, root / "archive", root / "model", model="inky-mini-stub",
                    mode="hypergame", split="train", max_roots=2, transport=fail_after_audit)
            prefix = (root / "model/events.jsonl").read_bytes()
            self.assertEqual(len(_read_lines(root / "model/events.jsonl")), 1)
            complete = run(self.native, root / "archive", root / "model",
                           model="inky-mini-stub", mode="hypergame", split="train",
                           max_roots=2, resume=True, transport=_public_oracle)
            self.assertTrue((root / "model/events.jsonl").read_bytes().startswith(prefix))
            self.assertEqual(complete["metrics"]["accepted"], 2)
            self.assertEqual(replay_run(self.native, root / "archive", root / "model")["status"], "verified")
            with self.assertRaisesRegex(ValueError, "configuration"):
                run(self.native, root / "archive", root / "model",
                    model="different-model", mode="hypergame", split="train",
                    max_roots=2, resume=True, transport=_public_oracle)

    def test_replay_rejects_tampered_actions_and_unsafe_proxy(self):
        with tempfile.TemporaryDirectory(prefix="dg-sw-replay-") as tmp:
            root = Path(tmp)
            collect(self.native, root / "archive", seed=23, per_mode=1)
            report = run(self.native, root / "archive", root / "model",
                         model="inky-mini-stub", mode="hypergame", split="train",
                         max_roots=2, transport=lambda _: _fake_response("shortcut"))
            self.assertEqual(report["metrics"]["accepted"], 0)
            self.assertEqual(report["metrics"]["unsafe_releases"], 2)
            self.assertEqual(replay_run(self.native, root / "archive", root / "model")["status"], "verified")
            path = root / "model/events.jsonl"
            path.write_bytes(path.read_bytes().replace(b'"action":"shortcut"', b'"action":"audit"', 1))
            with self.assertRaisesRegex(ValueError, "action does not match"):
                replay_run(self.native, root / "archive", root / "model")


if __name__ == "__main__":
    unittest.main()
