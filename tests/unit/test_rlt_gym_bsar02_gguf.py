"""Offline GGUF client controls using a fake loopback HTTP response."""

import json
from pathlib import Path

import pytest

from training.rlt_gym.bsar02 import local_gguf


class FakeReply:
    def __init__(self, text: str):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps({
            "choices": [{"message": {"content": self.text}}],
            "usage": {"total_tokens": 47},
        }).encode("utf-8")


def test_endpoint_accepts_loopback_only():
    for host in ("http://127.0.0.1:8080", "http://localhost:8000"):
        value = host + "/v1/chat/completions"
        assert local_gguf.validated_endpoint(value) == value
    for bad in (
        "https://api.example.com/v1/chat/completions",
        "http://10.0.0.1:8080/v1/chat/completions",
        "http://127.0.0.1:8080/other",
        "http://user:pass@localhost/v1/chat/completions",
    ):
        with pytest.raises(ValueError):
            local_gguf.validated_endpoint(bad)


def test_parse_strict_single_label_response():
    assert local_gguf.label_from_response("UNSUPPORTED_PDE_LIFT") == "UNSUPPORTED_PDE_LIFT"
    assert local_gguf.label_from_response("`MISSING_PREMISE`") == "MISSING_PREMISE"
    for raw in ("the theorem proves PDE regularity", "UNSUPPORTED_PDE_LIFT because...", ""):
        assert local_gguf.label_from_response(raw) == "ABSTAIN"


def test_mock_llama_server_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured = []

    def fake_open(request, timeout):
        request_data = json.loads(request.data.decode("utf-8"))
        captured.append((request_data, timeout))
        return FakeReply("KERNEL_CHECK_REQUIRED")

    monkeypatch.setattr(local_gguf, "urlopen", fake_open)
    path = tmp_path / "answers.jsonl"
    endpoint = "http://127.0.0.1:8080/v1/chat/completions"
    result = local_gguf.run(path, endpoint, "inky-mini", limit=2)
    assert result["new_requests"] == 2
    assert result["coverage"] == pytest.approx(2 / 164)
    assert len(captured) == 2
    assert all(c[0]["temperature"] == 0 for c in captured)
    assert all(c[0]["max_tokens"] == 48 for c in captured)
    assert all("gold_label" not in c[0]["messages"][-1]["content"] for c in captured)
    assert local_gguf.run(path, endpoint, "inky-mini", limit=1)["new_requests"] == 1
    with pytest.raises(ValueError, match="another or unknown model"):
        local_gguf.run(path, endpoint, "bon-8b", limit=1)
