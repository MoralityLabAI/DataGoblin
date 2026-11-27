"""Local HTTP fixtures for token, identity, deadline and credential boundaries."""

import copy
import json
import threading
import time
from contextlib import contextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from datagoblin.research_v2.transport import (
    MAX_PROMPT_BYTES,
    MAX_REPLY_BYTES,
    MAX_RESPONSE_BYTES,
    PROTOCOL_ID,
    SCHEMA,
    TransportError,
    request_action,
    validate_binding,
)


def identity(protocol="raw-completion"):
    entry = {
        "id": "fixture", "model_id": "test/fixture", "revision": "a" * 40,
        "weights_sha256": "1" * 64, "precision": "bf16", "recurrence": {"passes": 2},
        "protocol": protocol, "context_length": 4096,
        "config_sha256": "6" * 64, "tokenizer_config_sha256": "7" * 64,
    }
    binding = {
        "schema": SCHEMA, "registry_id": entry["id"], "model_id": entry["model_id"],
        "served_model": "fixture-served", "revision": entry["revision"],
        "weights_sha256": entry["weights_sha256"], "tokenizer_sha256": "2" * 64,
        "template_sha256": "3" * 64, "precision": entry["precision"],
        "recurrence": copy.deepcopy(entry["recurrence"]), "protocol": protocol,
        "base_url": "http://127.0.0.1:8000/v1", "api_key_env": None,
        "tokenize_path": "/tokenize", "prompt_role": "raw", "thinking_mode": "not_applicable",
        "context_limit": 2048, "resource_caps": {"wall_seconds": 10, "ram_mib": 512, "max_requests": 16},
        "cap_wrapper_receipt_sha256": "4" * 64, "config_sha256": "6" * 64,
        "tokenizer_config_sha256": "7" * 64,
    }
    if protocol == "chat-completions":
        binding.update(prompt_role="user", thinking_mode="disabled", chat_template_kwargs={"enable_thinking": False})
    if protocol == "prefix-lm-completion":
        binding.update(hrm_condition="synth,cot", prefix_lm_prefill_ones=True)
    return entry, binding


@contextmanager
def fixture_server(responder):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "body": body, "authorization": self.headers.get("Authorization")})
            status, payload, headers = responder(self.path, body)
            raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def response(path, body, *, count=12, reply='{"action":"inspect"}', overrides=None):
    if path == "/tokenize":
        return 200, {"count": count, "model": body["model"]}, {}
    choice = {"finish_reason": "stop"}
    choice.update({"message": {"role": "assistant", "content": reply}} if path.endswith("/chat/completions")
                  else {"text": reply})
    value = {"model": body["model"], "choices": [choice],
             "usage": {"prompt_tokens": count, "completion_tokens": 5, "total_tokens": count + 5}}
    value.update(overrides or {})
    return 200, value, {}


@pytest.mark.parametrize("protocol", ["raw-completion", "chat-completions", "prefix-lm-completion"])
def test_transport_primary_protocol_rendering_and_accounting(protocol):
    entry, binding = identity(protocol)
    prompt = 'Choose one JSON action for the public task: {"case":1}'
    with fixture_server(response) as (url, seen):
        binding["base_url"] = url
        result = request_action(entry, binding, prompt, max_tokens=64, timeout_seconds=2)
    assert result["text"] == '{"action":"inspect"}'
    assert result["usage"] == {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    assert result["http_requests"] == 2 and result["inference_requests"] == 1
    assert result["identity_verification"] == "operator_attested"
    assert result["protocol_id"] == PROTOCOL_ID
    before, after = [item["body"] for item in seen]
    if protocol == "chat-completions":
        assert before["messages"] == after["messages"] == [{"role": "user", "content": prompt}]
        assert before["chat_template_kwargs"] == after["chat_template_kwargs"] == {"enable_thinking": False}
        assert seen[1]["path"] == "/v1/chat/completions"
    else:
        expected = (f"<|im_start|><|quad_end|><|object_ref_end|>{prompt}<|im_end|>"
                    if protocol == "prefix-lm-completion" else prompt)
        assert before["prompt"] == after["prompt"] == expected
        assert seen[1]["path"] == "/v1/completions"
        if protocol == "prefix-lm-completion":
            assert after["token_type_ids"] == [1] * 12
    assert "max_tokens" not in before
    assert after["max_tokens"] == 64
    assert after["temperature"] == 0 and after["top_p"] == 1 and after["seed"] == 0


def test_transport_exact_context_preflight_prevents_generation():
    entry, binding = identity()
    with fixture_server(lambda p, b: response(p, b, count=2000)) as (url, seen):
        binding["base_url"] = url
        with pytest.raises(TransportError, match="context_budget_exceeded") as caught:
            request_action(entry, binding, "prompt", max_tokens=64, timeout_seconds=2)
        assert len(seen) == 1
    assert caught.value.receipt["http_requests"] == 1
    assert caught.value.receipt["inference_requests"] == 0


@pytest.mark.parametrize(("overrides", "code"), [
    ({"usage": None}, "token_usage_required"),
    ({"model": "another-model"}, "served_model_mismatch"),
    ({"usage": {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18}}, "tokenizer_usage_mismatch"),
    ({"usage": {"prompt_tokens": 12, "completion_tokens": 65, "total_tokens": 77}}, "generation_token_limit_exceeded"),
    ({"usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 18}}, "token_usage_total_mismatch"),
    ({"usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17,
                "completion_tokens_details": {"reasoning_tokens": 6}}}, "reasoning_tokens_exceed_completion_usage"),
    ({"choices": [{"text": "x", "finish_reason": "length"}]}, "generation_truncated"),
])
def test_transport_rejects_unaccounted_or_wrong_model_outputs(overrides, code):
    entry, binding = identity()
    with fixture_server(lambda p, b: response(p, b, overrides=overrides)) as (url, _):
        binding["base_url"] = url
        with pytest.raises(TransportError, match=code) as caught:
            request_action(entry, binding, "prompt", max_tokens=64, timeout_seconds=2)
    assert caught.value.receipt["http_requests"] == 2
    assert caught.value.receipt["inference_requests"] == 1


def test_transport_one_deadline_covers_tokenization_and_generation():
    def delayed(path, body):
        time.sleep(0.10 if path == "/tokenize" else 0.18)
        return response(path, body)

    entry, binding = identity()
    with fixture_server(delayed) as (url, _):
        binding["base_url"] = url
        started = time.monotonic()
        with pytest.raises(TransportError, match="request_deadline_exceeded") as caught:
            request_action(entry, binding, "prompt", max_tokens=64, timeout_seconds=0.20)
        elapsed = time.monotonic() - started
    assert elapsed < 0.26
    assert caught.value.receipt["http_requests"] == 2
    assert caught.value.receipt["inference_requests"] == 1


def test_transport_redirect_does_not_forward_credentials(monkeypatch):
    entry, binding = identity()
    monkeypatch.setenv("DATAGOBLIN_TRANSPORT_FIXTURE_KEY", "fixture-secret-value")
    binding["api_key_env"] = "DATAGOBLIN_TRANSPORT_FIXTURE_KEY"
    def redirect(_path, _body):
        return 302, {"error": "fixture-secret-value"}, {"Location": "/stolen"}

    with fixture_server(redirect) as (url, seen):
        binding["base_url"] = url
        with pytest.raises(TransportError, match="redirect_rejected") as caught:
            request_action(entry, binding, "prompt", max_tokens=64, timeout_seconds=2)
        assert len(seen) == 1
        assert seen[0]["authorization"] == "Bearer fixture-secret-value"
    assert "fixture-secret-value" not in str(caught.value)
    assert "fixture-secret-value" not in json.dumps(caught.value.receipt)
    assert caught.value.receipt["inference_requests"] == 0


def test_transport_bounds_wire_and_reply_bytes():
    entry, binding = identity()
    for responder, code in [
        (lambda _p, _b: (200, b"x" * (MAX_RESPONSE_BYTES + 1), {}), "response_byte_limit"),
        (lambda p, b: response(p, b, reply="x" * (MAX_REPLY_BYTES + 1)), "reply_byte_limit"),
    ]:
        with fixture_server(responder) as (url, _):
            binding["base_url"] = url
            with pytest.raises(TransportError, match=code):
                request_action(entry, binding, "prompt", max_tokens=64, timeout_seconds=2)
    with pytest.raises(TransportError, match="prompt_byte_limit") as caught:
        request_action(entry, binding, "x" * (MAX_PROMPT_BYTES + 1), max_tokens=64, timeout_seconds=2)
    assert caught.value.receipt["http_requests"] == 0


@pytest.mark.parametrize(("key", "value"), [
    ("revision", "main"), ("weights_sha256", "9" * 64), ("precision", "q4"),
    ("recurrence", {"passes": 3}), ("tokenizer_config_sha256", "8" * 64),
    ("cap_wrapper_receipt_sha256", None), ("resource_caps", {}),
    ("base_url", "http://example.com/v1"), ("base_url", "https://user:password@example.com/v1"),
    ("base_url", "https://example.com/v1?key=secret"), ("tokenize_path", "https://example.com/tokenize"),
    ("tokenize_path", "//example.com/tokenize"), ("tokenize_path", "/../tokenize"),
    ("context_limit", 4096), ("identity_verification", "verified"),
])
def test_transport_binding_rejects_identity_or_boundary_changes(key, value):
    entry, binding = identity()
    binding[key] = value
    with pytest.raises(TransportError):
        validate_binding(entry, binding)


def test_transport_requires_hrm_prefix_attention_and_condition():
    entry, binding = identity("prefix-lm-completion")
    for mutation in [{"prefix_lm_prefill_ones": False}, {"hrm_condition": "cot"}, {"prompt_role": "user"}]:
        with pytest.raises(TransportError):
            validate_binding(entry, {**binding, **mutation})


def test_transport_sampling_and_caps_cannot_hide_unbounded_requests():
    entry, binding = identity()
    for maximum, timeout in [(513, 2), (True, 2), (64, 31), (64, float("inf"))]:
        with pytest.raises(TransportError) as caught:
            request_action(entry, binding, "prompt", max_tokens=maximum, timeout_seconds=timeout)
        assert caught.value.receipt["http_requests"] == 0
    for sampling in [{"temperature": 0.6}, {"top_p": 0.95}, {"seed": 1}]:
        with pytest.raises(TransportError, match="primary_sampling_contract_mismatch"):
            validate_binding(entry, {**binding, "sampling": sampling})
