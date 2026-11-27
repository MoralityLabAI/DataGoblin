"""Bounded HTTP transport for explicitly bound research-model endpoints.

Checkpoint identity and server resource caps are operator attestations. HTTP
responses establish only the served alias, provider-reported token accounting
and the output of the endpoint's tokenizer. This module provisions no runtime.
"""

from __future__ import annotations

import copy
import http.client
import ipaddress
import json
import math
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

SCHEMA = "datagoblin-research-v2-binding-1"
PROTOCOL_ID = "research-v2-context2048"
MAX_CONTEXT_TOKENS = 2048
MAX_OUTPUT_TOKENS = 512
MAX_PROMPT_BYTES = 32768
MAX_REPLY_BYTES = 16384
MAX_RESPONSE_BYTES = 1048576
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ENV = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_PROTOCOLS = {"raw-completion", "chat-completions", "prefix-lm-completion"}
_ALLOWED_BINDING = {
    "schema", "registry_id", "model_id", "served_model", "revision", "weights_sha256",
    "tokenizer_sha256", "template_sha256", "precision", "recurrence", "protocol",
    "base_url", "api_key_env", "tokenize_path", "prompt_role", "thinking_mode",
    "chat_template_kwargs", "context_limit", "resource_caps", "cap_wrapper_receipt_sha256",
    "hrm_condition", "prefix_lm_prefill_ones", "config_sha256", "tokenizer_config_sha256",
    "sampling", "identity_verification", "resource_cap_verification", "protocol_id",
}


class TransportError(RuntimeError):
    """A stable code and a sanitized request receipt, with no provider body."""

    def __init__(self, code: str, receipt: dict[str, Any] | None = None):
        super().__init__(code)
        self.code = code
        self.receipt = {"error_code": code, **dict(receipt or {})}


def _fail(code: str) -> None:
    raise TransportError(code)


def _finite_number(value: Any, minimum: float, maximum: float, code: str) -> float:
    if type(value) not in {int, float} or not minimum <= value <= maximum or not math.isfinite(value):
        _fail(code)
    return float(value)


def _integer(value: Any, minimum: int, maximum: int, code: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(code)
    return value


def _sha(value: Any, code: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        _fail(code)
    return value


def _text(value: Any, maximum: int, code: str) -> str:
    if (type(value) is not str or not value or len(value) > maximum
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        _fail(code)
    return value


def _json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError, UnicodeError):
        _fail("invalid_json_value")


def _path(value: Any, code: str, *, empty_ok: bool = False) -> str:
    if empty_ok and value == "":
        return value
    if (type(value) is not str or not value.startswith("/") or len(value) > 512
            or re.fullmatch(r"/[A-Za-z0-9._~/-]*", value) is None
            or "//" in value or any(part in {".", ".."} for part in value.split("/"))):
        _fail(code)
    return value


def _endpoint(value: Any) -> str:
    url = _text(value, 1024, "invalid_base_url")
    try:
        parsed = urllib.parse.urlsplit(url)
        host, port = parsed.hostname, parsed.port
    except ValueError:
        _fail("invalid_base_url")
    if (parsed.scheme not in {"http", "https"} or not host or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment or "\\" in url):
        _fail("invalid_base_url")
    if port is not None and not 1 <= port <= 65535:
        _fail("invalid_base_url")
    if re.fullmatch(r"[A-Za-z0-9.:-]+", host) is None:
        _fail("invalid_base_url")
    if parsed.scheme == "http":
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host.lower() == "localhost"
        if not loopback:
            _fail("http_requires_loopback")
    _path(parsed.path, "invalid_base_path", empty_ok=True)
    return url.rstrip("/")


def validate_binding(entry: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any]:
    """Validate a complete operator binding without contacting its endpoint."""
    if type(entry) is not dict or type(binding) is not dict:
        _fail("invalid_binding")
    if set(binding) - _ALLOWED_BINDING or binding.get("schema") != SCHEMA:
        _fail("invalid_binding_schema")
    for key, expected in (("registry_id", entry.get("id")), ("model_id", entry.get("model_id")),
                          ("revision", entry.get("revision")), ("precision", entry.get("precision")),
                          ("protocol", entry.get("protocol")), ("recurrence", entry.get("recurrence"))):
        if key not in binding or _json(binding[key]) != _json(expected):
            _fail(f"binding_mismatch_{key}")
    if binding["protocol"] not in _PROTOCOLS:
        _fail("unsupported_protocol")
    _text(binding["model_id"], 256, "invalid_model_id")
    _text(binding.get("served_model"), 256, "invalid_served_model")
    revision = binding["revision"]
    if revision is not None and (type(revision) is not str or re.fullmatch(r"[0-9a-f]{40}", revision) is None):
        _fail("revision_must_be_immutable")
    for key in ("weights_sha256", "tokenizer_sha256", "template_sha256", "cap_wrapper_receipt_sha256"):
        _sha(binding.get(key), f"invalid_{key}")
        if entry.get(key) is not None and binding[key] != entry[key]:
            _fail(f"binding_mismatch_{key}")
    for key in ("config_sha256", "tokenizer_config_sha256"):
        if entry.get(key) is not None:
            _sha(binding.get(key), f"invalid_{key}")
            if binding[key] != entry[key]:
                _fail(f"binding_mismatch_{key}")
        elif key in binding:
            _sha(binding[key], f"invalid_{key}")
    if revision is None and entry.get("weights_sha256") is None:
        _fail("registry_requires_revision_or_weights_hash")
    base_url = _endpoint(binding.get("base_url"))
    tokenize_path = _path(binding.get("tokenize_path"), "invalid_tokenize_path")
    if tokenize_path.endswith("/"):
        _fail("invalid_tokenize_path")
    api_key_env = binding.get("api_key_env")
    if api_key_env is not None and (type(api_key_env) is not str or _ENV.fullmatch(api_key_env) is None):
        _fail("invalid_api_key_env")
    context = _integer(binding.get("context_limit"), 1, MAX_CONTEXT_TOKENS, "invalid_context_limit")
    if context > _integer(entry.get("context_length"), 1, 2**31 - 1, "invalid_registry_context"):
        _fail("context_exceeds_model_limit")
    if context != MAX_CONTEXT_TOKENS:
        _fail("primary_protocol_requires_context2048")
    mode = binding.get("thinking_mode")
    if mode not in {"enabled", "disabled", "model_default", "not_applicable"}:
        _fail("invalid_thinking_mode")
    protocol = binding["protocol"]
    kwargs = binding.get("chat_template_kwargs", {})
    if type(kwargs) is not dict or len(_json(kwargs)) > 2048:
        _fail("invalid_chat_template_kwargs")
    if protocol == "chat-completions":
        if binding.get("prompt_role") != "user" or mode == "not_applicable":
            _fail("chat_requires_user_role_and_thinking_mode")
        if mode in {"enabled", "disabled"} and kwargs.get("enable_thinking") is not (mode == "enabled"):
            _fail("thinking_mode_template_mismatch")
        if mode == "model_default" and "enable_thinking" in kwargs:
            _fail("thinking_mode_template_mismatch")
    elif binding.get("prompt_role") != "raw" or mode != "not_applicable" or kwargs:
        _fail("completion_requires_raw_framing")
    if protocol == "prefix-lm-completion":
        if (binding.get("hrm_condition") not in {"direct", "synth,cot"}
                or binding.get("prefix_lm_prefill_ones") is not True):
            _fail("prefix_lm_attestation_required")
    elif binding.get("hrm_condition") is not None or binding.get("prefix_lm_prefill_ones") is not None:
        _fail("unexpected_prefix_lm_fields")
    caps = binding.get("resource_caps")
    if type(caps) is not dict or set(caps) != {"wall_seconds", "ram_mib", "max_requests"}:
        _fail("resource_caps_required")
    _finite_number(caps["wall_seconds"], 0.001, 86400, "invalid_wall_cap")
    _integer(caps["ram_mib"], 1, 1048576, "invalid_ram_cap")
    _integer(caps["max_requests"], 2, 1000000, "invalid_request_cap")
    declared_sampling = binding.get("sampling", {})
    if type(declared_sampling) is not dict or set(declared_sampling) - {"temperature", "top_p", "seed"}:
        _fail("invalid_sampling")
    sampling = {"temperature": 0.0, "top_p": 1.0, "seed": 0, **declared_sampling}
    sampling["temperature"] = _finite_number(sampling["temperature"], 0.0, 2.0, "invalid_temperature")
    sampling["top_p"] = _finite_number(sampling["top_p"], 0.000001, 1.0, "invalid_top_p")
    _integer(sampling["seed"], 0, 2**32 - 1, "invalid_sampling_seed")
    if sampling != {"temperature": 0.0, "top_p": 1.0, "seed": 0}:
        _fail("primary_sampling_contract_mismatch")
    for key in ("identity_verification", "resource_cap_verification"):
        if binding.get(key, "operator_attested") != "operator_attested":
            _fail("unverified_operator_attestation")
    if binding.get("protocol_id", PROTOCOL_ID) != PROTOCOL_ID:
        _fail("protocol_id_mismatch")
    normalized = copy.deepcopy(binding)
    normalized.update(
        base_url=base_url, tokenize_path=tokenize_path, api_key_env=api_key_env,
        chat_template_kwargs=copy.deepcopy(kwargs), sampling=copy.deepcopy(sampling),
        identity_verification="operator_attested",
        resource_cap_verification="operator_attested", protocol_id=PROTOCOL_ID,
    )
    return normalized


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        _fail("request_deadline_exceeded")
    return value


def _reject_constant(_: str) -> None:
    _fail("invalid_response_json")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_response_key")
        result[key] = value
    return result


def _post(url: str, body: dict[str, Any], headers: dict[str, str], deadline: float) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    request = urllib.request.Request(url, data=_json(body), headers=headers, method="POST")
    try:
        with opener.open(request, timeout=_remaining(deadline)) as response:
            if response.status != 200:
                _fail(f"http_status_{response.status}")
            length = response.headers.get("Content-Length")
            if length is not None:
                try:
                    if not 0 <= int(length) <= MAX_RESPONSE_BYTES:
                        _fail("response_byte_limit")
                except ValueError:
                    _fail("invalid_content_length")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                _fail("encoded_response_unsupported")
            chunks, size = [], 0
            while True:
                remaining = _remaining(deadline)
                # urllib's timeout otherwise restarts for every socket read.
                # Tighten it as the shared tokenization/inference deadline advances.
                raw = getattr(getattr(response, "fp", None), "raw", None)
                connection = getattr(raw, "_sock", None)
                if isinstance(connection, socket.socket):
                    connection.settimeout(remaining)
                chunk = response.read1(min(65536, MAX_RESPONSE_BYTES + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    _fail("response_byte_limit")
            _remaining(deadline)
    except urllib.error.HTTPError as error:
        # Discard the provider's error body; it may echo request or credential data.
        code = error.code
        error.close()
        _fail("redirect_rejected" if 300 <= code < 400 else f"http_status_{code}")
    except TimeoutError:
        _fail("request_deadline_exceeded")
    except urllib.error.URLError as error:
        _fail("request_deadline_exceeded" if isinstance(error.reason, TimeoutError)
              else "transport_connection_failed")
    except (OSError, http.client.HTTPException):
        _fail("transport_connection_failed")
    try:
        value = json.loads(b"".join(chunks), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, TypeError, RecursionError, UnicodeError):
        _fail("invalid_response_json")
    if type(value) is not dict:
        _fail("response_must_be_object")
    return value


def _prompt_body(binding: dict[str, Any], prompt: str) -> dict[str, Any]:
    body: dict[str, Any] = {"model": binding["served_model"]}
    if binding["protocol"] == "chat-completions":
        body["messages"] = [{"role": "user", "content": prompt}]
        body["chat_template_kwargs"] = binding["chat_template_kwargs"]
    else:
        if binding["protocol"] == "prefix-lm-completion":
            condition = ("<|object_ref_start|>" if binding["hrm_condition"] == "direct"
                         else "<|quad_end|><|object_ref_end|>")
            prompt = f"<|im_start|>{condition}{prompt}<|im_end|>"
        body["prompt"] = prompt
    return body


def request_action(
    entry: dict[str, Any], binding: dict[str, Any], prompt: str, *, max_tokens: int, timeout_seconds: float,
) -> dict[str, Any]:
    """Token-count the rendered prompt, then request one bounded text action.

    A shared monotonic deadline covers both HTTP exchanges. The endpoint must
    return exact provider token accounting, including all reasoning tokens.
    Failure receipts count attempted HTTP and inference calls conservatively.
    """
    started = time.monotonic()
    receipt: dict[str, Any] = {"http_requests": 0, "inference_requests": 0, "stage": "validation"}
    try:
        bound = validate_binding(entry, binding)
        max_tokens = _integer(max_tokens, 1, MAX_OUTPUT_TOKENS, "invalid_max_tokens")
        timeout = _finite_number(timeout_seconds, 0.001, 30.0, "invalid_timeout_seconds")
        timeout = min(timeout, bound["resource_caps"]["wall_seconds"])
        deadline = started + timeout
        if type(prompt) is not str or not prompt:
            _fail("invalid_prompt")
        try:
            if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
                _fail("prompt_byte_limit")
        except UnicodeError:
            _fail("invalid_prompt")
        headers = {"Content-Type": "application/json", "Accept": "application/json", "Accept-Encoding": "identity"}
        if bound["api_key_env"]:
            credential = os.environ.get(bound["api_key_env"])
            if (not credential or len(credential) > 8192
                    or any(ord(c) < 32 or ord(c) == 127 for c in credential)):
                _fail("api_key_env_missing_or_invalid")
            headers["Authorization"] = "Bearer " + credential
        body = _prompt_body(bound, prompt)
        parsed = urllib.parse.urlsplit(bound["base_url"])
        tokenize_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, bound["tokenize_path"], "", ""))
        receipt.update(stage="tokenization", http_requests=1)
        preflight = _post(tokenize_url, body, headers, deadline)
        count = _integer(preflight.get("count"), 1, 2**31 - 1, "invalid_token_count")
        receipt["prompt_tokens"] = count
        if preflight.get("model", bound["served_model"]) != bound["served_model"]:
            _fail("tokenizer_model_mismatch")
        if count + max_tokens > bound["context_limit"]:
            _fail("context_budget_exceeded")
        _remaining(deadline)
        generation = {**body, **bound["sampling"], "max_tokens": max_tokens, "stream": False, "n": 1}
        if bound["protocol"] == "prefix-lm-completion":
            generation["token_type_ids"] = [1] * count
        if bound["protocol"] == "chat-completions":
            endpoint = bound["base_url"] + "/chat/completions"
        else:
            endpoint = bound["base_url"] + "/completions"
            generation["echo"] = False
        receipt.update(stage="inference", http_requests=2, inference_requests=1, max_tokens=max_tokens)
        response = _post(endpoint, generation, headers, deadline)
        if response.get("model") != bound["served_model"]:
            _fail("served_model_mismatch")
        usage = response.get("usage")
        if type(usage) is not dict:
            _fail("token_usage_required")
        accounting = {key: _integer(usage.get(key), 0, 2**31 - 1, "invalid_token_usage")
                      for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        receipt["usage"] = accounting
        if accounting["prompt_tokens"] != count:
            _fail("tokenizer_usage_mismatch")
        if accounting["completion_tokens"] > max_tokens:
            _fail("generation_token_limit_exceeded")
        if accounting["total_tokens"] != count + accounting["completion_tokens"]:
            _fail("token_usage_total_mismatch")
        if accounting["total_tokens"] > bound["context_limit"]:
            _fail("context_budget_exceeded")
        details = usage.get("completion_tokens_details")
        if details is not None:
            if type(details) is not dict:
                _fail("invalid_reasoning_token_usage")
            if "reasoning_tokens" in details:
                _integer(details["reasoning_tokens"], 0, accounting["completion_tokens"],
                         "reasoning_tokens_exceed_completion_usage")
        if "reasoning_tokens" in usage:
            _fail("unsupported_separate_reasoning_usage")
        choices = response.get("choices")
        if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
            _fail("invalid_completion_choices")
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            _fail("generation_truncated")
        if finish_reason != "stop":
            _fail("unsupported_finish_reason")
        if bound["protocol"] == "chat-completions":
            message = choice.get("message")
            if (type(message) is not dict or message.get("role") != "assistant"
                    or message.get("tool_calls") or message.get("function_call")):
                _fail("text_action_response_required")
            text = message.get("content")
        else:
            text = choice.get("text")
        if type(text) is not str or not text:
            _fail("empty_or_invalid_reply")
        if len(text.encode("utf-8")) > MAX_REPLY_BYTES:
            _fail("reply_byte_limit")
        if accounting["completion_tokens"] == 0:
            _fail("invalid_token_usage")
        return {
            "text": text, "usage": accounting, "latency_seconds": time.monotonic() - started,
            "finish_reason": finish_reason, "served_model": bound["served_model"],
            "http_requests": 2, "inference_requests": 1, "protocol_id": PROTOCOL_ID,
            "identity_verification": "operator_attested", "usage_verification": "provider_reported",
        }
    except TransportError as error:
        receipt["latency_seconds"] = time.monotonic() - started
        raise TransportError(error.code, receipt) from None
    except (ValueError, TypeError, KeyError, UnicodeError):
        receipt["latency_seconds"] = time.monotonic() - started
        raise TransportError("invalid_transport_input_or_response", receipt) from None
