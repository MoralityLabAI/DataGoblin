"""Bounded local inference worker for DataGoblin Research Verifiers v2.

Speaks newline-delimited JSON-RPC over stdin/stdout so a loopback HTTP bridge
(e.g. inside WSL) can serve `/tokenize`, `/v1/completions`, and
`/v1/chat/completions` without cross-VM firewall configuration while keeping
exact token accounting, model-specific recurrence/PrefixLM semantics, and
CPU/GPU/RAM/VRAM telemetry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import psutil
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from torch import nn
from torch.nn import functional as F

ALEPH_VISIT_SCHEDULE = [0, *range(1, 11), *range(1, 11), 11, 12, 13]
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _log(msg: str) -> None:
    sys.stderr.write(f"[local_worker] {msg}\n")
    sys.stderr.flush()


class AlephBlock(nn.Module):
    def __init__(self, width: int, heads: int = 16, kv: int = 4, ff: int | None = None) -> None:
        super().__init__()
        self.h, self.kv, self.k = heads, kv, width // heads
        self.n1 = nn.RMSNorm(width)
        self.n2 = nn.RMSNorm(width)
        self.q = nn.Linear(width, width, bias=False)
        self.kp = nn.Linear(width, kv * self.k, bias=False)
        self.v = nn.Linear(width, kv * self.k, bias=False)
        self.o = nn.Linear(width, width, bias=False)
        ff = 4 * width if ff is None else ff
        self.up = nn.Linear(width, ff, bias=False)
        self.gate = nn.Linear(width, ff, bias=False)
        self.down = nn.Linear(ff, width, bias=False)

    @staticmethod
    def rope(x: torch.Tensor, start_pos: int) -> torch.Tensor:
        t, k = x.shape[-2:]
        freq = 10000.0 ** (-torch.arange(0, k, 2, device=x.device, dtype=torch.float32) / k)
        positions = torch.arange(start_pos, start_pos + t, device=x.device, dtype=torch.float32)
        phase = positions[:, None] * freq
        c, s = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
        a, b = x[..., ::2], x[..., 1::2]
        return torch.stack((a * c - b * s, a * s + b * c), -1).flatten(-2)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, d = x.shape
        z = self.n1(x)
        q = self.rope(self.q(z).view(b, t, self.h, self.k).transpose(1, 2), start_pos)
        k = self.rope(self.kp(z).view(b, t, self.kv, self.k).transpose(1, 2), start_pos)
        v = self.v(z).view(b, t, self.kv, self.k).transpose(1, 2)
        if kv_cache is not None:
            k = torch.cat((kv_cache[0], k), dim=2)
            v = torch.cat((kv_cache[1], v), dim=2)
        new_cache = (k, v)
        k_exp = k.repeat_interleave(self.h // self.kv, dim=1)
        v_exp = v.repeat_interleave(self.h // self.kv, dim=1)
        attn = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=(t > 1 and kv_cache is None))
        x = x + self.o(attn.transpose(1, 2).reshape(b, t, d))
        z = self.n2(x)
        x = x + self.down(F.silu(self.gate(z)) * self.up(z))
        return x, new_cache


class AlephNativeModel(nn.Module):
    def __init__(
        self,
        width: int = 2048,
        vocab: int = 49152,
        heads: int = 16,
        kv: int = 4,
        core_ff: int = 8960,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList(
            AlephBlock(width, heads, kv, core_ff if 1 <= i <= 10 else None) for i in range(14)
        )
        self.norm = nn.RMSNorm(width)

    @torch.inference_mode()
    def forward_step(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x = self.embed(tokens)
        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for visit_idx, physical in enumerate(ALEPH_VISIT_SCHEDULE):
            cache = None if caches is None else caches[visit_idx]
            x, updated = self.blocks[physical](x, start_pos=start_pos, kv_cache=cache)
            new_caches.append(updated)
        logits = F.linear(self.norm(x[:, -1:, :]), self.embed.weight)
        return logits[:, -1, :], new_caches


class AlephNativeBackend:
    def __init__(
        self,
        weights_path: Path,
        tokenizer_path: Path,
        served_model: str,
        device: str,
        token_checks_path: Path | None = None,
        threads: int = 8,
    ) -> None:
        if device == "cpu":
            torch.set_num_threads(threads)
        self.device = torch.device(device)
        self.served_model = served_model
        self.eos_id = 0
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.model = AlephNativeModel().to(dtype=torch.float32)
        state = load_file(str(weights_path), device="cpu")
        missing, unexpected = self.model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"native key mismatch: missing={missing}, unexpected={unexpected}")
        self.parameters = sum(p.numel() for p in self.model.parameters())
        if self.parameters != 999_352_320:
            raise RuntimeError(f"expected 999352320 parameters, got {self.parameters}")
        self.model.to(self.device)
        self.model.eval()
        self.parity_checks: dict[str, Any] = {}
        self._verify_kv_parity()
        if token_checks_path is not None and token_checks_path.is_file():
            self._verify_token_checks(token_checks_path)

    @torch.inference_mode()
    def _verify_kv_parity(self) -> None:
        ids = torch.tensor([[504, 3575, 282, 4649, 314]], dtype=torch.long, device=self.device)
        full_logits, _ = self.model.forward_step(ids, start_pos=0, caches=None)
        _, caches = self.model.forward_step(ids[:, :-1], start_pos=0, caches=None)
        cached_logits, _ = self.model.forward_step(ids[:, -1:], start_pos=ids.shape[1] - 1, caches=caches)
        max_diff = float((full_logits - cached_logits).abs().max().item())
        if max_diff > 1e-4:
            raise RuntimeError(f"KV-cache parity failed: max_diff={max_diff}")
        self.parity_checks["kv_cache_max_abs_diff"] = max_diff

    @torch.inference_mode()
    def _verify_token_checks(self, path: Path) -> None:
        checks = json.loads(path.read_text(encoding="utf-8"))
        matched_ids, matched_top1 = 0, 0
        for item in checks:
            ids = self.tokenizer.encode(item["prompt"], add_special_tokens=False).ids
            if ids == item["tokenizers_ids"]:
                matched_ids += 1
            t_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            logits, _ = self.model.forward_step(t_ids, start_pos=0, caches=None)
            top1 = int(logits[0].argmax().item())
            if top1 == item["native_top1_id"]:
                matched_top1 += 1
        self.parity_checks["token_checks_total"] = len(checks)
        self.parity_checks["token_checks_ids_match"] = matched_ids
        self.parity_checks["token_checks_top1_match"] = matched_top1
        if matched_ids != len(checks) or matched_top1 != len(checks):
            raise RuntimeError(
                f"token_checks parity mismatch: ids={matched_ids}/{len(checks)} top1={matched_top1}/{len(checks)}"
            )

    def tokenize(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = payload["prompt"]
        ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        return {"count": len(ids), "model": self.served_model}

    @torch.inference_mode()
    def generate(self, payload: dict[str, Any], deadline: float) -> tuple[dict[str, Any], float]:
        prompt = payload["prompt"]
        max_tokens = int(payload["max_tokens"])
        ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        prompt_tokens = len(ids)
        t_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        gpu_ms = 0.0
        if self.device.type == "cuda":
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
        logits, caches = self.model.forward_step(t_ids, start_pos=0, caches=None)
        generated: list[int] = []
        finish_reason = "length"
        pos = prompt_tokens
        for _ in range(max_tokens):
            next_id = int(logits[0].argmax().item())
            generated.append(next_id)
            if next_id == self.eos_id:
                finish_reason = "stop"
                break
            if len(generated) >= max_tokens:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("worker_generation_deadline")
            step_t = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            logits, caches = self.model.forward_step(step_t, start_pos=pos, caches=caches)
            pos += 1
        if self.device.type == "cuda":
            end_ev.record()
            torch.cuda.synchronize(self.device)
            gpu_ms = float(start_ev.elapsed_time(end_ev))
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        if not text:
            text = self.tokenizer.decode(generated, skip_special_tokens=False)
        comp = len(generated)
        reply = {
            "id": f"cmpl-aleph-{int(time.time()*1000)}",
            "object": "text_completion",
            "model": self.served_model,
            "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": comp,
                "total_tokens": prompt_tokens + comp,
            },
        }
        return reply, gpu_ms / 1000.0


class HFTransformersBackend:
    def __init__(
        self,
        model_dir: Path,
        served_model: str,
        protocol: str,
        device: str,
        dtype_name: str = "bfloat16",
        weights_dir: Path | None = None,
        trust_remote_code: bool = False,
        threads: int = 8,
    ) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if device == "cpu":
            torch.set_num_threads(threads)
        self.device = torch.device(device)
        self.served_model = served_model
        self.protocol = protocol
        self.dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}[dtype_name]
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
        config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
        # Support staged config/code dir paired with external shard dir (zero-copy for Nanbeige)
        if weights_dir is not None and weights_dir != model_dir:
            _log(f"Loading weights from {weights_dir} with config/code from {model_dir}")
            with torch.device("meta"):
                meta_model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
            meta_model = meta_model.to_empty(device="cpu").to(dtype=self.dtype)
            index_file = weights_dir / "model.safetensors.index.json"
            if index_file.is_file():
                idx = json.loads(index_file.read_text(encoding="utf-8"))
                shard_files = sorted(set(idx["weight_map"].values()))
            else:
                shard_files = [p.name for p in sorted(weights_dir.glob("*.safetensors"))]
            state: dict[str, torch.Tensor] = {}
            for sf in shard_files:
                state.update(load_file(str(weights_dir / sf), device="cpu"))
            missing, unexpected = meta_model.load_state_dict(state, strict=False)
            if missing and getattr(config, "tie_word_embeddings", False) and missing == ["lm_head.weight"]:
                meta_model.tie_weights()
                missing = []
            if missing or unexpected:
                raise RuntimeError(f"weight load mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")
            self.model = meta_model.to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                config=config,
                torch_dtype=self.dtype,
                trust_remote_code=trust_remote_code,
            ).to(self.device)
        self.model.eval()
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.eos_ids = self._resolve_eos_ids(config)
        self.parity_checks = {
            "parameters": self.parameters,
            "model_type": getattr(config, "model_type", None),
            "dtype": str(self.dtype),
            "eos_ids": sorted(self.eos_ids),
        }

    def _resolve_eos_ids(self, config: Any) -> set[int]:
        ids: set[int] = set()
        for src in (getattr(config, "eos_token_id", None), getattr(self.tokenizer, "eos_token_id", None)):
            if isinstance(src, int):
                ids.add(src)
            elif isinstance(src, (list, tuple)):
                ids.update(int(x) for x in src)
        gen_cfg = getattr(self.model, "generation_config", None)
        if gen_cfg is not None:
            g_eos = getattr(gen_cfg, "eos_token_id", None)
            if isinstance(g_eos, int):
                ids.add(g_eos)
            elif isinstance(g_eos, (list, tuple)):
                ids.update(int(x) for x in g_eos)
        return ids

    def _render_ids(self, payload: dict[str, Any]) -> list[int]:
        if self.protocol == "chat-completions":
            messages = payload["messages"]
            kwargs = dict(payload.get("chat_template_kwargs") or {})
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
            return list(self.tokenizer.encode(rendered, add_special_tokens=False))
        prompt = payload["prompt"]
        return list(self.tokenizer.encode(prompt, add_special_tokens=False))

    def tokenize(self, payload: dict[str, Any]) -> dict[str, Any]:
        ids = self._render_ids(payload)
        return {"count": len(ids), "model": self.served_model}

    def _parse_chat_output(self, gen_ids: list[int], finish_reason: str) -> tuple[str, int | None]:
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        # Handle models whose chat template opens <think>\n inside the generation prompt
        if "</think>" in raw:
            before, after = raw.split("</think>", 1)
            think_text = before.split("<think>", 1)[-1] if "<think>" in before else before
            reasoning_toks = len(self.tokenizer.encode(think_text + "</think>", add_special_tokens=False))
            reasoning_toks = min(reasoning_toks, len(gen_ids))
            return after.strip(), reasoning_toks
        m = _THINK_RE.search(raw)
        if m:
            reasoning_toks = min(len(self.tokenizer.encode(m.group(0), add_special_tokens=False)), len(gen_ids))
            cleaned = _THINK_RE.sub("", raw).strip()
            return cleaned, reasoning_toks
        return raw.strip(), None

    @torch.inference_mode()
    def generate(self, payload: dict[str, Any], deadline: float) -> tuple[dict[str, Any], float]:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _DeadlineStop(StoppingCriteria):
            def __init__(self, limit: float) -> None:
                self.limit = limit
                self.timed_out = False

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
                if time.monotonic() > self.limit:
                    self.timed_out = True
                    return True
                return False

        ids = self._render_ids(payload)
        prompt_tokens = len(ids)
        max_tokens = int(payload["max_tokens"])
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "use_cache": True,
        }
        if self.protocol == "prefix-lm-completion":
            tt = payload.get("token_type_ids")
            if not isinstance(tt, list) or len(tt) != prompt_tokens or any(x != 1 for x in tt):
                raise ValueError("prefix-lm-completion requires all-ones token_type_ids matching prompt_tokens")
            gen_kwargs["token_type_ids"] = torch.tensor([tt], dtype=torch.long, device=self.device)
        if self.eos_ids:
            gen_kwargs["eos_token_id"] = sorted(self.eos_ids)
        if self.tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        elif self.eos_ids:
            gen_kwargs["pad_token_id"] = min(self.eos_ids)

        stopper = _DeadlineStop(deadline)
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([stopper])

        gpu_ms = 0.0
        if self.device.type == "cuda":
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
        if getattr(self.model.config, "model_type", None) == "nanbeige":
            from transformers.cache_utils import DynamicCache

            pkv = DynamicCache()
            cur_ids = input_ids
            cur_mask = attention_mask
            gen_ids_list: list[int] = []
            for _ in range(max_tokens):
                if time.monotonic() > deadline:
                    stopper.timed_out = True
                    break
                step_out = self.model(
                    input_ids=cur_ids,
                    attention_mask=cur_mask,
                    past_key_values=pkv,
                    use_cache=True,
                )
                pkv = step_out.past_key_values
                next_id = int(step_out.logits[0, -1, :].argmax().item())
                gen_ids_list.append(next_id)
                if next_id in self.eos_ids:
                    break
                cur_ids = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                cur_mask = torch.cat([cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype, device=self.device)], dim=1)
            gen_ids = gen_ids_list
        else:
            out = self.model.generate(**gen_kwargs)
            gen_ids = out[0, prompt_tokens:].tolist()
        if self.device.type == "cuda":
            end_ev.record()
            torch.cuda.synchronize(self.device)
            gpu_ms = float(start_ev.elapsed_time(end_ev))
        if stopper.timed_out:
            raise TimeoutError("worker_generation_deadline")

        comp = len(gen_ids)
        ended_with_eos = bool(gen_ids) and (gen_ids[-1] in self.eos_ids)
        finish_reason = "stop" if ended_with_eos else ("length" if comp >= max_tokens else "stop")
        usage: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp,
            "total_tokens": prompt_tokens + comp,
        }
        if self.protocol == "chat-completions":
            content, reasoning_toks = self._parse_chat_output(gen_ids, finish_reason)
            if reasoning_toks is not None:
                usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_toks}
            reply = {
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion",
                "model": self.served_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": usage,
            }
        else:
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            if not text:
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
            reply = {
                "id": f"cmpl-{int(time.time()*1000)}",
                "object": "text_completion",
                "model": self.served_model,
                "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
                "usage": usage,
            }
        return reply, gpu_ms / 1000.0


class LlamaServerBackend:
    def __init__(
        self,
        llama_bin: Path,
        gguf_path: Path,
        served_model: str,
        protocol: str,
        ctx_size: int = 2048,
        threads: int = 10,
        ngl: int = 0,
        extra_args: list[str] | None = None,
    ) -> None:
        self.served_model = served_model
        self.protocol = protocol
        self.port = self._free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        cmd = [
            str(llama_bin),
            "-m",
            str(gguf_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--alias",
            served_model,
            "--ctx-size",
            str(ctx_size),
            "--threads",
            str(threads),
            "-ngl",
            str(ngl),
            "--parallel",
            "1",
            "--seed",
            "0",
            "--no-warmup",
        ]
        if protocol == "chat-completions":
            cmd.append("--jinja")
        if extra_args:
            cmd.extend(extra_args)
        _log(f"Starting llama-server: {' '.join(cmd)}")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.ps_proc = psutil.Process(self.proc.pid)
        self.parameters = 0
        self.parity_checks = {"llama_server_port": self.port, "ngl": ngl, "threads": threads}
        deadline = time.monotonic() + 240.0
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                err = self.proc.stderr.read().decode("utf-8", errors="replace") if self.proc.stderr else ""
                raise RuntimeError(f"llama-server exited early ({self.proc.returncode}): {err[-1000:]}")
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=2) as r:
                    if r.status == 200:
                        _log(f"llama-server ready on {self.base_url}")
                        return
            except Exception:
                time.sleep(0.5)
        self.close()
        raise RuntimeError("llama-server startup timeout")

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    def _http_post(self, path: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=max(0.5, timeout)) as r:
            return json.loads(r.read().decode("utf-8"))

    def _apply_template_if_chat(self, payload: dict[str, Any], timeout: float) -> str:
        if self.protocol != "chat-completions":
            return str(payload["prompt"])
        body: dict[str, Any] = {"messages": payload["messages"]}
        if payload.get("chat_template_kwargs"):
            body["chat_template_kwargs"] = payload["chat_template_kwargs"]
        res = self._http_post("/apply-template", body, timeout)
        return str(res["prompt"])

    def tokenize(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.protocol == "chat-completions":
            rendered = self._apply_template_if_chat(payload, 10.0)
            tok_req = {"content": rendered, "add_special": False, "parse_special": True}
            res = self._http_post("/tokenize", tok_req, 10.0)
        else:
            tok_req = {"content": payload["prompt"], "add_special": False, "parse_special": False}
            res = self._http_post("/tokenize", tok_req, 10.0)
        return {"count": len(res["tokens"]), "model": self.served_model}

    def generate(self, payload: dict[str, Any], deadline: float) -> tuple[dict[str, Any], float]:
        timeout = max(0.5, deadline - time.monotonic())
        if self.protocol == "chat-completions":
            rendered = self._apply_template_if_chat(payload, timeout)
            tok_body = {"content": rendered, "add_special": False, "parse_special": True}
            tok_ids = self._http_post("/tokenize", tok_body, timeout)["tokens"]
            body = {
                "prompt": tok_ids,
                "n_predict": int(payload["max_tokens"]),
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 0,
                "cache_prompt": False,
            }
            res = self._http_post("/completion", body, max(0.5, deadline - time.monotonic()))
            raw = str(res.get("content", ""))
            comp = int(res.get("tokens_predicted", 0))
            prompt_tokens = len(tok_ids)
            stop_type = str(res.get("stop_type", ""))
            stopped_limit = bool(res.get("stopped_limit", False)) or stop_type == "limit"
            finish_reason = "length" if stopped_limit else "stop"
            usage: dict[str, Any] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": comp,
                "total_tokens": prompt_tokens + comp,
            }
            content = raw
            if "</think>" in raw:
                before, after = raw.split("</think>", 1)
                think_req = {"content": before + "</think>", "add_special": False, "parse_special": True}
                r_toks = len(self._http_post("/tokenize", think_req, 5.0)["tokens"])
                usage["completion_tokens_details"] = {"reasoning_tokens": min(r_toks, comp)}
                content = after.strip()
            else:
                content = raw.strip()
            reply = {
                "id": f"chatcmpl-llama-{int(time.time()*1000)}",
                "object": "chat.completion",
                "model": self.served_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": usage,
            }
        else:
            raw_tok_body = {"content": payload["prompt"], "add_special": False, "parse_special": False}
            tok_ids = self._http_post("/tokenize", raw_tok_body, timeout)["tokens"]
            body = {
                "prompt": tok_ids,
                "n_predict": int(payload["max_tokens"]),
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 0,
                "cache_prompt": False,
            }
            res = self._http_post("/completion", body, max(0.5, deadline - time.monotonic()))
            text = str(res.get("content", ""))
            comp = int(res.get("tokens_predicted", 0))
            prompt_tokens = len(tok_ids)
            stop_type = str(res.get("stop_type", ""))
            stopped_limit = bool(res.get("stopped_limit", False)) or stop_type == "limit"
            finish_reason = "length" if stopped_limit else "stop"
            reply = {
                "id": f"cmpl-llama-{int(time.time()*1000)}",
                "object": "text_completion",
                "model": self.served_model,
                "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": comp,
                    "total_tokens": prompt_tokens + comp,
                },
            }
        return reply, 0.0

    def child_stats(self) -> tuple[float, float]:
        try:
            ct = self.ps_proc.cpu_times()
            rss_mib = self.ps_proc.memory_info().rss / (1024 * 1024)
            return float(ct.user + ct.system), float(rss_mib)
        except Exception:
            return 0.0, 0.0

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("aleph_native", "hf_transformers", "llama_server"), required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument(
        "--protocol",
        choices=("raw-completion", "chat-completions", "prefix-lm-completion"),
        required=True,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--weights-path", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--token-checks-path", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--llama-bin", type=Path)
    parser.add_argument("--gguf-path", type=Path)
    parser.add_argument("--ngl", type=int, default=0)
    parser.add_argument("--extra-llama-arg", action="append", default=[])
    args = parser.parse_args()

    proc = psutil.Process(os.getpid())
    started = time.monotonic()
    load_start = time.monotonic()

    if args.backend == "aleph_native":
        backend: Any = AlephNativeBackend(
            weights_path=args.weights_path,
            tokenizer_path=args.tokenizer_path,
            served_model=args.served_model,
            device=args.device,
            token_checks_path=args.token_checks_path,
            threads=args.threads,
        )
    elif args.backend == "hf_transformers":
        backend = HFTransformersBackend(
            model_dir=args.model_dir,
            served_model=args.served_model,
            protocol=args.protocol,
            device=args.device,
            dtype_name=args.dtype,
            weights_dir=args.weights_dir,
            trust_remote_code=args.trust_remote_code,
            threads=args.threads,
        )
    else:
        backend = LlamaServerBackend(
            llama_bin=args.llama_bin,
            gguf_path=args.gguf_path,
            served_model=args.served_model,
            protocol=args.protocol,
            threads=args.threads,
            ngl=args.ngl,
            extra_args=args.extra_llama_arg,
        )

    load_seconds = time.monotonic() - load_start
    telemetry: dict[str, Any] = {
        "backend": args.backend,
        "served_model": args.served_model,
        "protocol": args.protocol,
        "device": args.device,
        "dtype": (
            args.dtype
            if args.backend == "hf_transformers"
            else ("float32" if args.backend == "aleph_native" else "gguf")
        ),
        "parameters": backend.parameters,
        "load_seconds": round(load_seconds, 4),
        "parity_checks": backend.parity_checks,
        "tokenize_requests": 0,
        "generate_requests": 0,
        "generate_errors": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "gpu_seconds": 0.0,
        "peak_ram_mib": 0.0,
        "peak_vram_mib": 0.0,
        "latencies_seconds": [],
    }

    def _update_peaks() -> None:
        rss = proc.memory_info().rss / (1024 * 1024)
        if isinstance(backend, LlamaServerBackend):
            _, child_rss = backend.child_stats()
            rss += child_rss
        telemetry["peak_ram_mib"] = round(max(telemetry["peak_ram_mib"], rss), 2)
        if args.device == "cuda" and torch.cuda.is_available():
            vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
            telemetry["peak_vram_mib"] = round(max(telemetry["peak_vram_mib"], vram), 2)
        ct = proc.cpu_times()
        cpu_s = float(ct.user + ct.system)
        if isinstance(backend, LlamaServerBackend):
            child_cpu, _ = backend.child_stats()
            cpu_s += child_cpu
        telemetry["cpu_seconds"] = round(cpu_s, 4)
        telemetry["wall_seconds"] = round(time.monotonic() - started, 4)

    _update_peaks()
    sys.stdout.write(json.dumps({"ready": True, "telemetry": telemetry}) + "\n")
    sys.stdout.flush()

    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            req = json.loads(line)
            op = req.get("op")
            try:
                if op == "tokenize":
                    telemetry["tokenize_requests"] += 1
                    res = backend.tokenize(req["payload"])
                    _update_peaks()
                    sys.stdout.write(json.dumps({"ok": True, "result": res}) + "\n")
                elif op == "generate":
                    telemetry["generate_requests"] += 1
                    t0 = time.monotonic()
                    timeout_s = float(req.get("timeout_seconds", 29.0))
                    res, gpu_s = backend.generate(req["payload"], deadline=t0 + timeout_s)
                    dt = time.monotonic() - t0
                    telemetry["gpu_seconds"] = round(telemetry["gpu_seconds"] + gpu_s, 4)
                    telemetry["total_prompt_tokens"] += int(res["usage"]["prompt_tokens"])
                    telemetry["total_completion_tokens"] += int(res["usage"]["completion_tokens"])
                    telemetry["latencies_seconds"].append(round(dt, 4))
                    _update_peaks()
                    sys.stdout.write(json.dumps({"ok": True, "result": res}) + "\n")
                elif op == "telemetry":
                    _update_peaks()
                    sys.stdout.write(json.dumps({"ok": True, "result": telemetry}) + "\n")
                elif op == "shutdown":
                    _update_peaks()
                    sys.stdout.write(json.dumps({"ok": True, "result": telemetry}) + "\n")
                    sys.stdout.flush()
                    break
                else:
                    sys.stdout.write(json.dumps({"ok": False, "error": f"unknown op: {op}"}) + "\n")
            except Exception as exc:
                telemetry["generate_errors"] += 1
                _update_peaks()
                sys.stdout.write(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
            sys.stdout.flush()
    finally:
        if isinstance(backend, LlamaServerBackend):
            backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
