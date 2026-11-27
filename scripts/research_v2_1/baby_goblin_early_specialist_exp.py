"""Early Baby Goblin 10M v2 specialist-training experiment at matched compute budgets.

Compares four arms starting from the exact native Baby Goblin 10M v2 checkpoint
(AlephFunk/aleph-direct-a-10m-v2, 999,352,320 tied parameters, SHA-256
2e40e32da6a6d975fbd7dc7bd56cd440321531e2e755b9bcc215a289c64232a8):
  1. arm0_original_10m         (0 additional optimizer updates)
  2. arm1_specialist_only      (N updates on the 211,906 supervised-token specialist pool)
  3. arm2_continued_base       (N updates on disjoint continued general-language pretraining)
  4. arm3_base_then_specialist (N/2 base pretraining + N/2 specialist adaptation; matched budget)

Strictly excludes ALL held/dev/prospective roots from every training slice.
Uses explicit BF16 compute on frozen base weights with FP32 trainable parameters
and FP32 AdamW optimizer states (matching Baby Goblin's training recipe without
BF16 update underflow) while staying strictly within bounded local RAM caps.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import psutil
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from torch import nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research_v2"))

from local_worker import ALEPH_VISIT_SCHEDULE, AlephNativeModel  # noqa: E402

from datagoblin.research_v2.suite import LEARNERS, canon, digest  # noqa: E402
from datagoblin.research_v2_1.suite import make_suite as make_suite_v21  # noqa: E402

NATIVE_WEIGHTS_SHA256 = "2e40e32da6a6d975fbd7dc7bd56cd440321531e2e755b9bcc215a289c64232a8"
TOKENIZER_SHA256 = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
EOS_ID = 0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FP32LoRALinear(nn.Module):
    """Frozen BF16 base projection + FP32 low-rank adapter (eliminates BF16 update underflow)."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.scale = alpha / rank
        dev = base.weight.device
        self.lora_a = nn.Parameter(torch.empty((rank, base.in_features), device=dev, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros((base.out_features, rank), device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_base = self.base(x.to(self.base.weight.dtype))
        delta = (x.float() @ self.lora_a.t() @ self.lora_b.t()) * self.scale
        return out_base.float() + delta


class FP32HeadAdapter(nn.Module):
    """Low-rank FP32 residual on the tied output projection (49,152 x 2,048)."""

    def __init__(
        self, width: int = 2048, vocab: int = 49152, rank: int = 64, device: torch.device | None = None
    ) -> None:
        super().__init__()
        self.scale = 1.0 / math.sqrt(rank)
        self.a = nn.Parameter(torch.empty((rank, width), device=device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros((vocab, rank), device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, h: torch.Tensor, base_embed_weight: torch.Tensor) -> torch.Tensor:
        base_logits = F.linear(h.to(base_embed_weight.dtype), base_embed_weight).float()
        delta = (h.float() @ self.a.t() @ self.b.t()) * self.scale
        return base_logits + delta


def _apply_trainable_scope(
    model: AlephNativeModel, scope: str, device: torch.device
) -> tuple[FP32HeadAdapter | None, int, int, int]:
    """Configure FP32 trainable parameters over BF16 base weights.

    Returns (head_adapter, freeze_prefix_visits, trainable_params, total_base_params).
    """
    base_params = sum(p.numel() for p in model.parameters())
    for p in model.parameters():
        p.requires_grad_(False)

    head_adapter: FP32HeadAdapter | None = None
    freeze_prefix = 0

    if scope == "exit_shell_fp32":
        # Frozen visits 0..20 (blocks 0..10) in BF16; FP32 LoRA on blocks[11,12]
        # + full FP32 block[13] + norm + rank-64 head adapter
        freeze_prefix = 21
        for idx in (11, 12):
            blk = model.blocks[idx]
            for attr in ("q", "kp", "v", "o", "up", "gate", "down"):
                setattr(blk, attr, FP32LoRALinear(getattr(blk, attr), rank=16, alpha=32.0))
            blk.n1.float()
            blk.n2.float()
            for p in blk.n1.parameters():
                p.requires_grad_(True)
            for p in blk.n2.parameters():
                p.requires_grad_(True)
        model.blocks[13].float()
        for p in model.blocks[13].parameters():
            p.requires_grad_(True)
        model.norm.float()
        for p in model.norm.parameters():
            p.requires_grad_(True)
        head_adapter = FP32HeadAdapter(width=2048, vocab=49152, rank=64, device=device)
    elif scope == "full":
        model.float()
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(f"unsupported train scope: {scope}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if head_adapter is not None:
        trainable += sum(p.numel() for p in head_adapter.parameters())
    return head_adapter, freeze_prefix, trainable, base_params


def _run_block_mixed(
    block: nn.Module, x: torch.Tensor, start_pos: int = 0, kv_cache: Any = None
) -> tuple[torch.Tensor, Any]:
    """Execute AlephBlock safely when norm/weights are FP32 or mixed BF16+FP32."""
    target_dt = block.n1.weight.dtype
    return block(x.to(target_dt), start_pos=start_pos, kv_cache=kv_cache)


def _forward_full_logits(
    model: AlephNativeModel,
    tokens: torch.Tensor,
    *,
    freeze_prefix_visits: int = 0,
    head_adapter: FP32HeadAdapter | None = None,
) -> torch.Tensor:
    if freeze_prefix_visits > 0:
        with torch.no_grad():
            x = model.embed(tokens)
            for physical in ALEPH_VISIT_SCHEDULE[:freeze_prefix_visits]:
                x, _ = _run_block_mixed(model.blocks[physical], x, start_pos=0, kv_cache=None)
        x = x.detach()
        for physical in ALEPH_VISIT_SCHEDULE[freeze_prefix_visits:]:
            x, _ = _run_block_mixed(model.blocks[physical], x, start_pos=0, kv_cache=None)
    else:
        x = model.embed(tokens)
        for physical in ALEPH_VISIT_SCHEDULE:
            x, _ = _run_block_mixed(model.blocks[physical], x, start_pos=0, kv_cache=None)
    h = model.norm(x.to(model.norm.weight.dtype))
    if head_adapter is not None:
        return head_adapter(h, model.embed.weight)
    return F.linear(h.to(model.embed.weight.dtype), model.embed.weight).float()


@torch.inference_mode()
def _generate_greedy(
    model: AlephNativeModel,
    tokenizer: Tokenizer,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    device: torch.device,
    head_adapter: FP32HeadAdapter | None = None,
) -> tuple[list[int], str, bool]:
    t_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    x = model.embed(t_ids)
    caches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for physical in ALEPH_VISIT_SCHEDULE:
        x, updated = _run_block_mixed(model.blocks[physical], x, start_pos=0, kv_cache=None)
        caches.append(updated)
    h_last = model.norm(x[:, -1:, :].to(model.norm.weight.dtype))
    logits = (
        head_adapter(h_last, model.embed.weight)[:, -1, :]
        if head_adapter is not None
        else F.linear(h_last.to(model.embed.weight.dtype), model.embed.weight)[:, -1, :].float()
    )

    generated: list[int] = []
    ended_eos = False
    pos = len(prompt_ids)
    for _ in range(max_new_tokens):
        nxt = int(logits[0].argmax().item())
        generated.append(nxt)
        if nxt == EOS_ID:
            ended_eos = True
            break
        step_t = torch.tensor([[nxt]], dtype=torch.long, device=device)
        sx = model.embed(step_t)
        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for v_idx, physical in enumerate(ALEPH_VISIT_SCHEDULE):
            sx, updated = _run_block_mixed(model.blocks[physical], sx, start_pos=pos, kv_cache=caches[v_idx])
            new_caches.append(updated)
        caches = new_caches
        sh = model.norm(sx[:, -1:, :].to(model.norm.weight.dtype))
        logits = (
            head_adapter(sh, model.embed.weight)[:, -1, :]
            if head_adapter is not None
            else F.linear(sh.to(model.embed.weight.dtype), model.embed.weight)[:, -1, :].float()
        )
        pos += 1
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return generated, text, ended_eos


def _load_exclusions(paths: list[Path]) -> tuple[set[str], set[str]]:
    roots: set[str] = set()
    sems: set[str] = set()
    for p in paths:
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            roots.update(data.get("held_root_ids", []))
            sems.update(data.get("held_semantic_sha256", []))
    return roots, sems


def _load_specialist_examples(
    public_jsonl: Path,
    private_jinn_jsonl: Path | None,
    tokenizer: Tokenizer,
    excl_roots: set[str],
    excl_sems: set[str],
    max_seq_len: int,
) -> tuple[list[tuple[list[int], list[int], str]], dict[str, Any]]:
    examples: list[tuple[list[int], list[int], str]] = []
    pub_sup = 0
    pub_count = 0
    for line in public_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["root_id"] in excl_roots or rec["semantic_sha256"] in excl_sems:
            raise RuntimeError(f"EXCLUDED ROOT IN PUBLIC SPECIALIST DATA: {rec['root_id']}")
        p_ids = tokenizer.encode(rec["prompt"], add_special_tokens=False).ids
        t_ids = [*tokenizer.encode(rec["target"], add_special_tokens=False).ids, EOS_ID]
        max_p = max(16, max_seq_len - len(t_ids))
        if len(p_ids) > max_p:
            p_ids = p_ids[-max_p:]
        examples.append((p_ids, t_ids, "research_v2_synthetic"))
        pub_sup += len(t_ids) - 1
        pub_count += 1

    jb_sup = 0
    jb_count = 0
    if private_jinn_jsonl is not None and private_jinn_jsonl.is_file():
        for line in private_jinn_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("root_id") in excl_roots or rec.get("semantic_sha256") in excl_sems:
                raise RuntimeError(f"EXCLUDED ROOT IN JINN/BEAST INTAKE: {rec.get('root_id')}")
            prompt_txt = canon(rec["input"]) + "\nTarget:\n"
            target_txt = canon(rec["target"])
            p_ids = tokenizer.encode(prompt_txt, add_special_tokens=False).ids
            t_ids = [*tokenizer.encode(target_txt, add_special_tokens=False).ids, EOS_ID]
            max_p = max(16, max_seq_len - len(t_ids))
            if len(p_ids) > max_p:
                p_ids = p_ids[-max_p:]
            examples.append((p_ids, t_ids, "jinn_beast_train"))
            jb_sup += len(t_ids) - 1
            jb_count += 1

    meta = {
        "public_synthetic_records": pub_count,
        "public_synthetic_supervised_tokens": pub_sup,
        "private_jinn_beast_records": jb_count,
        "private_jinn_beast_supervised_tokens": jb_sup,
        "total_supervised_tokens_pool": pub_sup + jb_sup,
        "exclusion_overlaps": 0,
    }
    return examples, meta


def _build_disjoint_general_corpus(
    repo_root: Path, tokenizer: Tokenizer, max_seq_len: int
) -> tuple[list[list[int]], list[list[int]], dict[str, Any]]:
    excluded_parts = {"research_v2", "research_v2_1", "research_v2_baby_goblin_mesh", "research_v2_specialist", ".git"}
    docs: list[tuple[str, str]] = []
    for folder in (repo_root / "docs", repo_root / "pre-trains", repo_root / "README.md"):
        candidates = [folder] if folder.is_file() else sorted(folder.rglob("*.md"))
        for p in candidates:
            if any(part in excluded_parts for part in p.parts) or "research_verifiers_v2" in p.name:
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore").strip()
            if len(txt) >= 300:
                docs.append((str(p.relative_to(repo_root)), txt))

    import collections
    import datetime
    import difflib
    import fractions
    import heapq
    import statistics

    for mod in (collections, datetime, difflib, fractions, heapq, statistics):
        docs.append((f"stdlib/{mod.__name__}.py", inspect.getsource(mod)))

    train_chunks: list[list[int]] = []
    eval_chunks: list[list[int]] = []
    train_doc_cnt = 0
    eval_doc_cnt = 0
    doc_hashes: list[str] = []
    for label, txt in sorted(docs, key=lambda x: x[0]):
        h = hashlib.sha256(label.encode("utf-8")).hexdigest()
        doc_hashes.append(h)
        ids = tokenizer.encode(txt, add_special_tokens=False).ids
        is_eval = int(h[:4], 16) % 5 == 0
        if is_eval:
            eval_doc_cnt += 1
        else:
            train_doc_cnt += 1
        step = max(64, max_seq_len - 16)
        for start in range(0, max(1, len(ids) - 64), step):
            chunk = ids[start : start + max_seq_len]
            if len(chunk) >= 64:
                if is_eval:
                    eval_chunks.append(chunk)
                else:
                    train_chunks.append(chunk)

    meta = {
        "train_docs": train_doc_cnt,
        "eval_docs": eval_doc_cnt,
        "train_chunks": len(train_chunks),
        "eval_chunks": len(eval_chunks),
        "train_eval_doc_overlap": 0,
        "corpus_manifest_sha256": digest(sorted(doc_hashes)),
    }
    return train_chunks, eval_chunks, meta


def _compute_repetition_metrics(token_ids: list[int]) -> dict[str, float]:
    if not token_ids:
        return {"top1_token_share": 1.0, "distinct_2": 0.0}
    cnt = Counter(token_ids)
    top1_share = cnt.most_common(1)[0][1] / len(token_ids)
    if len(token_ids) < 2:
        return {"top1_token_share": round(top1_share, 4), "distinct_2": 0.0}
    bigrams = list(itertools.pairwise(token_ids))
    return {"top1_token_share": round(top1_share, 4), "distinct_2": round(len(set(bigrams)) / len(bigrams), 4)}


def _parse_valid_action(text: str) -> tuple[bool, bool, str | None]:
    s = text.strip()
    try:
        obj = json.loads(s)
    except Exception:
        return False, False, None
    if not isinstance(obj, dict):
        return False, False, None
    act = obj.get("action")
    allowed = {
        "run_experiment": {"action", "config"},
        "check": {"action", "answer"},
        "submit": {"action", "answer"},
        "inspect": {"action"},
    }
    schema_ok = isinstance(act, str) and act in allowed and set(obj) == allowed[act]
    return True, schema_ok, act if isinstance(act, str) else None


def _evaluate_arm(
    model: AlephNativeModel,
    tokenizer: Tokenizer,
    *,
    device: torch.device,
    eval_cases: list[dict[str, Any]],
    general_eval_chunks: list[list[int]],
    token_checks_path: Path,
    max_seq_len: int,
    max_gen_tokens: int,
    head_adapter: FP32HeadAdapter | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    gen_nll_sum = 0.0
    gen_tok_cnt = 0
    with torch.inference_mode():
        for chunk in general_eval_chunks[:24]:
            t = torch.tensor([chunk], dtype=torch.long, device=device)
            logits = _forward_full_logits(model, t[:, :-1], head_adapter=head_adapter)
            targets = t[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum")
            gen_nll_sum += float(loss.item())
            gen_tok_cnt += int(targets.numel())
    gen_nll = gen_nll_sum / max(1, gen_tok_cnt)
    gen_ppl = float(math.exp(min(30.0, gen_nll)))

    tc_matched = 0
    tc_total = 0
    if token_checks_path.is_file():
        checks = json.loads(token_checks_path.read_text(encoding="utf-8"))
        tc_total = len(checks)
        with torch.inference_mode():
            for item in checks:
                ids = tokenizer.encode(item["prompt"], add_special_tokens=False).ids
                t = torch.tensor([ids], dtype=torch.long, device=device)
                logits = _forward_full_logits(model, t, head_adapter=head_adapter)[:, -1, :]
                if int(logits[0].argmax().item()) == item["native_top1_id"]:
                    tc_matched += 1

    tf_nll_sum = 0.0
    tf_tok_cnt = 0
    tf_correct_toks = 0
    eos_probs: list[float] = []
    valid_json_cnt = 0
    balanced_brace_cnt = 0
    schema_valid_cnt = 0
    eos_stop_cnt = 0
    verb_correct_cnt = 0
    gen_lengths: list[int] = []
    top1_shares: list[float] = []
    distinct2s: list[float] = []
    sample_outputs: list[dict[str, Any]] = []
    exported_generations: list[dict[str, Any]] = []

    for case in eval_cases:
        obs0 = {
            "done": False,
            "remaining": {"turns": 8, "experiments": 4, "checks": 2, "optimizer_updates": 256},
            "artifacts": [],
            "compact_state": {
                "family_kind": "learner" if case["family"] in LEARNERS else "symbolic",
                "turn_log": [],
                "disallowed_actions": [],
                "next_valid_actions": ["run_experiment"] if case["family"] in LEARNERS else ["check", "submit"],
                "verified_ready": False,
            },
            "status": "ready",
        }
        prompt_str = case["prompt"] + "\nCurrent observation:\n" + canon(obs0) + "\nAction:\n"
        expected_verb = "run_experiment" if case["family"] in LEARNERS else "check"
        target_probe = (
            '{"action":"run_experiment","config":{"steps":0}}'
            if case["family"] in LEARNERS
            else '{"action":"check","answer":{}}'
        )
        p_ids = tokenizer.encode(prompt_str, add_special_tokens=False).ids
        t_ids = [*tokenizer.encode(target_probe, add_special_tokens=False).ids, EOS_ID]
        max_p = max(16, max_seq_len - len(t_ids))
        p_trim = p_ids[-max_p:] if len(p_ids) > max_p else p_ids

        with torch.inference_mode():
            seq = torch.tensor([p_trim + t_ids], dtype=torch.long, device=device)
            logits = _forward_full_logits(model, seq[:, :-1], head_adapter=head_adapter)
            start_pos = len(p_trim) - 1
            t_logits = logits[0, start_pos : start_pos + len(t_ids), :]
            t_targets = torch.tensor(t_ids, dtype=torch.long, device=device)
            tf_nll_sum += float(F.cross_entropy(t_logits, t_targets, reduction="sum").item())
            tf_tok_cnt += len(t_ids)
            tf_correct_toks += int((t_logits.argmax(dim=-1) == t_targets).sum().item())
            eos_probs.append(float(F.softmax(t_logits[-1], dim=-1)[EOS_ID].item()))

        gen_ids, gen_text, ended_eos = _generate_greedy(
            model,
            tokenizer,
            p_trim,
            max_new_tokens=max_gen_tokens,
            device=device,
            head_adapter=head_adapter,
        )
        gen_lengths.append(len(gen_ids))
        if ended_eos:
            eos_stop_cnt += 1
        rep = _compute_repetition_metrics(gen_ids)
        top1_shares.append(rep["top1_token_share"])
        distinct2s.append(rep["distinct_2"])
        s = gen_text.strip()
        if s.startswith("{") and s.endswith("}") and s.count("{") == s.count("}"):
            balanced_brace_cnt += 1
        v_json, v_schema, act_verb = _parse_valid_action(gen_text)
        if v_json:
            valid_json_cnt += 1
        if v_schema:
            schema_valid_cnt += 1
        if act_verb == expected_verb or (case["family"] not in LEARNERS and act_verb in {"check", "submit"}):
            verb_correct_cnt += 1

        exported_generations.append(
            {
                "task_id": case["id"],
                "family": case["family"],
                "gen_text": gen_text,
                "ended_eos": ended_eos,
                "gen_tokens": len(gen_ids),
            }
        )
        if len(sample_outputs) < 4:
            sample_outputs.append(
                {
                    "family": case["family"],
                    "gen_tokens": len(gen_ids),
                    "ended_eos": ended_eos,
                    "valid_json": v_json,
                    "schema_valid": v_schema,
                    "action_verb": act_verb,
                    "top1_token_share": rep["top1_token_share"],
                    "text_excerpt": gen_text[:120],
                }
            )

    n_cases = max(1, len(eval_cases))
    metrics = {
        "syntax_acquisition": {
            "valid_json_rate": round(valid_json_cnt / n_cases, 4),
            "balanced_brace_rate": round(balanced_brace_cnt / n_cases, 4),
            "schema_valid_action_rate": round(schema_valid_cnt / n_cases, 4),
            "mean_top1_token_share": round(sum(top1_shares) / n_cases, 4),
            "mean_distinct_2": round(sum(distinct2s) / n_cases, 4),
        },
        "eos_termination": {
            "eos_stop_rate": round(eos_stop_cnt / n_cases, 4),
            "truncation_rate": round(1.0 - eos_stop_cnt / n_cases, 4),
            "mean_teacher_forced_eos_prob": round(sum(eos_probs) / n_cases, 6),
            "mean_generated_tokens": round(sum(gen_lengths) / n_cases, 2),
        },
        "action_selection": {
            "action_verb_accuracy": round(verb_correct_cnt / n_cases, 4),
            "teacher_forced_target_nll_nats": round(tf_nll_sum / max(1, tf_tok_cnt), 4),
            "teacher_forced_token_accuracy": round(tf_correct_toks / max(1, tf_tok_cnt), 4),
        },
        "general_language_retention": {
            "general_eval_tokens": gen_tok_cnt,
            "general_val_nll_nats": round(gen_nll, 4),
            "general_val_perplexity": round(gen_ppl, 2),
            "token_checks_top1_match": f"{tc_matched}/{tc_total}",
        },
        "sample_outputs": sample_outputs,
    }
    return metrics, exported_generations


def _train_steps(
    model: AlephNativeModel,
    *,
    optimizer: torch.optim.Optimizer,
    steps: int,
    mode: str,
    specialist_examples: list[tuple[list[int], list[int], str]],
    base_chunks: list[list[int]],
    device: torch.device,
    freeze_prefix_visits: int,
    head_adapter: FP32HeadAdapter | None,
    seed: int,
) -> dict[str, Any]:
    model.train()
    gen = torch.Generator().manual_seed(seed)
    total_seq_tokens = 0
    total_loss_tokens = 0
    curve: list[dict[str, Any]] = []
    t0 = time.monotonic()

    pub_ex = [e for e in specialist_examples if e[2] == "research_v2_synthetic"]
    jb_ex = [e for e in specialist_examples if e[2] == "jinn_beast_train"]

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss_val = 0.0
        if mode == "specialist":
            batch_items = [pub_ex[int(torch.randint(0, len(pub_ex), (1,), generator=gen).item())]]
            if jb_ex:
                batch_items.append(jb_ex[int(torch.randint(0, len(jb_ex), (1,), generator=gen).item())])
            else:
                batch_items.append(pub_ex[int(torch.randint(0, len(pub_ex), (1,), generator=gen).item())])
            for p_ids, t_ids, _src in batch_items:
                seq = torch.tensor([p_ids + t_ids], dtype=torch.long, device=device)
                logits = _forward_full_logits(
                    model, seq[:, :-1], freeze_prefix_visits=freeze_prefix_visits, head_adapter=head_adapter
                )
                start = len(p_ids) - 1
                t_logits = logits[0, start : start + len(t_ids), :]
                t_targets = torch.tensor(t_ids, dtype=torch.long, device=device)
                loss = F.cross_entropy(t_logits, t_targets) / len(batch_items)
                loss.backward()
                step_loss_val += float(loss.item())
                total_seq_tokens += int(seq.numel())
                total_loss_tokens += len(t_ids)
        else:
            for _ in range(2):
                chunk = base_chunks[int(torch.randint(0, len(base_chunks), (1,), generator=gen).item())]
                seq = torch.tensor([chunk], dtype=torch.long, device=device)
                logits = _forward_full_logits(
                    model, seq[:, :-1], freeze_prefix_visits=freeze_prefix_visits, head_adapter=head_adapter
                )
                targets = seq[:, 1:]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) / 2.0
                loss.backward()
                step_loss_val += float(loss.item())
                total_seq_tokens += int(seq.numel())
                total_loss_tokens += int(targets.numel())

        params = [p for p in model.parameters() if p.requires_grad]
        if head_adapter is not None:
            params.extend(list(head_adapter.parameters()))
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0).item())
        optimizer.step()

        if step in (1, max(1, steps // 4), max(1, steps // 2), steps):
            curve.append(
                {
                    "step": step,
                    "mode": mode,
                    "loss_nats": round(step_loss_val, 4),
                    "grad_norm": round(grad_norm, 4),
                    "elapsed_s": round(time.monotonic() - t0, 2),
                }
            )
    return {
        "steps": steps,
        "mode": mode,
        "total_forward_tokens": total_seq_tokens,
        "total_supervised_loss_tokens": total_loss_tokens,
        "wall_seconds": round(time.monotonic() - t0, 2),
        "learning_curve": curve,
    }


def _weights_digest(model: AlephNativeModel, head_adapter: FP32HeadAdapter | None) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if p.requires_grad:
            h.update(name.encode("utf-8"))
            h.update(p.detach().float().cpu().numpy().tobytes()[:4096])
    if head_adapter is not None:
        for name, p in sorted(head_adapter.named_parameters()):
            h.update(("head." + name).encode("utf-8"))
            h.update(p.detach().float().cpu().numpy().tobytes()[:4096])
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--token-checks", type=Path, required=True)
    parser.add_argument("--public-specialist", type=Path, required=True)
    parser.add_argument("--private-jinn-train", type=Path, default=None)
    parser.add_argument("--exclude-registry", type=Path, action="append", default=[])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--train-scope", choices=("exit_shell_fp32", "full"), default="exit_shell_fp32")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-seq-len", type=int, default=224)
    parser.add_argument("--max-gen-tokens", type=int, default=64)
    parser.add_argument("--eval-seed", type=int, default=2026112799)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--generations-out", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    proc = psutil.Process(os.getpid())
    started = time.monotonic()

    weights_sha = _sha256_file(args.weights)
    tok_sha = _sha256_file(args.tokenizer)
    if weights_sha != NATIVE_WEIGHTS_SHA256:
        raise RuntimeError(f"native weights SHA mismatch: {weights_sha}")
    if tok_sha != TOKENIZER_SHA256:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    excl_roots, excl_sems = _load_exclusions(args.exclude_registry)
    specialist_examples, spec_meta = _load_specialist_examples(
        args.public_specialist, args.private_jinn_train, tokenizer, excl_roots, excl_sems, args.max_seq_len
    )
    base_train_chunks, general_eval_chunks, base_meta = _build_disjoint_general_corpus(
        REPO_ROOT, tokenizer, args.max_seq_len
    )

    eval_cases = make_suite_v21(seed=args.eval_seed, per_family=2, split="development")
    spec_roots = {
        json.loads(line)["root_id"]
        for line in args.public_specialist.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if {c["root_id"] for c in eval_cases} & (spec_roots | excl_roots):
        raise RuntimeError("eval cases overlapped training or excluded roots")

    print(
        f"[bg-exp] Loading base weights {weights_sha[:12]} (BF16 base + FP32 trainable on {args.device})...",
        flush=True,
    )
    raw_state = load_file(str(args.weights), device="cpu")
    base_state_bf16 = {k: v.to(dtype=torch.bfloat16) for k, v in raw_state.items()}
    del raw_state

    def make_fresh_model() -> tuple[AlephNativeModel, FP32HeadAdapter | None, int, int, int]:
        m = AlephNativeModel().to(dtype=torch.bfloat16)
        m.load_state_dict({k: v.clone() for k, v in base_state_bf16.items()}, strict=True)
        m.to(device)
        ha, fp, tr, tb = _apply_trainable_scope(m, args.train_scope, device)
        return m, ha, fp, tr, tb

    arm_plan = [
        ("arm0_original_10m", []),
        ("arm1_specialist_only", [("specialist", args.steps)]),
        ("arm2_continued_base", [("base", args.steps)]),
        ("arm3_base_then_specialist", [("base", args.steps // 2), ("specialist", args.steps - args.steps // 2)]),
    ]

    arms_report: dict[str, Any] = {}
    all_generations: dict[str, Any] = {"eval_seed": args.eval_seed, "arms": {}}
    peak_ram_mib = 0.0
    peak_vram_mib = 0.0
    freeze_prefix_used = 0

    for arm_idx, (arm_name, schedule) in enumerate(arm_plan):
        print(f"[bg-exp] === {arm_name} schedule={schedule} ===", flush=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model, head_adapter, freeze_prefix, trainable_params, total_base_params = make_fresh_model()
        freeze_prefix_used = freeze_prefix
        init_digest = _weights_digest(model, head_adapter)
        train_stages = []
        if schedule:
            opt_params = [p for p in model.parameters() if p.requires_grad]
            if head_adapter is not None:
                opt_params.extend(list(head_adapter.parameters()))
            optimizer = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=0.01)
            for s_idx, (stage_mode, stage_steps) in enumerate(schedule):
                stage_rec = _train_steps(
                    model,
                    optimizer=optimizer,
                    steps=stage_steps,
                    mode=stage_mode,
                    specialist_examples=specialist_examples,
                    base_chunks=base_train_chunks,
                    device=device,
                    freeze_prefix_visits=freeze_prefix,
                    head_adapter=head_adapter,
                    seed=20261127 + arm_idx * 100 + s_idx,
                )
                train_stages.append(stage_rec)
        after_digest = _weights_digest(model, head_adapter)
        ev, gens = _evaluate_arm(
            model,
            tokenizer,
            device=device,
            eval_cases=eval_cases,
            general_eval_chunks=general_eval_chunks,
            token_checks_path=args.token_checks,
            max_seq_len=args.max_seq_len,
            max_gen_tokens=args.max_gen_tokens,
            head_adapter=head_adapter,
        )
        rss = proc.memory_info().rss / (1024 * 1024)
        peak_ram_mib = max(peak_ram_mib, rss)
        if device.type == "cuda":
            peak_vram_mib = max(peak_vram_mib, torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        arms_report[arm_name] = {
            "trainable_parameters": trainable_params,
            "total_base_parameters": total_base_params,
            "weights_changed": after_digest != init_digest,
            "trainable_weights_sha256": after_digest,
            "total_optimizer_updates": sum(s["steps"] for s in train_stages),
            "total_forward_tokens": sum(s["total_forward_tokens"] for s in train_stages),
            "total_supervised_loss_tokens": sum(s["total_supervised_loss_tokens"] for s in train_stages),
            "training_stages": train_stages,
            "evaluation": ev,
        }
        all_generations["arms"][arm_name] = gens
        print(
            f"[bg-exp] {arm_name} done (peak_ram={rss:.0f} MiB): "
            f"json={ev['syntax_acquisition']['valid_json_rate']} "
            f"eos={ev['eos_termination']['eos_stop_rate']} "
            f"p_eos={ev['eos_termination']['mean_teacher_forced_eos_prob']} "
            f"act_nll={ev['action_selection']['teacher_forced_target_nll_nats']} "
            f"gen_nll={ev['general_language_retention']['general_val_nll_nats']}",
            flush=True,
        )
        del model, head_adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    base_nll = arms_report["arm0_original_10m"]["evaluation"]["general_language_retention"]["general_val_nll_nats"]
    for rec in arms_report.values():
        nll = rec["evaluation"]["general_language_retention"]["general_val_nll_nats"]
        rec["evaluation"]["general_language_retention"]["delta_nll_vs_10m"] = round(nll - base_nll, 4)

    report = {
        "schema": "research-v2.1-baby-goblin-early-specialist-exp-v1",
        "base_checkpoint": {
            "model_id": "AlephFunk/aleph-direct-a-10m-v2",
            "weights_sha256": weights_sha,
            "tokenizer_sha256": tok_sha,
            "parameters": 999352320,
            "visit_schedule": ALEPH_VISIT_SCHEDULE,
        },
        "compute_config": {
            "device": args.device,
            "base_dtype": "bfloat16",
            "trainable_dtype": "float32",
            "optimizer": "AdamW",
            "lr": args.lr,
            "threads": args.threads,
            "train_scope": args.train_scope,
            "freeze_prefix_visits": freeze_prefix_used,
            "matched_optimizer_updates": args.steps,
            "max_seq_len": args.max_seq_len,
            "max_gen_tokens": args.max_gen_tokens,
            "eval_seed": args.eval_seed,
            "peak_ram_mib": round(peak_ram_mib, 2),
            "peak_vram_mib": round(peak_vram_mib, 2),
            "wall_seconds": round(time.monotonic() - started, 2),
        },
        "specialist_corpus_accounting": spec_meta,
        "continued_base_corpus_accounting": base_meta,
        "arms": arms_report,
    }
    report["report_sha256"] = digest(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.generations_out is not None:
        args.generations_out.parent.mkdir(parents=True, exist_ok=True)
        args.generations_out.write_text(json.dumps(all_generations, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[bg-exp] DONE report_sha256={report['report_sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
