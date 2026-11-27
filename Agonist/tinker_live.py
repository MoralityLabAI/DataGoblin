"""Live DG-AGN-TINKER-01 runner. All paid actions require --live and a cap."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Agonist.tinker_gym import (
    ROLES,
    Journal,
    Task,
    canonical,
    digest,
    episode,
    make_tasks,
    parse_action,
)

MODEL = "thinkingmachines/Inkling-2-Mini"
EFFORT = 0.9
CONTEXT = 12_288
FABLE = "claude-fable-5-2"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def cap_usd(value: float) -> int:
    if not math.isfinite(value) or value < 0:
        raise ValueError("invalid USD cap")
    return math.floor(value * 1_000_000)


class Tinker:
    """Text-only provider with matched native Inkling renderer and effort."""
    def __init__(self, journal: Journal, cap: int, max_tokens: int):
        if not os.environ.get("TINKER_API_KEY"):
            raise RuntimeError("TINKER_API_KEY is missing")
        if cap <= 0 or not 128 <= max_tokens <= 8192:
            raise ValueError("explicit positive cap and bounded generation required")
        import tinker
        from tinker_cookbook import model_info
        from tinker_cookbook.renderers import get_renderer, get_text_content
        from tinker_cookbook.supervised.common import datum_from_model_input_weights
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        self.journal, self.cap, self.max_tokens = journal, cap, max_tokens
        self.tinker = tinker
        self.service = tinker.ServiceClient()
        self.renderer_name = model_info.get_recommended_renderer_name(MODEL)
        self.renderer = get_renderer(self.renderer_name, get_tokenizer(MODEL))
        self.get_text = get_text_content
        self.to_datum = datum_from_model_input_weights
        self.samplers: dict[str, Any] = {}

    def sample(self, key: str, prompt: str, snapshot: str | None) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        rendered = self.renderer.build_generation_prompt(messages, effort=EFFORT)
        n = rendered.length
        if n + self.max_tokens > CONTEXT:
            raise ValueError("rendered prompt + completion exceeds bounded context")
        identity = snapshot or MODEL
        reserve = math.ceil(1.2 * (0.15 * n + 0.50 * self.max_tokens))
        def invoke() -> dict[str, Any]:
            if identity not in self.samplers:
                kwargs = {"model_path": snapshot} if snapshot else {"base_model": MODEL}
                self.samplers[identity] = self.service.create_sampling_client(**kwargs).result()
            sampled = self.samplers[identity].sample(
                prompt=rendered, num_samples=1,
                sampling_params=self.tinker.SamplingParams(
                    max_tokens=self.max_tokens, temperature=0.2, stop=self.renderer.get_stop_sequences(),
                ),
            ).result().sequences[0]
            message, termination = self.renderer.parse_response(sampled.tokens)
            clean = bool(termination.is_clean)
            return {
                "actor": identity, "text": self.get_text(message) if clean else "",
                "termination": str(termination), "stop_reason": str(sampled.stop_reason),
                "usage": {"input_tokens": n, "output_tokens": len(sampled.tokens),
                          "estimated_usd": (0.15 * n + 0.50 * len(sampled.tokens)) / 1_000_000},
                "renderer": self.renderer_name, "effort": EFFORT, "utc": utc(),
            }
        return self.journal.call(
            key, "tinker", [identity, prompt, self.max_tokens, self.renderer_name, EFFORT],
            reserve, self.cap, invoke,
        )

    def train_step(
        self, key: str, tasks: list[Task], state: str | None, rank: int, lr: float,
    ) -> dict[str, Any]:
        if self.journal.mode != "clean":
            raise ValueError("advisory journal cannot train")
        if not tasks or len({t.root for t in tasks}) != len(tasks):
            raise ValueError("empty or repeated-root batch")
        examples = [t.demonstration() for t in tasks]
        data, tokens = [], 0
        for example in examples:
            model_input, weights = self.renderer.build_supervised_example(example["messages"], effort=EFFORT)
            if model_input.length > CONTEXT or not bool(weights.sum() > 0):
                raise ValueError("invalid SFT example: overflow or no supervised tokens")
            tokens += model_input.length
            data.append(self.to_datum(model_input, weights, max_length=CONTEXT))
        reserve = math.ceil(1.2 * 0.75 * tokens)
        identity = [MODEL, self.renderer_name, EFFORT, state, rank, lr, examples]
        name = "dg-" + digest([key, identity])[:24]
        def invoke() -> dict[str, Any]:
            client = (
                self.service.create_training_client_from_state_with_optimizer(state).result()
                if state else self.service.create_lora_training_client(base_model=MODEL, rank=rank).result()
            )
            fb = client.forward_backward(data, loss_fn="cross_entropy").result()
            client.optim_step(self.tinker.AdamParams(learning_rate=lr)).result()
            # Explicit expiry bounds the accumulation of checkpoint storage.
            ttl = 45 * 24 * 3600
            full = client.save_state(name, ttl_seconds=ttl).result()
            sampler = client.save_weights_for_sampler(name, ttl_seconds=ttl).result()
            return {
                "state": full.path, "sampler": sampler.path, "task_ids": [t.key for t in tasks],
                "rank": rank, "learning_rate": lr, "renderer": self.renderer_name,
                "effort": EFFORT, "training_tokens": tokens,
                "estimated_usd": 0.75 * tokens / 1_000_000,
                "metrics": {str(k): float(v) for k, v in fb.metrics.items() if math.isfinite(float(v))},
                "utc": utc(),
            }
        return self.journal.call(key, "tinker", identity, reserve, self.cap, invoke)


class LocalDG:
    """llama.cpp completion API; no assumed Aleph chat template or tool access."""
    def __init__(self, journal: Journal, url: str, model_id: str):
        # A fixed loopback origin prevents model-controlled URL or network access.
        if url not in ("http://127.0.0.1:8080", "http://127.0.0.1:8000"):
            raise ValueError("use a supported loopback llama.cpp server")
        if not model_id:
            raise ValueError("provide an actual local checkpoint identity")
        self.journal, self.url, self.model_id = journal, url, model_id

    def sample(self, key: str, prompt: str) -> dict[str, Any]:
        # The local checkpoint is a completion model; provide an explicit text format.
        text = "DataGoblin task:\n" + prompt + "\nAnswer as JSON:\n"
        def invoke() -> dict[str, Any]:
            payload = {"prompt": text, "n_predict": 256, "temperature": 0.2, "stream": False,
                       "cache_prompt": False}
            req = urllib.request.Request(
                self.url + "/completion", data=canonical(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as response:
                reply = json.loads(response.read(1_000_000))
            return {
                "actor": self.model_id, "text": str(reply.get("content", "")),
                "usage": {"input_tokens": reply.get("tokens_evaluated"),
                          "output_tokens": reply.get("tokens_predicted")}, "utc": utc(),
            }
        return self.journal.call(key, "local", [self.model_id, self.url, text], 0, 0, invoke)


class FableAdvisor:
    """Separate operational advisory experiment; never a training teacher."""
    def __init__(self, journal: Journal, cap: int, max_tokens: int):
        if journal.mode != "advisory" or cap <= 0 or not 512 <= max_tokens <= 16384:
            raise ValueError("advisory mode, positive cap and bounded output required")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is missing")
        import anthropic

        self.client = anthropic.Anthropic(max_retries=0, timeout=1800.0)
        self.journal, self.cap, self.max_tokens = journal, cap, max_tokens

    def consult(self, key: str, task: Task, history: list[dict[str, Any]]) -> str:
        prompt = canonical({
            "purpose": "Independent operational advisory evaluation. Suggest a concise correction "
                       "to the agent's visible decision process. No tool actions or training data.",
            "public": task.public, "role": task.role, "allowed_actions": sorted(task.scores),
            "attempts": history,
        })
        messages = [{"role": "user", "content": prompt}]
        # Byte count plus margin is a conservative bound for this ASCII JSON prompt.
        # Account ceilings remain authoritative; no SDK automatic retries.
        n = len(prompt.encode()) + 1024
        reserve = 8 * n + 40 * self.max_tokens
        def invoke() -> dict[str, Any]:
            with self.client.messages.stream(
                model=FABLE, max_tokens=self.max_tokens, output_config={"effort": "xhigh"},
                messages=messages,
            ) as stream:
                response = stream.get_final_message()
            return {
                "text": "\n".join(b.text for b in response.content if b.type == "text"),
                "actor": FABLE, "effort": "xhigh",
                "usage": {"input_tokens": response.usage.input_tokens,
                          "output_tokens": response.usage.output_tokens},
                "utc": utc(),
            }
        result = self.journal.call(key, "fable", [FABLE, prompt, self.max_tokens], reserve, self.cap, invoke)
        return result["text"][:4000]


def active(journal: Journal, role: str, before: int) -> dict[str, Any] | None:
    records = [
        r for r in journal.events()
        if r.get("type") == "promotion" and r["role"] == role and r["round"] < before and r["promoted"]
    ]
    return max(records, key=lambda r: r["round"])["candidate"] if records else None


def choose_batch(
    journal: Journal, key: str, pool: list[Task], count: int,
    curator: str, rng: random.Random, dg: LocalDG | None, provider: Tinker, curriculum: str | None,
) -> list[Task]:
    existing = journal.get_event(key)
    if existing:
        ids = existing["selected"]
        by_id = {t.key: t for t in pool}
        if len(ids) != count or len(set(ids)) != count or any(x not in by_id for x in ids):
            raise ValueError("stored selection incompatible with pool")
        return [by_id[x] for x in ids]
    # All rows are train-only. Selection sees task identities and verified public inputs,
    # but no development labels, model-generated reward or Fable material.
    if curator in ("dg", "inky"):
        # Keep the curation prompt compact. Observe prior clean train failures only.
        misses = {
            r["task_id"] for r in journal.events()
            if r.get("type") == "episode" and r.get("split") == "train" and not r["success"]
        }
        visible = [{"id": t.key, "role": t.role, "previous_failure": t.key in misses,
                    "features": {"candidate_count": len(t.scores), "public_size": len(canonical(t.public)),
                                 "task_kind": t.public.get("request", t.public.get("goal", t.role))}}
                   for t in pool]
        prompt = canonical({"instruction": f"Choose exactly {count} IDs from the pool for the next "
                                          "specialist SFT batch. Return JSON with selected list.",
                            "candidates": visible})
        if curator == "dg":
            if dg is None:
                raise ValueError("DG curator requires local endpoint")
            result = dg.sample(key + ":curator", prompt)
        else:
            result = provider.sample(key + ":curator", prompt, curriculum)
        try:
            selected = json.loads(result["text"])["selected"]
            if (not isinstance(selected, list) or not all(isinstance(x, str) for x in selected)
                    or len(selected) != count or len(set(selected)) != count
                    or not set(selected) <= {t.key for t in pool}):
                raise ValueError("invalid selection")
            method = curator
        except (ValueError, KeyError, TypeError):
            selected = [t.key for t in pool[:count]]
            method = curator + "_invalid_fixed_fallback"
    else:
        selected = [t.key for t in (rng.sample(pool, count) if curator == "random" else pool[:count])]
        method = curator
    journal.event(key, {"type": "selection", "selected": selected, "method": method,
                        "pool_hash": digest([t.key for t in pool])})
    by_id = {t.key: t for t in pool}
    return [by_id[x] for x in selected]


def train(
    journal: Journal, provider: Tinker, dg: LocalDG | None, args: argparse.Namespace, role: str, rnd: int,
) -> dict[str, Any]:
    if journal.mode != "clean":
        raise ValueError("no advisory training")
    key = f"candidate:{rnd}:{role}"
    previous = journal.get_event(key)
    if previous:
        return previous
    pool = [t for t in make_tasks(rnd, args.groups, "train") if t.role == role]
    if args.steps * args.batch_size > len(pool):
        raise ValueError("insufficient fresh roots; increase --groups or reduce steps")
    incumbent = active(journal, role, rnd)
    state = incumbent["state"] if incumbent else None
    curriculum = active(journal, "curriculum", rnd)
    rng = random.Random(rnd)
    for step in range(args.steps):
        step_key = f"train:{rnd}:{role}:{step}"
        selected = choose_batch(
            journal, step_key + ":selection", pool, args.batch_size, args.curator, rng, dg, provider,
            curriculum["sampler"] if curriculum else None,
        )
        selected_ids = {t.key for t in selected}
        pool = [t for t in pool if t.key not in selected_ids]
        result = provider.train_step(step_key, selected, state, args.rank, args.lr)
        state = result["state"]
    candidate = {
        "type": "candidate", "role": role, "round": rnd, "state": result["state"],
        "sampler": result["sampler"], "incumbent": incumbent["sampler"] if incumbent else None,
        "steps": args.steps, "batch_size": args.batch_size,
    }
    journal.event(key, candidate)
    return candidate


def evaluate(
    journal: Journal, provider: Tinker, args: argparse.Namespace, role: str, rnd: int,
) -> dict[str, Any]:
    key = f"promotion:{rnd}:{role}"
    existing = journal.get_event(key)
    if existing:
        return existing
    candidate = journal.get_event(f"candidate:{rnd}:{role}")
    if candidate is None:
        raise ValueError("train candidate before evaluation")
    tasks = [t for t in make_tasks(rnd, args.eval_groups, "dev") if t.role == role]
    scores: dict[str, list[dict[str, Any]]] = {"incumbent": [], "candidate": []}
    for task in tasks:
        for arm, snapshot in (("incumbent", candidate["incumbent"]), ("candidate", candidate["sampler"])):
            result = provider.sample(f"eval:{rnd}:{role}:{arm}:{task.key}", task.prompt(), snapshot)
            action = parse_action(result["text"])
            grade = task.grade(action)
            grade["false_release"] = role == "route" and task.gold == "STOP" and action not in (None, "STOP")
            scores[arm].append(grade)
    summary = {
        arm: {"success": sum(r["success"] for r in rows) / len(rows),
              "valid": sum(r["valid"] for r in rows) / len(rows),
              "false_release": sum(r["false_release"] for r in rows)}
        for arm, rows in scores.items()
    }
    gain = summary["candidate"]["success"] - summary["incumbent"]["success"]
    promoted = (gain >= args.min_gain and summary["candidate"]["valid"] >= 0.95
                and summary["candidate"]["false_release"] == 0)
    report = {"type": "promotion", "role": role, "round": rnd, "candidate": candidate,
              "promoted": promoted, "gain": gain, "scores": summary, "eval_roots": len(tasks),
              "claim": "development fixture gate only"}
    journal.event(key, report)
    return report


def rollout(
    journal: Journal, provider: Tinker, dg: LocalDG | None, advisor: FableAdvisor | None,
    args: argparse.Namespace, rnd: int,
) -> list[dict[str, Any]]:
    tasks = make_tasks(rnd, args.groups, "train")
    snapshots = {role: active(journal, role, rnd + 1) for role in ROLES}
    def actor(key: str, task: Task, prompt: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        if dg and (not history or any("advisory_note" in h for h in history)):
            return dg.sample(key + ":dg", prompt)
        if task.role != "route":
            route_prompt = canonical({
                "instruction": "Choose the adapter for the request. Return JSON action only.",
                "request": task.public.get("request", task.public.get("goal", task.public.get("evidence"))),
                "allowed_actions": list(ROLES),
                "adapter_descriptions": {
                    "curriculum": "select training roots", "route": "choose a specialist",
                    "diagnose": "identify failure prerequisite", "repair": "revise a decision",
                },
                "previous_attempts": history,
            })
            router = snapshots["route"]
            route_result = provider.sample(key + ":route", route_prompt, router["sampler"] if router else None)
            route_action = parse_action(route_result["text"])
            journal.event(key + ":route-decision", {"type": "route_decision", "task_id": task.key,
                          "chosen": route_action, "required": task.role, "actor": route_result["actor"]})
            if route_action != task.role:
                return {"actor": route_result["actor"], "text": "", "usage": route_result["usage"]}
        snapshot = snapshots[task.role]
        return provider.sample(key + ":specialist", prompt, snapshot["sampler"] if snapshot else None)
    return [episode(t, journal, f"rollout-{rnd}", actor, args.attempts,
                    advisor.consult if advisor else None) for t in tasks]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("train", "evaluate", "rollout", "campaign", "status", "trace"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--live", action="store_true")
    p.add_argument("--mode", choices=("clean", "advisory"), default="clean")
    p.add_argument("--advisory-authorized", action="store_true",
                   help="operator has permission for the separate Fable advisory experiment")
    p.add_argument("--tinker-usd", type=float, default=0)
    p.add_argument("--fable-usd", type=float, default=0)
    p.add_argument("--fable-max-tokens", type=int, default=8192)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--dg-url")
    p.add_argument("--dg-model-id", default="")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--groups", type=int, default=16)
    p.add_argument("--eval-groups", type=int, default=8)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min-gain", type=float, default=0.05)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--role", choices=ROLES, default="curriculum")
    p.add_argument("--curator", choices=("fixed", "random", "dg", "inky"), default="fixed")
    args = p.parse_args()
    if (args.round < 0 or not 1 <= args.rounds <= 1000 or not 1 <= args.groups <= 256
            or not 1 <= args.eval_groups <= 256 or args.steps < 1 or args.batch_size < 1
            or not 1 <= args.rank <= 64 or not 0 < args.lr <= 0.001 or not 0 < args.min_gain <= 1
            or not 1 <= args.attempts <= 8):
        p.error("invalid experiment limits")
    journal = Journal(args.out, args.mode)
    if args.command == "status":
        print(json.dumps(journal.summary(), indent=2))
        return
    if args.command == "trace":
        for record in journal.events():
            print(canonical(record))
        return
    if not args.live:
        p.error("remote calls require --live and explicit --tinker-usd")
    if args.mode == "advisory" and (
        args.command != "rollout" or not args.advisory_authorized or not args.dg_url
        or args.attempts < 2 or args.fable_usd <= 0
    ):
        p.error("advisory requires isolated authorized rollout, local DG, and positive Fable cap")
    if args.mode == "clean" and args.fable_usd != 0:
        p.error("Fable calls cannot enter clean lane")
    if args.command in ("train", "evaluate", "campaign") and args.mode != "clean":
        p.error("training and promotion require clean journal")
    if (args.out / "STOP").exists():
        p.error("operator STOP file")
    dg = LocalDG(journal, args.dg_url, args.dg_model_id) if args.dg_url else None
    provider = Tinker(journal, cap_usd(args.tinker_usd), args.max_tokens)
    advisor = FableAdvisor(journal, cap_usd(args.fable_usd), args.fable_max_tokens) if args.mode == "advisory" else None
    if args.command == "train":
        print(canonical(train(journal, provider, dg, args, args.role, args.round)))
    elif args.command == "evaluate":
        print(canonical(evaluate(journal, provider, args, args.role, args.round)))
    elif args.command == "rollout":
        print(canonical({"episodes": rollout(journal, provider, dg, advisor, args, args.round)}))
    else:
        for rnd in range(args.round, args.round + args.rounds):
            for role in ROLES:
                train(journal, provider, dg, args, role, rnd)
                print(canonical(evaluate(journal, provider, args, role, rnd)), flush=True)
            rows = rollout(journal, provider, dg, None, args, rnd)
            print(canonical({"round": rnd, "episodes": len(rows), "success": sum(r["success"] for r in rows),
                             "ledger": journal.summary()}), flush=True)
    print(json.dumps(journal.summary(), indent=2))


if __name__ == "__main__":
    main()
