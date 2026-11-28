"""Explicitly authorized two-wave Fable batch pilot. No paid calls on import."""
from __future__ import annotations

import argparse
import os
from decimal import Decimal
from pathlib import Path

from gym import (SYSTEM, continuation_prompt, digest, initial_prompt, observe,
                 parse_initial, read, write)

MODEL = "claude-fable-5-2"
# USD per million tokens = integer microdollars per token.
INPUT_RATE, OUTPUT_RATE = 4, 20
RATES_URL = "https://platform.claude.com/docs/en/models/fable-5-2/whats-new-fable-5-2"


def reserved_micro(input_tokens, max_output):
    # Token counting is an estimate. Buffer plus uncached base batch pricing.
    return INPUT_RATE * ((5 * input_tokens + 3) // 4 + 1024) + OUTPUT_RATE * max_output


def make_requests(out, wave):
    manifest = read(out / "manifest.json")
    if not manifest["source_verified"]:
        raise ValueError("paid run requires source-verified packets")
    if wave == 2 and not (out / "wave1" / "receipt.json").exists():
        raise ValueError("collect wave 1 first")
    result = []
    for rid in manifest["roots"]:
        packet = read(out / "public" / (rid + ".json"))
        if not packet["source_verified"] or packet["training_eligible"] or packet["custody"] != "research-only":
            raise ValueError("invalid packet custody")
        if wave == 1:
            prompt = initial_prompt(packet)
        else:
            rd = out / "wave1" / rid
            if not (rd / "initial.json").exists() or not (rd / "observation.json").exists():
                continue
            initial, obs = read(rd / "initial.json"), read(rd / "observation.json")
            if obs != observe(packet, initial):
                raise ValueError("observation changed")
            prompt = continuation_prompt(packet, initial, obs)
        result.append({"custom_id": rid, "params": {"model": MODEL,
                       "max_tokens": 8192 if wave == 1 else 16384,
                       "system": SYSTEM, "messages": [{"role": "user", "content": prompt}],
                       "output_config": {"effort": "high" if wave == 1 else "xhigh"}}})
    return result


def submit(client, out, wave, cap, confirm):
    if not confirm:
        raise ValueError("Confirm remaining promotional eligibility, dedicated account/workspace spend limit, and price first")
    if cap <= 0 or cap > 10_000_000:
        raise ValueError("pilot cap must be >0 and <= $10")
    folder = out / f"wave{wave}"
    folder.mkdir(parents=True, exist_ok=True)
    # O_EXCL serializes submission and leaves an ambiguity fence on a crash.
    lock = folder / "submit.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        if (folder / "reservation.json").exists():
            raise ValueError("submission already reserved; reconcile manually if batch ID missing")
        budget = out / "budget.json"
        write(budget, {"cap_micro_usd": cap, "model": MODEL, "rate_source": RATES_URL,
                       "input_micro_per_token": INPUT_RATE, "output_micro_per_token": OUTPUT_RATE,
                       "custody": "research-only"})
        requests = make_requests(out, wave)
        if not requests:
            raise ValueError("no valid requests")
        estimates = {}
        for request in requests:
            p = request["params"]
            count = client.messages.count_tokens(model=p["model"], system=p["system"], messages=p["messages"])
            estimates[request["custom_id"]] = reserved_micro(count.input_tokens, p["max_tokens"])
        prior = 0
        if wave == 2:
            receipt = read(out / "wave1" / "receipt.json")
            if receipt["budget_anomaly"]:
                raise ValueError("first-wave usage exceeded estimate or had unexpected pricing; reconcile")
            prior = receipt["charged_or_reserved_micro_usd"]
        reserve = sum(estimates.values())
        if prior + reserve > cap * 95 // 100:
            raise ValueError(f"planned charge {prior + reserve} micro-USD exceeds 95% of pilot cap")
        write(folder / "requests.json", requests)
        write(folder / "reservation.json", {"requests_sha256": digest(requests),
              "per_root_micro_usd": estimates, "total_micro_usd": reserve,
              "prior_micro_usd": prior})
        # SDK retries disabled at client creation. A failed create call is ambiguous.
        batch = client.messages.batches.create(requests=requests)
        write(folder / "batch.json", {"id": batch.id})
        print(batch.id)
    finally:
        lock.unlink()


def collect(client, out, wave):
    folder = out / f"wave{wave}"
    batch = read(folder / "batch.json")
    reservation = read(folder / "reservation.json")
    requests = read(folder / "requests.json")
    if digest(requests) != reservation["requests_sha256"]:
        raise ValueError("request file changed")
    estimates = reservation["per_root_micro_usd"]
    status = client.messages.batches.retrieve(batch["id"])
    if status.processing_status != "ended":
        print(status.processing_status)
        return
    outcomes = {}
    anomaly = False
    for row in client.messages.batches.results(batch["id"]):
        rid = row.custom_id
        if rid not in estimates or rid in outcomes:
            raise ValueError("unexpected or duplicate provider result")
        result = row.result
        rd = folder / rid
        charge = estimates[rid]
        entry = {"type": result.type, "accounting": "reserve held"}
        if result.type == "succeeded":
            msg = result.message
            usage = msg.usage.model_dump()
            # No caching or server tools requested. Unexpected usage must be reconciled.
            unexpected = any(usage.get(k) for k in ("cache_creation_input_tokens", "cache_read_input_tokens", "server_tool_use"))
            known = int(usage["input_tokens"]) * INPUT_RATE + int(usage["output_tokens"]) * OUTPUT_RATE
            if unexpected or known > estimates[rid]:
                anomaly = True
                charge = max(known, estimates[rid])
            else:
                charge = known
                entry["accounting"] = "provider reported token usage; verify invoice"
            visible = "\n".join(b.text for b in msg.content if b.type == "text")
            write(rd / "response.json", {"visible_text": visible, "usage": usage,
                  "message_id": msg.id, "stop_reason": msg.stop_reason,
                  "custody": "research-only", "training_eligible": False})
            if msg.stop_reason == "end_turn":
                if wave == 1:
                    packet = read(out / "public" / (rid + ".json"))
                    try:
                        initial = parse_initial(visible)
                    except (ValueError, TypeError, ZeroDivisionError):
                        entry["status"] = "invalid initial response"
                    else:
                        write(rd / "initial.json", initial)
                        observation = observe(packet, initial)
                        write(rd / "observation.json", observation)
                        entry["status"] = "verified finite fixture"
                        entry["prediction_correct"] = observation["prediction_correct"]
                else:
                    write(rd / "research_print.json", {"markdown": visible,
                          "evidence": "candidate research print; independent review pending",
                          "training_eligible": False})
                    print_path = rd / "research_print.md"
                    if print_path.exists() and print_path.read_text(encoding="utf-8") != visible + "\n":
                        raise ValueError("immutable research print differs")
                    print_path.write_text(visible + "\n", encoding="utf-8")
                    entry["status"] = "candidate print"
            else:
                entry["status"] = "incomplete or refused"
        entry["charged_or_reserved_micro_usd"] = charge
        write(rd / "accounting.json", entry)
        outcomes[rid] = entry
    missing = set(estimates) - set(outcomes)
    charge = sum(x["charged_or_reserved_micro_usd"] for x in outcomes.values()) + sum(estimates[r] for r in missing)
    write(folder / "receipt.json", {"batch_id": batch["id"], "wave": wave,
          "outcomes": outcomes, "missing": sorted(missing), "budget_anomaly": anomaly or bool(missing),
          "charged_or_reserved_micro_usd": charge, "custody": "research-only"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["submit", "collect"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--wave", type=int, choices=[1,2], required=True)
    p.add_argument("--budget-usd", type=Decimal)
    p.add_argument("--confirm-credit-and-limit", action="store_true")
    args = p.parse_args()
    if args.command == "submit" and (args.budget_usd is None or not args.confirm_credit_and_limit):
        p.error("explicit pilot cap and credit/limit confirmation required")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        p.error("load ANTHROPIC_API_KEY locally; never include it in run artifacts")
    import anthropic
    client = anthropic.Anthropic(max_retries=0)
    if args.command == "submit":
        submit(client, args.out, args.wave, int(args.budget_usd * 1_000_000),
               args.confirm_credit_and_limit)
    else:
        collect(client, args.out, args.wave)


if __name__ == "__main__":
    main()
