"""AGN-03 development-only bridge to an operator-pinned SW-PROV-05 checkout."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from fractions import Fraction as F
from pathlib import Path

FILES = ("provenance_access.py", "provenance_characters.py", "provenance_session.py", "provenance_design_audit.py")
FAMILY = "north_quay_provenance_access"
PUBLIC = {"schema", "title", "remaining", "turn", "terminal", "text", "ledger", "actions", "swmd_card"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def source_digest(source):
    hashes = {p: hashlib.sha256((source / p).read_bytes()).hexdigest() for p in FILES}
    return digest(hashes), hashes


def check_public(packet):
    if set(packet) != PUBLIC or packet["schema"] != "SW-PROV/session/v1":
        raise ValueError("unexpected public packet")
    if any(set(row) != {"query", "outcome", "text"} for row in packet["ledger"]):
        raise ValueError("unexpected ledger field")
    if any(set(row) != {"id", "label", "cost"} for row in packet["actions"]):
        raise ValueError("unexpected action field")
    if packet["terminal"] and packet["actions"]:
        raise ValueError("terminal exposes actions")
    return packet


def rollout(api, index, budget, mode, sign):
    # Replay/grade in host. Reconstruct all oracle state from visible facts only.
    session = api.AccessSession(budget, index, mode)
    public = check_public(session.public())
    # Opening text is part of the pinned fixture contract. The supplied sign
    # is checked against the public text before it enters the oracle.
    if not public["text"].startswith(f"The harbour report {api.SIGNS[sign]} Captain Vale."):
        raise ValueError("opening report mismatch")
    state = api.root(sign, budget)
    steps = []
    while not public["terminal"]:
        if state.budget != public["remaining"]:
            raise ValueError("budget mismatch")
        legal = {r["id"] for r in public["actions"]}
        if legal != set(api.legal(state, mode)):
            raise ValueError("legal action mismatch")
        values = api.values(state, state.budget, mode)
        if set(values) != legal:
            raise ValueError("oracle action mismatch")
        best = max(values.values())
        optimum = sorted(a for a, value in values.items() if value == best)
        action = api.choose(state, "full", mode)
        if action not in optimum:
            raise ValueError("suboptimal oracle action")
        after = check_public(session.act(action))
        steps.append({"observation": public, "action": action, "optimal_actions": optimum, "next_observation": after})
        if action in api.COST:
            last = after["ledger"][-1]
            if last["query"] != action or after["remaining"] != state.budget - api.COST[action]:
                raise ValueError("inconsistent acquisition")
            y = int(last["outcome"]) if action in api.FIELDS else last["outcome"]
            state = api.step(state, action, y, mode)
        public = after
    return {"mode": mode, "steps": steps, "grade": session.grade()}


def make_views(rows, revision):
    # All branches share a single exposed family. Four candidate presentations
    # across every lane combined; no confirmed or train-ready export.
    base = {"family_id": FAMILY, "root_id": FAMILY, "split": "dev", "training_ready": False,
            "source_digest": revision, "rights": "review_required"}
    samples = [step for row in rows for step in row["social"]["steps"]]
    # Pick a real gated action, and a different factual acquisition.
    gate = next(x for x in samples if x["action"] == "apply")
    evidence = next(x for x in samples if x["action"] == "registry")
    return [
        {**base, "lane": "action", "evidence_class": "finite_replay_checked",
         "input": gate["observation"], "target": {"optimal_actions": gate["optimal_actions"]}},
        {**base, "lane": "dynamics", "evidence_class": "finite_replay_checked",
         "input": {"observation": evidence["observation"], "action": evidence["action"]},
         "target": evidence["next_observation"]},
        {**base, "lane": "induction", "evidence_class": "proposed",
         "input": {"question": "When can documented provenance enable access? State the assumptions and seek a counterexample."},
         "target": {"hypothesis": "If briefing sets trust to 1/4 and grant requires trust > 1/8, briefing enables grant.",
                    "scope": "permission only; acquisition cost, available budget and application history remain relevant"}},
        {**base, "lane": "lean", "evidence_class": "proposed", "kernel_checked": False,
         "input": {"statement": "Given trust = 1/4 and grant iff trust > 1/8, prove permission; rationals."},
         "target": {"lean_draft": "import Mathlib\nexample (trust : ℚ) (h : trust = 1/4) : trust > 1/8 := by\n  rw [h]\n  norm_num\n",
                    "status": "unverified_draft"}},
    ]


def run(api, revision, budget=5):
    if budget != 5:
        raise ValueError("only the frozen five-unit development fixture is supported")
    rows = []
    total = {mode: F(0) for mode in ("social", "blocked")}
    histories = {}
    for i, (weight, latent) in enumerate(api.LAW):
        item = {"weight": str(weight), "latent_assignment_index": i}
        for mode in total:
            result = rollout(api, i, budget, mode, latent["A"])
            item[mode] = result
            total[mode] += weight * F(result["grade"]["utility"])
            for step in result["steps"]:
                key = (mode, digest(step["observation"]))
                previous = histories.setdefault(key, step["optimal_actions"])
                if previous != step["optimal_actions"]:
                    raise ValueError("identical public packets have inconsistent action labels")
        rows.append(item)
    if total != {"social": F(25, 36), "blocked": F(11, 24)}:
        raise ValueError("pinned fixture scores changed")
    views = make_views(rows, revision)
    summary = {"schema": "AGN-03/dev-v1", "family_id": FAMILY, "split": "dev", "independent_families": 1,
               "latent_assignments": len(rows), "paired_rollouts": 2 * len(rows), "decision_rows": sum(
                   len(r[m]["steps"]) for r in rows for m in total), "unique_mode_public_packets": len(histories),
               "utility": {k: str(v) for k, v in total.items()}, "utility_gain": str(total["social"] - total["blocked"]),
               "candidate_views": len(views), "training_ready_views": 0, "kernel_checked": False,
               "source_digest": revision, "evidence_class": "finite_synthetic_replay", "confirmation": False}
    return rows, views, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--print-source-digest", action="store_true")
    p.add_argument("--expected-source-digest")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    revision, hashes = source_digest(args.source)
    if args.print_source_digest:
        print(revision)
        return
    if revision != args.expected_source_digest or args.out is None:
        p.error("verified source digest and fresh output directory are required")
    if args.out.exists():
        p.error("refusing to overwrite archive")
    # Only import a reviewed, pinned host implementation. Never execute proposals.
    sys.path.insert(0, str(args.source.resolve()))
    api = importlib.import_module("provenance_access")
    rows, views, summary = run(api, revision)
    args.out.mkdir(parents=True)
    for name, records in (("private_replays.jsonl", rows), ("development_candidates.jsonl", views)):
        (args.out / name).write_text("".join(canonical(x) + "\n" for x in records), encoding="utf-8")
    manifest = {**summary, "sources": hashes, "artifacts": {n: hashlib.sha256((args.out / n).read_bytes()).hexdigest()
                 for n in ("private_replays.jsonl", "development_candidates.jsonl")}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
