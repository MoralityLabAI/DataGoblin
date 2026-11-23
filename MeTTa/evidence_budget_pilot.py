"""Compare public evidence-check policies with independently replayed MeTTa receipts."""
from __future__ import annotations

import argparse
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from functools import cache
from importlib.metadata import version
from math import prod
from pathlib import Path

import numpy as np

from .evidence_budget import (
    BUDGET,
    CURRENT,
    ROUTES,
    UNCHECKED,
    WIDTH,
    Episode,
    PublicObservation,
    bayes_check,
    check_run,
    greedy_check,
    make_episode,
)
from .evidence_proofs import storyworld_rules
from .ram_hillclimb import canonical, digest, install_memory_ceiling, peak_rss_mib
from .ram_hillclimb_native import PACKAGE_VERSION
from .ram_trajectory_policy import ActionPolicy

GATE_COMMIT = "1ed89dee0363b6964739cc6f14f0078b7de56341"
GATE_PATH = "metta-storyworld/engine/evidence_receipt_gate.metta"
GATE_SHA256 = "1c6f48179259298f1920c501dd46e4f8a9844cbe93b7031fa665cf1fed74dc8d"
OPTIMIZER_STEPS = 180
TRAIN_ROOTS_PER_FAMILY = 72
EVAL_ROOTS_PER_FAMILY = 24
SEEDS = (1101, 1102)
TRAIN_FAMILIES = ("fork", "chain", "bottleneck")


def features(public: PublicObservation) -> tuple[np.ndarray, np.ndarray]:
    catalog, state = public.catalog, public.state
    viable = catalog.available(state, public.remaining)
    trials = [(tuple(i for i in route if state[i] == UNCHECKED), len(route)) for route in viable]
    probabilities = [prod(catalog.priors[i] / 100 for i in pending) for pending, _ in trials]
    rows = []
    for slot, prior in enumerate(catalog.priors):
        relevant = [(pending, size, p) for (pending, size), p in zip(trials, probabilities, strict=True)
                    if slot in pending]
        scores = sorted((p for _, _, p in relevant), reverse=True)
        rows.append((
            1.0, prior / 100, public.remaining / BUDGET, len(relevant) / len(viable),
            max(scores, default=0.0), sum(scores) / len(viable), [*scores, 0.0, 0.0][1],
            min((len(pending) / BUDGET for pending, _, _ in relevant), default=1.0),
            max(((size - len(pending)) / BUDGET for pending, size, _ in relevant), default=0.0),
            sum(len(pending) == 1 for pending, _, _ in relevant) / len(viable),
            max(probabilities), sum(s == CURRENT for s in state) / BUDGET,
        ))
    return np.asarray(rows), np.asarray([slot in public.legal for slot in range(WIDTH)])


def train_rows(episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs, targets, masks = [], [], []
    for episode in episodes:
        teacher = bayes_check(episode.catalog)

        def collect(public: PublicObservation, teacher=teacher) -> int:
            action = teacher(public)
            x, legal = features(public)
            inputs.append(x)
            targets.append(action)
            masks.append(legal)
            return action

        check_run(episode, collect)
    return np.stack(inputs), np.asarray(targets), np.stack(masks)


def route_rules(routes: tuple[tuple[int, ...], ...]) -> tuple[list[str], list[str]]:
    facts, trees = [], []
    for number, route in enumerate(routes):
        rule = f"Route{number}"
        left, right = (f"(Evidence Note{i} Source{i})" for i in route[:2])
        a, b = (f"Claim{i}" for i in route[:2])
        if len(route) == 3:
            c = route[2]
            facts.append(f"(SupportsBoth First{number} {a} {b} Partial{number})")
            left, right = f"(Because First{number} {left} {right})", f"(Evidence Note{c} Source{c})"
            a, b = f"Partial{number}", f"Claim{c}"
        facts.append(f"(SupportsBoth {rule} {a} {b} Review)")
        trees.append(f"(Because {rule} {left} {right})")
    return facts, trees


def verified_proofs(
    rules: str, episode: Episode, state: tuple[int, ...],
    *, native_snapshot: Callable[[list[str], tuple[str, ...]], tuple[list, ...]] | None = None,
) -> tuple[set[str], str]:
    facts = []
    for i, current in enumerate(episode.current):
        if current:
            facts.extend((f"(Witness Reviewer Claim{i} Note{i})", f"(RecordState Note{i} current)",
                          f"(RecordRoot Note{i} Source{i})"))
        if state[i] == CURRENT:
            facts.append(f"(CheckedRecord Reviewer Epoch1 Note{i} Source{i})")
    route_facts, trees = route_rules(episode.catalog.routes)
    facts.extend(route_facts)
    expected = {tree for route, tree in zip(episode.catalog.routes, trees, strict=True)
                if all(state[i] == CURRENT for i in route)}
    query = "(SupportWithReceipts Reviewer Epoch1 Review (Succ (Succ Zero)))"
    source = "\n".join((rules, *facts, f"!{query}"))
    if native_snapshot is None:
        from hyperon import Environment, MeTTa

        runtime = MeTTa(env_builder=Environment.custom_env(create_config=False, is_test=True))
        (atoms,) = runtime.run(source)
    else:
        (atoms,) = native_snapshot(facts, (query,))
    actual = {str(atom) for atom in atoms}
    if expected != actual:
        raise ValueError("Native receipt gate differs from the public route verifier")
    return actual, digest(source.encode())


def gate_rules(checkout: Path) -> str:
    if version("hyperon") != PACKAGE_VERSION:
        raise ValueError(f"Use hyperon=={PACKAGE_VERSION}")
    gate = subprocess.check_output(["git", "-C", str(checkout), "show", f"{GATE_COMMIT}:{GATE_PATH}"])
    if digest(gate) != GATE_SHA256:
        raise ValueError("The native receipt rule must match its source pin")
    return storyworld_rules(checkout) + "\n" + gate.decode("utf-8")


def run(checkout: Path, out: Path) -> dict:
    limits = install_memory_ceiling(8192)
    started = time.perf_counter()
    rules = gate_rules(checkout)

    train = [make_episode(family, "teacher", seed) for family in TRAIN_FAMILIES
             for seed in range(TRAIN_ROOTS_PER_FAMILY)]
    x, y, masks = train_rows(train)
    fitted = []
    for seed in SEEDS:
        for recurrent, depth in ((False, 1), (True, 3)):
            model = ActionPolicy(seed, x.shape[-1], recurrent=recurrent)
            initial = model.digest()
            losses = model.fit(x, y, masks, steps=OPTIMIZER_STEPS, lr=0.15, depth=depth)
            name = "tied_recurrent" if recurrent else "direct"
            fitted.append((name, seed, model, depth, {
                "arm": name, "seed": seed, "depth": depth, "parameters": sum(p.size for p in model.parameters),
                "initial_sha256": initial, "checkpoint_sha256": model.digest(), "first_loss": losses[0],
                "final_loss": losses[-1], "optimizer_updates": len(losses), "supervised_examples": len(y),
                "dense_mac_proxy_per_update": model.work_per_update(x, depth),
            }))

    @cache
    def audit(episode: Episode, state: tuple[int, ...]) -> tuple[set[str], str]:
        return verified_proofs(rules, episode, state)

    counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    receipts = []
    used_roots = {episode.catalog.source_root for episode in train}
    if len(used_roots) != len(train):
        raise ValueError("Duplicate public training catalog")
    for family in ROUTES:
        for ordinal in range(EVAL_ROOTS_PER_FAMILY):
            episode = make_episode(family, "fresh_pilot", ordinal)
            if episode.catalog.source_root in used_roots:
                raise ValueError("Public catalog overlaps with previous TRAIN or scoring roots")
            used_roots.add(episode.catalog.source_root)
            policies = [("index_order", None, lambda public: public.legal[0], 0),
                        ("route_greedy", None, greedy_check, 0),
                        ("bayes_public", None, bayes_check(episode.catalog), 0)]
            for arm, seed, model, depth, _ in fitted:
                def propose(public, model=model, depth=depth):
                    feats, legal = features(public)
                    return int(np.argmax(np.where(legal, list(model.states(feats, depth))[-1] @ model.wo, -np.inf)))
                policies.append((arm, seed, propose, depth))
            for arm, seed, policy, depth in policies:
                row = check_run(episode, policy)
                keys, receipt_hash = audit(episode, row["final_state"])
                if bool(keys) != row["completed"]:
                    raise ValueError("Host completion differs from independent native acceptance")
                tally = counts[family][arm]
                tally.update(trials=1, completed=int(row["completed"]), checks=row["checks"],
                             inference_passes=depth * row["checks"])
                receipts.append({**row, "family": family, "arm": arm, "model_seed": seed,
                                 "native_input_sha256": receipt_hash, "native_accepted_proofs": sorted(keys)})

    payload = "".join(canonical(row) + "\n" for row in receipts).encode()
    out.mkdir(parents=True, exist_ok=False)
    (out / "check_receipts.jsonl").write_bytes(payload)
    report = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "training_roots": sorted(episode.catalog.source_root for episode in train),
        "fits": [info for *_, info in fitted], "families": counts, "check_budget": BUDGET,
        "eval_roots_per_family": EVAL_ROOTS_PER_FAMILY, "transfer_family": "overlap_transfer",
        "native_gate_commit": GATE_COMMIT, "native_rule_sha256": digest(rules.encode()),
        "hyperon": PACKAGE_VERSION, "numpy": np.__version__,
        "native_snapshot_checks": audit.cache_info().misses, "receipts_sha256": digest(payload),
        "memory": limits, "peak_mib": peak_rss_mib(), "wall_seconds": time.perf_counter() - started,
        "evidence_class": "synthetic_development_pilot",
    }
    (out / "budget_report.json").write_text(canonical(report) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storyworld", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.storyworld, args.out)
    keys = ("families", "native_snapshot_checks", "receipts_sha256", "peak_mib")
    print(canonical({key: report[key] for key in keys}))


if __name__ == "__main__":
    main()
