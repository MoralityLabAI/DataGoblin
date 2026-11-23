"""Matched CPU Looped Transformers for full and minimal public MeTTa obligations."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from collections import Counter, defaultdict
from functools import cache
from itertools import product
from math import prod
from pathlib import Path

import numpy as np
import torch

from metta_research_bench.looped.model import (
    BATCH_SIZE,
    BUCKETS,
    FIELDS,
    MAX_ITEMS,
    LoopedPointer,
    fit,
    upstream_revision,
)

from .evidence_budget import BUDGET, CURRENT, ROUTES, Catalog, Episode, PublicObservation, bayes_check, check_run
from .evidence_budget_confirmation import FAMILIES, primary_intervals, relabel, topology
from .evidence_budget_pilot import gate_rules, verified_proofs
from .evidence_obligations import ScopedNative, obligation_reader, planning_rules
from .ram_hillclimb import canonical, digest, install_memory_ceiling, peak_rss_mib

TRAIN_FAMILIES = {
    **{name: ROUTES[name] for name in ("fork", "chain", "bottleneck")},
    "side_ring": ((0, 1), (1, 2), (2, 3), (0, 3), (3, 4, 5), (4, 6)),
    "late_bridge": ((0, 1), (0, 2, 3), (1, 3, 4), (2, 5), (4, 5, 6)),
    "shortcuts": ((0, 1), (1, 2), (2, 3), (3, 4, 5), (0, 4, 6), (1, 5, 6)),
}
PROTOCOL = {
    "schema": "metta-obligation-representation-pilot-v1",
    "train_roots_per_family": 48, "eval_roots_per_family": 18, "budgets": (3, 4),
    "seeds": (1101, 1102, 1103), "updates": 200,
    "prior_confirmation_catalogs_sha256": "0006cb77a7a1abbf3a1447f1013793f1e4353be77fee07fbaefeb051167d6032",
}
MODES = ("full_frontier", "minimal_frontier")


def sample(family: str, ordinal: int, *, training: bool) -> Episode:
    rng = random.Random(f"{PROTOCOL['schema']}/{training}/{family}/{ordinal}")
    templates = TRAIN_FAMILIES if training else FAMILIES
    width = 7
    routes = relabel(templates[family], tuple(rng.sample(range(width), width)))
    priors = tuple(rng.randrange(18, 93) for _ in range(width))
    return Episode(Catalog(routes, priors), tuple(rng.randrange(100) < p for p in priors))


def encode(public: PublicObservation, frontier: tuple[tuple[int, ...], ...]) -> tuple[torch.Tensor, torch.Tensor]:
    catalog, remaining = public.catalog, public.remaining
    scores = [prod(catalog.priors[i] / 100 for i in path) for path in frontier]
    rows = np.zeros((MAX_ITEMS + 1, FIELDS))
    for i, prior in enumerate(catalog.priors):
        paths = [(path, p) for path, p in zip(frontier, scores, strict=True) if i in path]
        local = sorted((p for _, p in paths), reverse=True)
        row = (
            1.0, prior / 100, remaining / BUDGET, len(paths) / len(catalog.routes),
            max(local, default=0.0), sum(local) / len(catalog.routes), [*local, 0.0, 0.0][1],
            min((len(path) / BUDGET for path, _ in paths), default=1.0),
            sum(len(path) == 1 for path, _ in paths) / len(catalog.routes), max(scores, default=0.0),
            len(frontier) / len(catalog.routes), sum(status == CURRENT for status in public.state) / BUDGET,
        )
        rows[i, :len(row)] = row
    allowed = np.zeros(MAX_ITEMS + 1, dtype=bool)
    allowed[list(public.legal)] = True
    codes = np.rint((rows + 2) * 8).astype(np.int64) + np.arange(FIELDS) * BUCKETS
    return torch.from_numpy(codes), torch.from_numpy(allowed)


def fit_pairs(train: list[Episode], project) -> list[tuple]:
    inputs: list[list[torch.Tensor]] = [[], []]
    masks, targets = [], []
    for ordinal, episode in enumerate(train):
        teacher = bayes_check(episode.catalog, minimize_work=True)

        def collect(public: PublicObservation, teacher=teacher) -> int:
            slot = teacher(public)
            for bucket, frontier in zip(inputs, project(public), strict=True):
                codes, mask = encode(public, frontier)
                bucket.append(codes)
            masks.append(mask)
            targets.append(slot)
            return slot

        check_run(episode, collect, PROTOCOL["budgets"][ordinal % len(PROTOCOL["budgets"])])
    mask_batch, target_batch = torch.stack(masks), torch.as_tensor(targets)
    batches = [torch.stack(bucket) for bucket in inputs]
    fitted = []
    for seed in PROTOCOL["seeds"]:
        torch.manual_seed(seed)
        initial = LoopedPointer()
        for index, mode in enumerate(MODES):
            model, metadata = fit(initial, (batches[index], mask_batch, target_batch), seed, PROTOCOL["updates"])
            fitted.append((mode, seed, model, metadata))
        matched = ("initial_sha256", "supervised_examples", "supervised_action_tokens",
                   "optimizer_updates", "parameters")
        for key in matched:
            if fitted[-2][3][key] != fitted[-1][3][key]:
                raise ValueError(f"Matched fit field changed across representations: {key}")
    return fitted


def preflight(prior: Path) -> tuple[dict, set[str]]:
    data = (prior / "public_catalogs.jsonl").read_bytes()
    report = json.loads((prior / "confirmation_report.json").read_bytes())
    lock_raw = (prior / "locked_protocol.json").read_bytes()
    if (digest(data) != PROTOCOL["prior_confirmation_catalogs_sha256"]
            or report["locked_protocol_sha256"] != digest(lock_raw)):
        raise ValueError("The prior topology archive must match the preregistered digests")
    train_structures = {topology(routes) for routes in TRAIN_FAMILIES.values()}
    if train_structures.intersection(topology(routes) for routes in FAMILIES.values()):
        raise ValueError("Training contains a held-out topology up to slot permutation")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("Commit the experiment before opening fresh roots")
    excluded = set(json.loads(lock_raw)["excluded_source_hashes"])
    excluded.update(json.loads(line)["source_root_sha256"] for line in data.splitlines())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return {"commit": commit, "protocol": PROTOCOL, "train_topologies": TRAIN_FAMILIES,
            "eval_topologies": FAMILIES, "excluded_source_hashes": sorted(excluded)}, excluded


@torch.inference_mode()
def select(model, public: PublicObservation, pending: tuple[tuple[int, ...], ...]) -> int:
    codes, allowed = encode(public, pending)
    return int(model(codes[None], allowed[None])[0].argmax().item())


def run(checkout: Path, prior: Path, out: Path) -> dict:
    limits = install_memory_ceiling(8192)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    upstream = upstream_revision()
    started = time.perf_counter()
    lock, known = preflight(prior)
    public_rules = planning_rules(checkout)
    acceptance_rules = gate_rules(checkout)
    planner, verifier = ScopedNative(public_rules), ScopedNative(acceptance_rules)
    project = obligation_reader(planner)
    seen_states: set[PublicObservation] = set()
    compressed_states = set()

    @cache
    def tracked(public: PublicObservation) -> tuple:
        full, minimal = project(public)
        seen_states.add(public)
        if full != minimal:
            compressed_states.add(public)
        return full, minimal

    train = [sample(family, ordinal, training=True)
             for family, ordinal in product(TRAIN_FAMILIES, range(PROTOCOL["train_roots_per_family"]))]
    train_roots = {episode.catalog.source_root for episode in train}
    if len(train_roots) != len(train) or train_roots & known:
        raise ValueError("Training catalog overlaps an earlier source")
    known.update(train_roots)
    out.mkdir(parents=True, exist_ok=False)
    protocol_file = out / "locked_protocol.json"
    lock.update(train_source_hashes=sorted(train_roots), upstream=upstream, torch=str(torch.__version__),
                obligation_rules_sha256=digest(public_rules.encode()),
                receipt_rules_sha256=digest(acceptance_rules.encode()), optimizer_batch_size=BATCH_SIZE)
    protocol_file.write_bytes((canonical(lock) + "\n").encode())
    models = fit_pairs(train, tracked)
    fit_file = out / "locked_fits.json"
    fit_file.write_bytes((canonical([meta for _, _, _, meta in models]) + "\n").encode())

    @cache
    def accept(episode: Episode, state: tuple[int, ...]) -> tuple[set[str], str]:
        return verified_proofs(acceptance_rules, episode, state, native_snapshot=verifier.evaluate)

    outcomes: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    paired: dict[str, list[list[float]]] = defaultdict(list)
    receipts = []
    for family, ordinal in product(FAMILIES, range(PROTOCOL["eval_roots_per_family"])):
        episode = sample(family, ordinal, training=False)
        root = episode.catalog.source_root
        if root in known:
            raise ValueError("Evaluation catalog overlaps a prior or training source")
        known.add(root)
        differences = []
        for budget in PROTOCOL["budgets"]:
            results = {}
            policies = [("exact_work_tie", None, bayes_check(episode.catalog, minimize_work=True))]
            for mode, seed, model, _ in models:
                mode_index = MODES.index(mode)

                def choose(public, model=model, index=mode_index):
                    return select(model, public, tracked(public)[index])
                policies.append((mode, seed, choose))
            for mode, seed, choose in policies:
                row = check_run(episode, choose, budget)
                proofs, native_hash = accept(episode, row["final_state"])
                if bool(proofs) != row["completed"]:
                    raise ValueError("Native acceptance disagrees with an evaluation receipt")
                outcomes[family][mode].update(cases=1, completed=int(row["completed"]), checks=row["checks"])
                receipts.append({**row, "family": family, "budget": budget, "arm": mode, "model_seed": seed,
                                 "native_input_sha256": native_hash, "native_accepted_proofs": sorted(proofs)})
                results[mode, seed] = row
            for seed in PROTOCOL["seeds"]:
                full, minimal = (results[mode, seed] for mode in MODES)
                differences.append((full["checks"] - minimal["checks"], minimal["completed"] - full["completed"]))
        paired[family].append(np.mean(differences, axis=0).tolist())

    payload = "".join(canonical(row) + "\n" for row in receipts).encode()
    (out / "receipts.jsonl").write_bytes(payload)
    report = {"protocol_sha256": digest(protocol_file.read_bytes()), "fits_sha256": digest(fit_file.read_bytes()),
              "outcomes": outcomes, "paired_minimal_minus_full": primary_intervals(paired),
              "projected_public_states": len(seen_states), "compressed_public_states": len(compressed_states),
              "native_obligation_calls": planner.calls, "native_acceptance_calls": verifier.calls,
              "receipts_sha256": digest(payload), "peak_mib": peak_rss_mib(), "memory": limits,
              "wall_seconds": time.perf_counter() - started, "training_ready": False}
    (out / "obligation_report.json").write_bytes((canonical(report) + "\n").encode())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("storyworld", "prior", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    print(canonical(run(args.storyworld, args.prior, args.out)))


if __name__ == "__main__":
    main()
