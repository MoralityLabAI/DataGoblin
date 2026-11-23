"""Cross training-time and inference-time incidence bias on fresh evidence catalogs."""
from __future__ import annotations

import argparse
import inspect
import json
import random
import subprocess
import time
from collections import Counter, defaultdict
from functools import cache
from itertools import product
from pathlib import Path

import numpy as np
import torch
from research_gym.neural.obligation_pointer import MODES, ObligationPointer

from metta_research_bench.looped.model import BATCH_SIZE

from .evidence_budget import ROUTES, WIDTH, Catalog, Episode, PublicObservation, bayes_check, check_run
from .evidence_budget_confirmation import primary_intervals, relabel, topology
from .evidence_budget_pilot import gate_rules, verified_proofs
from .evidence_incidence_pilot import EVAL as PRIOR_INCIDENCE
from .evidence_incidence_pilot import PRIOR_EVAL, SLOTS, TRAIN_FAMILIES, fit_pairs, graph_batch
from .evidence_obligations import ScopedNative, obligation_reader, planning_rules
from .ram_hillclimb import canonical, digest, install_memory_ceiling, peak_rss_mib

EVAL = {
    "square_tail": ((0, 1), (1, 2), (2, 3), (0, 3), (3, 4), (4, 5), (5, 6)),
    "shared_triangle": ((0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (3, 4), (4, 5), (4, 6)),
    "pair_ladder": ((0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5), (5, 6)),
    "paired_gate": ((0, 1), (1, 2), (0, 3, 4), (2, 4, 5), (1, 5, 6)),
    "hub_bridge": ((0, 1), (0, 2), (0, 3, 4), (0, 5, 6), (1, 4, 6)),
    "four_triad_cycle": ((0, 1, 2), (0, 3, 4), (2, 4, 5), (1, 5, 6)),
    "overlap_mesh": ((0, 2, 3), (0, 4, 5), (1, 2, 5), (1, 3, 6), (2, 4, 6)),
    "braided_tail": ((0, 1), (2, 3), (0, 2, 4), (1, 3, 5), (0, 5, 6), (3, 4, 6)),
}
ARMS = {
    "neutral": ("dense", "dense"),
    "inference_only": ("dense", "incidence"),
    "training_only": ("incidence", "dense"),
    "both": ("incidence", "incidence"),
    "inference_sham": ("dense", "rewired"),
}
CONTRASTS = {
    "both_minus_neutral": {"both": 1, "neutral": -1},
    "inference_without_training_bias": {"inference_only": 1, "neutral": -1},
    "training_without_inference_bias": {"training_only": 1, "neutral": -1},
    "inference_with_training_bias": {"both": 1, "training_only": -1},
    "training_with_inference_bias": {"both": 1, "inference_only": -1},
    "inference_minus_sham": {"inference_only": 1, "inference_sham": -1},
    "training_inference_interaction": {"both": 1, "training_only": -1, "inference_only": -1, "neutral": 1},
}
PROTOCOL = {
    "schema": "metta-incidence-training-inference-v1",
    "train_roots_per_family": 40, "eval_roots_per_family": 24, "budgets": (3, 4),
    "seeds": (1101, 1102, 1103), "updates": 180, "train_modes": ("dense", "incidence"),
    "arms": ARMS, "contrasts": CONTRASTS, "primary": "both_minus_neutral",
    "minimum_meaningful_completion_effect": 0.02, "evidence_class": "development_pilot",
    "upstream_commit": "207d92ff62f72a818cb3530e064270258e211be4",
    "prior_protocol_sha256": "31542fb3e17333f98674d0991b95e66a147e41bb707f167a5435ed1c04330e04",
    "prior_receipts_sha256": "2ac36954d60b266bdbe16d0e33033549e10f725637c97050cf16ccd360e4eed4",
}


def catalog_draw(family: str, ordinal: int, *, training: bool) -> tuple[Catalog, random.Random]:
    rng = random.Random(f"{PROTOCOL['schema']}/{training}/{family}/{ordinal}")
    templates = TRAIN_FAMILIES if training else EVAL
    routes = relabel(templates[family], tuple(rng.sample(range(WIDTH), WIDTH)))
    return Catalog(routes, tuple(rng.randrange(18, 93) for _ in range(WIDTH))), rng


def open_episode(draw: tuple[Catalog, random.Random]) -> Episode:
    catalog, rng = draw
    return Episode(catalog, tuple(rng.randrange(100) < prior for prior in catalog.priors))


def preflight(archive: Path) -> tuple[dict, set[str]]:
    prior_lock = (archive / "locked_protocol.json").read_bytes()
    prior_receipts = (archive / "receipts.jsonl").read_bytes()
    if digest(prior_lock) != PROTOCOL["prior_protocol_sha256"] or digest(prior_receipts) != PROTOCOL[
        "prior_receipts_sha256"
    ]:
        raise ValueError("Prior incidence protocol and receipts must match the registered archive")
    old = json.loads(prior_lock)
    known = set(old["excluded_source_hashes"]) | set(old["train_source_hashes"])
    known.update(json.loads(row)["source_root_sha256"] for row in prior_receipts.splitlines())
    structures = {topology(routes) for routes in EVAL.values()}
    previous = (*ROUTES.values(), *TRAIN_FAMILIES.values(), *PRIOR_EVAL.values(), *PRIOR_INCIDENCE.values())
    if len(structures) != len(EVAL) or structures.intersection(topology(routes) for routes in previous):
        raise ValueError("Evaluation topologies must be new up to a slot permutation")
    upstream = Path(inspect.getfile(ObligationPointer)).resolve().parents[2]
    if (subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
            != PROTOCOL["upstream_commit"] or subprocess.check_output(
                ["git", "-C", str(upstream), "diff", "HEAD", "--", "research_gym", "lsa"]
            )):
        raise ValueError("Use the pinned LoopedTransformers checkout")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("Commit the experiment before opening the cohort")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return {"commit": commit, "protocol": PROTOCOL, "excluded_source_hashes": sorted(known),
            "train_topologies": TRAIN_FAMILIES, "eval_topologies": EVAL}, known


def root_contrasts(results: dict[tuple[str, int, int], dict]) -> dict[str, list[float]]:
    """Use one paired unit per root, averaging budgets and seeds before resampling."""
    return {name: np.mean([
        [-sum(coef * results[arm, budget, seed]["checks"] for arm, coef in weights.items()),
         sum(coef * results[arm, budget, seed]["completed"] for arm, coef in weights.items())]
        for budget, seed in product(PROTOCOL["budgets"], PROTOCOL["seeds"])
    ], axis=0).tolist() for name, weights in CONTRASTS.items()}


def run(checkout: Path, prior: Path, out: Path) -> dict:
    limits = install_memory_ceiling(8192)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    lock, known = preflight(prior)
    public_rules, acceptance_rules = planning_rules(checkout), gate_rules(checkout)
    planner, verifier = ScopedNative(public_rules), ScopedNative(acceptance_rules)
    project = obligation_reader(planner)

    @cache
    def encode(public: PublicObservation) -> tuple:
        full, _ = project(public)
        return graph_batch(public, full)

    train_draws = [catalog_draw(family, ordinal, training=True) for family, ordinal in
                   product(TRAIN_FAMILIES, range(PROTOCOL["train_roots_per_family"]))]
    evaluation = [(family, catalog_draw(family, ordinal, training=False)) for family, ordinal in
                  product(EVAL, range(PROTOCOL["eval_roots_per_family"]))]
    train_roots = {catalog.source_root for catalog, _ in train_draws}
    eval_roots = {catalog.source_root for _, (catalog, _) in evaluation}
    if (len(train_roots) != len(train_draws) or len(eval_roots) != len(evaluation)
            or (train_roots | eval_roots) & known or train_roots & eval_roots):
        raise ValueError("Fresh source-root allocation overlaps another cohort")
    lock.update(train_source_hashes=sorted(train_roots), eval_source_hashes=sorted(eval_roots),
                torch=str(torch.__version__), slots=SLOTS, batch_size=BATCH_SIZE,
                obligation_rules_sha256=digest(public_rules.encode()),
                receipt_rules_sha256=digest(acceptance_rules.encode()))
    out.mkdir(parents=True, exist_ok=False)
    protocol_file = out / "locked_protocol.json"
    protocol_file.write_bytes((canonical(lock) + "\n").encode())
    fitted = fit_pairs([open_episode(draw) for draw in train_draws], encode,
                       modes=PROTOCOL["train_modes"], protocol=PROTOCOL)
    fit_file = out / "locked_fits.json"
    fit_file.write_bytes((canonical([{**meta, "train_mode": mode} for mode, _, _, meta in fitted]) + "\n").encode())
    models = {(mode, seed): model for mode, seed, model, _ in fitted}
    visits = {model.core.rounds for model in models.values()}
    (visits_per_check,) = visits

    def policy(train_mode: str, inference_mode: str, seed: int):
        model = models[train_mode, seed]
        view = MODES.index(inference_mode)

        @torch.inference_mode()
        def choose(public: PublicObservation) -> int:
            packs, allowed, _ = encode(public)
            return int(model(packs[view][None], allowed[None])[0].argmax().item())

        return choose

    learned = [(arm, seed, policy(*modes, seed)) for arm, modes in ARMS.items() for seed in PROTOCOL["seeds"]]

    @cache
    def accept(episode: Episode, state: tuple[int, ...]) -> tuple[set[str], str]:
        return verified_proofs(acceptance_rules, episode, state, native_snapshot=verifier.evaluate)

    outcomes: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    paired: dict[str, dict[str, list[list[float]]]] = {name: defaultdict(list) for name in CONTRASTS}
    receipts = []
    for family, draw in evaluation:
        episode = open_episode(draw)
        exact = bayes_check(episode.catalog, minimize_work=True)
        results = {}
        for budget in PROTOCOL["budgets"]:
            for arm, seed, choose in [*learned, ("exact_work_tie", None, exact)]:
                row = check_run(episode, choose, budget)
                proofs, native_hash = accept(episode, row["final_state"])
                if bool(proofs) != row["completed"]:
                    raise ValueError("Native proof acceptance disagrees with the host receipt")
                outcomes[family][arm].update(cases=1, completed=int(row["completed"]), checks=row["checks"],
                                             unsuccessful_checks=row["checks"] * (not row["completed"]))
                receipts.append({**row, "family": family, "budget": budget, "arm": arm, "model_seed": seed,
                                 "recurrent_visits": visits_per_check * row["checks"] if seed is not None else 0,
                                 "native_input_sha256": native_hash, "native_accepted_proofs": sorted(proofs)})
                results[arm, budget, seed] = row
        for name, difference in root_contrasts(results).items():
            paired[name][family].append(difference)

    payload = "".join(canonical(row) + "\n" for row in receipts).encode()
    (out / "receipts.jsonl").write_bytes(payload)
    report = {
        "protocol_sha256": digest(protocol_file.read_bytes()), "fits_sha256": digest(fit_file.read_bytes()),
        "outcomes": outcomes, "paired": {name: primary_intervals(rows) for name, rows in paired.items()},
        "native_obligation_calls": planner.calls, "native_acceptance_calls": verifier.calls,
        "receipts_sha256": digest(payload), "peak_mib": peak_rss_mib(), "memory": limits,
        "wall_seconds": time.perf_counter() - started, "training_ready": False,
    }
    (out / "stages_report.json").write_bytes((canonical(report) + "\n").encode())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("storyworld", "prior", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    print(canonical(run(args.storyworld, args.prior, args.out)))


if __name__ == "__main__":
    main()
