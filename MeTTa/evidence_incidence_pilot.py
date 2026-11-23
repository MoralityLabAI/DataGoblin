"""Compare graph-aligned and control attention in CPU Looped Transformers."""
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
from research_gym.neural.obligation_pointer import MODES, ObligationPointer, pack_graph

from metta_research_bench.looped.model import BATCH_SIZE, BUCKETS, FIELDS, fit

from .evidence_budget import BUDGET, WIDTH, Catalog, Episode, PublicObservation, bayes_check, check_run
from .evidence_budget_confirmation import FAMILIES as PRIOR_EVAL
from .evidence_budget_confirmation import primary_intervals, relabel, topology
from .evidence_budget_pilot import gate_rules, verified_proofs
from .evidence_obligation_pilot import TRAIN_FAMILIES
from .evidence_obligations import ScopedNative, obligation_reader, planning_rules
from .ram_hillclimb import canonical, digest, install_memory_ceiling, peak_rss_mib

EVAL = {
    "double_fan": ((0, 1), (0, 2), (0, 3), (3, 4), (3, 5), (3, 6)),
    "matched_chords": ((0, 1), (2, 3), (4, 5), (5, 6), (0, 6), (1, 4)),
    "necklace": ((0, 1), (0, 2, 3), (1, 4, 5), (2, 4, 6), (3, 4, 6)),
    "double_bridge": ((0, 1), (2, 3), (1, 2, 4), (0, 3, 5), (4, 5, 6)),
    "branch_chain": ((0, 1), (1, 2), (2, 3), (3, 4, 5), (1, 5, 6)),
    "triple_closure": ((0, 1), (0, 2, 3), (0, 3, 4), (1, 4, 5), (1, 5, 6)),
    "layered_spokes": ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5, 6), (3, 5, 6)),
    "triangle_chain": ((0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4), (4, 5), (5, 6), (4, 6)),
}
ROUTE_SLOTS = 9
SLOTS = WIDTH + ROUTE_SLOTS
PROTOCOL = {
    "schema": "metta-incidence-attention-pilot-v1",
    "train_roots_per_family": 40, "eval_roots_per_family": 24, "budgets": (3, 4),
    "seeds": (1101, 1102, 1103), "updates": 180,
    "minimum_meaningful_completion_effect": 0.02, "evidence_class": "development_pilot",
    "upstream_commit": "207d92ff62f72a818cb3530e064270258e211be4",
    "prior_receipts_sha256": "4b6cc691e67b6993a37666584e044e9942d3e65a05fa180eb97ddf84ba84a8a0",
}


def sample(family: str, ordinal: int, *, training: bool) -> Episode:
    rng = random.Random(f"{PROTOCOL['schema']}/{training}/{family}/{ordinal}")
    templates = TRAIN_FAMILIES if training else EVAL
    routes = relabel(templates[family], tuple(rng.sample(range(WIDTH), WIDTH)))
    priors = tuple(rng.randrange(18, 93) for _ in range(WIDTH))
    return Episode(Catalog(routes, priors), tuple(rng.randrange(100) < p for p in priors))


def graph_batch(public: PublicObservation, pending: tuple[tuple[int, ...], ...]) -> tuple[tuple, torch.Tensor, int]:
    if len(pending) > ROUTE_SLOTS:
        raise ValueError("The public frontier exceeds the registered route slots")
    incidence = torch.zeros((ROUTE_SLOTS, WIDTH), dtype=torch.bool)
    for i, route in enumerate(pending):
        incidence[i, list(route)] = True
    priors = public.catalog.priors
    rows = np.zeros((SLOTS, FIELDS))
    for slot, (prior, status) in enumerate(zip(priors, public.state, strict=True)):
        rows[slot, :4] = (1, prior / 100, status, public.remaining / BUDGET)
        rows[slot, 4:4 + ROUTE_SLOTS] = incidence[:, slot]
    for index, route in enumerate(pending):
        rows[WIDTH + index, :4] = (2, len(route) / BUDGET, 0, public.remaining / BUDGET)
        rows[WIDTH + index, 4:4 + WIDTH] = incidence[index]
    codes = torch.from_numpy(np.rint((rows + 2) * 8).astype(np.int64) + np.arange(FIELDS) * BUCKETS)
    present = torch.arange(SLOTS) < WIDTH + len(pending)
    allowed = torch.zeros(SLOTS, dtype=torch.bool)
    allowed[list(public.legal)] = True
    packs = tuple(pack_graph(codes, present, incidence, mode) for mode in MODES)
    changed = int((packs[1][:, FIELDS + 1:] != packs[2][:, FIELDS + 1:]).sum().item())
    return packs, allowed, changed


def preflight(prior: Path) -> tuple[dict, set[str]]:
    archive_lock = (prior / "locked_protocol.json").read_bytes()
    archived = (prior / "receipts.jsonl").read_bytes()
    report = json.loads((prior / "obligation_report.json").read_bytes())
    if digest(archived) != PROTOCOL["prior_receipts_sha256"] or digest(archive_lock) != report["protocol_sha256"]:
        raise ValueError("The archived obligation study must match the registered digests")
    old = json.loads(archive_lock)
    known = set(old["excluded_source_hashes"]) | set(old["train_source_hashes"])
    known.update(json.loads(line)["source_root_sha256"] for line in archived.splitlines())
    structures = {topology(templates) for templates in EVAL.values()}
    previous_structures = {topology(templates) for templates in (*TRAIN_FAMILIES.values(), *PRIOR_EVAL.values())}
    if len(structures) != len(EVAL) or structures.intersection(previous_structures):
        raise ValueError("Evaluation topologies must be new up to a slot permutation")
    upstream = Path(inspect.getfile(ObligationPointer)).resolve().parents[2]
    actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != PROTOCOL["upstream_commit"] or subprocess.check_output(
        ["git", "-C", str(upstream), "diff", "HEAD", "--", "research_gym", "lsa"]
    ):
        raise ValueError("Use the registered LoopedTransformers checkout without edits")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("Commit the experiment before opening fresh roots")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return {"commit": commit, "protocol": PROTOCOL, "excluded_source_hashes": sorted(known),
            "train_topologies": TRAIN_FAMILIES, "eval_topologies": EVAL}, known


def fit_pairs(train: list[Episode], encode, *, modes=MODES, protocol=PROTOCOL) -> list[tuple]:
    mode_indices = [MODES.index(mode) for mode in modes]
    inputs = [[] for _ in modes]
    masks, targets = [], []
    for ordinal, episode in enumerate(train):
        teacher = bayes_check(episode.catalog, minimize_work=True)

        def collect(public: PublicObservation, teacher=teacher) -> int:
            slot = teacher(public)
            packs, mask, _ = encode(public)
            for bucket, index in zip(inputs, mode_indices, strict=True):
                bucket.append(packs[index])
            masks.append(mask)
            targets.append(slot)
            return slot

        check_run(episode, collect, protocol["budgets"][ordinal % len(protocol["budgets"])])
    mask_batch, target_batch = torch.stack(masks), torch.as_tensor(targets)
    batches = [torch.stack(bucket) for bucket in inputs]
    fitted = []
    for seed in protocol["seeds"]:
        torch.manual_seed(seed)
        initial = ObligationPointer(SLOTS, FIELDS, BUCKETS)
        for mode, codes in zip(modes, batches, strict=True):
            model, metadata = fit(initial, (codes, mask_batch, target_batch), seed, protocol["updates"])
            fitted.append((mode, seed, model, metadata))
        matched = ("initial_sha256", "supervised_examples", "supervised_action_tokens",
                   "optimizer_updates", "parameters")
        for key in matched:
            if len({row[3][key] for row in fitted[-len(modes):]}) != 1:
                raise ValueError(f"Matched training field changed across arms: {key}")
    return fitted


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
    sham_changes = Counter()

    @cache
    def encode(public: PublicObservation) -> tuple:
        full, _ = project(public)
        packs, allowed, changed = graph_batch(public, full)
        sham_changes.update(states=1, changes=int(changed > 0), changed_directed_pairs=changed)
        return packs, allowed, changed

    train = [sample(family, ordinal, training=True)
             for family, ordinal in product(TRAIN_FAMILIES, range(PROTOCOL["train_roots_per_family"]))]
    train_roots = {episode.catalog.source_root for episode in train}
    if len(train_roots) != len(train) or train_roots & known:
        raise ValueError("Training overlaps the archived public roots")
    known.update(train_roots)
    lock.update(train_source_hashes=sorted(train_roots), torch=str(torch.__version__), slots=SLOTS,
                obligation_rules_sha256=digest(public_rules.encode()), batch_size=BATCH_SIZE,
                receipt_rules_sha256=digest(acceptance_rules.encode()))
    out.mkdir(parents=True, exist_ok=False)
    protocol_file = out / "locked_protocol.json"
    protocol_file.write_bytes((canonical(lock) + "\n").encode())
    models = fit_pairs(train, encode)
    fit_file = out / "locked_fits.json"
    fit_file.write_bytes((canonical([{**meta, "arm": arm} for arm, _, _, meta in models]) + "\n").encode())
    train_changes = dict(sham_changes)

    @cache
    def accept(episode: Episode, state: tuple[int, ...]) -> tuple[set[str], str]:
        return verified_proofs(acceptance_rules, episode, state, native_snapshot=verifier.evaluate)

    outcomes: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    paired: dict[str, dict[str, list[list[float]]]] = {key: defaultdict(list) for key in ("dense", "rewired")}
    receipts = []
    for family, ordinal in product(EVAL, range(PROTOCOL["eval_roots_per_family"])):
        episode = sample(family, ordinal, training=False)
        root = episode.catalog.source_root
        if root in known:
            raise ValueError("Evaluation overlaps a previous public source")
        known.add(root)
        differences: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for budget in PROTOCOL["budgets"]:
            results = {}
            policies = [("exact_work_tie", None, bayes_check(episode.catalog, minimize_work=True))]
            for mode, seed, model, _ in models:
                mode_index = MODES.index(mode)

                @torch.inference_mode()
                def choose(public, model=model, index=mode_index):
                    packs, allowed, _ = encode(public)
                    return int(model(packs[index][None], allowed[None])[0].argmax().item())

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
            for control in paired:
                for seed in PROTOCOL["seeds"]:
                    trial, baseline = results["incidence", seed], results[control, seed]
                    differences[control].append((baseline["checks"] - trial["checks"],
                                                 trial["completed"] - baseline["completed"]))
        for control, values in differences.items():
            paired[control][family].append(np.mean(values, axis=0).tolist())

    payload = "".join(canonical(row) + "\n" for row in receipts).encode()
    (out / "receipts.jsonl").write_bytes(payload)
    report = {
        "protocol_sha256": digest(protocol_file.read_bytes()), "fits_sha256": digest(fit_file.read_bytes()),
        "outcomes": outcomes, "paired_incidence": {key: primary_intervals(values) for key, values in paired.items()},
        "training_sham": train_changes, "sham_all_states": sham_changes,
        "native_obligation_calls": planner.calls, "native_acceptance_calls": verifier.calls,
        "receipts_sha256": digest(payload), "peak_mib": peak_rss_mib(), "memory": limits,
        "wall_seconds": time.perf_counter() - started, "training_ready": False,
    }
    (out / "incidence_report.json").write_bytes((canonical(report) + "\n").encode())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("storyworld", "prior", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    print(canonical(run(args.storyworld, args.prior, args.out)))


if __name__ == "__main__":
    main()
