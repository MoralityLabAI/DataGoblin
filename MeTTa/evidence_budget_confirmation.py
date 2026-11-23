"""Prospective paired test of a lower-work policy for MeTTa evidence checks."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from functools import cache
from itertools import permutations, product
from math import prod
from pathlib import Path

import numpy as np

from .evidence_budget import WIDTH, Catalog, Episode, bayes_check, check_run
from .evidence_budget_pilot import gate_rules, verified_proofs
from .ram_hillclimb import canonical, digest, install_memory_ceiling, peak_rss_mib

FAMILIES = {
    "pair_path": ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)),
    "pair_ring": ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6)),
    "triad_bridge": ((0, 1), (1, 2, 3), (0, 4, 5), (3, 5, 6)),
    "pair_tripod": ((0, 1), (0, 2), (0, 3), (1, 4, 5), (2, 5, 6)),
    "paired_overlap": ((0, 1), (2, 3), (4, 5), (1, 3, 6), (0, 2, 6)),
    "triple_mix": ((0, 1), (2, 3, 4), (2, 5, 6), (0, 4, 6), (1, 3, 5)),
    "dense_hub": ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)),
    "mixed_hub": ((0, 1), (0, 2, 3), (0, 4, 5), (0, 2, 6)),
}
PROTOCOL = {
    "schema": "metta-check-cost-confirmation-v1",
    "roots_per_family": 24, "budgets": (2, 3, 4), "correlated_budget": 3,
    "completion_margin": -0.03, "bootstrap_draws": 4000, "bootstrap_seed": 2101,
    "prior_receipts_sha256": "e34bc5266fd1797280b3f3b73895c46b77d24b2e890e168ca3b2271197f900ce",
}
POLICIES = ("index_tie", "work_tie")


def relabel(routes: tuple[tuple[int, ...], ...], order: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(order[i] for i in route)) for route in routes))


def topology(routes: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    """Canonicalize the seven-record hypergraph across every slot permutation."""
    return min(relabel(routes, order) for order in permutations(range(WIDTH)))


def sample_root(family: str, ordinal: int) -> tuple[Catalog, tuple[bool, ...], tuple[bool, ...]]:
    rng = random.Random(f"{PROTOCOL['schema']}/{family}/{ordinal}")
    routes = relabel(FAMILIES[family], tuple(rng.sample(range(WIDTH), WIDTH)))
    priors = tuple(rng.randrange(18, 93) for _ in range(WIDTH))
    independent = tuple(rng.randrange(100) < p for p in priors)
    # Shared docket state retains calibrated marginals and changes the joint law.
    shared_draw = rng.randrange(100)
    correlated = tuple((shared_draw if rng.randrange(5) < 3 else rng.randrange(100)) < p for p in priors)
    return Catalog(routes, priors), independent, correlated


def exact_expectations(catalog: Catalog, policy, budget: int) -> tuple[int, int]:
    """Independently enumerate the private worlds, weighted by public marginals."""
    successes, checks = 0, 0
    for current in product((False, True), repeat=len(catalog.priors)):
        world_mass = prod(p if status else 100 - p for p, status in zip(catalog.priors, current, strict=True))
        result = check_run(Episode(catalog, current), policy, budget)
        successes += world_mass * result["completed"]
        checks += world_mass * result["checks"]
    return successes, checks


def primary_intervals(paired: dict[str, list[list[float]]]) -> dict:
    """Resample source roots within each fixed topology, keeping budgets paired."""
    rng = np.random.default_rng(PROTOCOL["bootstrap_seed"])
    strata = [np.asarray(values) for values in paired.values()]
    draws = []
    for values in strata:
        index = rng.integers(0, len(values), size=(PROTOCOL["bootstrap_draws"], len(values)))
        draws.append(values[index].mean(axis=1))
    lower, upper = np.quantile(np.mean(draws, axis=0), [0.025, 0.975], axis=0)
    means = np.mean([values.mean(axis=0) for values in strata], axis=0)
    return {name: {"estimate": float(mean), "paired_root_bootstrap_95": [float(lo), float(hi)]}
            for name, mean, lo, hi in zip(("calls_saved", "completion_difference"), means, lower, upper, strict=True)}


def preflight(prior: Path, native_rules: str) -> tuple[dict, set[str]]:
    from .evidence_budget import ROUTES

    data = (prior / "check_receipts.jsonl").read_bytes()
    report = json.loads((prior / "budget_report.json").read_bytes())
    if (digest(data) != PROTOCOL["prior_receipts_sha256"]
            or report["native_rule_sha256"] != digest(native_rules.encode())):
        raise ValueError("The preregistered prior and native receipt rules must match the archive")
    registered = {topology(routes) for routes in FAMILIES.values()}
    old = {topology(routes) for routes in ROUTES.values()}
    if len(registered) != len(FAMILIES) or registered & old:
        raise ValueError("The confirmation topology families must be distinct from all earlier families")
    known = set(report["training_roots"])
    known.update(json.loads(line)["source_root_sha256"] for line in data.splitlines())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("Commit the protocol before opening the fresh cohort")
    lock = {"commit": commit, "protocol": PROTOCOL, "topologies": FAMILIES,
            "excluded_source_hashes": sorted(known), "native_rule_sha256": digest(native_rules.encode())}
    return lock, known


def run(checkout: Path, prior: Path, out: Path) -> dict:
    bounds = install_memory_ceiling(8192)
    started = time.perf_counter()
    rules = gate_rules(checkout)
    lock, used = preflight(prior, rules)
    out.mkdir(parents=True, exist_ok=False)
    lock_path = out / "locked_protocol.json"
    lock_path.write_bytes((canonical(lock) + "\n").encode())

    @cache
    def audit(episode: Episode, state: tuple[int, ...]) -> tuple[set[str], str]:
        return verified_proofs(rules, episode, state)

    stats: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    paired: dict[str, list[list[float]]] = defaultdict(list)
    receipts, catalogs = [], []
    model_expected = Counter()
    for family, ordinal in product(FAMILIES, range(PROTOCOL["roots_per_family"])):
        catalog, iid, shared = sample_root(family, ordinal)
        root = catalog.source_root
        if root in used:
            raise ValueError("Fresh public catalog overlaps a prior source")
        used.add(root)
        catalogs.append({"source_root_sha256": root, "family": family, **asdict(catalog)})
        original, revised = (cache(bayes_check(catalog, minimize_work=(arm == "work_tie"))) for arm in POLICIES)
        budgets = [("independent", iid, b) for b in PROTOCOL["budgets"]]
        budgets.append(("correlated", shared, PROTOCOL["correlated_budget"]))
        deltas = []
        for condition, private, budget in budgets:
            if condition == "independent":
                old_success, old_checks = exact_expectations(catalog, original, budget)
                new_success, new_checks = exact_expectations(catalog, revised, budget)
                if old_success != new_success or new_checks > old_checks:
                    raise ValueError("Tie-break failed exact modeled completion preservation or work minimization")
                model_expected.update(cases=1, saved_numerator=old_checks - new_checks)
            outcomes = []
            for arm, policy in zip(POLICIES, (original, revised), strict=True):
                episode = Episode(catalog, private)
                row = check_run(episode, policy, budget)
                proofs, native_hash = audit(episode, row["final_state"])
                if bool(proofs) != row["completed"]:
                    raise ValueError("Sample outcome disagrees with native acceptance")
                cell = f"{condition}/budget_{budget}"
                stats[cell][arm].update(cases=1, completed=int(row["completed"]), checks=row["checks"])
                receipts.append({**row, "family": family, "condition": condition, "budget": budget, "arm": arm,
                                 "native_input_sha256": native_hash, "native_accepted_proofs": sorted(proofs)})
                outcomes.append(row)
            if condition == "independent":
                old, new = outcomes
                deltas.append((old["checks"] - new["checks"], int(new["completed"]) - int(old["completed"])))
        paired[family].append(np.mean(deltas, axis=0).tolist())

    receipt_payload = "".join(canonical(row) + "\n" for row in receipts).encode()
    catalog_payload = "".join(canonical(row) + "\n" for row in catalogs).encode()
    (out / "paired_check_receipts.jsonl").write_bytes(receipt_payload)
    (out / "public_catalogs.jsonl").write_bytes(catalog_payload)
    primary = primary_intervals(paired)
    primary["passed"] = (primary["calls_saved"]["paired_root_bootstrap_95"][0] > 0
                         and primary["completion_difference"]["paired_root_bootstrap_95"][0]
                         >= PROTOCOL["completion_margin"])
    result = {
        "locked_protocol_sha256": digest(lock_path.read_bytes()), "roots": len(catalogs), "stats": stats,
        "primary": primary, "exhaustive_model_cases": model_expected["cases"],
        "exhaustive_model_calls_saved": model_expected["saved_numerator"] / (100**WIDTH * model_expected["cases"]),
        "native_snapshot_checks": audit.cache_info().misses, "receipts_sha256": digest(receipt_payload),
        "public_catalogs_sha256": digest(catalog_payload), "peak_mib": peak_rss_mib(), "memory": bounds,
        "wall_seconds": time.perf_counter() - started, "training_ready": False,
    }
    (out / "confirmation_report.json").write_bytes((canonical(result) + "\n").encode())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("storyworld", "prior", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    print(canonical(run(args.storyworld, args.prior, args.out)))


if __name__ == "__main__":
    main()
