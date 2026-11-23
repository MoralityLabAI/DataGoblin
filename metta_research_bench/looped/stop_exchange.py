"""Test one bounded, verified remove-add exchange at a proposed STOP."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from importlib.metadata import version
from itertools import product
from pathlib import Path

import torch

from MeTTa.evidence_obligations import ScopedNative
from MeTTa.ram_hillclimb import FAMILIES, OOD_FAMILY, digest, install_memory_ceiling, peak_rss_mib, root_name
from MeTTa.ram_hillclimb_native import PACKAGE_VERSION
from MeTTa.ram_trajectory import typed_proposal
from MeTTa.sft_oracle import make_task

from ..experiment import VERIFIERS_VERSION
from ..taskset import MAX_DECISIONS
from .__main__ import revision
from .action_ties import save_lines
from .model import LOOPS, MAX_PROPOSALS, upstream_revision, weight_hash
from .stop_policy import analyze, evaluate, load_archive, public_rank, rank_fact, rank_order, refit, save

PROTOCOL = {
    "schema": "metta-verified-stop-exchange-v1", "frozen_start": 160000, "conflict_start": 170000,
    "frozen_roots_per_family": 80, "conflict_roots": 80, "primary": "exchange_minus_raw",
    "minimum_meaningful_completion_effect": 0.02, "proposal_cap": MAX_PROPOSALS, "decision_cap": MAX_DECISIONS,
    "recurrent_visits": LOOPS, "witness_verifier_cap": 2, "max_exchanges": 1,
    "prior_stop_lock_sha256": "1940811894c58f7303fbfef6011b60c9ae55bc988230fde791342e93fcc5c68e",
    "prior_selection_sha256": "78b9315ac96bbe40da78f1102f9fdaf191d90675e07c4b4c9f24b4f656501fcf",
}
CONTRASTS = (("raw", "exchange"), ("calibrated", "calibrated_exchange"),
             ("raw", "calibrated"), ("raw", "calibrated_exchange"))
QUERY = "(BudgetedStopExchanges CurrentState)"
RESERVED_QUERY = "(ReservedStopExchanges CurrentState)"


def exchange_candidates(world: dict, selected: list[str], visited: set, attempted: set):
    """Screen public constraints before spending authoritative typed checks."""
    lookup = {item["id"]: item for item in world["items"]}
    chosen = set(selected)
    absent = sorted(lookup.keys() - chosen)
    budget = world["token_budget"]
    revision = world["family"] == "revision"
    latest = {}
    for item in lookup.values():
        if item["eligible"]:
            latest[item["key"]] = max(item["epoch"], latest.get(item["key"], 0))

    def usable(item):
        return item["eligible"] and (not revision or item["epoch"] == latest[item["key"]])

    original_rank = public_rank(lookup, tuple(selected))
    original_missing = any(set(lookup[key]["requires"]) - chosen for key in chosen)
    conflicts = [set(pair) for pair in world["conflicts"]]
    proposals, screened, ranked = [], 0, 0
    for drop in selected:
        rest = chosen - {drop}
        middle = tuple(sorted(rest))
        if drop in attempted or middle in visited:
            continue
        items = [lookup[key] for key in rest]
        missing = {dep for item in items for dep in item["requires"]} - rest
        lineages, keys = {item["lineage"] for item in items}, {item["key"] for item in items}
        blocked = {other for pair in conflicts if pair & rest for other in (pair - rest)}
        remaining = budget - sum(item["cost"] for item in items)
        viable = (len(selected) <= world["max_items"] and (original_missing or not missing)
                  and all(usable(item) for item in items) and len(lineages) == len(items)
                  and (not revision or len(keys) == len(items)) and not any(pair <= rest for pair in conflicts))
        for add in absent:
            screened += 1
            item = lookup[add]
            if (not viable or item["cost"] > remaining or not usable(item) or item["lineage"] in lineages
                    or (revision and item["key"] in keys) or add in blocked
                    or not missing <= {add} or not set(item["requires"]) <= rest):
                continue
            endpoint = tuple(sorted(rest | {add}))
            if endpoint in visited:
                continue
            rank = public_rank(lookup, endpoint)
            ranked += 1
            if rank_order(rank) > rank_order(original_rank):
                proposals.append((rank, drop, add))
    proposals.sort(key=lambda row: rank_order(row[0]), reverse=True)
    return proposals, original_rank, screened, ranked


class StopExchange:
    def __init__(self, rules: str):
        self.native = ScopedNative(rules)

    def __call__(self, world, selected, visited, attempted, remaining, already_spent, refine, *,
                 reserved: bool = False, preferred_remove: str | None = None) -> dict:
        result = {"checks": [], "rank_probes": 0, "screened_pairs": 0, "native": None, "chosen": None}
        cap = PROTOCOL["witness_verifier_cap"]
        if min(remaining, cap - already_spent) < 2:
            return result
        candidates, current_rank, screened, ranked = exchange_candidates(world, selected, visited, attempted)
        result.update(rank_probes=ranked, screened_pairs=screened)
        if not candidates:
            return result
        exchange_rank, drop, add = candidates[0]
        if reserved and (remaining != 2 or (preferred_remove is not None and drop != preferred_remove)):
            return result
        drop_action = {"op": "remove", "id": drop}
        drop_receipt = refine(typed_proposal(drop_action), selected)
        result["checks"].append({"before": selected, "action": drop_action, "receipt": drop_receipt})
        drop_status = "Accepted" if drop_receipt["accepted"] else "Vetoed"
        facts = [rank_fact("CurrentState", current_rank),
                 f"(StopExchangeAllowance CurrentState {remaining} {already_spent} {cap})",
                 f"(CheckedExchangeEdit Attempt CurrentState remove {drop} {drop_status} Middle)"]
        if reserved:
            permission = "AnyRemoval" if preferred_remove is None else f"(ModelRemoval {preferred_remove})"
            facts.append(f"(ExchangeReservation CurrentState {remaining} {permission})")
        expected = set()
        if drop_receipt["accepted"]:
            middle = drop_receipt["selected_after"]
            add_action = {"op": "add", "id": add}
            add_receipt = refine(typed_proposal(add_action), middle)
            result["checks"].append({"before": middle, "action": add_action, "receipt": add_receipt})
            feasible = not add_receipt["result_violations"]
            state = "Accepted" if add_receipt["accepted"] else "Vetoed"
            endpoint_status = "Feasible" if feasible else "Invalid"
            facts.extend((f"(CheckedExchangeEdit Attempt Middle add {add} {state} Endpoint)",
                          f"(ExchangeEndpoint Attempt Endpoint {endpoint_status})",
                          rank_fact("Endpoint", exchange_rank)))
            if add_receipt["accepted"] and feasible:
                expected.add(f"(ContinueWithExchange Attempt {drop} {add})")
        query = RESERVED_QUERY if reserved else QUERY
        (offers,) = self.native.evaluate(facts, (query,))
        actual = {str(offer) for offer in offers}
        if actual != expected:
            raise ValueError("Native STOP exchange disagrees with the two checked transitions")
        if actual:
            result["chosen"] = (0, 1)
        result["native"] = {"facts": facts, "query": query, "offers": sorted(actual), "facts_sha256": digest(facts)}
        return result


def source_cohorts(archive: dict, *prior_runs: dict, protocol=PROTOCOL) -> dict:
    excluded = set(archive["excluded_source_sha256"])
    excluded.update(row["source_sha256"] for group in archive["allocation"].values() for row in group)
    excluded.update(row["source_sha256"] for run in prior_runs for group in run["cohorts"].values() for row in group)
    cohorts = {}
    for split, families, count in (("frozen", FAMILIES, protocol["frozen_roots_per_family"]),
                                   ("conflict", (OOD_FAMILY,), protocol["conflict_roots"])):
        start = protocol[f"{split}_start"]
        rows = []
        for family, seed in product(families, range(start, start + count)):
            root = root_name(family, seed)
            sha = digest(make_task(family, seed)["program"].encode())
            if sha in excluded:
                raise ValueError("Exchange witness cohort overlaps a previous source")
            excluded.add(sha)
            rows.append({"root": root, "seed": seed, "family": family, "source_sha256": sha})
        cohorts[split] = rows
    return cohorts


def read_stop_policy(path: Path) -> tuple[dict, float]:
    lock = (path / "locked_protocol.json").read_bytes()
    selection = (path / "locked_selection.json").read_bytes()
    if (digest(lock), digest(selection)) != (PROTOCOL["prior_stop_lock_sha256"], PROTOCOL["prior_selection_sha256"]):
        raise ValueError("Use the archived development-selected STOP offset and its locked source allocation")
    return json.loads(lock), json.loads(selection)["offset"]


async def run(prior: Path, stop_archive: Path, out: Path) -> dict:
    limits = install_memory_ceiling(8192)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    commit, dirty = revision()
    if dirty or version("verifiers") != VERIFIERS_VERSION or version("hyperon") != PACKAGE_VERSION:
        raise ValueError("Use the committed exchange protocol with pinned Prime Verifiers and Hyperon")
    upstream = upstream_revision()
    archived, fits, frames = load_archive(prior)
    prior_stop, selected_offset = read_stop_policy(stop_archive)
    cohorts = source_cohorts(archived, prior_stop)
    witness_rules = Path(__file__).with_name("stop_witness.metta").read_text(encoding="utf-8")
    rules = witness_rules + "\n" + Path(__file__).with_suffix(".metta").read_text(encoding="utf-8")
    witness = StopExchange(rules)
    out.mkdir(parents=True, exist_ok=False)
    protocol_sha = save(out / "locked_protocol.json", {
        "commit": commit, "upstream": upstream, "torch": str(torch.__version__), "verifiers": VERIFIERS_VERSION,
        "hyperon": PACKAGE_VERSION, "protocol": PROTOCOL, "native_rules_sha256": digest(rules.encode()),
        "action_set_lock_sha256": digest((prior / "locked_protocol.json").read_bytes()),
        "offset": selected_offset, "cohorts": cohorts,
    })
    models, receipts = refit(frames, fits)
    fit_sha = save(out / "locked_fits.json", receipts)
    modes = {"raw": (0.0, None), "calibrated": (selected_offset, None), "exchange": (0.0, witness),
             "calibrated_exchange": (selected_offset, witness)}
    result = {}
    for split in ("frozen", "conflict"):
        rows, prime_sha = await evaluate(cohorts[split], split, models, receipts, modes, out, exchange=True)
        receipt_sha = save_lines(out / f"{split}_receipts.jsonl", rows)
        result[split] = {**analyze(rows, CONTRASTS), "receipts_sha256": receipt_sha,
                         "prime_verifiers_sha256": prime_sha}
    if any(weight_hash(model) != receipts[seed]["checkpoint_sha256"] for seed, model in models.items()):
        raise ValueError("A frozen model changed during exchange scoring")
    report = {"protocol_sha256": protocol_sha, "fits_sha256": fit_sha, "offset": selected_offset,
              "result": result, "native_exchange_calls": witness.native.calls, "memory": limits,
              "peak_mib": peak_rss_mib(), "wall_seconds": time.perf_counter() - started, "training_ready": False}
    save(out / "stop_exchange_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("prior", "stop-archive", "out"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    params = parser.parse_args()
    print(asyncio.run(run(params.prior, params.stop_archive, params.out)))


if __name__ == "__main__":
    main()
