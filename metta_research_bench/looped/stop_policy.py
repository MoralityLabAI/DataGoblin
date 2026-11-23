"""Freeze the action-set students and calibrate or verify when they should continue."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from importlib.metadata import version
from itertools import product
from pathlib import Path

import torch

from MeTTa.airis02_student import MAX_ITEMS
from MeTTa.evidence_budget_confirmation import primary_intervals
from MeTTa.evidence_obligations import ScopedNative
from MeTTa.ram_hillclimb import FAMILIES, OOD_FAMILY, canonical, digest, install_memory_ceiling, peak_rss_mib, root_name
from MeTTa.ram_hillclimb_native import PACKAGE_VERSION
from MeTTa.ram_trajectory import typed_proposal
from MeTTa.sft_oracle import first_state, make_task

from ..experiment import VERIFIERS_VERSION, score_submission
from ..taskset import MAX_DECISIONS, BenchConfig, CurationData, CurationTask
from .__main__ import revision, summarize
from .action_ties import edit_index, save_lines
from .model import LOOPS, MAX_PROPOSALS, LoopedPointer, fit, solve, tokenize, upstream_revision, weight_hash

PROTOCOL = {
    "schema": "metta-verified-rectification-stop-v1", "dev_start": 130000, "frozen_start": 140000,
    "conflict_start": 150000, "dev_roots_per_family": 12, "frozen_roots_per_family": 80, "conflict_roots": 80,
    "offsets": (0.0, -0.25, -0.5, -0.75, -1.0, -1.5), "max_dev_call_increase_percent": 15,
    "witness_calls_per_episode": 2, "primary": "witness_minus_raw", "minimum_meaningful_completion_effect": 0.02,
    "decision_cap": MAX_DECISIONS, "proposal_cap": MAX_PROPOSALS, "recurrent_visits": LOOPS,
    "archived_protocol_sha256": "b53dfd255a917d3ed1dc988628deac4c15a3f91b5095bba44d205499613b1178",
    "archived_fits_sha256": "bb0e177af435310d6fdca1306dd8ff8a9a441264ec9bea1af913d13a5492ec86",
    "archived_frames_sha256": "ea08c5ddb3dee3a20fb218255d6b24e0d71a4ed834bc95e1d98f28174704268a",
}
CONTRASTS = (("raw", "witness"), ("raw", "calibrated"), ("raw", "calibrated_witness"),
             ("calibrated", "calibrated_witness"))
QUERY = "(StopDeferralWitnesses CurrentState)"


def public_rank(lookup: dict, selected: tuple[str, ...]) -> tuple[int, int, int, int]:
    """Match the benchmark's public objective and exact lexicographic final tie-break."""
    chosen = [lookup[key] for key in selected]
    lex = 0
    for key in selected:
        lex = lex * (MAX_ITEMS + 1) + int(key[1:]) + 1
    lex *= (MAX_ITEMS + 1) ** (MAX_ITEMS - len(selected))
    return (len({topic for record in chosen for topic in record["topics"]}),
            sum(record["quality"] for record in chosen), sum(record["cost"] for record in chosen), lex)


def rank_order(rank: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return rank[0], rank[1], -rank[2], -rank[3]


def rank_fact(state: str, rank: tuple[int, int, int, int]) -> str:
    return f"(PublicRank {state} (Rank {' '.join(map(str, rank))}))"


class StopWitness:
    def __init__(self, rules: str):
        self.native = ScopedNative(rules)

    def __call__(self, world, selected, visited, attempted, remaining, already_spent, refine) -> dict:
        cap = min(remaining, PROTOCOL["witness_calls_per_episode"] - already_spent)
        result = {"checks": [], "rank_probes": 0, "native": None, "chosen": None}
        if cap <= 0:
            return result
        lookup = {item["id"]: item for item in world["items"]}
        present = set(selected)
        current_rank = public_rank(lookup, tuple(selected))
        candidates = []
        for key in lookup.keys() - attempted:
            after = tuple(sorted(present ^ {key}))
            if after in visited:
                continue
            candidate_rank = public_rank(lookup, after)
            result["rank_probes"] += 1
            if rank_order(candidate_rank) > rank_order(current_rank):
                candidates.append((candidate_rank, key))
        candidates.sort(key=lambda row: rank_order(row[0]), reverse=True)
        if not candidates:
            return result
        facts = [rank_fact("CurrentState", current_rank)]
        expected = set()
        for index, (rank, key) in enumerate(candidates[:cap]):
            action = {"op": "remove" if key in present else "add", "id": key}
            receipt = refine(typed_proposal(action), selected)
            accepted, feasible = receipt["accepted"], not receipt["result_violations"]
            result["checks"].append({"action": action, "rank": rank, "receipt": receipt})
            status = "Accepted" if accepted else "Vetoed"
            safety = "Feasible" if feasible else "Invalid"
            facts.append(f"(CheckedStopNeighbor CurrentState Move{index} {status} {safety} Next{index})")
            facts.append(rank_fact(f"Next{index}", rank))
            if accepted and feasible:
                expected.add(f"(ContinueWith Move{index})")
                result["chosen"] = index
                break
        (offers,) = self.native.evaluate(facts, (QUERY,))
        actual = {str(offer) for offer in offers}
        if actual != expected:
            raise ValueError("Native STOP witness disagrees with a checked public rank improvement")
        result["native"] = {"facts": facts, "query": QUERY, "offers": sorted(actual), "facts_sha256": digest(facts)}
        return result


def load_archive(prior: Path) -> tuple[dict, list, list]:
    lock, fits, frames = ((prior / name).read_bytes() for name in
                          ("locked_protocol.json", "locked_fits.json", "training_frames.jsonl"))
    if (digest(lock), digest(fits), digest(frames)) != (
        PROTOCOL["archived_protocol_sha256"], PROTOCOL["archived_fits_sha256"], PROTOCOL["archived_frames_sha256"],
    ):
        raise ValueError("Use the exact archived action-set training bundle")
    return json.loads(lock), json.loads(fits), [json.loads(line) for line in frames.splitlines()]


def allocate(archived: dict) -> dict:
    reserved = set(archived["excluded_source_sha256"])
    reserved.update(source["source_sha256"] for group in archived["allocation"].values() for source in group)
    cohorts = {}
    for split, families, count in (("dev", FAMILIES, PROTOCOL["dev_roots_per_family"]),
                                   ("frozen", FAMILIES, PROTOCOL["frozen_roots_per_family"]),
                                   ("conflict", (OOD_FAMILY,), PROTOCOL["conflict_roots"])):
        start = PROTOCOL[f"{split}_start"]
        rows = []
        for family, seed in product(families, range(start, start + count)):
            root = root_name(family, seed)
            sha = digest(make_task(family, seed)["program"].encode())
            if sha in reserved:
                raise ValueError("STOP-policy cohort overlaps a previous source")
            reserved.add(sha)
            rows.append({"root": root, "seed": seed, "family": family, "source_sha256": sha})
        cohorts[split] = rows
    return cohorts


def refit(frames: list, fits: list) -> tuple[dict, dict]:
    tokens, masks, targets = [], [], []
    worlds = {}
    for row in frames:
        if row["root"] not in worlds:
            world = make_task(row["family"], row["seed"])
            if digest(world["program"].encode()) != row["source_sha256"]:
                raise ValueError("The archived training state differs from its source")
            worlds[row["root"]] = world
        code, mask = tokenize(worlds[row["root"]], row["selected"])
        support = row["labels"]["uniform_ties"]
        target = torch.zeros(MAX_ITEMS + 1)
        target[[edit_index(action) for action in support]] = 1 / len(support)
        tokens.append(code)
        masks.append(mask)
        targets.append(target)
    batch = tuple(torch.stack(values) for values in (tokens, masks, targets))
    models, receipts = {}, {}
    for old in fits:
        if old["arm"] != "uniform_ties":
            continue
        seed = old["seed"]
        torch.manual_seed(seed)
        models[seed], current = fit(LoopedPointer(), batch, seed, old["optimizer_updates"])
        if any(current[key] != old[key] for key in ("checkpoint_sha256", "initial_sha256", "parameters",
                                                    "supervised_examples", "supervised_action_tokens")):
            raise ValueError("Recreated checkpoint differs from the archived student")
        receipts[seed] = current
    return models, receipts


async def evaluate(cohort, split, models, fits, modes, out: Path) -> tuple[list, str]:
    task_config = BenchConfig().task
    spent = []
    prime = out / f"{split}_verifiers_traces.jsonl"
    with prime.open("w", encoding="utf-8", newline="\n") as stream:
        for source in cohort:
            world = make_task(source["family"], source["seed"])
            selected = first_state(world, source["seed"])
            prompt = ("Repair the public evidence selection, prioritizing coverage, quality and then lower cost. "
                      f"Respect the typed constraints. Start at {canonical(selected)}.\n\n{world['program']}")
            data = CurationData(id=source["root"], family=source["family"], source_seed=source["seed"], split=split,
                                source_sha256=source["source_sha256"], prompt=prompt)
            task = CurationTask(data, task_config)
            for (seed, model), (arm, (offset, witness)) in product(models.items(), modes.items()):
                events = []
                actions, policy = solve(model, data, stop_offset=offset, stop_guard=witness, audit=events)
                result = await score_submission(task, actions, f"stop-policy/{arm}/{seed}")
                result.info.update(policy=policy, checkpoint_sha256=fits[seed]["checkpoint_sha256"])
                stream.write(result.model_dump_json() + "\n")
                spent.append({**source, "arm": arm, "train_seed": seed, "offset": offset, "actions": actions,
                              "events": events, "policy": policy, "grade": result.info["curation_grade"],
                              "reward": int(result.reward)})
    return spent, digest(prime.read_bytes())


def metrics(rows: list[dict]) -> dict:
    witness_fields = (
        "witness_verifier_calls", "witness_interventions", "witness_rank_probes", "witness_native_queries",
    )
    return {**summarize(rows), **{field: sum(row["policy"].get(field, 0) for row in rows) for field in witness_fields}}


def select_offset(summaries: dict, offsets: dict) -> tuple[str, list[str]]:
    baseline = summaries["raw"]
    cap = 100 + PROTOCOL["max_dev_call_increase_percent"]
    eligible = [arm for arm, row in summaries.items()
                if row["policy_verifier_calls"] * 100 <= baseline["policy_verifier_calls"] * cap
                and row["unsafe_additions"] <= baseline["unsafe_additions"]]
    selected = max(eligible, key=lambda arm: (summaries[arm]["safe_completions"],
                                              -summaries[arm]["policy_verifier_calls"],
                                              -summaries[arm]["premature_stops"], offsets[arm]))
    return selected, eligible


def analyze(rows: list[dict]) -> dict:
    groups = defaultdict(dict)
    for row in rows:
        groups[row["root"], row["train_seed"]][row["arm"]] = row
    contrasts = {}
    for baseline, candidate in CONTRASTS:
        differences = defaultdict(list)
        families = {}
        for (root, _), group in groups.items():
            left, right = group[baseline], group[candidate]
            families[root] = left["family"]
            calls = left["policy"]["policy_verifier_calls"] - right["policy"]["policy_verifier_calls"]
            differences[root].append((calls, right["reward"] - left["reward"]))
        strata = defaultdict(list)
        for root, values in differences.items():
            strata[families[root]].append([sum(metric) / len(metric) for metric in zip(*values, strict=True)])
        contrasts[f"{candidate}_minus_{baseline}"] = primary_intervals(strata)
    arms = {row["arm"] for row in rows}
    return {"arms": {arm: metrics([row for row in rows if row["arm"] == arm]) for arm in arms},
            "contrasts": contrasts}


def save(path: Path, data: dict) -> str:
    raw = (canonical(data) + "\n").encode()
    path.write_bytes(raw)
    return digest(raw)


async def run(prior: Path, out: Path) -> dict:
    limit = install_memory_ceiling(8192)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    commit, dirty = revision()
    if dirty or version("verifiers") != VERIFIERS_VERSION or version("hyperon") != PACKAGE_VERSION:
        raise ValueError("Use the committed protocol with pinned Prime Verifiers and Hyperon")
    upstream = upstream_revision()
    archived, fits, frames = load_archive(prior)
    cohorts = allocate(archived)
    rules = Path(__file__).with_name("stop_witness.metta").read_text(encoding="utf-8")
    witness = StopWitness(rules)
    out.mkdir(parents=True, exist_ok=False)
    lock = {"commit": commit, "upstream": upstream, "torch": str(torch.__version__), "verifiers": VERIFIERS_VERSION,
            "hyperon": PACKAGE_VERSION, "protocol": PROTOCOL, "native_rule_sha256": digest(rules.encode()),
            "cohorts": cohorts}
    protocol_sha = save(out / "locked_protocol.json", lock)
    models, receipts = refit(frames, fits)
    fit_sha = save(out / "locked_fits.json", receipts)
    offsets = {"raw" if offset == 0 else f"offset_{offset}": offset for offset in PROTOCOL["offsets"]}
    dev_modes = {name: (offset, None) for name, offset in offsets.items()}
    rows, prime_sha = await evaluate(cohorts["dev"], "dev", models, receipts, dev_modes, out)
    dev_scores = {name: metrics([row for row in rows if row["arm"] == name]) for name in dev_modes}
    chosen, eligible = select_offset(dev_scores, offsets)
    selection = {"dev_scores": dev_scores, "eligible": eligible, "chosen": chosen, "offset": offsets[chosen],
                 "dev_receipts_sha256": save_lines(out / "dev_receipts.jsonl", rows), "dev_prime_sha256": prime_sha}
    selection_sha = save(out / "locked_selection.json", selection)
    modes = {"raw": (0.0, None), "calibrated": (offsets[chosen], None),
             "witness": (0.0, witness), "calibrated_witness": (offsets[chosen], witness)}
    result = {}
    for split in ("frozen", "conflict"):
        rows, prime_sha = await evaluate(cohorts[split], split, models, receipts, modes, out)
        result[split] = {**analyze(rows), "receipts_sha256": save_lines(out / f"{split}_receipts.jsonl", rows),
                         "prime_verifiers_sha256": prime_sha}
    if any(weight_hash(model) != receipts[seed]["checkpoint_sha256"] for seed, model in models.items()):
        raise ValueError("A frozen model changed during scoring")
    report = {"protocol_sha256": protocol_sha, "fits_sha256": fit_sha, "selection_sha256": selection_sha,
              "selected_offset": offsets[chosen], "result": result, "native_witness_calls": witness.native.calls,
              "peak_mib": peak_rss_mib(), "memory": limit, "wall_seconds": time.perf_counter() - started,
              "training_ready": False}
    save(out / "stop_policy_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("prior", "out"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args()
    print(canonical(asyncio.run(run(args.prior, args.out))))


if __name__ == "__main__":
    main()
