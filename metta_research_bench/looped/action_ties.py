"""Transfer verified action-set supervision to the paired MeTTa curation benchmark."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import Counter, defaultdict, deque
from importlib.metadata import version
from itertools import product
from pathlib import Path

import torch

from MeTTa.airis02_student import MAX_ITEMS
from MeTTa.evidence_budget_confirmation import primary_intervals
from MeTTa.evidence_obligations import ScopedNative
from MeTTa.ram_hillclimb import FAMILIES, OOD_FAMILY, canonical, digest, install_memory_ceiling, peak_rss_mib, root_name
from MeTTa.ram_hillclimb_native import PACKAGE_VERSION
from MeTTa.ram_trajectory import STOP, RepairVerifier, typed_proposal
from MeTTa.sft_oracle import first_state, make_task

from ..experiment import VERIFIERS_VERSION, score_submission
from ..taskset import MAX_DECISIONS, BenchConfig, CurationData, CurationTask
from .__main__ import revision, summarize
from .model import BATCH_SIZE, LOOPS, MAX_PROPOSALS, LoopedPointer, fit, solve, tokenize, upstream_revision, weight_hash

ARMS = ("canonical_hard", "sampled_hard", "uniform_ties")
PROTOCOL = {
    "schema": "metta-looped-verified-action-ties-v2", "train_start": 100000,
    "frozen_start": 110000, "conflict_start": 120000, "train_roots_per_family": 48,
    "frozen_roots_per_family": 80, "conflict_roots": 80, "train_seeds": (1101, 1102, 1103),
    "optimizer_updates": 240, "presentations_per_root": 2, "primary": "uniform_ties_minus_canonical_hard",
    "minimum_meaningful_completion_effect": 0.02, "evidence_class": "frozen_synthetic_transfer_pilot",
    "evaluation_decision_cap": MAX_DECISIONS, "evaluation_proposal_cap": MAX_PROPOSALS,
}
ARCHIVES = (
    "metta-looped-quality/locked_development_report.json", "metta-looped-quality/frozen_report.json",
    "metta-looped-ablation/locked_ablation.json", "metta-looped-ablation/frozen_ablation_report.json",
    "metta-looped-interaction/locked_interaction.json", "metta-looped-interaction/frozen_interaction_report.json",
    "metta-looped-revision-replication/frozen_revision_replication.json",
    "metta-looped-lineage-replication/frozen_lineage_replication.json",
    "metta-looped-lineage-800-replication/locked_lineage_replication.json",
    "metta-looped-lineage-800-replication/frozen_lineage_replication.json",
    "metta-research-bench/development_report.json", "metta-research-bench/frozen_report.json",
    "metta11-development/report.json", "metta11-frozen/frozen_conflict_receipts.jsonl",
    "metta12-pilot/development_report.json", "metta12-pilot/frozen_conflict_receipts.jsonl",
    "metta13-pilot/development_report.json", "metta13-pilot/frozen_conflict_receipts.jsonl",
    "metta13-pilot/retrospective_report.json",
)
QUERY = "(VerifiedShortestActions CurrentState)"


def edit_index(action: dict) -> int:
    return MAX_ITEMS if action == STOP else int(action["id"][1:])


class ShortestRepair:
    """Traverse the bounded state graph backwards through actual gated edits."""

    def __init__(self, verifier: RepairVerifier):
        self.verifier = verifier
        target = tuple(verifier.optimum["selected"])
        self.records = tuple(item["id"] for item in verifier.world["items"])
        self.distances = {target: 0}
        self.pending = deque([target])
        self.receipts = {}

    def edge(self, state: tuple[str, ...], action: dict) -> dict:
        key = state, action["op"], action["id"]
        if key not in self.receipts:
            self.receipts[key] = self.verifier.refine(typed_proposal(action), list(state))
        return self.receipts[key]

    def neighbors(self, state: tuple[str, ...]):
        present = set(state)
        # The ordering preserves the conventional remove-first canonical target for exact ties.
        for record in (*state, *(key for key in self.records if key not in present)):
            yield tuple(sorted(present ^ {record})), {"op": "remove" if record in present else "add", "id": record}

    def distance(self, state: tuple[str, ...]) -> int:
        while state not in self.distances and self.pending:
            following = self.pending.popleft()
            distance = self.distances[following] + 1
            for previous, inverse in self.neighbors(following):
                if previous in self.distances:
                    continue
                forward = {"op": "remove" if inverse["op"] == "add" else "add", "id": inverse["id"]}
                if self.edge(previous, forward)["accepted"]:
                    self.distances[previous] = distance
                    self.pending.append(previous)
        if state not in self.distances:
            raise ValueError("No repair path through the environment gate")
        return self.distances[state]

    def offers(self, selected: list[str]) -> tuple[list[dict], list[dict]]:
        state = tuple(selected)
        distance = self.distance(state)
        if not distance:
            stop = self.verifier.check(STOP, selected)
            if not stop["correct_stop"]:
                raise ValueError("The verified terminal state rejected STOP")
            return [STOP], [{"action": STOP, "receipt": stop, "distance_before": 0, "distance_after": 0}]
        # Every closer layer has been discovered when the source is reached by the reverse breadth-first search.
        inspected = [{"action": action, "receipt": self.edge(state, action),
                      "distance_before": distance, "distance_after": distance - 1}
                     for neighbor, action in self.neighbors(state) if self.distances.get(neighbor) == distance - 1]
        allowed = [item["action"] for item in inspected if item["receipt"]["accepted"]]
        if not allowed:
            raise ValueError("No accepted first edit on a shortest path")
        return allowed, inspected


def native_certificate(native: ScopedNative, allowed: list[dict], checked: list[dict]) -> dict:
    distance = checked[0]["distance_before"]
    facts = [f"(RepairDistance CurrentState {distance})"]
    for index, item in enumerate(checked):
        action, receipt = item["action"], item["receipt"]
        if action == STOP:
            facts.append("(CheckedRepairStop CurrentState Accepted)")
        else:
            status = "Accepted" if receipt["accepted"] else "Vetoed"
            facts.append(f'(CheckedRepairEdit CurrentState {action["op"]} {action["id"]} {status} Next{index})')
            facts.append(f'(RepairDistance Next{index} {item["distance_after"]})')
    (answers,) = native.evaluate(facts, (QUERY,))
    actual = sorted({str(row) for row in answers})
    expected = sorted(f'(RepairChoice {action["op"]} {action["id"]})' for action in allowed)
    if actual != expected:
        raise ValueError("Native action-set query disagrees with checked shortest edits")
    return {"facts": facts, "query": QUERY, "offers": actual, "facts_sha256": digest(facts)}


def archived_hashes(node, key: str = "") -> set[str]:
    if isinstance(node, dict):
        return set().union(*(archived_hashes(value, name) for name, value in node.items()))
    if isinstance(node, list):
        return set().union(*(archived_hashes(value, key) for value in node))
    source_keys = ("source_sha256", "sha256", "source_root_sha256", "train_hashes", "root_hashes",
                   "source_hashes", "excluded_source_hashes")
    return {node} if key in source_keys and isinstance(node, str) and re.fullmatch(r"[0-9a-f]{64}", node) else set()


def registry(archive: Path) -> tuple[dict, set[str]]:
    receipts, excluded = {}, set()
    for name in ARCHIVES:
        raw = (archive / name).read_bytes()
        contents = [json.loads(line) for line in raw.splitlines()] if name.endswith(".jsonl") else json.loads(raw)
        sources = archived_hashes(contents)
        receipts[name] = {"sha256": digest(raw), "source_count": len(sources)}
        excluded.update(sources)
    return receipts, excluded


def allocate() -> dict:
    schedules = {
        "train": (FAMILIES, PROTOCOL["train_roots_per_family"]),
        "frozen": (FAMILIES, PROTOCOL["frozen_roots_per_family"]),
        "conflict": ((OOD_FAMILY,), PROTOCOL["conflict_roots"]),
    }
    strata = {}
    for split, (families, count) in schedules.items():
        start = PROTOCOL[f"{split}_start"]
        strata[split] = [{"family": family, "seed": seed, "root": root_name(family, seed),
                          "source_sha256": digest(make_task(family, seed)["program"].encode())}
                         for family, seed in product(families, range(start, start + count))]
    return strata


def pack_training(train: list[dict], native: ScopedNative):
    tokens, masks, targets, frames, traces = [], [], defaultdict(list), [], []
    totals = Counter()
    for source in train:
        world = make_task(source["family"], source["seed"])
        start = first_state(world, source["seed"])
        verifier = RepairVerifier(world)
        search = ShortestRepair(verifier)
        prior = {}
        for view in range(PROTOCOL["presentations_per_root"]):
            selected, transitions = start, []
            for _ in range(MAX_DECISIONS):
                key = tuple(selected)
                if key not in prior:
                    allowed, checked = search.offers(selected)
                    certificate = native_certificate(native, allowed, checked)
                    codes, mask = tokenize(world, selected)
                    tokens.append(codes)
                    masks.append(mask)
                    pick = int(digest([source["source_sha256"], selected])[:16], 16) % len(allowed)
                    labels = {ARMS[0]: allowed[:1], ARMS[1]: [allowed[pick]], ARMS[2]: allowed}
                    for arm, actions in labels.items():
                        distribution = torch.zeros(MAX_ITEMS + 1)
                        distribution[[edit_index(action) for action in actions]] = 1 / len(actions)
                        targets[arm].append(distribution)
                    frames.append({**source, "selected": selected, "labels": labels, "checked": checked,
                                   "distance": checked[0]["distance_before"], "native": certificate})
                    prior[key] = (allowed, {edit_index(item["action"]): item["receipt"] for item in checked})
                allowed, indexed = prior[key]
                action = allowed[0] if view == 0 else allowed[-1]
                receipt = indexed[edit_index(action)]
                transitions.append({"selected": selected, "action": action, "receipt": receipt})
                if action == STOP:
                    break
                selected = receipt["selected_after"]
            else:
                raise ValueError("A training presentation exceeded the frozen decision cap")
            traces.append({**source, "presentation": view, "transitions": transitions})
        totals.update(verifier_calls=verifier.checks, constraint_checks=verifier.constraint_checks,
                      enumerated_subsets=verifier.optimum["checked_subsets"])
    return torch.stack(tokens), torch.stack(masks), targets, frames, traces, totals


def fit_all(tokens, masks, targets) -> tuple[dict, list]:
    models, receipts = {}, []
    fields = ("initial_sha256", "parameters", "supervised_examples", "optimizer_updates", "supervised_action_tokens")
    for seed in PROTOCOL["train_seeds"]:
        torch.manual_seed(seed)
        initial = LoopedPointer()
        group = []
        for arm in ARMS:
            batch = tokens, masks, torch.stack(targets[arm])
            model, receipt = fit(initial, batch, seed, PROTOCOL["optimizer_updates"])
            models[arm, seed] = model
            group.append({**receipt, "arm": arm})
        if any(len({item[field] for item in group}) != 1 for field in fields):
            raise ValueError("The matched fits have different initialization or training budgets")
        receipts.extend(group)
    return models, receipts


async def evaluate(sources: list[dict], split: str, models: dict, checkpoints: dict, out: Path) -> tuple[list, str]:
    task_config = BenchConfig().task
    rows = []
    trace = out / f"{split}_verifiers_traces.jsonl"
    with trace.open("w", encoding="utf-8", newline="\n") as stream:
        for source in sources:
            world = make_task(source["family"], source["seed"])
            selected = first_state(world, source["seed"])
            verifier = RepairVerifier(world)
            shortest, checks = ShortestRepair(verifier).offers(selected)
            distance = checks[0]["distance_before"]
            prompt = ("Repair the evidence selection using the public MeTTa facts. Maximize distinct goal coverage, "
                      "then quality, then minimize token cost. Break exact ties by sorted record IDs. "
                      f"Respect all constraints and end with STOP within {MAX_DECISIONS} decisions.\n"
                      f"Initial selection: {canonical(selected)}.\n\n{world['program']}")
            data = CurationData(id=source["root"], family=source["family"], source_seed=source["seed"], split=split,
                                source_sha256=source["source_sha256"], prompt=prompt)
            task = CurationTask(data, task_config)
            for (arm, seed), model in models.items():
                actions, costs = solve(model, task.data)
                result = await score_submission(task, actions, f"action-ties/{arm}/{seed}")
                result.info.update(policy=costs, checkpoint_sha256=checkpoints[arm, seed])
                stream.write(result.model_dump_json() + "\n")
                rows.append({**source, "train_seed": seed, "arm": arm, "actions": actions, "initial_distance": distance,
                             "optimal_first_edit": int(bool(actions) and actions[0] in shortest),
                             "initial_reference_checks": verifier.checks, "initial_optimal_actions": shortest,
                             "reward": int(result.reward), "grade": result.info["curation_grade"], "policy": costs})
    return rows, digest(trace.read_bytes())


def analyze(rows: list[dict]) -> dict:
    groups = defaultdict(dict)
    for row in rows:
        groups[row["root"], row["train_seed"]][row["arm"]] = row
    summaries = {}
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        multi = [row for row in subset if row["initial_distance"] >= 2]
        summaries[arm] = {**summarize(subset), "optimal_first_edits": sum(row["optimal_first_edit"] for row in subset),
                          "multi_edit_cases": len(multi), "multi_edit_completions": sum(row["reward"] for row in multi)}
    contrasts = {}
    for baseline, candidate in ((ARMS[0], ARMS[2]), (ARMS[1], ARMS[2]), (ARMS[0], ARMS[1])):
        differences = defaultdict(list)
        families = {}
        for (root, _), trio in groups.items():
            left, right = trio[baseline], trio[candidate]
            families[root] = left["family"]
            calls_saved = left["policy"]["policy_verifier_calls"] - right["policy"]["policy_verifier_calls"]
            differences[root].append((calls_saved, right["reward"] - left["reward"]))
        strata = defaultdict(list)
        for root, values in differences.items():
            strata[families[root]].append([sum(metric) / len(metric) for metric in zip(*values, strict=True)])
        contrasts[f"{candidate}_minus_{baseline}"] = primary_intervals(strata)
    return {"arms": summaries, "contrasts": contrasts}


def save_lines(path: Path, rows: list[dict]) -> str:
    raw = "".join(canonical(row) + "\n" for row in rows).encode()
    path.write_bytes(raw)
    return digest(raw)


async def run(archive: Path, out: Path) -> dict:
    limit = install_memory_ceiling(8192)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    commit, dirty = revision()
    if dirty or version("verifiers") != VERIFIERS_VERSION or version("hyperon") != PACKAGE_VERSION:
        raise ValueError("Use the committed study with pinned Prime Verifiers and Hyperon")
    upstream = upstream_revision()
    archives, protected = registry(archive)
    strata = allocate()
    identities = [source["source_sha256"] for split in strata.values() for source in split]
    if len(identities) != len(set(identities)) or protected.intersection(identities):
        raise ValueError("The study overlaps a prior or current source allocation")
    rules = Path(__file__).with_suffix(".metta").read_text(encoding="utf-8")
    native = ScopedNative(rules)
    out.mkdir(parents=True, exist_ok=False)
    lock = {"commit": commit, "upstream": upstream, "torch": str(torch.__version__), "batch_size": BATCH_SIZE,
            "hyperon": PACKAGE_VERSION, "verifiers": VERIFIERS_VERSION, "recurrent_visits": LOOPS,
            "protocol": PROTOCOL, "archives": archives, "excluded_source_sha256": sorted(protected),
            "native_rule_sha256": digest(rules.encode()), "allocation": strata}
    lock_path = out / "locked_protocol.json"
    lock_path.write_bytes((canonical(lock) + "\n").encode())
    tokens, masks, targets, frames, traces, totals = pack_training(strata["train"], native)
    frames_sha = save_lines(out / "training_frames.jsonl", frames)
    traces_sha = save_lines(out / "training_trajectories.jsonl", traces)
    models, fits = fit_all(tokens, masks, targets)
    fit_path = out / "locked_fits.json"
    fit_path.write_bytes((canonical(fits) + "\n").encode())
    checkpoints = {(row["arm"], row["seed"]): row["checkpoint_sha256"] for row in fits}
    result = {}
    for split in ("frozen", "conflict"):
        rows, prime_sha = await evaluate(strata[split], split, models, checkpoints, out)
        result[split] = {**analyze(rows), "receipts_sha256": save_lines(out / f"{split}_receipts.jsonl", rows),
                         "prime_verifiers_sha256": prime_sha}
    if any(weight_hash(model) != checkpoints[key] for key, model in models.items()):
        raise ValueError("A checkpoint changed during frozen scoring")
    report = {"protocol_sha256": digest(lock_path.read_bytes()), "fits_sha256": digest(fit_path.read_bytes()),
              "training_frames_sha256": frames_sha, "training_trajectories_sha256": traces_sha,
              "training_labels_with_ties": sum(len(row["labels"][ARMS[2]]) > 1 for row in frames),
              "training_native_calls": native.calls, "training_verifier_work": totals, "result": result,
              "wall_seconds": time.perf_counter() - started, "peak_mib": peak_rss_mib(), "memory": limit,
              "training_ready": False}
    (out / "action_ties_report.json").write_bytes((canonical(report) + "\n").encode())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(canonical(asyncio.run(run(args.archive, args.out))))


if __name__ == "__main__":
    main()
