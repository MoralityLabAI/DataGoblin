"""Compare warm proof-cell invalidation with full cold computation on CPU."""
from __future__ import annotations

import argparse
import random
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from importlib.metadata import version
from itertools import product
from pathlib import Path

from .evidence_index import ProofIndex
from .evidence_proofs import TARGET, Record, Rule, Snapshot
from .evidence_revision import FAMILIES, checked_snapshot, episode, storyworld_lineage_rules
from .ram_hillclimb import canonical, digest, install_memory_ceiling, peak_rss_mib
from .ram_hillclimb_native import PACKAGE_VERSION

REPLAY_COMMIT = "0282f6e67f546a56a3e992cf24de59b97758faac"
REPLAY_PATH = "metta-storyworld/engine/evidence_replay.metta"
REPLAY_SHA256 = "246cd5dfe11db2ad515a07aaa69115bafadb8ddedb5574e8645f1cd6216b5f7f"
PILOT_FAMILIES = (*FAMILIES, "wide_alternatives", "rule_rewire")


def stress_episode(family: str, seed: int) -> tuple[Snapshot, ...]:
    rng = random.Random(f"proof-index-pilot-v1/{family}/{seed}")
    tag = f"{rng.getrandbits(48):012x}"
    if family == "wide_alternatives":
        width = rng.randrange(4, 9)
        records = tuple(Record(f"{side}Record{i}x{tag}", f"{side}Claim{i}",
                               f"{side}Source{i}x{tag}") for i in range(width) for side in ("Left", "Right"))
        rules = tuple(Rule(f"Branch{i}", (f"LeftClaim{i}", f"RightClaim{i}"), TARGET) for i in range(width))
        first, second = rng.sample(range(len(records)), 2)
        removed = tuple(replace(r, current=False) if i == first else r for i, r in enumerate(records))
        changed = tuple(replace(r, current=False) if i == second else r for i, r in enumerate(removed))
        restored = tuple(replace(r, name=f"Replacement{tag}", root=f"SourceReplacement{tag}", current=True)
                         if i == first else r for i, r in enumerate(changed))
        return tuple(Snapshot(rows, rules, 1) for rows in (records, removed, changed, restored))
    if family == "rule_rewire":
        records = (Record(f"Primary{tag}", "Mark", f"SourceA{tag}"),
                   Record(f"Old{tag}", "Gate", f"SourceB{tag}"))
        rules = (Rule("ReviewRule", ("Mark", "Gate"), TARGET), Rule("Backedge", (TARGET,), "NewGate"))
        rewired = (replace(rules[0], premises=("Mark", "NewGate")), rules[1])
        arrived = (*records, Record(f"New{tag}", "NewGate", f"SourceC{tag}"))
        removed = (replace(arrived[0], visible=False), *arrived[1:])
        return (Snapshot(records, rules, 3), Snapshot(records, rewired, 3),
                Snapshot(arrived, rewired, 3), Snapshot(removed, rewired, 3))
    return episode(family, seed)


def replay_survivors(rules: str, public: str, prior: list[str]) -> tuple[set[str], str | None]:
    """Submit old trees from the independently checked receipt to the new snapshot."""
    from hyperon import Environment, MeTTa

    if not prior:
        return set(), None
    facts, query = public.strip().rsplit("\n", 1)
    prefix = query.removesuffix(")").replace("!(Support ", "!(RecheckEvidenceProof ", 1)
    source = "\n".join((rules, facts, *(f"{prefix} {proof})" for proof in prior)))
    runtime = MeTTa(env_builder=Environment.custom_env(create_config=False, is_test=True))
    results = runtime.run(source)
    return {str(atom) for result in results for atom in result}, digest(source.encode())


def run(checkout: Path, out: Path, roots_per_family: int) -> dict:
    if version("hyperon") != PACKAGE_VERSION or not 1 <= roots_per_family <= 8:
        raise ValueError(f"Use hyperon=={PACKAGE_VERSION} and 1 to 8 roots per family")
    bounds = install_memory_ceiling(8192)
    full_rules = storyworld_lineage_rules(checkout)
    replay = subprocess.check_output(["git", "-C", str(checkout), "show", f"{REPLAY_COMMIT}:{REPLAY_PATH}"])
    if digest(replay) != REPLAY_SHA256:
        raise ValueError("Storyworld replay rules differ from the pinned bytes")
    replay_rules = replay.decode("utf-8")
    rows = []
    work: dict[str, dict[str, Counter]] = defaultdict(lambda: {"warm": Counter(), "cold": Counter()})
    timing: Counter = Counter()
    outcomes: Counter = Counter()
    started = time.perf_counter()
    for family, seed in product(PILOT_FAMILIES, range(roots_per_family)):
        snapshots = stress_episode(family, seed)
        root = digest([asdict(snapshot) for snapshot in snapshots])
        warm = ProofIndex()
        prior: list[str] = []
        for epoch, snapshot in enumerate(snapshots):
            start = time.process_time()
            updated, warm_work = warm.evaluate(snapshot)
            timing["warm_cpu"] += time.process_time() - start
            start = time.process_time()
            recomputed, cold_work = ProofIndex().evaluate(snapshot)
            timing["cold_cpu"] += time.process_time() - start
            expected = checked_snapshot(full_rules, snapshot)
            if updated != recomputed or updated != expected["target"]:
                raise ValueError(f"Proof index mismatch at {family} {seed} epoch {epoch}")
            if epoch:
                survivors, replay_sha256 = replay_survivors(replay_rules, expected["public_query"], prior)
                if survivors != set(prior).intersection(updated["proofs"]):
                    raise ValueError("Native proof replay disagrees with the full-snapshot intersection")
                outcomes["replayed_proofs"] += len(prior)
                outcomes["surviving_proofs"] += len(survivors)
                outcomes["transitions"] += 1
                outcomes["warm_fewer_cells"] += warm_work.get("computed_cells", 0) < cold_work["computed_cells"]
                for arm, arm_work in (("warm", warm_work), ("cold", cold_work)):
                    work[family][arm].update(arm_work)
                rows.append({
                    "source_root_sha256": root, "family": family, "epoch": epoch,
                    "public_query": expected["public_query"],
                    "full_native_sha256": expected["native_input_sha256"], "replay_native_sha256": replay_sha256,
                    "cold_work": cold_work, "warm_work": warm_work, "proof_target": updated,
                    "surviving_proofs": sorted(survivors), "training_ready": False,
                })
            prior = updated["proofs"]
    payload = "".join(canonical(row) + "\n" for row in rows).encode()
    out.mkdir(parents=True, exist_ok=False)
    (out / "index_audit.jsonl").write_bytes(payload)
    report = {
        "roots": len(PILOT_FAMILIES) * roots_per_family,
        "snapshots": len(PILOT_FAMILIES) * roots_per_family * 4,
        "outcomes": dict(outcomes), "work_by_family": work, "cpu_seconds_including_initialization": timing,
        "full_rules_sha256": digest(full_rules.encode()), "native_replay_commit": REPLAY_COMMIT,
        "native_replay_sha256": REPLAY_SHA256, "hyperon": PACKAGE_VERSION,
        "wall_seconds": time.perf_counter() - started,
        "memory": bounds, "peak_mib": peak_rss_mib(), "audit_sha256": digest(payload), "training_ready": False,
    }
    (out / "index_report.json").write_text(canonical(report) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storyworld", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roots-per-family", type=int, default=6)
    args = parser.parse_args()
    print(canonical(run(args.storyworld, args.out, args.roots_per_family)))


if __name__ == "__main__":
    main()
