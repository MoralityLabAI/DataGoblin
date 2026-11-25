"""COG-02: immutable attempts and reusable lemmas in finite Horn proof search.

The generated worlds are synthetic propositional exercises. Their source and
checker are independent of Cogentic's papers. The checker only applies declared
implications to facts already established in the episode. All model features
come from the public world and the current proved facts.

This module uses only the Python standard library so downstream repositories
can vendor an exact, source-pinned copy. Its JSON encoding matches DataGoblin's
``canonical_json`` for the plain JSON records used here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "cogentic.ledger.v1"
SPLITS = ("train", "calibration", "ordinary_test", "ood_test")
FAMILIES = ("chain", "fork_join", "redundant", "distractor")
FEATURE_NAMES = (
    "bias", "premise_fraction", "available_fraction", "missing_fraction",
    "conclusion_known", "conclusion_is_goal", "goal_premise_known_fraction",
    "conclusion_outdegree", "premise_mean_outdegree", "initial_premise_fraction",
    "goal_predecessor", "progress_fraction",
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _world(rng: random.Random, family: str, depth: int) -> dict[str, Any]:
    facts = list(range(rng.randint(3, 5)))
    atoms = list(facts)
    rules: list[dict[str, Any]] = []
    prior = list(facts)
    previous = list(facts)
    for _ in range(depth):
        width = 1 if family == "chain" else rng.randint(2, 3)
        layer = []
        for _ in range(width):
            chosen = {rng.choice(previous)}
            extra = 0 if family == "chain" else rng.randint(1, 2)
            chosen.update(rng.sample(prior, min(extra, len(prior))))
            conclusion = len(atoms)
            atoms.append(conclusion)
            rules.append({"premises": sorted(chosen), "conclusion": conclusion})
            if family == "redundant" and rng.random() < 0.65:
                alternate = sorted({rng.choice(previous), rng.choice(facts)})
                if alternate != sorted(chosen):
                    rules.append({"premises": alternate, "conclusion": conclusion})
            layer.append(conclusion)
        previous, prior = layer, prior + layer
    goal = len(atoms)
    atoms.append(goal)
    rules.append({"premises": sorted(previous), "conclusion": goal})
    # Reachable irrelevant lemmas share the same syntax as useful lemmas.
    decoys = rng.randint(5, 9) + (depth if family == "distractor" else 0)
    pool = list(facts)
    for _ in range(decoys):
        conclusion = len(atoms)
        atoms.append(conclusion)
        premises = sorted(rng.sample(pool, min(rng.randint(1, 3), len(pool))))
        rules.append({"premises": premises, "conclusion": conclusion})
        pool.append(conclusion)
    # These atoms are unproved hypotheses of tempting rules. The environment
    # will reject applications until every declared premise is available.
    for _ in range(rng.randint(2, 4)):
        unavailable = len(atoms)
        atoms.extend([unavailable, unavailable + 1])
        rules.append({"premises": [rng.choice(facts), unavailable], "conclusion": unavailable + 1})
    return {"atoms": atoms, "facts": facts, "rules": rules, "goal": goal}


def generate_tasks(per_split: int = 64, seed: int = 20261126) -> list[dict[str, Any]]:
    """Generate distinct root worlds, with deeper roots reserved for OOD.

    ``root_id`` binds the unpermuted world. Presentation IDs and rule order are
    randomized independently. All variants of one root retain one split.
    This is a public development benchmark; its seeds are public.
    """
    if type(per_split) is not int or not 1 <= per_split <= 4096:
        raise ValueError("per_split must be in 1..4096")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    tasks = []
    roots: set[str] = set()
    for split_index, split in enumerate(SPLITS):
        rng = random.Random(seed + split_index * 1_000_003)
        for index in range(per_split):
            family = FAMILIES[index % len(FAMILIES)]
            depth = rng.randint(6, 9) if split == "ood_test" else rng.randint(2, 5)
            world = _world(rng, family, depth)
            root = digest(world)
            while root in roots:
                world = _world(rng, family, depth)
                root = digest(world)
            roots.add(root)
            permutation = list(world["atoms"])
            rng.shuffle(permutation)
            presented = {
                "atoms": list(world["atoms"]),
                "facts": sorted(permutation[x] for x in world["facts"]),
                "rules": [
                    {"premises": sorted(permutation[x] for x in r["premises"]),
                     "conclusion": permutation[r["conclusion"]]}
                    for r in world["rules"]
                ],
                "goal": permutation[world["goal"]],
            }
            rng.shuffle(presented["rules"])
            task = {
                "schema_version": SCHEMA, "root_id": root,
                "task_id": digest(presented), "split": split, "family": family,
                "world": presented,
                "provenance": {"kind": "synthetic_horn_proof_dag", "generator": SCHEMA,
                               "public_seed": seed, "depth": depth, "split_index": index,
                               "cogentic_paper_trace": False, "source_world_sha256": root},
            }
            task["task_digest"] = digest(task)
            validate_task(task)
            tasks.append(task)
    return tasks


def validate_task(task: dict[str, Any]) -> None:
    if task.get("schema_version") != SCHEMA or task.get("split") not in SPLITS:
        raise ValueError("invalid task schema or split")
    if task.get("task_digest") != digest({k: v for k, v in task.items() if k != "task_digest"}):
        raise ValueError("task content digest mismatch")
    world = task["world"]
    atoms = world["atoms"]
    if atoms != list(range(len(atoms))) or not 2 <= len(atoms) <= 128:
        raise ValueError("atoms must be bounded contiguous integer IDs")
    facts = world["facts"]
    if facts != sorted(set(facts)) or not facts or not set(facts) <= set(atoms):
        raise ValueError("invalid facts")
    if world["goal"] not in atoms or world["goal"] in facts:
        raise ValueError("invalid goal")
    if not 1 <= len(world["rules"]) <= 128:
        raise ValueError("invalid rule count")
    for rule in world["rules"]:
        premises = rule["premises"]
        if (set(rule) != {"premises", "conclusion"} or not premises
                or premises != sorted(set(premises)) or not set(premises) <= set(atoms)
                or rule["conclusion"] not in atoms):
            raise ValueError("invalid rule")


def initial_state(task: dict[str, Any]) -> list[int]:
    return list(task["world"]["facts"])


def public_observation(
    task: dict[str, Any], known: list[int], history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Explicit allowlist prevents metadata, future verdicts and teachers entering inputs."""
    prior = []
    for step in history or []:
        prior.append({key: copy.deepcopy(step[key]) for key in ("action", "accepted", "code", "new_fact")})
    return {"world": copy.deepcopy(task["world"]), "known": sorted(set(known)), "history": prior}


def check_step(task: dict[str, Any], known: list[int], action: int) -> dict[str, Any]:
    """One environment-owned verifier call, charged on every attempted rule.

    The caller holds the state. ``replay_trace`` independently reconstructs it
    from the initial facts before a trace may enter an exported corpus.
    """
    world = task["world"]
    if any(type(x) is not int for x in known) or not set(known) <= set(world["atoms"]):
        raise ValueError("unknown or noninteger state atom")
    before = set(known)
    accepted, code, new_fact = False, "invalid_action", None
    if type(action) is int and 0 <= action < len(world["rules"]):
        rule = world["rules"][action]
        if set(rule["premises"]) <= before:
            accepted = True
            new_fact = None if rule["conclusion"] in before else rule["conclusion"]
            code = "duplicate_lemma" if new_fact is None else "proved_lemma"
            before.add(rule["conclusion"])
        else:
            code = "missing_premise"
    return {"accepted": accepted, "code": code, "next_known": sorted(before),
            "new_fact": new_fact, "solved": world["goal"] in before, "verifier_calls": 1}


def useful_atoms(task: dict[str, Any]) -> set[int]:
    """Structural goal ancestors, used by the teacher and the named structural control."""
    needed = {task["world"]["goal"]}
    changed = True
    while changed:
        old = len(needed)
        for rule in task["world"]["rules"]:
            if rule["conclusion"] in needed:
                needed.update(rule["premises"])
        changed = len(needed) != old
    return needed


def reference_action(task: dict[str, Any], known: list[int]) -> int | None:
    """An explicit symbolic teacher; its choices are training labels only."""
    needed, present = useful_atoms(task), set(known)
    for index, rule in enumerate(task["world"]["rules"]):
        if (rule["conclusion"] in needed and rule["conclusion"] not in present
                and set(rule["premises"]) <= present):
            return index
    return None


def candidate_features(task: dict[str, Any], known: list[int], action: int) -> list[float]:
    """Twelve public local features. The complete graph remains available to models."""
    world = task["world"]
    rule = world["rules"][action]
    present, facts = set(known), set(world["facts"])
    premises = set(rule["premises"])
    outdegree = Counter(x for item in world["rules"] for x in item["premises"])
    goal_pre = {x for item in world["rules"] if item["conclusion"] == world["goal"] for x in item["premises"]}
    count = len(premises)
    scale = max(1, len(world["rules"]))
    return [
        1.0, count / 3, len(premises & present) / count, len(premises - present) / count,
        float(rule["conclusion"] in present), float(rule["conclusion"] == world["goal"]),
        len(goal_pre & present) / max(1, len(goal_pre)), outdegree[rule["conclusion"]] / scale,
        sum(outdegree[x] for x in premises) / (count * scale), len(premises & facts) / count,
        float(rule["conclusion"] in goal_pre), len(present) / len(world["atoms"]),
    ]


def ranking_row(task: dict[str, Any], known: list[int]) -> dict[str, Any]:
    """Collect checked candidates from one public state; labels live in their own field."""
    needed = useful_atoms(task)
    receipts = [check_step(task, known, i) for i in range(len(task["world"]["rules"]))]
    return {
        "schema_version": SCHEMA, "root_id": task["root_id"], "task_id": task["task_id"],
        "split": task["split"], "family": task["family"],
        "public": public_observation(task, known),
        "candidates": [{"action": i, "features": candidate_features(task, known, i)} for i in range(len(receipts))],
        "labels": {
            "accepted": [r["accepted"] for r in receipts],
            "useful": [r["new_fact"] is not None and r["new_fact"] in needed for r in receipts],
        },
        "collection_cost": {"verifier_calls": len(receipts)},
        "label_source": "exact_step_checker_and_public_symbolic_goal_ancestry",
    }


def collect_trace(task: dict[str, Any], max_calls: int = 64) -> dict[str, Any]:
    """A fixed teacher collection policy inserts one rejection and one duplicate.

    These controls provide both outcomes in every episode. They are plainly
    labeled and never treated as evidence that a model learned repair.
    """
    known, history, steps = initial_state(task), [], []
    previous = digest({"task_digest": task["task_digest"], "policy": "teacher_with_sentinels"})
    rejected = next(i for i, rule in enumerate(task["world"]["rules"]) if not set(rule["premises"]) <= set(known))
    pending = [rejected]
    duplicated = False
    for visit in range(max_calls):
        action = pending.pop(0) if pending else reference_action(task, known)
        if action is None:
            break
        receipt = check_step(task, known, action)
        step = {"visit": visit, "observation": public_observation(task, known, history),
                "action": action, "verification": receipt, "previous_digest": previous,
                "cost": {"verifier_calls": 1, "proposal_calls": 1}}
        step["step_digest"] = digest(step)
        previous = step["step_digest"]
        steps.append(step)
        history.append({"action": action, **{k: receipt[k] for k in ("accepted", "code", "new_fact")}})
        known = receipt["next_known"]
        if receipt["solved"]:
            break
        if receipt["new_fact"] is not None and not duplicated:
            pending.append(action)
            duplicated = True
    result = {"schema_version": SCHEMA, "root_id": task["root_id"], "task_id": task["task_id"],
              "task_digest": task["task_digest"], "split": task["split"], "family": task["family"],
              "policy": "teacher_with_sentinels", "provenance": copy.deepcopy(task["provenance"]),
              "steps": steps, "solved": task["world"]["goal"] in known,
              "verifier_calls": len(steps), "final_digest": previous}
    result["trace_digest"] = digest(result)
    return result


def replay_trace(task: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """Reject changed verdicts, cost erasure, skipped states and broken hash chains."""
    validate_task(task)
    if trace.get("schema_version") != SCHEMA or trace.get("provenance") != task["provenance"]:
        raise ValueError("trace schema or provenance mismatch")
    if trace.get("trace_digest") != digest({k: v for k, v in trace.items() if k != "trace_digest"}):
        raise ValueError("trace digest mismatch")
    for key in ("root_id", "task_id", "task_digest", "split", "family"):
        if trace.get(key) != task[key]:
            raise ValueError(f"trace task binding mismatch: {key}")
    known, history = initial_state(task), []
    previous = digest({"task_digest": task["task_digest"], "policy": trace["policy"]})
    for visit, step in enumerate(trace["steps"]):
        if step["visit"] != visit or step["previous_digest"] != previous:
            raise ValueError("trace order or chain mismatch")
        if step["observation"] != public_observation(task, known, history):
            raise ValueError("trace public state mismatch")
        if step["step_digest"] != digest({k: v for k, v in step.items() if k != "step_digest"}):
            raise ValueError("step digest mismatch")
        receipt = check_step(task, known, step["action"])
        if step["verification"] != receipt or step["cost"] != {"verifier_calls": 1, "proposal_calls": 1}:
            raise ValueError("trace checker or cost mismatch")
        history.append({"action": step["action"], **{k: receipt[k] for k in ("accepted", "code", "new_fact")}})
        known, previous = receipt["next_known"], step["step_digest"]
    if (trace["solved"] != (task["world"]["goal"] in known)
            or trace["verifier_calls"] != len(trace["steps"]) or trace["final_digest"] != previous):
        raise ValueError("trace terminal or cost mismatch")
    return {"passed": True, "trace_digest": trace["trace_digest"], "verified_steps": len(trace["steps"])}


def _training_row(step: dict[str, Any], meta: dict[str, Any], target: Any, kind: str) -> dict[str, Any]:
    return {"id": digest([step["step_digest"], kind]), **meta,
            "input": copy.deepcopy(step["observation"]), "target": target,
            "loss": {"boundary": "response_only", "field": "target", "eligible": True}}


def export_streams(task: dict[str, Any], trace: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Use RLT corpus semantics: outcomes retain failures; SFT admits proved lemmas."""
    replay_trace(task, trace)
    streams: dict[str, list[dict[str, Any]]] = {name: [] for name in ("sft", "repairs", "outcomes", "transitions")}
    if task["split"] != "train":
        return streams
    failed_before = False
    for step in trace["steps"]:
        receipt = step["verification"]
        meta = {"root": task["root_id"], "split": task["split"], "family": task["family"],
                "trace_digest": trace["trace_digest"], "step_digest": step["step_digest"],
                "verification_kind": "finite_horn_step_replay", "source": "synthetic_horn_proof_dag"}
        outcome = _training_row(step, meta, {"accepted": receipt["accepted"], "code": receipt["code"]}, "outcomes")
        outcome["input"]["candidate"] = step["action"]
        streams["outcomes"].append(outcome)
        streams["transitions"].append(
            _training_row(step, meta, {"action": step["action"], "checker": receipt}, "transitions"),
        )
        if receipt["new_fact"] is not None:
            streams["sft"].append(_training_row(step, meta, {"action": step["action"]}, "sft"))
            if failed_before:
                streams["repairs"].append(_training_row(step, meta, {"action": step["action"]}, "repairs"))
        failed_before |= not receipt["accepted"]
    return streams


def lean_source(task: dict[str, Any], trace: dict[str, Any], name: str) -> str:
    """Translate replayed derivation steps to constructive Lean terms using Init."""
    replay_trace(task, trace)
    if not trace["solved"] or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None:
        raise ValueError("solved trace and safe theorem name required")
    world = task["world"]
    params = ["(" + " ".join(f"p{x}" for x in world["atoms"]) + " : Prop)"]
    params.extend(f"(f{x} : p{x})" for x in world["facts"])
    for index, rule in enumerate(world["rules"]):
        proposition = " → ".join([*(f"p{x}" for x in rule["premises"]), f"p{rule['conclusion']}"])
        params.append(f"(r{index} : {proposition})")
    lines = [f"theorem {name} {' '.join(params)} : p{world['goal']} := by"]
    terms = {x: f"f{x}" for x in world["facts"]}
    for step in trace["steps"]:
        receipt = step["verification"]
        if receipt["new_fact"] is not None:
            index = step["action"]
            rule = world["rules"][index]
            atom = rule["conclusion"]
            lines.append(f"  have h{atom} : p{atom} := r{index} " + " ".join(terms[x] for x in rule["premises"]))
            terms[atom] = f"h{atom}"
    lines.extend([f"  exact {terms[world['goal']]}", f"#print axioms {name}"])
    return "\n".join(lines) + "\n"


def native_replay(
    tasks: list[dict[str, Any]], traces: list[dict[str, Any]], executable: Path, out: Path,
) -> dict[str, Any]:
    """Compile complete proofs, audit axiom output, then reject a wrong-term sentinel."""
    version = subprocess.run([str(executable), "--version"], check=True, capture_output=True,
                             text=True, timeout=15).stdout.strip()
    source = out / "certificates.lean"
    names = [f"cogCheck{i}" for i in range(len(tasks))]
    source.write_text("import Init\n\n" + "\n".join(
        lean_source(task, trace, name) for task, trace, name in zip(tasks, traces, names, strict=True)
    ), encoding="utf-8")
    compiled = subprocess.run([str(executable), str(source)], capture_output=True, text=True, timeout=120)
    output = compiled.stdout + compiled.stderr
    log = out / "lean_stdout.txt"
    log.write_text(output, encoding="utf-8")
    audited = {m.group(1) for m in re.finditer(r"'(cogCheck\d+)' does not depend on any axioms", output)}
    negative = out / "rejected_sentinel.lean"
    negative.write_text("import Init\ntheorem cogRejected (P Q : Prop) (h : P) : Q := h\n", encoding="utf-8")
    rejected = subprocess.run([str(executable), str(negative)], capture_output=True, text=True, timeout=15)
    passed = compiled.returncode == 0 and audited == set(names) and rejected.returncode != 0
    return {"passed": passed, "kind": "Lean4_full_proof_replay", "version": version,
            "lean_executable_sha256": file_digest(executable), "source_sha256": file_digest(source),
            "stdout_sha256": file_digest(log), "compiled_proofs": len(audited),
            "expected_proofs": len(tasks), "axioms": [], "negative_sentinel_rejected": rejected.returncode != 0,
            "negative_source_sha256": file_digest(negative), "compiler_exit_code": compiled.returncode,
            "cost": {"positive_batch_processes": 1, "negative_processes": 1}}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")


def build(out: Path, per_split: int = 64, seed: int = 20261126, lean: Path | None = None) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    tasks = generate_tasks(per_split, seed)
    traces = [collect_trace(task) for task in tasks]
    streams: dict[str, list[dict[str, Any]]] = {name: [] for name in ("sft", "repairs", "outcomes", "transitions")}
    ranking = []
    codes: Counter[str] = Counter()
    for task, trace in zip(tasks, traces, strict=True):
        replay_trace(task, trace)
        seen = set()
        for step in trace["steps"]:
            codes[step["verification"]["code"]] += 1
            known = step["observation"]["known"]
            token = tuple(known)
            if token not in seen:
                ranking.append(ranking_row(task, known))
                seen.add(token)
        for key, rows in export_streams(task, trace).items():
            streams[key].extend(rows)
    rows_by_file = {"tasks": tasks, "traces": traces, "ranking": ranking, **streams}
    for name, rows in rows_by_file.items():
        _write_jsonl(out / f"{name}.jsonl", rows)
    files = {name + ".jsonl": {"sha256": file_digest(out / f"{name}.jsonl"), "rows": len(rows)}
             for name, rows in rows_by_file.items()}
    receipt = native_replay(tasks, traces, lean, out) if lean else {"passed": False, "status": "not_run"}
    manifest = {
        "schema_version": SCHEMA, "generator_sha256": file_digest(Path(__file__)),
        "public_seed": seed, "per_split": per_split, "tasks": len(tasks),
        "splits": dict(Counter(t["split"] for t in tasks)),
        "families": dict(Counter(t["family"] for t in tasks)), "feature_names": list(FEATURE_NAMES),
        "max_atoms": max(len(t["world"]["atoms"]) for t in tasks),
        "max_rules": max(len(t["world"]["rules"]) for t in tasks),
        "all_teacher_traces_solved": all(t["solved"] for t in traces),
        "exact_replayed_steps": sum(t["verifier_calls"] for t in traces),
        "outcome_counts": dict(codes),
        "ranking_collection_verifier_calls": sum(r["collection_cost"]["verifier_calls"] for r in ranking),
        "root_overlap_across_splits": 0, "native_replay": receipt, "files": files,
        "evidence_scope": "synthetic finite Horn proof-search infrastructure; public development splits",
        "source_paper": "https://github.com/google-research/google-research/tree/4a7bc8ab0bc7/cogentic_research_papers",
        "source_paper_use": "research motivation only; zero paper-derived proof traces",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--lean", type=Path)
    args = parser.parse_args()
    result = build(args.out, args.per_split, args.seed, args.lean)
    selected = ("tasks", "exact_replayed_steps", "outcome_counts", "native_replay")
    print(json.dumps({k: result[k] for k in selected}, indent=2))
    if args.lean and not result["native_replay"]["passed"]:
        raise SystemExit("native Lean replay failed")


if __name__ == "__main__":
    main()
