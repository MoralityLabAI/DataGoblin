"""Build a split-safe MeTTa curation SFT / recurrent correction curriculum.

This is a *synthetic, exact Python-oracle* corpus. The model sees declarative
MeTTa facts, a current evidence selection and an action contract. Targets
teach a canonical bounded repair-and-stop policy. The labels do not claim to
be model activations, native Hyperon execution, or independently admitted
training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .collect import ROOT, sha256_bytes
from .sft_oracle import (
    FAMILIES,
    OOD_FAMILY,
    edit,
    exact_optimum,
    first_state,
    make_task,
    next_edit,
    replay,
)

SCHEMA = "metta-looped-sft-v1"
GENERATED_BASE = ROOT / "runs" / "metta" / "looped_sft"


def canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def split_for(family: str, seed: int) -> str:
    """Source-family OOD split; per-root hashed train/development/test splits."""
    if family == OOD_FAMILY:
        return "ood_reserved"
    sample = int(hashlib.sha256(f"metta-curation-split-v1/{family}/{seed}".encode()).hexdigest()[:8], 16) % 100
    if sample < 76:
        return "train_candidate"
    return "development_reserved" if sample < 88 else "test_reserved"


def root_id(family: str, seed: int) -> str:
    return f"metta-sft-v1:{family}:{seed:06d}"


def prompt(task: dict[str, Any], selected: list[str]) -> dict[str, Any]:
    return {
        "objective": (
            "Curate a small evidence set from the MeTTa fact store. Maximize goal coverage, "
            "then verified quality, then minimize token cost. Respect eligibility, "
            "lineage, dependencies, revisions, conflicts and selection budgets. "
            "Apply one edit toward the certified optimum. Stop at the fixed point."
        ),
        "language": "MeTTa",
        "program": task["program"],
        "current_selected": list(selected),
        "action_format": {"op": "add|remove|stop", "id": "record_id|none"},
        "verification_scope": "bounded synthetic collection problem",
    }


def reject_action(selected: list[str], chosen: dict[str, str], task: dict[str, Any]) -> dict[str, str]:
    if chosen["op"] != "stop":
        return {"op": "stop", "id": "none"}
    if selected:
        return {"op": "remove", "id": selected[0]}
    return {"op": "add", "id": task["items"][0]["id"]}


def selected_views(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # One root contributes at most four *distinct* presentations across all
    # supervision heads, including STOP; no separate duplicate preference file.
    if not trace:
        raise ValueError("empty curation trace")
    stops = (0, min(1, len(trace) - 1), max(0, len(trace) - 2), len(trace) - 1)
    return [trace[index] for index in dict.fromkeys(stops)]


def query_supervision(task: dict[str, Any], step_number: int) -> dict[str, Any]:
    """Native-shaped MeTTa match query with an independently computed static target."""
    skill = task["goals"][step_number % len(task["goals"])]
    query = (
        f"!(match &self (, (Provides $record {skill}) "
        "(Eligible $record True)) $record)"
    )
    matches = sorted(
        item["id"] for item in task["items"]
        if item["eligible"] and skill in item["topics"]
    )
    return {
        "query": query,
        "expected_record_ids": matches,
        "calculated_from": "source_structured_facts",
        "verified_native_metta": False,
    }


def checked_training_row(
    task: dict[str, Any],
    root: str,
    step: dict[str, Any],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    selected = step["selected"]
    chosen = step["action"]
    if chosen != next_edit(selected, oracle["selected"]):
        raise ValueError("teacher edit does not match canonical next action")
    wrong = reject_action(selected, chosen, task)
    if wrong == chosen:
        raise ValueError("preference candidate duplicates teacher edit")
    next_selected = edit(selected, chosen)
    expected_distance = len(set(selected) ^ set(oracle["selected"]))
    next_distance = len(set(next_selected) ^ set(oracle["selected"]))
    if next_distance != max(0, expected_distance - 1):
        raise ValueError("oracle action fails to decrease canonical correction distance")
    action = {"op": chosen["op"], "id": chosen["id"]}
    user = prompt(task, selected)
    return {
        "schema_version": SCHEMA,
        "root": root,
        "presentation": step["step"],
        "split": "train_candidate",
        "family": task["family"],
        "model_generated": True,
        "generator": "MeTTa.looped_sft.v1",
        "teacher": "exhaustive_python_subset_oracle",
        "evidence_type": "SIMULATED",
        "training_ready": False,
        "verified_native_metta": False,
        "messages": [
            {"role": "user", "content": canon(user)},
            {"role": "assistant", "content": canon(action)},
        ],
        "objective": "SFT_NEXT_EDIT_AND_STOP",
        "metta_query_supervision": query_supervision(task, step["step"]),
        "preference": {"chosen": action, "rejected": wrong},
        "recurrent_supervision": {
            "visible_step": step["step"],
            "selected_before": selected,
            "selected_after": next_selected,
            "canonical_edits_remaining_before": expected_distance,
            "canonical_edits_remaining_after": next_distance,
            "stop_target": chosen["op"] == "stop",
            "synthetic_trace": True,
        },
        "verifier_receipt": {
            "method": oracle["method"],
            "checked_subsets": oracle["checked_subsets"],
            "feasible_subsets": oracle["feasible_subsets"],
            "optimal_subset": oracle["selected"],
            "optimal_rank": oracle["value"],
            "target_matches_first_canonical_edit": True,
        },
        "curriculum": {
            "num_candidates": len(task["items"]),
            "num_goals": len(task["goals"]),
            "num_dependencies": sum(len(item["requires"]) for item in task["items"]),
            "num_conflicts": len(task["conflicts"]),
            "structural_depth": expected_distance,
            "remaining_loop_budget": max(0, 2 * len(task["items"]) - step["step"]),
            "difficulty_definition": "synthetic-canonical-edit-distance",
        },
    }


def heldout_record(task: dict[str, Any], root: str, initial: list[str], split: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "root": root,
        "split": split,
        "family": task["family"],
        "presentation": 0,
        "model_input": prompt(task, initial),
        "teacher_labels_included": False,
        "training_ready": False,
        "source_kind": "synthetic_symbolic_world",
    }


def make_bundle(out: Path, *, roots_per_family: int = 512, families: tuple[str, ...] = FAMILIES) -> dict[str, Any]:
    if roots_per_family < 1 or roots_per_family > 100_000:
        raise ValueError("roots_per_family out of bounds")
    if not families or len(set(families)) != len(families) or any(f not in FAMILIES for f in families):
        raise ValueError("invalid family selection")
    if out.resolve().is_relative_to((ROOT / "MeTTa").resolve()) or out.resolve().is_relative_to(
        (ROOT / "training").resolve()
    ):
        raise ValueError("generated training candidates must remain outside committed source and training plans")
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train_candidates.jsonl"
    reserved_path = out / "reserved_public.jsonl"
    split_counts: Counter[str] = Counter()
    sft_counts: Counter[str] = Counter()
    seen_roots = set()
    root_presentations: Counter[str] = Counter()
    observed_depths = []
    with train_path.open("w", encoding="utf-8") as train_out, reserved_path.open("w", encoding="utf-8") as heldout:
        for family in families:
            for seed in range(roots_per_family):
                root = root_id(family, seed)
                if root in seen_roots:
                    raise ValueError("duplicate synthetic root")
                seen_roots.add(root)
                split = split_for(family, seed)
                split_counts[split] += 1
                task = make_task(family, seed)
                # Public initial state is independent of the oracle solution.
                initial = first_state(task, seed)
                if split != "train_candidate":
                    heldout.write(canon(heldout_record(task, root, initial, split)) + "\n")
                    continue
                oracle = exact_optimum(task)
                trace = replay(task, initial, oracle["selected"])
                views = selected_views(trace)
                if not 1 <= len(views) <= 4:
                    raise ValueError("invalid presentation count")
                observed_depths.append(len(trace) - 1)
                for step in views:
                    row = checked_training_row(task, root, step, oracle)
                    train_out.write(canon(row) + "\n")
                    root_presentations[root] += 1
                    sft_counts[family] += 1
                    sft_counts["stop"] += int(step["action"]["op"] == "stop")
                    sft_counts["continue"] += int(step["action"]["op"] != "stop")
    if not root_presentations or any(n > 4 for n in root_presentations.values()):
        raise ValueError("invalid four-presentation cap")
    train_sha = sha256_bytes(train_path.read_bytes())
    heldout_sha = sha256_bytes(reserved_path.read_bytes())
    manifest = {
        "schema_version": "metta-looped-sft-bundle-v1",
        "generator_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "oracle_sha256": sha256_bytes((Path(__file__).parent / "sft_oracle.py").read_bytes()),
        "roots_per_family": roots_per_family,
        "families": list(families),
        "root_count": len(seen_roots),
        "split_roots": dict(sorted(split_counts.items())),
        "training_presentations": sum(root_presentations.values()),
        "train_rows_by_family": {f: sft_counts[f] for f in families},
        "continue_targets": sft_counts["continue"],
        "stop_targets": sft_counts["stop"],
        "mean_canonical_edits_to_fixed_point": (
            round(sum(observed_depths) / len(observed_depths), 3) if observed_depths else None
        ),
        "max_presentations_per_root": max(root_presentations.values()),
        "files": {
            "train_candidates.jsonl": train_sha,
            "reserved_public.jsonl": heldout_sha,
        },
        "policy": {
            "model_supervision": "synthetic_checked_policy",
            "native_metta_verified": False,
            "training_ready": False,
            "eval_labels_written": 0,
            "evaluation_custody": "reserved_public_unlabeled_not_sealed",
            "verifier_authority": "bounded_exhaustive_python_oracle",
            "no_architecture_gain_claimed": True,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def audit_bundle(out: Path, *, full_replay: bool = True) -> dict[str, Any]:
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "metta-looped-sft-bundle-v1":
        raise ValueError("unsupported manifest")
    for name, expected in manifest["files"].items():
        path = out / name
        if sha256_bytes(path.read_bytes()) != expected:
            raise ValueError(f"changed artifact: {name}")
    seen: dict[str, set[int]] = {}
    split_roots: dict[str, str] = {}
    count = 0
    stop = 0
    root_cache: dict[str, tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]]]] = {}
    for line in (out / "train_candidates.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        root = row["root"]
        if row["training_ready"] is not False or row["split"] != "train_candidate":
            raise ValueError("unexpected released or split row in training")
        if row["presentation"] in seen.setdefault(root, set()):
            raise ValueError("duplicate root presentation")
        seen[root].add(row["presentation"])
        if len(seen[root]) > 4:
            raise ValueError("root presentation cap exceeded")
        if split_roots.setdefault(root, row["split"]) != row["split"]:
            raise ValueError("root crosses splits")
        if full_replay:
            if root not in root_cache:
                family, seed = root.split(":")[1:]
                task = make_task(family, int(seed))
                oracle = exact_optimum(task)
                trace_by_step = {
                    event["step"]: event for event in replay(
                        task, first_state(task, int(seed)), oracle["selected"]
                    )
                }
                root_cache[root] = (task, oracle, trace_by_step)
            task, oracle, trace_by_step = root_cache[root]
            family, seed = root.split(":")[1:]
            if split_for(family, int(seed)) != "train_candidate":
                raise ValueError("train row belongs to a reserved root")
            allowed = {event["step"] for event in selected_views(list(trace_by_step.values()))}
            if row["presentation"] not in allowed:
                raise ValueError("unregistered training presentation")
            p = json.loads(row["messages"][0]["content"])
            selected = p["current_selected"]
            if p != prompt(task, selected):
                raise ValueError("prompt or synthetic world drift")
            chosen = json.loads(row["messages"][1]["content"])
            if chosen != row["preference"]["chosen"]:
                raise ValueError("preference target differs from SFT completion")
            event = trace_by_step.get(row["presentation"])
            if event is None or event["selected"] != selected or event["action"] != chosen:
                raise ValueError("training presentation differs from canonical trajectory")
            rebuilt = checked_training_row(task, root, event, oracle)
            if row != rebuilt:
                raise ValueError("oracle target / receipt or recurrent labels differ on replay")
        count += 1
        stop += int(row["recurrent_supervision"]["stop_target"])
    reserved = 0
    for line in (out / "reserved_public.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        root = row["root"]
        if row["split"] not in ("development_reserved", "test_reserved", "ood_reserved"):
            raise ValueError("train or unknown row in reserved public")
        if any(key in row for key in ("messages", "preference", "verifier_receipt", "target", "oracle")):
            raise ValueError("heldout target leakage")
        if split_roots.setdefault(root, row["split"]) != row["split"]:
            raise ValueError("root crosses heldout and training")
        if full_replay:
            family, seed = root.split(":")[1:]
            task = make_task(family, int(seed))
            if row["split"] != split_for(family, int(seed)):
                raise ValueError("reserved root partition differs from frozen split rule")
            if row != heldout_record(task, root, first_state(task, int(seed)), row["split"]):
                raise ValueError("reserved task drift")
        reserved += 1
    if count != manifest["training_presentations"] or stop != manifest["stop_targets"]:
        raise ValueError("training count / stop targets differ")
    if sum(manifest["split_roots"].values()) != len(split_roots):
        raise ValueError("split root registry incomplete")
    if reserved != sum(n for s, n in manifest["split_roots"].items() if s != "train_candidate"):
        raise ValueError("reserved root count differs")
    return {
        "valid": True,
        "training_presentations": count,
        "train_roots": len(seen),
        "reserved_roots": reserved,
        "stop_targets": stop,
        "replayed": full_replay,
        "training_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--out", type=Path, default=GENERATED_BASE)
    build.add_argument("--roots-per-family", type=int, default=512)
    build.add_argument("--family", action="append", choices=FAMILIES)
    audit = sub.add_parser("audit")
    audit.add_argument("--out", type=Path, default=GENERATED_BASE)
    audit.add_argument("--fast", action="store_true", help="hash and custody checks without full oracle replay")
    args = parser.parse_args()
    if args.command == "build":
        result = make_bundle(args.out, roots_per_family=args.roots_per_family,
                             families=tuple(args.family) if args.family else FAMILIES)
    else:
        result = audit_bundle(args.out, full_replay=not args.fast)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
