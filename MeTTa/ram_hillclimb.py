"""RAM-only MeTTa-guided hill climb over tiny TRM and typed-LDT proxy students.

The weights genuinely fit on CPU using the existing NumPy AIRIS-02 learner.
A typed LDT proxy trains a linear proposal network and routes its edits through
the environment-owned constraint verifier; it is not a native neural LDT.
GPT proposals are explicit JSONL imports. This runner never claims to have
called a frontier model or to have trained a full transformer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from .airis02_student import (
    LinearStudent,
    RecurrentStudent,
    _matrix,
    _next_action,
    evaluate,
)
from .airis02_stop import StopHead
from .sft_oracle import (
    edit,
    evaluate_subset,
    exact_optimum,
    first_state,
    make_task,
    next_edit,
)

SCHEMA = "metta-ram-hillclimb-v1"
ROOT_NAMESPACE = "metta-ram10-v1"
TRAIN_START = 20000
DEV_START = 30000
PUBLIC_TEST_START = 40000
FAMILIES = ("lineage", "coverage", "dependency", "revision", "budget")
OOD_FAMILY = "conflict"
ARMS = ("trm", "ldt_proxy")
DEFAULT_OUT = Path("runs/metta/ram_hillclimb")
OPTIONS: dict[str, dict[str, tuple[Any, ...]]] = {
    "trm": {
        "steps": (12, 16, 24, 32),
        "lr": (0.04, 0.08, 0.12, 0.20),
        "hidden": (8, 12, 16, 24),
        "train_loops": (2, 4, 6),
        "inference_loops": (2, 4, 6, 8),
        "stop_policy": ("action_class", "train_calibrated"),
    },
    "ldt_proxy": {
        "steps": (12, 16, 24, 32),
        "lr": (0.04, 0.08, 0.12, 0.20),
        "stop_policy": ("action_class", "train_calibrated"),
    },
}
DEFAULTS: dict[str, dict[str, Any]] = {
    "trm": {
        "steps": 16, "lr": 0.20, "hidden": 12, "train_loops": 4,
        "inference_loops": 4, "stop_policy": "action_class",
    },
    "ldt_proxy": {"steps": 16, "lr": 0.20, "stop_policy": "action_class"},
}
DEFAULT_PROPOSALS = (
    {"arm": "trm", "field": "stop_policy", "value": "train_calibrated"},
    {"arm": "ldt_proxy", "field": "stop_policy", "value": "train_calibrated"},
    {"arm": "trm", "field": "lr", "value": 0.08},
    {"arm": "ldt_proxy", "field": "lr", "value": 0.08},
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def root_name(family: str, seed: int) -> str:
    if family not in (*FAMILIES, OOD_FAMILY) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Invalid synthetic source root")
    return f"{ROOT_NAMESPACE}:{family}:{seed:06d}"


def training_views(task: dict[str, Any], seed: int) -> list[list[str]]:
    """At most four distinct selected-set states from one synthetic source root."""
    optimum = list(exact_optimum(task)["selected"])
    state = list(first_state(task, seed))
    views = [state.copy()]
    for _ in range(2):
        action = next_edit(state, optimum)
        if action["op"] == "stop":
            break
        state = edit(state, action)
        if state not in views:
            views.append(list(state))
    if optimum not in views:
        views.append(optimum)
    if not 1 <= len(views) <= 4:
        raise ValueError("Training root exceeded four-view quota")
    return views


def make_row(family: str, seed: int, state: list[str], partition: str, index: int) -> dict[str, Any]:
    task = make_task(family, seed)
    optimum = exact_optimum(task)["selected"]
    label = next_edit(state, optimum)
    return {
        "schema_version": "metta-ram10-synthetic-root-v1",
        "root": root_name(family, seed),
        "family": family,
        "seed": seed,
        "split": partition,
        "presentation": index,
        "public_world": task,
        "selected": list(state),
        "oracle_next_edit": label,
        "source_sha256": digest(task["program"].encode("utf-8")),
        "oracle_method": "exhaustive_independent_bounded_subset",
        "label_exposed_to_student": partition == "train_candidate",
        "training_ready": False,
        "synthetic_research_fit_eligible": partition == "train_candidate",
    }


def make_bundle(train_roots: int, dev_roots: int) -> dict[str, Any]:
    if not 1 <= train_roots <= 24 or not 1 <= dev_roots <= 8:
        raise ValueError("Root counts exceed bounded CPU study")
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    for family in FAMILIES:
        for seed in range(TRAIN_START, TRAIN_START + train_roots):
            task = make_task(family, seed)
            for index, state in enumerate(training_views(task, seed)):
                train.append(make_row(family, seed, state, "train_candidate", index))
        for seed in range(DEV_START, DEV_START + dev_roots):
            task = make_task(family, seed)
            dev.append(make_row(family, seed, first_state(task, seed), "public_development", 0))
    train_ids = {row["root"] for row in train}
    dev_ids = {row["root"] for row in dev}
    public_test_roots = [
        root_name(family, seed)
        for family in (*FAMILIES, OOD_FAMILY)
        for seed in range(PUBLIC_TEST_START, PUBLIC_TEST_START + dev_roots)
    ]
    if train_ids & dev_ids or (train_ids | dev_ids) & set(public_test_roots):
        raise ValueError("Root split collision")
    counts = Counter(row["root"] for row in train)
    if any(number > 4 for number in counts.values()):
        raise ValueError("Source root presentation cap breached")
    for row in [*train, *dev]:
        regenerated = make_task(row["family"], row["seed"])
        if row["public_world"] != regenerated:
            raise ValueError("Synthetic source world changed")
        oracle = exact_optimum(regenerated)
        if next_edit(row["selected"], oracle["selected"]) != row["oracle_next_edit"]:
            raise ValueError("Teacher repair target failed independent replay")
        if row["training_ready"] is not False:
            raise ValueError("Synthetic research root was promoted")
    return {
        "train": train,
        "dev": dev,
        "public_test_root_names_only": public_test_roots,
        "train_roots_sha256": digest(sorted(train_ids)),
        "development_roots_sha256": digest(sorted(dev_ids)),
        "public_test_root_registry_sha256": digest(public_test_roots),
        "train_presentations": len(train),
        "dev_cases": len(dev),
        "train_source_roots": len(train_ids),
        "dev_source_roots": len(dev_ids),
        "max_presentations_per_root": max(counts.values()),
    }


def check_config(arm: str, config: dict[str, Any]) -> None:
    if arm not in ARMS or set(config) != set(OPTIONS[arm]):
        raise ValueError("Unsupported architecture or training config")
    for field, value in config.items():
        if field == "stop_policy":
            valid_type = type(value) is str
        else:
            valid_type = type(value) in (float, int)
        if not valid_type or value not in OPTIONS[arm][field]:
            raise ValueError(f"Configuration field {field} outside registered values")


def validate_proposal(proposal: dict[str, Any], current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(proposal, dict):
        raise ValueError("Teacher proposal must be an object")
    if set(proposal) - {"arm", "field", "value", "source", "model_id"}:
        raise ValueError("Unexpected proposal keys")
    arm, field, value = proposal.get("arm"), proposal.get("field"), proposal.get("value")
    if not isinstance(arm, str) or arm not in ARMS or not isinstance(field, str):
        raise ValueError("Unknown architecture or edit field")
    if field not in OPTIONS[arm]:
        raise ValueError("Unsupported edit")
    allowed_type = type(value) is str if field == "stop_policy" else type(value) in (int, float)
    if not allowed_type or value not in OPTIONS[arm][field]:
        raise ValueError("Edit value outside bounded search grammar")
    if current[arm][field] == value:
        raise ValueError("No-op edit is not a hill-climb proposal")
    for name in ("source", "model_id"):
        label = proposal.get(name, "not_reported")
        if not isinstance(label, str) or not 1 <= len(label) <= 80:
            raise ValueError("Invalid or missing teacher provenance label")
    updated = dict(current[arm])
    updated[field] = value
    check_config(arm, updated)
    return updated


def _metta_atom(value: Any) -> str:
    if isinstance(value, float):
        return "v" + str(value).replace(".", "_")
    if isinstance(value, int):
        return f"n{value}"
    if isinstance(value, str) and value.replace("_", "").isalnum():
        return value
    raise ValueError("Unsafe MeTTa atom")


def metta_proposal(proposal: dict[str, Any], ram_limit_mib: int) -> str:
    """Ground facts plus a native match query, never executable user code."""
    arm, field = proposal["arm"], proposal["field"]
    value = _metta_atom(proposal["value"])
    if arm not in ARMS or field not in OPTIONS[arm]:
        raise ValueError("Unregistered proposal")
    return "\n".join([
        "; MeTTa-10 proposal snapshot, research-only data",
        f"(RamCeilingMiB n{ram_limit_mib})",
        "(SourceKind synthetic_research)",
        f"(Candidate {arm} {field} {value})",
        "!(match &self (Candidate $arm $field $value) (Proposal $arm $field $value))",
        "",
    ])


def peak_rss_mib() -> float:
    if resource is None:
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError(
                "Windows needs psutil for RSS monitoring (pip install psutil), "
                "or run under WSL with a Linux hard memory cap"
            ) from exc
        return round(psutil.Process().memory_info().rss / 1048576, 2)
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(maximum / (1024 * 1024 if sys.platform == "darwin" else 1024), 2)


def install_memory_ceiling(ram_limit_mib: int) -> dict[str, Any]:
    if not 512 <= ram_limit_mib <= 12288:
        raise ValueError("RAM limit must be between 512 MiB and 12 GiB")
    maximum_bytes = ram_limit_mib * 1024 * 1024
    enabled = False
    if sys.platform.startswith("linux") and resource is not None:
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_soft = min(maximum_bytes, hard) if hard != resource.RLIM_INFINITY else maximum_bytes
        if new_soft < maximum_bytes:
            raise ValueError("Existing process hard limit is below the requested RAM ceiling")
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
        enabled = True
    return {
        "ram_limit_mib": ram_limit_mib,
        "linux_hard_virtual_address_space_limit": enabled,
        "other_platform_memory_check": not enabled,
        "gpu_used": False,
    }


def select_key(metrics: dict[str, Any]) -> tuple[int, ...]:
    return (
        metrics["solved_and_stopped"],
        -metrics["new_violation_additions"],
        -metrics["unsafe_additions"],
        metrics["exact_first_edit"],
        -metrics["premature_stops"],
        -metrics["verifier_queries"],
        -round(metrics["mean_remaining_canonical_edits"] * 1000),
        -round(metrics["mean_inference_iterations"] * 1000),
    )


def fit_candidate(
    arm: str,
    config: dict[str, Any],
    train: list[dict[str, Any]],
    dev: list[dict[str, Any]],
    *,
    seed: int,
    ram_limit_mib: int,
) -> tuple[LinearStudent | RecurrentStudent, dict[str, Any]]:
    check_config(arm, config)
    if not train or not dev or {x["root"] for x in train} & {x["root"] for x in dev}:
        raise ValueError("Invalid or overlapping fit/evaluation roots")
    x, y, masks = _matrix(train)
    start = time.monotonic()
    if arm == "trm":
        model: LinearStudent | RecurrentStudent = RecurrentStudent(seed, hidden=config["hidden"])
        history = model.fit(
            x, y, legal_masks=masks, steps=config["steps"],
            lr=config["lr"], train_loops=config["train_loops"],
        )
        loops = config["inference_loops"]
        gated = False
        model_scope = "weight_tied_tanh_trm_style_numpy_student"
    else:
        model = LinearStudent(seed)
        history = model.fit(
            x, y, legal_masks=masks, steps=config["steps"], lr=config["lr"],
        )
        loops = 1
        gated = True
        model_scope = "linear_learner_plus_host_verified_lattice_refinement_proxy"
    elapsed_ms = int((time.monotonic() - start) * 1000)
    family_metrics = {
        family: evaluate(
            model, [row for row in dev if row["family"] == family],
            loops=loops, verifier_guard=gated,
        )
        for family in FAMILIES
    }
    additive = (
        "solved_and_stopped", "new_violation_additions", "unsafe_additions",
        "exact_first_edit", "premature_stops", "verifier_queries",
        "abstentions", "verifier_vetoes",
    )
    aggregate: dict[str, Any] = {
        field: sum(value[field] for value in family_metrics.values())
        for field in additive
    }
    aggregate["cases"] = len(dev)
    for field in ("mean_remaining_canonical_edits", "mean_inference_iterations"):
        aggregate[field] = round(
            sum(family_metrics[f][field] * family_metrics[f]["cases"] for f in FAMILIES)
            / len(dev), 5,
        )
    max_rss = peak_rss_mib()
    if max_rss > ram_limit_mib:
        raise MemoryError("CPU research gym exceeded the registered RSS ceiling")
    result = {
        "config": dict(config),
        "model_scope": model_scope,
        "model_trained": True,
        "real_cpu_gradient_updates": config["steps"],
        "training_examples": len(train),
        "parameter_count": model.params,
        "checkpoint_sha256": model.digest(),
        "final_training_loss": history[-1],
        "training_wall_ms": elapsed_ms,
        "peak_process_rss_mib": max_rss,
        "family_metrics": family_metrics,
        "dev_aggregate": aggregate,
        "selection_key": list(select_key(aggregate)),
        "native_metta_training": False,
        "public_synthetic_only": True,
        "training_ready": False,
    }
    return model, result


def read_proposals(path: Path | None, limit: int) -> list[dict[str, Any]]:
    if not 1 <= limit <= 8:
        raise ValueError("Teacher edits must be bounded at 1..8")
    if path is None:
        return [
            {**proposal, "source": "deterministic_control", "model_id": "none"}
            for proposal in DEFAULT_PROPOSALS[:limit]
        ]
    if path.stat().st_size > 16384:
        raise ValueError("Teacher proposal file exceeds byte budget")
    proposals: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("Teacher proposal JSONL expects objects")
            proposals.append(item)
    if not proposals or len(proposals) > limit:
        raise ValueError("Teacher proposal count exceeds registered budget")
    return proposals


def export_repair_traces(
    model: LinearStudent | RecurrentStudent,
    arm: str,
    config: dict[str, Any],
    train: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One replayed train-root presentation, including a checked wrong action."""
    result: list[dict[str, Any]] = []
    loops = config["inference_loops"] if arm == "trm" else 1
    for row in train:
        task = row["public_world"]
        proposed, vetoed, queries = _next_action(
            model, task, row["selected"], loops=loops, stop_head=None,
            verifier_guard=(arm == "ldt_proxy"),
        )
        expected = row["oracle_next_edit"]
        allowed = proposed["op"] in ("add", "remove", "stop")
        valid = False
        introduced_violations = 0
        if proposed["op"] == "stop":
            valid = proposed == expected
        elif allowed:
            before = set(evaluate_subset(task, row["selected"])["violations"])
            new_state = edit(row["selected"], proposed)
            after = set(evaluate_subset(task, new_state)["violations"])
            introduced_violations = len(after - before)
            valid = not introduced_violations
        record = {
            "schema_version": "metta-ram10-checked-repair-trace-v1",
            "root": row["root"],
            "split": "train_candidate",
            "family": row["family"],
            "presentation": row["presentation"],
            "metta_program": task["program"],
            "source_sha256": row["source_sha256"],
            "observation": {"selected": row["selected"]},
            "model_proposal": proposed,
            "verified_canonical_next_edit": expected,
            "exact_teacher_match": proposed == expected,
            "proposed_transition_introduced_violations": introduced_violations,
            "proposed_transition_public_constraint_valid": valid,
            "host_verifier_vetoes": vetoed,
            "host_verifier_calls": queries,
            "reward": 1 if proposed == expected else -1 if introduced_violations else 0,
            "chosen": expected,
            "rejected": proposed if proposed != expected else None,
            "oracle": "bounded_synthetic_subset_exact_optimum",
            "model_kind": arm,
            "model_checkpoint_sha256": model.digest(),
            "data_source": "fresh_generator_owned_synthetic",
            "teacher_source": "independent_python_oracle",
            "synthetic_research_fit_eligible": True,
            "training_ready": False,
        }
        result.append(record)
    if len(result) != len(train):
        raise ValueError("Per-root exposure count changed during trace export")
    return result


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def run_gym(
    out: Path,
    *,
    train_roots: int = 8,
    dev_roots: int = 3,
    proposals_path: Path | None = None,
    max_proposals: int = 4,
    ram_limit_mib: int = 8192,
    seed: int = 20261122,
) -> dict[str, Any]:
    memory = install_memory_ceiling(ram_limit_mib)
    bundle = make_bundle(train_roots, dev_roots)
    current = {arm: dict(DEFAULTS[arm]) for arm in ARMS}
    models: dict[str, LinearStudent | RecurrentStudent] = {}
    best: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        models[arm], best[arm] = fit_candidate(
            arm, current[arm], bundle["train"], bundle["dev"],
            seed=seed, ram_limit_mib=ram_limit_mib,
        )
    history: list[dict[str, Any]] = []
    proposals = read_proposals(proposals_path, max_proposals)
    for turn, proposal in enumerate(proposals):
        arm = proposal.get("arm")
        if arm not in ARMS:
            raise ValueError("Unknown teacher-proposed arm")
        prior = best[arm]
        updated = validate_proposal(proposal, current)
        source = metta_proposal(proposal, ram_limit_mib)
        candidate_model, candidate = fit_candidate(
            arm, updated, bundle["train"], bundle["dev"],
            seed=seed, ram_limit_mib=ram_limit_mib,
        )
        previous_metrics = prior["dev_aggregate"]
        new_metrics = candidate["dev_aggregate"]
        safety_regression = (
            new_metrics["new_violation_additions"] > previous_metrics["new_violation_additions"]
            or new_metrics["unsafe_additions"] > previous_metrics["unsafe_additions"]
        )
        improved = select_key(new_metrics) > select_key(previous_metrics)
        accepted = improved and not safety_regression
        if accepted:
            current[arm] = updated
            best[arm] = candidate
            models[arm] = candidate_model
        history.append({
            "schema_version": "metta-ram10-hillclimb-step-v1",
            "turn": turn,
            "arm": arm,
            "teacher_proposal": proposal,
            "teacher_identity_unverified": proposal.get("model_id", "not_reported") != "none",
            "metta_candidate_program": source,
            "metta_candidate_sha256": digest(source.encode()),
            "incumbent_config": prior["config"],
            "incumbent_metrics": previous_metrics,
            "candidate_fit": candidate,
            "accepted": accepted,
            "decision": "reject_safety_regression" if safety_regression else (
                "accept_verified_development_gain" if accepted else "reject_no_gain"
            ),
            "verified_solved_delta": new_metrics["solved_and_stopped"] - previous_metrics["solved_and_stopped"],
            "development_selection_used": True,
            "public_test_roots_opened": False,
            "external_teacher_model_invoked": False,
            "training_ready": False,
        })
    champion = max(ARMS, key=lambda arm: (select_key(best[arm]["dev_aggregate"]), arm))
    repair = export_repair_traces(models[champion], champion, current[champion], bundle["train"])
    out.mkdir(parents=True, exist_ok=True)
    steps_text = "".join(canonical(row) + "\n" for row in history)
    repair_text = "".join(canonical(row) + "\n" for row in repair)
    (out / "hillclimb_steps.jsonl").write_text(steps_text, encoding="utf-8")
    (out / "train_repair_traces.jsonl").write_text(repair_text, encoding="utf-8")
    report = {
        "schema_version": SCHEMA,
        "model_trained_on_cpu": True,
        "gpu_required": False,
        "external_teacher_model_invoked": False,
        "teacher_mode": "imported_jsonl" if proposals_path else "deterministic_control",
        "teacher_model_identity_verified": False,
        "measured_source": "fresh_generator_owned_synthetic_only",
        "ram": memory,
        "observed_peak_rss_mib": peak_rss_mib(),
        "train_roots_sha256": bundle["train_roots_sha256"],
        "development_roots_sha256": bundle["development_roots_sha256"],
        "public_test_root_registry_sha256": bundle["public_test_root_registry_sha256"],
        "public_test_roots_materialized": 0,
        "ood_conflict_roots_materialized": 0,
        "max_source_presentations_per_root": bundle["max_presentations_per_root"],
        "train_cases": bundle["train_presentations"],
        "development_cases": bundle["dev_cases"],
        "training_arms": {arm: best[arm] for arm in ARMS},
        "baseline_and_search_candidates": len(ARMS) + len(history),
        "proposals_executed": len(history),
        "accepted_edits": sum(row["accepted"] for row in history),
        "champion_arm": champion,
        "champion_model_checkpoint_sha256": models[champion].digest(),
        "hillclimb_steps_sha256": digest(steps_text.encode()),
        "train_repair_traces_sha256": digest(repair_text.encode()),
        "train_repair_traces": len(repair),
        "independent_synthetic_oracle": True,
        "native_metta_query_verified": False,
        "training_ready": False,
        "synthetic_research_fit_eligible": True,
        "sealed_eval_eligible": False,
        "results_scope": "public development selected after repeated hillclimb observations",
        "model_comparison_warning": "TRM-style recurrent and LDT-proxy linear have unequal parameters",
        "next_teacher_prompt": {
            "instruction": (
                "Propose one JSON edit for the next CPU training trial. "
                "Restrict action to listed knobs; maximize verified solved-and-stopped "
                "on held-out public development while minimizing unsafe additions "
                "and verifier calls. Keep claims limited to measured results."
            ),
            "current_configs": current,
            "public_development_metrics": {
                arm: best[arm]["dev_aggregate"] for arm in ARMS
            },
            "action_grammar": {
                arm: {field: list(values) for field, values in OPTIONS[arm].items()}
                for arm in ARMS
            },
            "ram_limit_mib": ram_limit_mib,
            "labels_in_prompt": False,
        },
    }
    atomic_json(out / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-roots", type=int, default=8)
    parser.add_argument("--dev-roots", type=int, default=3)
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--max-proposals", type=int, default=4)
    parser.add_argument("--ram-limit-mib", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args()
    result = run_gym(
        args.out, train_roots=args.train_roots, dev_roots=args.dev_roots,
        proposals_path=args.proposals, max_proposals=args.max_proposals,
        ram_limit_mib=args.ram_limit_mib, seed=args.seed,
    )
    print(json.dumps({
        "training_arms": {
            arm: {
                "config": item["config"],
                "development": item["dev_aggregate"],
                "parameter_count": item["parameter_count"],
            } for arm, item in result["training_arms"].items()
        },
        "proposals_executed": result["proposals_executed"],
        "accepted_edits": result["accepted_edits"],
        "champion_arm": result["champion_arm"],
        "observed_peak_rss_mib": result["observed_peak_rss_mib"],
        "train_repair_traces": result["train_repair_traces"],
        "training_ready": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
