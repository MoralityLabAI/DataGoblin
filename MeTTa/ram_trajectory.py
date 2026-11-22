"""Exact synthetic repair trajectories and the environment-owned typed proposal check."""
from __future__ import annotations

import json
from pathlib import Path

from .airis02_student import action_index, decode_action, legal_action_mask
from .ram_hillclimb import canonical, digest, root_name
from .sft_oracle import edit, evaluate_subset, exact_optimum, first_state, make_task, replay

STOP = {"op": "stop", "id": "none"}
VIOLATIONS = (
    "item_limit", "token_budget", "ineligible", "repeated_lineage",
    "missing_dependency", "conflicting_sources", "stale_revision", "duplicate_revision_key",
)


class RepairVerifier:
    """Keep exact objectives private and count each check, including audit-only calls."""

    def __init__(self, world: dict):
        self.world = world
        self.checks = 0
        self.constraint_checks = 0
        self.optimum = exact_optimum(world)

    def check(self, action: dict, selected: list[str], *, gate: bool = False) -> dict:
        """A typed set-membership refinement of one open include/exclude decision."""
        self.checks += 1
        index = action_index(action)
        applicable = bool(legal_action_mask(self.world, selected)[index])
        proposed = edit(selected, action) if applicable else selected
        new_violations = []
        result_violations = []
        if applicable and action != STOP:
            prior = set(evaluate_subset(self.world, selected)["violations"])
            result_violations = evaluate_subset(self.world, proposed)["violations"]
            self.constraint_checks += 2
            new_violations = sorted(set(result_violations) - prior)
        unsafe_addition = action["op"] == "add" and bool(result_violations)
        accepted = applicable and (not gate or not (new_violations or unsafe_addition))
        target = self.optimum["selected"]
        after = proposed if accepted else selected
        veto = set(new_violations)
        if unsafe_addition:
            veto.update(result_violations)
        return {
            "method": "exact_python_curation_v1",
            "accepted": accepted,
            "structurally_applicable": applicable,
            "reason": (
                "inapplicable" if not applicable else "unsafe_addition" if gate and unsafe_addition
                else "new_public_violation" if not accepted else "accepted"
            ),
            "new_violations": new_violations,
            "result_violations": result_violations,
            "veto_violations": sorted(veto) if not accepted else [],
            "selected_after": after,
            "proposed_selection": proposed,
            "remaining_before": len(set(selected) ^ set(target)),
            "remaining_after": len(set(after) ^ set(target)),
            "correct_stop": action == STOP and selected == target,
        }

    def refine(self, proposal: dict, selected: list[str]) -> dict:
        """Only an environment-certified narrowing can update public selection."""
        domain = proposal["domain_proposed"]
        if proposal["domain_before"] != ["excluded", "included"] or domain not in (["included"], ["excluded"]):
            self.checks += 1
            return {"accepted": False, "reason": "not_a_singleton_refinement", "selected_after": selected}
        action = {"op": "add" if domain == ["included"] else "remove", "id": proposal["slot"]}
        if proposal["slot"] not in {item["id"] for item in self.world["items"]}:
            self.checks += 1
            return {"accepted": False, "reason": "unknown_slot", "selected_after": selected}
        receipt = self.check(action, selected, gate=True)
        receipt["verified_provenance"] = "environment_constraint_check" if receipt["accepted"] else None
        return receipt


def typed_proposal(action: dict) -> dict:
    """Project a repair edit to a singleton decision in a public Boolean lattice."""
    if action["op"] not in ("add", "remove"):
        raise ValueError("Only membership edits refine the candidate lattice")
    return {
        "schema_version": "metta11-membership-refinement-v1",
        "slot": action["id"],
        "domain_before": ["excluded", "included"],
        "domain_proposed": ["included" if action["op"] == "add" else "excluded"],
        "claimed_provenance": "unknown",
    }


def training_trajectory(family: str, seed: int) -> dict:
    world = make_task(family, seed)
    verifier = RepairVerifier(world)
    target = verifier.optimum["selected"]
    start = first_state(world, seed)
    transitions = []
    for frame in replay(world, start, target):
        selected, action = frame["selected"], frame["action"]
        candidates = [
            decode_action(int(index)) for index in legal_action_mask(world, selected).nonzero()[0]
        ]
        correct = verifier.check(action, selected)
        if action == STOP:
            if not correct["correct_stop"]:
                raise ValueError("Invalid terminal trajectory")
        elif correct["remaining_after"] != correct["remaining_before"] - 1:
            raise ValueError("Canonical edit did not repair one membership")
        # Prefer an environment-vetoed counterfactual. All inspected proposals
        # count toward the generation verifier budget, even those not exported.
        wrong = None
        for candidate in candidates:
            if candidate == action:
                continue
            receipt = (
                verifier.check(candidate, selected)
                if candidate == STOP else verifier.refine(typed_proposal(candidate), selected)
            )
            if not receipt["accepted"] or (wrong is None and receipt["remaining_after"] >= correct["remaining_before"]):
                wrong = {"action": candidate, "receipt": receipt}
            if not receipt["accepted"]:
                break
        transitions.append({
            "step": frame["step"],
            "selected": selected,
            "legal_actions": candidates,
            "next_edit": action,
            "receipt": correct,
            "counterfactual": wrong,
            "remaining_distance": correct["remaining_before"],
            "stop_target": "STOP" if action == STOP else "CONTINUE",
        })
    return {
        "schema_version": "metta11-verified-repair-trajectory-v1",
        "root": root_name(family, seed),
        "source_sha256": digest(world["program"].encode()),
        "presentation": 0,
        "split": "train_candidate",
        "family": family,
        "world": world,
        "transitions": transitions,
        "generation_verifier_calls": verifier.checks,
        "generation_constraint_checks": verifier.constraint_checks + len(transitions),
        "generation_exact_optimum_calls": 1,
        "generation_subsets_enumerated": verifier.optimum["checked_subsets"],
        "training_ready": False,
    }


def metta_transitions(row: dict) -> str:
    """Serialize generated public states and checked transitions as ground facts."""
    source = row["world"]["program"].splitlines()
    for frame in row["transitions"]:
        step = frame["step"]
        action = frame["next_edit"]
        post = " ".join(frame["receipt"]["selected_after"])
        before = " ".join(frame["selected"])
        source.extend((
            f"(RepairState {step} ({before}))",
            f'(RepairTransition {step} ({action["op"]} {action["id"]}) ({post}))',
            f'(RepairDistance {step} {frame["remaining_distance"]})',
            f'(RepairTarget {step} {frame["stop_target"]})',
        ))
        for candidate in frame["legal_actions"]:
            source.append(f'(LegalRepairAction {step} ({candidate["op"]} {candidate["id"]}))')
        wrong = frame["counterfactual"]
        action, receipt = wrong["action"], wrong["receipt"]
        reasons = " ".join(receipt["veto_violations"])
        source.append(f'(RepairFeedback {step} ({action["op"]} {action["id"]}) '
                      f'({receipt["reason"]} ({reasons})))')
    return "\n".join(source) + "\n"


def export_trajectories(rows: list[dict], out: Path) -> dict:
    """A whole trajectory is one source presentation in either training view."""
    sft, rlt = [], []
    for row in rows:
        metadata = {key: row[key] for key in ("root", "source_sha256", "presentation", "split", "training_ready")}
        metadata["presentation_sha256"] = digest(metadata)
        public = row["world"]["program"]
        frames = row["transitions"]
        sft.append({
            "schema_version": "metta11-sft-trajectory-v1",
            "instruction": "Produce a verified repair trajectory from the public MeTTa state.",
            "input": canonical({"program": public, "selected": frames[0]["selected"]}),
            "output": canonical([{"action": f["next_edit"], "stop_target": f["stop_target"]} for f in frames]),
            "metadata": metadata,
            "verifier_receipts": [f["receipt"] for f in frames],
        })
        rlt.append({
            "schema_version": "metta11-rlt-trajectory-v1",
            **metadata,
            "observation": {"program": public, "selected": frames[0]["selected"]},
            "transitions": frames,
            "metta_trajectory": metta_transitions(row),
        })
    if len({row["root"] for row in rows}) != len(rows):
        raise ValueError("Expected one unique presentation per source root")
    artifacts = {
        "train_repair_trajectories.jsonl": rows,
        "train_sft_alternate.jsonl": sft,
        "train_rlt_alternate.jsonl": rlt,
    }
    hashes = {}
    for name, records in artifacts.items():
        data = "".join(canonical(record) + "\n" for record in records).encode()
        (out / name).write_bytes(data)
        hashes[name] = digest(data)
    manifest = {
        "schema_version": "metta11-training-views-v1",
        "sha256": hashes,
        "roots": len(rows),
        "presentations_per_root": 1,
        "consume_only_one_view_per_training_run": True,
        "development_labels_exported": False,
        "training_ready": False,
    }
    (out / "training_views_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_metta_transitions(rows: list[dict], out: Path) -> dict:
    """Independently query host-generated trajectory facts in the pinned runtime."""
    from importlib.metadata import version

    from .ram_hillclimb_native import PACKAGE_VERSION

    if version("hyperon") != PACKAGE_VERSION:
        raise RuntimeError(f"Native trajectory queries require hyperon=={PACKAGE_VERSION}")
    from hyperon import Environment, MeTTa

    receipts = []
    query = "!(match &self (RepairTransition $step $action $after) ($step $action $after))"
    for row in rows:
        program = metta_transitions(row)
        runtime = MeTTa(env_builder=Environment.custom_env(create_config=False, is_test=True))
        observations = sorted(map(str, runtime.run(program + query)[-1]))
        expected = []
        for frame in row["transitions"]:
            action = frame["next_edit"]
            after = " ".join(frame["receipt"]["selected_after"])
            expected.append(f'({frame["step"]} ({action["op"]} {action["id"]}) ({after}))')
        if observations != sorted(expected):
            raise ValueError(f"Native trajectory query disagreed for {row['root']}")
        receipts.append({
            "root": row["root"], "source_sha256": row["source_sha256"],
            "program_sha256": digest(program.encode()), "transitions_matched": len(observations),
            "matched_atom_sha256": digest(observations),
        })
    report = {
        "runtime": f"hyperon=={PACKAGE_VERSION}", "scope": "ground_transition_fact_query",
        "roots_matched": len(rows), "transitions_matched": sum(row["transitions_matched"] for row in receipts),
        "receipts": receipts,
    }
    (out / "native_trajectory_receipts.json").write_text(canonical(report) + "\n", encoding="utf-8")
    return report
