from __future__ import annotations

import copy
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from datagoblin.domain.recipes import DataRecipe
from datagoblin.hashing import sha256_json
from datagoblin.serialization import to_jsonable


@dataclass(frozen=True)
class MutationReceipt:
    operator: str
    deterministic_seed: int
    expected_violation_mask: tuple[str, ...]
    minimal_repair: str
    before_hash: str
    after_hash: str

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


def _inject_eval_leakage(value: dict[str, Any], _: random.Random) -> None:
    value["split_contract"]["evaluation_seeds_exposed"] = True
    value["split_contract"]["builder_exposed_fields"].append("hidden_evaluation_seed")


def _remove_independent_verifier(value: dict[str, Any], _: random.Random) -> None:
    value["independent_verifier_requirements"] = {"required": False, "component": ""}


def _merge_builder_and_evaluator(value: dict[str, Any], _: random.Random) -> None:
    value["independent_verifier_requirements"]["component"] = value["builder_permissions"]["component"]


def _remove_falsifier(value: dict[str, Any], _: random.Random) -> None:
    value["falsification_criteria"] = []


def _unbound_compute(value: dict[str, Any], _: random.Random) -> None:
    value["resource_budget"]["wall_clock_seconds"] = 0


def _confound_ablation(value: dict[str, Any], _: random.Random) -> None:
    value["ablations"] = [{"name": "everything", "changed_variables": ["data", "optimizer", "features"]}]


def _claim_unsupported_gain(value: dict[str, Any], _: random.Random) -> None:
    value["predicted_outcomes"][0].update({"lower": 0.42, "upper": 0.42, "evidence_refs": []})


def _expose_hidden_metric(value: dict[str, Any], _: random.Random) -> None:
    value["split_contract"]["hidden_result_exposed"] = True


def _collapse_diversity(value: dict[str, Any], _: random.Random) -> None:
    value["curriculum_controller"]["diversity_collapsed"] = True
    value["curriculum_controller"]["unique_fraction"] = 0.1


def _cherry_pick_seeds(value: dict[str, Any], _: random.Random) -> None:
    value["split_contract"]["cherry_picked_seeds"] = True


def _request_unnecessary_network(value: dict[str, Any], _: random.Random) -> None:
    value["builder_permissions"]["network"] = True
    value["builder_permissions"]["network_justification"] = ""


def _self_certify(value: dict[str, Any], _: random.Random) -> None:
    proposer = value["provenance"]["asserted_by"]
    value["provenance"].update({"evidence_type": "ENVIRONMENT_VERIFIED", "verified_by": proposer})


def _generator_overlap(value: dict[str, Any], _: random.Random) -> None:
    value["split_contract"]["evaluation_generator_family"] = value["split_contract"]["train_generator_family"]


def _remove_solver(value: dict[str, Any], _: random.Random) -> None:
    value["reference_solver_requirements"]["required"] = False


def _alter_metric(value: dict[str, Any], _: random.Random) -> None:
    value["split_contract"]["metric_changed_after_run"] = True


_OPERATORS: dict[str, tuple[Callable[[dict[str, Any], random.Random], None], str, str]] = {
    "inject_eval_leakage": (
        _inject_eval_leakage,
        "evaluation_leakage",
        "Remove evaluation-only builder fields.",
    ),
    "remove_independent_verifier": (
        _remove_independent_verifier,
        "missing_independent_verifier",
        "Restore an independent verifier.",
    ),
    "merge_builder_and_evaluator": (
        _merge_builder_and_evaluator,
        "builder_evaluator_identity",
        "Separate builder and evaluator identities.",
    ),
    "remove_falsifier": (_remove_falsifier, "missing_falsifier", "Restore falsification criteria."),
    "unbound_compute": (
        _unbound_compute,
        "unbounded_budget",
        "Restore positive bounded resource limits.",
    ),
    "confound_ablation": (
        _confound_ablation,
        "confounded_ablation",
        "Split into one-variable ablations.",
    ),
    "claim_unsupported_gain": (
        _claim_unsupported_gain,
        "unsupported_expected_gain",
        "Use a bounded, evidenced estimate.",
    ),
    "expose_hidden_metric": (
        _expose_hidden_metric,
        "hidden_result_exposed",
        "Remove hidden metric access.",
    ),
    "collapse_diversity": (
        _collapse_diversity,
        "curriculum_diversity_collapse",
        "Restore curriculum diversity.",
    ),
    "cherry_pick_seeds": (
        _cherry_pick_seeds,
        "cherry_picked_seeds",
        "Use the frozen seed namespace.",
    ),
    "request_unnecessary_network": (
        _request_unnecessary_network,
        "unnecessary_network",
        "Deny network access.",
    ),
    "self_certify_environment_evidence": (
        _self_certify,
        "self_certified_environment_evidence",
        "Downgrade or independently verify evidence.",
    ),
    "create_train_eval_generator_overlap": (
        _generator_overlap,
        "seed_namespace_overlap",
        "Separate generator families.",
    ),
    "remove_solver_certificate": (
        _remove_solver,
        "uncertified_task",
        "Restore solver certification.",
    ),
    "alter_metric_after_freeze": (
        _alter_metric,
        "metric_mutated_after_freeze",
        "Restore the frozen metric.",
    ),
}

MUTATION_NAMES = tuple(_OPERATORS)


def apply_mutation(recipe: DataRecipe, operator: str, seed: int) -> tuple[DataRecipe, MutationReceipt]:
    if operator not in _OPERATORS:
        raise ValueError(f"unknown mutation operator: {operator}")
    value = copy.deepcopy(recipe.to_jsonable())
    before = sha256_json(value)
    function, violation, repair = _OPERATORS[operator]
    function(value, random.Random(seed))
    corrupted = DataRecipe.from_jsonable(value)
    after = sha256_json(corrupted.to_jsonable())
    receipt = MutationReceipt(operator, seed, (violation,), repair, before, after)
    return corrupted, receipt
