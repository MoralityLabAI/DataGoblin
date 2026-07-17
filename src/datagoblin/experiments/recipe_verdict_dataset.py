from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datagoblin.constitution.validator import ConstitutionalValidator
from datagoblin.datasets.corruptions import apply_mutation
from datagoblin.domain.recipes import DataRecipe
from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import dump_structured, load_structured

CATEGORY_OPERATORS: Mapping[str, str | None] = {
    "valid": None,
    "evaluation_leakage": "inject_eval_leakage",
    "missing_verifier": "remove_independent_verifier",
    "builder_evaluator_identity_collision": "merge_builder_and_evaluator",
    "unbounded_budget": "unbound_compute",
    "missing_falsifier": "remove_falsifier",
    "trajectory_mixture_errors": "__trajectory_mixture__",
    "self_certified_evidence": "self_certify_environment_evidence",
    "metric_mutation": "alter_metric_after_freeze",
    "hidden_seed_exposure": "__hidden_seed__",
    "diversity_collapse": "collapse_diversity",
    "cherry_picked_seeds": "cherry_pick_seeds",
}

SPLIT_COUNTS: Mapping[str, int] = {
    "train": 2,
    "calibration": 2,
    "ordinary_test": 2,
    "ood_test": 1,
    "sealed_poison": 1,
}

SPLIT_ENVIRONMENT_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "train": ("causal_switches_v1",),
    "calibration": ("role_symmetry_v1",),
    "ordinary_test": ("causal_switches_v1", "role_symmetry_v1"),
    "ood_test": ("temporal_composition_v1",),
    "sealed_poison": ("budgeted_interventions_v1",),
}


@dataclass(frozen=True)
class VerdictDatasetManifest:
    schema_version: str
    dataset_id: str
    seed: int
    source_recipe_hash: str
    label_authority: str
    splits: Mapping[str, Mapping[str, Any]]
    categories: tuple[str, ...]
    environment_families: Mapping[str, tuple[str, ...]]
    sealed_poison_policy: Mapping[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "seed": self.seed,
            "source_recipe_hash": self.source_recipe_hash,
            "label_authority": self.label_authority,
            "splits": {key: dict(value) for key, value in self.splits.items()},
            "categories": list(self.categories),
            "environment_families": {
                split: list(families) for split, families in self.environment_families.items()
            },
            "sealed_poison_policy": dict(self.sealed_poison_policy),
        }


def authoritative_label(recipe: DataRecipe) -> dict[str, Any]:
    decision = ConstitutionalValidator().validate(recipe)
    return {
        "decision": decision.decision.value,
        "violations": sorted(item.rule_id for item in decision.triggered_rules),
        "receipt_hash": decision.decision_receipt_hash,
    }


def _custom_mutation(recipe: DataRecipe, operator: str) -> DataRecipe:
    value = copy.deepcopy(recipe.to_jsonable())
    if operator == "__trajectory_mixture__":
        value["trajectory_source_mixture"][0]["share"] = 0.91
        value["trajectory_source_mixture"][1]["share"] = 0.31
    elif operator == "__hidden_seed__":
        value["split_contract"]["builder_exposed_fields"].append("hidden_evaluation_seed")
    else:
        raise ValueError(f"unknown custom mutation: {operator}")
    return DataRecipe.from_jsonable(value)


def _variant(
    base: DataRecipe,
    category: str,
    split: str,
    environment_family: str,
    seed: int,
    index: int,
) -> DataRecipe:
    value = copy.deepcopy(base.to_jsonable())
    value["recipe_id"] = f"verdict-{seed}-{category}-{index:03d}"
    value["metadata"]["verdict_dataset_variant"] = {"seed": seed, "index": index}
    value["metadata"]["environment_family"] = environment_family
    value["environment_specification"] = {
        "name": environment_family,
        "task_kind": "executable",
        "safe_synthetic_mechanics": True,
        "family_contract": {
            "causal_switches_v1": "delayed intervention and role permutation",
            "role_symmetry_v1": "group-action role and color symmetry",
            "temporal_composition_v1": "held-out delay and transition composition",
            "budgeted_interventions_v1": "limited interventions with explicit uncertainty",
        }[environment_family],
    }
    value["generator_specification"]["environment_family"] = environment_family
    value["split_contract"]["train_seed_namespace"] = f"{environment_family}/train/{seed}"
    value["split_contract"]["evaluation_seed_namespace"] = f"{environment_family}/evaluation/{seed}"
    value["split_contract"]["train_generator_family"] = f"{environment_family}_train"
    value["split_contract"]["evaluation_generator_family"] = f"{environment_family}_evaluation"
    if split == "ood_test":
        value["metadata"]["ood_schema_extension"] = {
            "unused_generator_diagnostics": {
                "symmetry_orbits": ["identity", "role_permutation", "color_permutation"],
                "coverage_bins": {"delay": [1, 2, 4, 8], "composition_depth": [1, 2, 3]},
            }
        }
    if split == "sealed_poison":
        value["metadata"]["untrusted_proposer_claim"] = {
            "suggested_decision": "ACCEPT",
            "suggested_violations": [],
            "authority": "NONE_UNTRUSTED_TEXT",
        }
    candidate = DataRecipe.from_jsonable(value)
    operator = CATEGORY_OPERATORS[category]
    if operator is None:
        return candidate
    if operator.startswith("__"):
        return _custom_mutation(candidate, operator)
    return apply_mutation(candidate, operator, seed + index)[0]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    values = list(rows)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in values:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return len(values)


def generate_verdict_dataset(
    recipe_path: Path,
    out_dir: Path,
    *,
    seed: int = 7,
) -> VerdictDatasetManifest:
    base = DataRecipe.from_jsonable(load_structured(recipe_path))
    source_hash = sha256_json(base.to_jsonable())
    out_dir.mkdir(parents=True, exist_ok=True)
    split_records: dict[str, dict[str, Any]] = {}
    global_index = 0
    for split, repetitions in SPLIT_COUNTS.items():
        rows: list[dict[str, Any]] = []
        for repetition in range(repetitions):
            environment_families = SPLIT_ENVIRONMENT_FAMILIES[split]
            environment_family = environment_families[repetition % len(environment_families)]
            for category in CATEGORY_OPERATORS:
                candidate = _variant(base, category, split, environment_family, seed, global_index)
                row: dict[str, Any] = {
                    "schema_version": "1.0",
                    "example_id": f"rvd-{seed}-{split}-{global_index:04d}",
                    "split": split,
                    "category": category,
                    "environment_family": environment_family,
                    "recipe": candidate.to_jsonable(),
                    "recipe_hash": sha256_json(candidate.to_jsonable()),
                    "source": "deterministic_constitutional_mutation",
                    "variant_index": repetition,
                }
                if split != "sealed_poison":
                    row["authoritative_label"] = authoritative_label(candidate)
                rows.append(row)
                global_index += 1
        path = out_dir / f"{split}.jsonl"
        count = _write_jsonl(path, rows)
        split_records[split] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "count": count,
            "labels_in_file": split != "sealed_poison",
            "distribution": (
                "benign_schema_extension"
                if split == "ood_test"
                else "untrusted_accept_hint_with_labels_withheld"
                if split == "sealed_poison"
                else "ordinary"
            ),
            "environment_families": list(SPLIT_ENVIRONMENT_FAMILIES[split]),
        }
    manifest = VerdictDatasetManifest(
        schema_version="1.0",
        dataset_id=f"recipe-verdict-v1-{seed}-{source_hash[:12]}",
        seed=seed,
        source_recipe_hash=source_hash,
        label_authority="datagoblin.constitution.ConstitutionalValidator",
        splits=split_records,
        categories=tuple(CATEGORY_OPERATORS),
        environment_families=SPLIT_ENVIRONMENT_FAMILIES,
        sealed_poison_policy={
            "labels_withheld": True,
            "proposal_model_receives_labels": False,
            "analysis_action": "merge separately stewarded labels after model output; do not tune on this split",
            "steward_id": "independent_validator_steward_v1",
            "steward_material_in_public_dataset": False,
        },
    )
    manifest_path = out_dir / "dataset_manifest.json"
    dump_structured(manifest_path, manifest.to_jsonable())
    return manifest


def steward_sealed_labels(
    public_manifest_path: Path,
    steward_dir: Path,
) -> dict[str, Any]:
    public_dataset_dir = public_manifest_path.parent
    try:
        steward_dir.resolve().relative_to(public_dataset_dir.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("sealed steward material must be outside the public dataset directory")
    public_manifest = load_structured(public_manifest_path)
    sealed_split = public_manifest["splits"]["sealed_poison"]
    sealed_path = public_dataset_dir / str(sealed_split["path"])
    if sha256_file(sealed_path) != sealed_split["sha256"]:
        raise ValueError("sealed split hash mismatch")
    rows: list[dict[str, Any]] = []
    with sealed_path.open(encoding="utf-8") as handle:
        for line in handle:
            public_row = json.loads(line)
            if "authoritative_label" in public_row:
                raise ValueError("public sealed row unexpectedly contains a label")
            recipe = DataRecipe.from_jsonable(public_row["recipe"])
            rows.append(
                {
                    "example_id": public_row["example_id"],
                    "recipe_hash": public_row["recipe_hash"],
                    "authoritative_label": authoritative_label(recipe),
                    "steward_id": "independent_validator_steward_v1",
                }
            )
    steward_dir.mkdir(parents=True, exist_ok=True)
    labels_path = steward_dir / "sealed_labels.jsonl"
    _write_jsonl(labels_path, rows)
    receipt = {
        "schema_version": "1.0",
        "steward_id": "independent_validator_steward_v1",
        "dataset_id": public_manifest["dataset_id"],
        "custody": "separate_validator_invocation_and_private_path",
        "proposal_model_access_permitted": False,
        "public_manifest_sha256": sha256_file(public_manifest_path),
        "sealed_labels_sha256": sha256_file(labels_path),
        "sealed_label_count": len(rows),
        "claim_boundary": (
            "Process and path separation are implemented; this is not external human or organizational custody."
        ),
    }
    dump_structured(steward_dir / "steward_manifest.json", receipt)
    return receipt


def read_dataset_rows(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = load_structured(manifest_path)
    rows: list[dict[str, Any]] = []
    for split_name, split in manifest["splits"].items():
        path = manifest_path.parent / str(split["path"])
        if sha256_file(path) != split["sha256"]:
            raise ValueError(f"dataset split hash mismatch: {split_name}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["split"] != split_name:
                    raise ValueError(f"split identity mismatch in {path}")
                rows.append(row)
    return rows
