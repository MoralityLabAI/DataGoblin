from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.experiments.ldt_array_confirmation import (
    CONFIRMATION_FEATURE_WHITELIST,
    ConfirmationPairEncoder,
    confirmation_input_sha256,
    make_confirmation_split,
    reduce_confirmation_array,
    validate_confirmation_pair,
    validate_confirmation_side,
)
from datagoblin.experiments.ldt_array_hard_surface import (
    ARRAY_STATUSES,
    HARD_INVERT_PREFERENCE,
    HARD_PREFERENCES,
    HARD_RATIONALES,
)
from datagoblin.experiments.ldt_array_sorter import (
    LoopedJointArray,
    classification_metrics,
    make_model,
    predict_model,
    sha256_file,
    sha256_json,
)
from datagoblin.serialization import dump_structured, load_structured

PAIR_RECOMMENDATION_KEYS = {
    "schema_version",
    "shadow_run_id",
    "candidate_id",
    "seed",
    "example_id",
    "array_id",
    "left_index",
    "right_index",
    "left_tournament_report_sha256",
    "right_tournament_report_sha256",
    "lineage_group_sha256",
    "input_sha256",
    "shadow_manifest_sha256",
    "contract_sha256",
    "model_sha256",
    "trainer_summary_sha256",
    "preference",
    "rationale_code",
    "confidence",
    "preference_probabilities",
    "swapped_preference",
    "swap_consistent",
    "calibration_temperature",
    "authority",
    "decision_effect",
}

ARRAY_RECOMMENDATION_KEYS = {
    "schema_version",
    "shadow_run_id",
    "candidate_id",
    "seed",
    "array_id",
    "lineage_group_sha256",
    "game",
    "fixture_pressure",
    "candidate_count",
    "tournament_report_sha256",
    "statuses",
    "selected_indices",
    "dominated_indices",
    "unresolved_indices",
    "unsafe_indices",
    "minimum_pair_confidence",
    "shadow_manifest_sha256",
    "contract_sha256",
    "model_sha256",
    "trainer_summary_sha256",
    "authority",
    "decision_effect",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _public_without_target(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"target", "authority"}
    }


def _shadow_split_config(contract: Mapping[str, Any]) -> dict[str, Any]:
    surface = contract["fresh_shadow_surface"]
    return {
        "surface": {
            "games": surface["games"],
            "pressures": surface["pressures"],
            "heldout_array_sizes": surface["array_sizes"],
            "heldout_seed": surface["seed"],
            "heldout_context_prefix": surface["context_prefix"],
            "metric_equivalence_tolerance": surface[
                "metric_equivalence_tolerance"
            ],
        }
    }


def generate_shadow_fixture(
    contract_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    contract = load_structured(contract_path)
    if contract["status"] != "frozen_before_shadow_generation":
        raise ValueError("shadow contract is not frozen")
    surface = contract["fresh_shadow_surface"]
    if int(surface["seed"]) in {
        int(value) for value in surface["forbidden_prior_seeds"]
    }:
        raise ValueError("shadow seed reuses a confirmation seed")
    out_dir.mkdir(parents=True, exist_ok=False)
    public_dir = out_dir / "public_shadow"
    verifier_dir = out_dir / "private_verifier"
    public_dir.mkdir()
    verifier_dir.mkdir()
    pairs, arrays, reports = make_confirmation_split(
        _shadow_split_config(contract),
        split="heldout",
    )
    observed = {
        "pairs": len(pairs),
        "arrays": len(arrays),
        "reports": len(reports),
    }
    expected = {
        "pairs": int(surface["required_pairs"]),
        "arrays": int(surface["required_arrays"]),
        "reports": int(surface["required_reports"]),
    }
    if observed != expected:
        raise ValueError(f"shadow fixture counts differ: {observed}")
    pair_path = public_dir / "shadow_pairs.jsonl"
    array_path = public_dir / "shadow_arrays.jsonl"
    report_path = public_dir / "shadow_tournament_reports.jsonl"
    pair_labels_path = verifier_dir / "verifier_pair_labels.jsonl"
    array_labels_path = verifier_dir / "verifier_array_labels.jsonl"
    public_pairs = [_public_without_target(row) for row in pairs]
    public_arrays = [_public_without_target(row) for row in arrays]
    _write_jsonl(pair_path, public_pairs)
    _write_jsonl(array_path, public_arrays)
    _write_jsonl(report_path, reports)
    _write_jsonl(
        pair_labels_path,
        [
            {
                "schema_version": "ldt-shadow-verifier-pair-label.v4",
                "example_id": row["example_id"],
                "array_id": row["array_id"],
                "lineage_group_sha256": row["lineage_group_sha256"],
                "target": row["target"],
                "authority": "ENVIRONMENT_OWNED",
            }
            for row in pairs
        ],
    )
    _write_jsonl(
        array_labels_path,
        [
            {
                "schema_version": "ldt-shadow-verifier-array-label.v4",
                "array_id": row["array_id"],
                "lineage_group_sha256": row["lineage_group_sha256"],
                "target": row["target"],
                "authority": "ENVIRONMENT_OWNED",
            }
            for row in arrays
        ],
    )
    lineages = {str(row["lineage_group_sha256"]) for row in pairs}
    forbidden_lineages = {
        str(value) for value in surface["forbidden_prior_lineage_sha256"]
    }
    report_hashes = {
        str(report["tournament_report_sha256"]) for report in reports
    }
    bound_hashes = {
        str(row[key])
        for row in pairs
        for key in (
            "left_tournament_report_sha256",
            "right_tournament_report_sha256",
        )
    }
    orientation_complete = all(
        {
            (int(pair["left_index"]), int(pair["right_index"]))
            for pair in pairs
            if pair["array_id"] == array_row["array_id"]
        }
        == {
            (left, right)
            for left in range(int(array_row["candidate_count"]))
            for right in range(int(array_row["candidate_count"]))
            if left != right
        }
        for array_row in arrays
    )
    checks = {
        "counts_match_contract": observed == expected,
        "fresh_lineages": not bool(lineages & forbidden_lineages),
        "report_hash_bindings_complete": report_hashes == bound_hashes,
        "orientation_complete": orientation_complete,
        "all_reports_content_free": all(
            not bool(report["contains_row_level_game_records"])
            and not bool(report["contains_hidden_reasoning"])
            for report in reports
        ),
        "public_pairs_have_no_targets": all(
            "target" not in row and "authority" not in row
            for row in public_pairs
        ),
        "public_arrays_have_no_targets": all(
            "target" not in row and "authority" not in row
            for row in public_arrays
        ),
        "all_array_sizes_present": {
            int(row["candidate_count"]) for row in arrays
        }
        == {int(value) for value in surface["array_sizes"]},
    }
    splits = {
        "shadow_pairs": {
            "path": pair_path.name,
            "count": len(public_pairs),
            "sha256": sha256_file(pair_path),
            "targets_in_file": False,
        },
        "shadow_arrays": {
            "path": array_path.name,
            "count": len(public_arrays),
            "sha256": sha256_file(array_path),
            "targets_in_file": False,
        },
        "shadow_reports": {
            "path": report_path.name,
            "count": len(reports),
            "sha256": sha256_file(report_path),
            "targets_in_file": False,
        },
    }
    manifest = {
        "schema_version": "ldt-shadow-fixture.v4",
        "experiment_id": contract["experiment_id"],
        "shadow_run_id": contract["shadow_run_id"],
        "contract_sha256": sha256_file(contract_path),
        "source_confirmation": contract["source_confirmation"],
        "frozen_model_binding": {
            key: contract["frozen_model"][key]
            for key in (
                "candidate_id",
                "seed",
                "trainable_parameters",
                "model_sha256",
                "trainer_summary_sha256",
            )
        },
        "feature_whitelist": list(CONFIRMATION_FEATURE_WHITELIST),
        "preference_labels": list(HARD_PREFERENCES),
        "rationale_labels": list(HARD_RATIONALES),
        "splits": splits,
        "verifier_label_custody": {
            "pair_labels_sha256": sha256_file(pair_labels_path),
            "array_labels_sha256": sha256_file(array_labels_path),
            "paths_disclosed_to_shadow_runner": False,
            "authority": "ENVIRONMENT_OWNED",
        },
        "lineage_audit": {
            "shadow_groups": len(lineages),
            "forbidden_prior_groups": len(forbidden_lineages),
            "overlap": len(lineages & forbidden_lineages),
        },
        "array_size_distribution": dict(
            sorted(Counter(int(row["candidate_count"]) for row in arrays).items())
        ),
        "report_binding_audit": {
            "report_count": len(report_hashes),
            "bound_report_count": len(bound_hashes),
            "complete": report_hashes == bound_hashes,
        },
        "checks": checks,
        "shadow_decision_effect": "NONE",
        "independent_verifier_remains_decision_owner": True,
        "training_authorized": False,
        "corpus_promotion_authorized": False,
    }
    manifest_path = public_dir / "shadow_manifest.json"
    dump_structured(manifest_path, manifest)
    generation_report = {
        "schema_version": "ldt-shadow-generation-report.v4",
        "experiment_id": contract["experiment_id"],
        "shadow_run_id": contract["shadow_run_id"],
        "contract_sha256": sha256_file(contract_path),
        "shadow_manifest_sha256": sha256_file(manifest_path),
        "counts": observed,
        "checks": checks,
        "generation_passed": all(checks.values()),
        "verifier_targets_exported_to_shadow_runner": False,
        "raw_game_records_read": False,
        "network_used": False,
        "claim_boundary": contract["claim_scope"],
    }
    dump_structured(out_dir / "generation_report.json", generation_report)
    return generation_report


def load_shadow_fixture(
    manifest_path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != "ldt-shadow-fixture.v4":
        raise ValueError("unsupported shadow manifest")
    if tuple(manifest["feature_whitelist"]) != CONFIRMATION_FEATURE_WHITELIST:
        raise ValueError("shadow feature whitelist differs")
    if not all(bool(value) for value in manifest["checks"].values()):
        raise ValueError("shadow fixture generation checks failed")
    loaded = {}
    for name, split in manifest["splits"].items():
        path = manifest_path.parent / str(split["path"])
        if sha256_file(path) != split["sha256"]:
            raise ValueError(f"shadow split hash mismatch: {name}")
        rows = _read_jsonl(path)
        if len(rows) != int(split["count"]):
            raise ValueError(f"shadow split count mismatch: {name}")
        loaded[name] = rows
    for row in loaded["shadow_pairs"]:
        validate_confirmation_pair(row, targets_required=False)
    report_map = {}
    for report in loaded["shadow_reports"]:
        if report["contains_row_level_game_records"] or report[
            "contains_hidden_reasoning"
        ]:
            raise ValueError("shadow report is not content-free")
        report_hash = str(report["tournament_report_sha256"])
        payload = {
            key: value
            for key, value in report.items()
            if key != "tournament_report_sha256"
        }
        if report_hash != sha256_json(payload) or report_hash in report_map:
            raise ValueError("shadow report hash is invalid")
        validate_confirmation_side(report["feature_summary"])
        report_map[report_hash] = report
    referenced = {
        str(row[key])
        for row in loaded["shadow_pairs"]
        for key in (
            "left_tournament_report_sha256",
            "right_tournament_report_sha256",
        )
    }
    if referenced != set(report_map):
        raise ValueError("shadow pair/report bindings differ")
    arrays = loaded["shadow_arrays"]
    for row in arrays:
        if "target" in row or "authority" in row:
            raise ValueError("public shadow array contains verifier state")
        if int(row["candidate_count"]) != len(row["candidates"]):
            raise ValueError("shadow array candidate count mismatch")
        for candidate in row["candidates"]:
            validate_confirmation_side(candidate["features"])
            if candidate["tournament_report_sha256"] not in report_map:
                raise ValueError("shadow array report binding is unknown")
    return (
        manifest,
        loaded["shadow_pairs"],
        arrays,
        loaded["shadow_reports"],
    )


def _as_array(
    value: Any,
    shape: tuple[int, ...],
    *,
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"frozen model tensor is invalid: {name}")
    return result


def restore_latent_model(payload: Mapping[str, Any]) -> LoopedJointArray:
    if payload["family"] != "latent_array" or payload["feature_view"] != "typed":
        raise ValueError("shadow runner accepts only the retained typed latent family")
    state = payload["state"]
    if state["architecture"] != "latent_array" or int(state["loop_count"]) != 1:
        raise ValueError("shadow model is not the one-pass latent architecture")
    input_width = int(state["input_width"])
    hidden_width = int(state["hidden_width"])
    model = make_model(
        {
            "family": "latent_array",
            "hidden_width": hidden_width,
            "loop_count": 1,
        },
        input_width,
        0,
        preference_count=len(HARD_PREFERENCES),
        rationale_count=len(HARD_RATIONALES),
    )
    if not isinstance(model, LoopedJointArray):
        raise TypeError("retained model did not restore as a latent array")
    model.input_weights = _as_array(
        state["input_weights"],
        (hidden_width, input_width),
        name="input_weights",
    )
    if state["recurrent_weights"] not in ([], None):
        raise ValueError("one-pass latent state contains recurrent weights")
    model.recurrent_weights = None
    model.hidden_bias = _as_array(
        state["hidden_bias"],
        (hidden_width,),
        name="hidden_bias",
    )
    model.preference_weights = _as_array(
        state["preference_weights"],
        (len(HARD_PREFERENCES), hidden_width),
        name="preference_weights",
    )
    model.preference_bias = _as_array(
        state["preference_bias"],
        (len(HARD_PREFERENCES),),
        name="preference_bias",
    )
    model.rationale_weights = _as_array(
        state["rationale_weights"],
        (len(HARD_RATIONALES), hidden_width),
        name="rationale_weights",
    )
    model.rationale_bias = _as_array(
        state["rationale_bias"],
        (len(HARD_RATIONALES),),
        name="rationale_bias",
    )
    if model.parameter_count != int(payload["trainable_parameters"]):
        raise ValueError("restored model parameter count differs")
    return model


def _encoder_from_payload(payload: Mapping[str, Any]) -> ConfirmationPairEncoder:
    receipt = payload["encoder"]
    encoder = ConfirmationPairEncoder(
        games=tuple(str(value) for value in receipt["games"]),
        pressures=tuple(str(value) for value in receipt["pressures"]),
        context_modes=tuple(str(value) for value in receipt["context_modes"]),
        optimization_pressures=tuple(
            str(value) for value in receipt["optimization_pressures"]
        ),
        tolerance=float(receipt["tolerance"]),
    )
    if list(encoder.feature_names("typed")) != payload["feature_names"]:
        raise ValueError("frozen encoder feature names differ from model payload")
    return encoder


@dataclass(frozen=True)
class FrozenShadowBundle:
    contract: dict[str, Any]
    contract_sha256: str
    model_payload: dict[str, Any]
    model_sha256: str
    trainer_summary_sha256: str
    encoder: ConfirmationPairEncoder
    model: LoopedJointArray
    calibration_temperature: float


def load_frozen_shadow_bundle(
    contract_path: Path,
    model_path: Path,
    trainer_summary_path: Path,
) -> FrozenShadowBundle:
    contract = load_structured(contract_path)
    if contract["status"] != "frozen_before_shadow_generation":
        raise ValueError("shadow contract is not frozen")
    binding = contract["frozen_model"]
    model_sha256 = sha256_file(model_path)
    trainer_sha256 = sha256_file(trainer_summary_path)
    if model_sha256 != binding["model_sha256"]:
        raise ValueError("operator model does not match frozen hash")
    if trainer_sha256 != binding["trainer_summary_sha256"]:
        raise ValueError("operator trainer summary does not match frozen hash")
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    trainer = json.loads(trainer_summary_path.read_text(encoding="utf-8"))
    if (
        trainer["status"] != "completed"
        or not trainer["hard_caps_verified"]
        or trainer["heldout_pair_targets_seen"]
        or trainer["heldout_array_targets_seen"]
    ):
        raise ValueError("source trainer receipt is incomplete or contaminated")
    if (
        payload["schema_version"] != "ldt-array-confirmation-model.v3"
        or payload["candidate_id"] != binding["candidate_id"]
        or int(payload["seed"]) != int(binding["seed"])
        or payload["family"] != binding["family"]
        or payload["feature_view"] != binding["feature_view"]
        or int(payload["trainable_parameters"])
        != int(binding["trainable_parameters"])
        or payload["heldout_targets_seen"]
        or payload["authority"] != "MODEL_RECOMMENDATION_ONLY"
    ):
        raise ValueError("operator model metadata differs from frozen binding")
    matching = [
        receipt
        for receipt in trainer["candidate_receipts"]
        if receipt["candidate_id"] == binding["candidate_id"]
        and int(receipt["seed"]) == int(binding["seed"])
    ]
    if len(matching) != 1 or matching[0]["model_sha256"] != model_sha256:
        raise ValueError("trainer receipt does not bind the frozen model")
    expected_calibration = binding["calibration"]
    calibration = payload["calibration"]
    temperature = float(calibration["selected_temperature"])
    if (
        calibration["method"] != expected_calibration["method"]
        or calibration["fit_split"] != expected_calibration["fit_split"]
        or not math.isclose(
            temperature,
            float(expected_calibration["temperature"]),
            abs_tol=1e-12,
        )
    ):
        raise ValueError("frozen calibration binding differs")
    model = restore_latent_model(payload)
    encoder = _encoder_from_payload(payload)
    return FrozenShadowBundle(
        contract=contract,
        contract_sha256=sha256_file(contract_path),
        model_payload=payload,
        model_sha256=model_sha256,
        trainer_summary_sha256=trainer_sha256,
        encoder=encoder,
        model=model,
        calibration_temperature=temperature,
    )


def _temperature_scale(
    probabilities: Mapping[str, float],
    temperature: float,
) -> dict[str, float]:
    logits = {
        label: math.log(max(float(value), 1e-12)) / temperature
        for label, value in probabilities.items()
    }
    maximum = max(logits.values())
    exponentials = {
        label: math.exp(value - maximum) for label, value in logits.items()
    }
    total = sum(exponentials.values())
    return {label: value / total for label, value in exponentials.items()}


def run_shadow(
    *,
    contract_path: Path,
    manifest_path: Path,
    model_path: Path,
    trainer_summary_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    bundle = load_frozen_shadow_bundle(
        contract_path,
        model_path,
        trainer_summary_path,
    )
    manifest, pair_rows, array_rows, _ = load_shadow_fixture(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    if manifest["contract_sha256"] != bundle.contract_sha256:
        raise ValueError("shadow manifest was generated under another contract")
    if manifest["frozen_model_binding"]["model_sha256"] != bundle.model_sha256:
        raise ValueError("shadow manifest binds another model")
    out_dir.mkdir(parents=True, exist_ok=False)
    pair_recommendations = []
    by_array: dict[str, dict[tuple[int, int], str]] = {}
    confidence_by_array: dict[str, list[float]] = {}
    for row in pair_rows:
        prediction = predict_model(
            bundle.model,
            bundle.encoder,
            row,
            view="typed",
            preference_labels=HARD_PREFERENCES,
            rationale_labels=HARD_RATIONALES,
        )
        probabilities = _temperature_scale(
            prediction["preference_probabilities"],
            bundle.calibration_temperature,
        )
        preference = max(
            HARD_PREFERENCES,
            key=lambda label: probabilities[label],
        )
        array_id = str(row["array_id"])
        by_array.setdefault(array_id, {})[
            (int(row["left_index"]), int(row["right_index"]))
        ] = preference
        confidence_by_array.setdefault(array_id, []).append(
            float(probabilities[preference])
        )
        pair_recommendations.append(
            {
                "schema_version": "ldt-shadow-pair-recommendation.v4",
                "shadow_run_id": bundle.contract["shadow_run_id"],
                "candidate_id": bundle.model_payload["candidate_id"],
                "seed": bundle.model_payload["seed"],
                "example_id": row["example_id"],
                "array_id": row["array_id"],
                "left_index": row["left_index"],
                "right_index": row["right_index"],
                "left_tournament_report_sha256": row[
                    "left_tournament_report_sha256"
                ],
                "right_tournament_report_sha256": row[
                    "right_tournament_report_sha256"
                ],
                "lineage_group_sha256": row["lineage_group_sha256"],
                "input_sha256": confirmation_input_sha256(row),
                "shadow_manifest_sha256": manifest_sha256,
                "contract_sha256": bundle.contract_sha256,
                "model_sha256": bundle.model_sha256,
                "trainer_summary_sha256": bundle.trainer_summary_sha256,
                "preference": preference,
                "rationale_code": prediction["rationale_code"],
                "confidence": probabilities[preference],
                "preference_probabilities": probabilities,
                "swapped_preference": prediction["swapped_preference"],
                "swap_consistent": prediction["swap_consistent"],
                "calibration_temperature": bundle.calibration_temperature,
                "authority": "MODEL_RECOMMENDATION_ONLY",
                "decision_effect": "NONE",
            }
        )
    array_recommendations = []
    for row in array_rows:
        array_id = str(row["array_id"])
        candidates = [
            candidate["features"] for candidate in row["candidates"]
        ]
        target = reduce_confirmation_array(candidates, by_array[array_id])
        array_recommendations.append(
            {
                "schema_version": "ldt-shadow-array-recommendation.v4",
                "shadow_run_id": bundle.contract["shadow_run_id"],
                "candidate_id": bundle.model_payload["candidate_id"],
                "seed": bundle.model_payload["seed"],
                "array_id": array_id,
                "lineage_group_sha256": row["lineage_group_sha256"],
                "game": row["game"],
                "fixture_pressure": row["fixture_pressure"],
                "candidate_count": row["candidate_count"],
                "tournament_report_sha256": [
                    candidate["tournament_report_sha256"]
                    for candidate in row["candidates"]
                ],
                **target,
                "minimum_pair_confidence": min(
                    confidence_by_array[array_id]
                ),
                "shadow_manifest_sha256": manifest_sha256,
                "contract_sha256": bundle.contract_sha256,
                "model_sha256": bundle.model_sha256,
                "trainer_summary_sha256": bundle.trainer_summary_sha256,
                "authority": "MODEL_RECOMMENDATION_ONLY",
                "decision_effect": "NONE",
            }
        )
    pair_path = out_dir / "shadow_pair_recommendations.jsonl"
    array_path = out_dir / "shadow_array_recommendations.jsonl"
    _write_jsonl(
        pair_path,
        sorted(pair_recommendations, key=lambda row: str(row["example_id"])),
    )
    _write_jsonl(
        array_path,
        sorted(array_recommendations, key=lambda row: str(row["array_id"])),
    )
    receipt = {
        "schema_version": "ldt-shadow-run-receipt.v4",
        "experiment_id": bundle.contract["experiment_id"],
        "shadow_run_id": bundle.contract["shadow_run_id"],
        "status": "completed",
        "contract_sha256": bundle.contract_sha256,
        "shadow_manifest_sha256": manifest_sha256,
        "model_sha256": bundle.model_sha256,
        "trainer_summary_sha256": bundle.trainer_summary_sha256,
        "calibration_temperature": bundle.calibration_temperature,
        "pair_recommendations_sha256": sha256_file(pair_path),
        "array_recommendations_sha256": sha256_file(array_path),
        "pair_count": len(pair_recommendations),
        "array_count": len(array_recommendations),
        "abstain_count": sum(
            row["preference"] == "ABSTAIN" for row in pair_recommendations
        ),
        "unresolved_candidate_count": sum(
            len(row["unresolved_indices"]) for row in array_recommendations
        ),
        "verifier_pair_targets_seen": False,
        "verifier_array_targets_seen": False,
        "raw_game_records_read": False,
        "network_used": False,
        "model_weights_persisted_to_repository": False,
        "authority": "MODEL_RECOMMENDATION_ONLY",
        "decision_effect": "NONE",
        "training_authorized": False,
        "corpus_promotion_authorized": False,
        "claim_boundary": bundle.contract["claim_scope"],
    }
    _write_json(out_dir / "shadow_run_receipt.json", receipt)
    return receipt


def _validate_pair_recommendations(
    rows: Sequence[Mapping[str, Any]],
    public_pairs: Sequence[Mapping[str, Any]],
    *,
    manifest_sha256: str,
    contract_sha256: str,
    model_binding: Mapping[str, Any],
    shadow_run_id: str,
    temperature: float,
) -> dict[str, Mapping[str, Any]]:
    public = {str(row["example_id"]): row for row in public_pairs}
    result = {}
    for row in rows:
        if set(row) != PAIR_RECOMMENDATION_KEYS:
            raise ValueError("shadow pair recommendation fields differ")
        if row["schema_version"] != "ldt-shadow-pair-recommendation.v4":
            raise ValueError("unsupported shadow pair recommendation schema")
        if (
            row["shadow_run_id"] != shadow_run_id
            or row["candidate_id"] != model_binding["candidate_id"]
            or int(row["seed"]) != int(model_binding["seed"])
            or row["shadow_manifest_sha256"] != manifest_sha256
            or row["contract_sha256"] != contract_sha256
            or row["model_sha256"] != model_binding["model_sha256"]
            or row["trainer_summary_sha256"]
            != model_binding["trainer_summary_sha256"]
        ):
            raise ValueError("shadow pair recommendation binding differs")
        if (
            row["authority"] != "MODEL_RECOMMENDATION_ONLY"
            or row["decision_effect"] != "NONE"
        ):
            raise ValueError("shadow pair recommendation asserted authority")
        example_id = str(row["example_id"])
        if example_id in result or example_id not in public:
            raise ValueError("shadow pair recommendation IDs differ")
        source = public[example_id]
        binding_fields = (
            "array_id",
            "left_index",
            "right_index",
            "left_tournament_report_sha256",
            "right_tournament_report_sha256",
            "lineage_group_sha256",
        )
        if any(row[field] != source[field] for field in binding_fields):
            raise ValueError("shadow pair source binding differs")
        if row["input_sha256"] != confirmation_input_sha256(source):
            raise ValueError("shadow pair input hash differs")
        if row["preference"] not in HARD_PREFERENCES:
            raise ValueError("shadow pair preference is unknown")
        if row["rationale_code"] not in HARD_RATIONALES:
            raise ValueError("shadow pair rationale is unknown")
        if row["swapped_preference"] not in HARD_PREFERENCES:
            raise ValueError("shadow swapped preference is unknown")
        probabilities = row["preference_probabilities"]
        if set(probabilities) != set(HARD_PREFERENCES):
            raise ValueError("shadow probability classes differ")
        values = [float(probabilities[label]) for label in HARD_PREFERENCES]
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in values
        ) or not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise ValueError("shadow pair probabilities are invalid")
        if not math.isclose(
            float(row["confidence"]),
            max(values),
            abs_tol=1e-8,
        ):
            raise ValueError("shadow pair confidence differs")
        expected_swap = (
            row["swapped_preference"]
            == HARD_INVERT_PREFERENCE[str(row["preference"])]
        )
        if bool(row["swap_consistent"]) != expected_swap:
            raise ValueError("shadow pair swap flag differs")
        if not math.isclose(
            float(row["calibration_temperature"]),
            temperature,
            abs_tol=1e-12,
        ):
            raise ValueError("shadow pair temperature differs")
        result[example_id] = row
    if set(result) != set(public):
        raise ValueError("shadow pair recommendations do not cover fixture")
    return result


def _validate_array_recommendations(
    rows: Sequence[Mapping[str, Any]],
    public_arrays: Sequence[Mapping[str, Any]],
    pair_recommendations: Mapping[str, Mapping[str, Any]],
    *,
    manifest_sha256: str,
    contract_sha256: str,
    model_binding: Mapping[str, Any],
    shadow_run_id: str,
) -> dict[str, Mapping[str, Any]]:
    public = {str(row["array_id"]): row for row in public_arrays}
    by_array: dict[str, dict[tuple[int, int], str]] = {}
    confidence_by_array: dict[str, list[float]] = {}
    for pair in pair_recommendations.values():
        array_id = str(pair["array_id"])
        by_array.setdefault(array_id, {})[
            (int(pair["left_index"]), int(pair["right_index"]))
        ] = str(pair["preference"])
        confidence_by_array.setdefault(array_id, []).append(
            float(pair["confidence"])
        )
    result = {}
    for row in rows:
        if set(row) != ARRAY_RECOMMENDATION_KEYS:
            raise ValueError("shadow array recommendation fields differ")
        if row["schema_version"] != "ldt-shadow-array-recommendation.v4":
            raise ValueError("unsupported shadow array recommendation schema")
        if (
            row["shadow_run_id"] != shadow_run_id
            or row["candidate_id"] != model_binding["candidate_id"]
            or int(row["seed"]) != int(model_binding["seed"])
            or row["shadow_manifest_sha256"] != manifest_sha256
            or row["contract_sha256"] != contract_sha256
            or row["model_sha256"] != model_binding["model_sha256"]
            or row["trainer_summary_sha256"]
            != model_binding["trainer_summary_sha256"]
        ):
            raise ValueError("shadow array recommendation binding differs")
        if (
            row["authority"] != "MODEL_RECOMMENDATION_ONLY"
            or row["decision_effect"] != "NONE"
        ):
            raise ValueError("shadow array recommendation asserted authority")
        array_id = str(row["array_id"])
        if array_id in result or array_id not in public:
            raise ValueError("shadow array recommendation IDs differ")
        source = public[array_id]
        binding_fields = (
            "lineage_group_sha256",
            "game",
            "fixture_pressure",
            "candidate_count",
        )
        if any(row[field] != source[field] for field in binding_fields):
            raise ValueError("shadow array source binding differs")
        report_hashes = [
            candidate["tournament_report_sha256"]
            for candidate in source["candidates"]
        ]
        if row["tournament_report_sha256"] != report_hashes:
            raise ValueError("shadow array report hashes differ")
        statuses = [str(value) for value in row["statuses"]]
        if (
            len(statuses) != int(source["candidate_count"])
            or any(status not in ARRAY_STATUSES for status in statuses)
        ):
            raise ValueError("shadow array statuses are invalid")
        expected = reduce_confirmation_array(
            [candidate["features"] for candidate in source["candidates"]],
            by_array[array_id],
        )
        for field in (
            "statuses",
            "selected_indices",
            "dominated_indices",
            "unresolved_indices",
            "unsafe_indices",
        ):
            if row[field] != expected[field]:
                raise ValueError("shadow array reduction differs from pairs")
        if not math.isclose(
            float(row["minimum_pair_confidence"]),
            min(confidence_by_array[array_id]),
            abs_tol=1e-12,
        ):
            raise ValueError("shadow array minimum confidence differs")
        result[array_id] = row
    if set(result) != set(public):
        raise ValueError("shadow array recommendations do not cover fixture")
    return result


def _validate_verifier_pair_labels(
    rows: Sequence[Mapping[str, Any]],
    public_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    public = {str(row["example_id"]): row for row in public_pairs}
    expected = {
        "schema_version",
        "example_id",
        "array_id",
        "lineage_group_sha256",
        "target",
        "authority",
    }
    result = {}
    for row in rows:
        if set(row) != expected:
            raise ValueError("shadow verifier pair label fields differ")
        if row["schema_version"] != "ldt-shadow-verifier-pair-label.v4":
            raise ValueError("unsupported shadow verifier pair label schema")
        if row["authority"] != "ENVIRONMENT_OWNED":
            raise ValueError("shadow verifier pair label lacks authority")
        example_id = str(row["example_id"])
        if example_id in result or example_id not in public:
            raise ValueError("shadow verifier pair label IDs differ")
        source = public[example_id]
        if (
            row["array_id"] != source["array_id"]
            or row["lineage_group_sha256"] != source["lineage_group_sha256"]
            or row["target"]["preference"] not in HARD_PREFERENCES
            or row["target"]["rationale_code"] not in HARD_RATIONALES
        ):
            raise ValueError("shadow verifier pair label binding differs")
        result[example_id] = row
    if set(result) != set(public):
        raise ValueError("shadow verifier pair labels do not cover fixture")
    return result


def _validate_verifier_array_labels(
    rows: Sequence[Mapping[str, Any]],
    public_arrays: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    public = {str(row["array_id"]): row for row in public_arrays}
    expected = {
        "schema_version",
        "array_id",
        "lineage_group_sha256",
        "target",
        "authority",
    }
    result = {}
    for row in rows:
        if set(row) != expected:
            raise ValueError("shadow verifier array label fields differ")
        if row["schema_version"] != "ldt-shadow-verifier-array-label.v4":
            raise ValueError("unsupported shadow verifier array label schema")
        if row["authority"] != "ENVIRONMENT_OWNED":
            raise ValueError("shadow verifier array label lacks authority")
        array_id = str(row["array_id"])
        if array_id in result or array_id not in public:
            raise ValueError("shadow verifier array label IDs differ")
        source = public[array_id]
        statuses = row["target"]["statuses"]
        if (
            row["lineage_group_sha256"] != source["lineage_group_sha256"]
            or len(statuses) != int(source["candidate_count"])
            or any(status not in ARRAY_STATUSES for status in statuses)
        ):
            raise ValueError("shadow verifier array label binding differs")
        result[array_id] = row
    if set(result) != set(public):
        raise ValueError("shadow verifier array labels do not cover fixture")
    return result


def _expected_calibration_error(
    actual: Sequence[str],
    recommendations: Sequence[Mapping[str, Any]],
    *,
    bins: int,
) -> float:
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for target, row in zip(actual, recommendations, strict=True):
        confidence = float(row["confidence"])
        index = min(int(confidence * bins), bins - 1)
        buckets[index].append(
            (confidence, float(row["preference"] == target))
        )
    result = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_confidence = sum(item[0] for item in bucket) / len(bucket)
        mean_accuracy = sum(item[1] for item in bucket) / len(bucket)
        result += (
            len(bucket)
            / len(recommendations)
            * abs(mean_accuracy - mean_confidence)
        )
    return result


def _supported_macro_f1(metrics: Mapping[str, Any]) -> float:
    values = [
        float(row["f1"])
        for row in metrics["per_class"].values()
        if int(row["support"]) > 0
    ]
    if not values:
        raise ValueError("support-aware macro F1 requires observed classes")
    return sum(values) / len(values)


def _score_pair_slice(
    public_rows: Sequence[Mapping[str, Any]],
    recommendations: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    *,
    bins: int,
) -> dict[str, Any]:
    ordered = sorted(public_rows, key=lambda row: str(row["example_id"]))
    predicted_rows = [
        recommendations[str(row["example_id"])] for row in ordered
    ]
    actual_preference = [
        str(labels[str(row["example_id"])]["target"]["preference"])
        for row in ordered
    ]
    actual_rationale = [
        str(labels[str(row["example_id"])]["target"]["rationale_code"])
        for row in ordered
    ]
    predicted_preference = [str(row["preference"]) for row in predicted_rows]
    predicted_rationale = [
        str(row["rationale_code"]) for row in predicted_rows
    ]
    preference = classification_metrics(
        actual_preference,
        predicted_preference,
        HARD_PREFERENCES,
    )
    rationale = classification_metrics(
        actual_rationale,
        predicted_rationale,
        HARD_RATIONALES,
    )
    safety_indices = [
        index
        for index, value in enumerate(actual_rationale)
        if value == "safety_gate"
    ]
    brier = sum(
        sum(
            (
                float(row["preference_probabilities"][label])
                - float(label == actual_preference[index])
            )
            ** 2
            for label in HARD_PREFERENCES
        )
        for index, row in enumerate(predicted_rows)
    ) / len(predicted_rows)
    negative_log_likelihood = sum(
        -math.log(
            max(
                float(
                    row["preference_probabilities"][
                        actual_preference[index]
                    ]
                ),
                1e-12,
            )
        )
        for index, row in enumerate(predicted_rows)
    ) / len(predicted_rows)
    abstain_support = sum(
        value == "ABSTAIN" for value in actual_preference
    )
    safety_accuracy = (
        sum(
            actual_preference[index] == predicted_preference[index]
            for index in safety_indices
        )
        / len(safety_indices)
        if safety_indices
        else None
    )
    return {
        "count": len(ordered),
        "preference_accuracy": preference["accuracy"],
        "preference_macro_f1": _supported_macro_f1(preference),
        "rationale_accuracy": rationale["accuracy"],
        "rationale_macro_f1": _supported_macro_f1(rationale),
        "joint_exact_accuracy": sum(
            actual_preference[index] == predicted_preference[index]
            and actual_rationale[index] == predicted_rationale[index]
            for index in range(len(ordered))
        )
        / len(ordered),
        "abstain_recall": (
            preference["per_class"]["ABSTAIN"]["recall"]
            if abstain_support
            else None
        ),
        "safety_gate_accuracy": safety_accuracy,
        "swap_consistency": sum(
            bool(row["swap_consistent"]) for row in predicted_rows
        )
        / len(predicted_rows),
        "multiclass_brier": brier,
        "multiclass_negative_log_likelihood": negative_log_likelihood,
        "expected_calibration_error": _expected_calibration_error(
            actual_preference,
            predicted_rows,
            bins=bins,
        ),
        "preference_disagreement_count": sum(
            actual != predicted
            for actual, predicted in zip(
                actual_preference,
                predicted_preference,
                strict=True,
            )
        ),
        "rationale_disagreement_count": sum(
            actual != predicted
            for actual, predicted in zip(
                actual_rationale,
                predicted_rationale,
                strict=True,
            )
        ),
    }


def _score_array_slice(
    public_rows: Sequence[Mapping[str, Any]],
    recommendations: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    actual_flat = []
    predicted_flat = []
    exact = 0
    selected_exact = 0
    for row in public_rows:
        array_id = str(row["array_id"])
        actual = [
            str(value) for value in labels[array_id]["target"]["statuses"]
        ]
        predicted = [
            str(value) for value in recommendations[array_id]["statuses"]
        ]
        actual_flat.extend(actual)
        predicted_flat.extend(predicted)
        exact += actual == predicted
        selected_exact += {
            index for index, value in enumerate(actual) if value == "SELECTED"
        } == {
            index
            for index, value in enumerate(predicted)
            if value == "SELECTED"
        }
    status = classification_metrics(
        actual_flat,
        predicted_flat,
        ARRAY_STATUSES,
    )
    return {
        "array_count": len(public_rows),
        "candidate_count": len(actual_flat),
        "status_accuracy": status["accuracy"],
        "status_macro_f1": _supported_macro_f1(status),
        "exact_array_accuracy": exact / len(public_rows),
        "selected_set_exact_accuracy": selected_exact / len(public_rows),
        "unresolved_recall": status["per_class"]["UNRESOLVED"]["recall"],
        "unsafe_exclusion_accuracy": status["per_class"]["UNSAFE"]["recall"],
        "array_disagreement_count": len(public_rows) - exact,
    }


def evaluate_shadow(
    *,
    contract_path: Path,
    manifest_path: Path,
    pair_recommendations_path: Path,
    array_recommendations_path: Path,
    shadow_run_receipt_path: Path,
    verifier_pair_labels_path: Path,
    verifier_array_labels_path: Path,
    out_dir: Path,
    supersedes_comparison_path: Path | None = None,
) -> dict[str, Any]:
    contract = load_structured(contract_path)
    if contract["status"] != "frozen_before_shadow_generation":
        raise ValueError("shadow contract is not frozen")
    contract_sha256 = sha256_file(contract_path)
    manifest, public_pairs, public_arrays, _ = load_shadow_fixture(
        manifest_path
    )
    manifest_sha256 = sha256_file(manifest_path)
    if manifest["contract_sha256"] != contract_sha256:
        raise ValueError("shadow fixture used another contract")
    run_receipt = json.loads(
        shadow_run_receipt_path.read_text(encoding="utf-8")
    )
    if (
        run_receipt["status"] != "completed"
        or run_receipt["verifier_pair_targets_seen"]
        or run_receipt["verifier_array_targets_seen"]
        or run_receipt["decision_effect"] != "NONE"
        or run_receipt["authority"] != "MODEL_RECOMMENDATION_ONLY"
        or run_receipt["shadow_manifest_sha256"] != manifest_sha256
        or run_receipt["contract_sha256"] != contract_sha256
        or run_receipt["pair_recommendations_sha256"]
        != sha256_file(pair_recommendations_path)
        or run_receipt["array_recommendations_sha256"]
        != sha256_file(array_recommendations_path)
    ):
        raise ValueError("shadow run receipt is invalid or contaminated")
    model_binding = manifest["frozen_model_binding"]
    pair_recommendations = _validate_pair_recommendations(
        _read_jsonl(pair_recommendations_path),
        public_pairs,
        manifest_sha256=manifest_sha256,
        contract_sha256=contract_sha256,
        model_binding=model_binding,
        shadow_run_id=str(manifest["shadow_run_id"]),
        temperature=float(contract["frozen_model"]["calibration"]["temperature"]),
    )
    array_recommendations = _validate_array_recommendations(
        _read_jsonl(array_recommendations_path),
        public_arrays,
        pair_recommendations,
        manifest_sha256=manifest_sha256,
        contract_sha256=contract_sha256,
        model_binding=model_binding,
        shadow_run_id=str(manifest["shadow_run_id"]),
    )
    custody = manifest["verifier_label_custody"]
    if (
        sha256_file(verifier_pair_labels_path)
        != custody["pair_labels_sha256"]
        or sha256_file(verifier_array_labels_path)
        != custody["array_labels_sha256"]
    ):
        raise ValueError("shadow verifier labels differ from custody receipt")
    verifier_pairs = _validate_verifier_pair_labels(
        _read_jsonl(verifier_pair_labels_path),
        public_pairs,
    )
    verifier_arrays = _validate_verifier_array_labels(
        _read_jsonl(verifier_array_labels_path),
        public_arrays,
    )
    bins = int(contract["monitoring"]["calibration_bins"])
    pair_aggregate = _score_pair_slice(
        public_pairs,
        pair_recommendations,
        verifier_pairs,
        bins=bins,
    )
    array_aggregate = _score_array_slice(
        public_arrays,
        array_recommendations,
        verifier_arrays,
    )
    pair_by_game = {}
    array_by_game = {}
    for game in contract["approved_input_scope"]["allowed_games"]:
        game_pairs = [
            row
            for row in public_pairs
            if row["input"]["left"]["game"] == game
        ]
        game_arrays = [
            row for row in public_arrays if row["game"] == game
        ]
        pair_by_game[str(game)] = _score_pair_slice(
            game_pairs,
            pair_recommendations,
            verifier_pairs,
            bins=bins,
        )
        array_by_game[str(game)] = _score_array_slice(
            game_arrays,
            array_recommendations,
            verifier_arrays,
        )
    array_by_size = {}
    for size in sorted(
        {int(row["candidate_count"]) for row in public_arrays}
    ):
        selected = [
            row
            for row in public_arrays
            if int(row["candidate_count"]) == size
        ]
        array_by_size[str(size)] = _score_array_slice(
            selected,
            array_recommendations,
            verifier_arrays,
        )
    monitoring = contract["monitoring"]
    monitoring_checks = {
        "pair_preference_macro_f1": (
            pair_aggregate["preference_macro_f1"]
            >= float(monitoring["minimum_pair_preference_macro_f1"])
        ),
        "abstain_recall": (
            pair_aggregate["abstain_recall"]
            >= float(monitoring["minimum_abstain_recall"])
        ),
        "expected_calibration_error": (
            pair_aggregate["expected_calibration_error"]
            <= float(monitoring["maximum_ece"])
        ),
        "safety_gate_accuracy": (
            pair_aggregate["safety_gate_accuracy"]
            == float(monitoring["require_safety_accuracy"])
        ),
        "swap_consistency": (
            pair_aggregate["swap_consistency"]
            == float(monitoring["require_swap_consistency"])
        ),
        "selected_set_exact_accuracy": (
            array_aggregate["selected_set_exact_accuracy"]
            >= float(monitoring["minimum_selected_set_exact_accuracy"])
        ),
    }
    review_rows = []
    array_sizes = {
        str(row["array_id"]): int(row["candidate_count"])
        for row in public_arrays
    }
    for public in public_pairs:
        example_id = str(public["example_id"])
        recommendation = pair_recommendations[example_id]
        verifier = verifier_pairs[example_id]
        mismatch_axes = []
        if (
            recommendation["preference"]
            != verifier["target"]["preference"]
        ):
            mismatch_axes.append("preference")
        if (
            recommendation["rationale_code"]
            != verifier["target"]["rationale_code"]
        ):
            mismatch_axes.append("rationale")
        if mismatch_axes:
            review_rows.append(
                {
                    "schema_version": "ldt-shadow-review-key.v4",
                    "review_key_sha256": sha256_json(
                        {
                            "kind": "pair",
                            "example_id": example_id,
                            "manifest_sha256": manifest_sha256,
                        }
                    ),
                    "kind": "pair_disagreement",
                    "mismatch_axes": mismatch_axes,
                    "game": public["input"]["left"]["game"],
                    "fixture_pressure": public["input"]["left"][
                        "fixture_pressure"
                    ],
                    "array_size": array_sizes[str(public["array_id"])],
                    "contains_verifier_label": False,
                }
            )
    for public in public_arrays:
        array_id = str(public["array_id"])
        if (
            array_recommendations[array_id]["statuses"]
            != verifier_arrays[array_id]["target"]["statuses"]
        ):
            review_rows.append(
                {
                    "schema_version": "ldt-shadow-review-key.v4",
                    "review_key_sha256": sha256_json(
                        {
                            "kind": "array",
                            "array_id": array_id,
                            "manifest_sha256": manifest_sha256,
                        }
                    ),
                    "kind": "array_disagreement",
                    "mismatch_axes": ["statuses"],
                    "game": public["game"],
                    "fixture_pressure": public["fixture_pressure"],
                    "array_size": int(public["candidate_count"]),
                    "contains_verifier_label": False,
                }
            )
    report = {
        "schema_version": "ldt-shadow-comparison.v4",
        "comparison_revision": 1,
        "verifier_join_count": 1,
        "supersedes_comparison_sha256": None,
        "logic_only_correction": None,
        "experiment_id": contract["experiment_id"],
        "shadow_run_id": contract["shadow_run_id"],
        "contract_sha256": contract_sha256,
        "shadow_manifest_sha256": manifest_sha256,
        "model_sha256": model_binding["model_sha256"],
        "trainer_summary_sha256": model_binding["trainer_summary_sha256"],
        "shadow_run_receipt_sha256": sha256_file(shadow_run_receipt_path),
        "pair_recommendations_sha256": sha256_file(
            pair_recommendations_path
        ),
        "array_recommendations_sha256": sha256_file(
            array_recommendations_path
        ),
        "verifier_pair_labels_sha256": sha256_file(
            verifier_pair_labels_path
        ),
        "verifier_array_labels_sha256": sha256_file(
            verifier_array_labels_path
        ),
        "verifier_authority": "ENVIRONMENT_OWNED",
        "shadow_authority": "MODEL_RECOMMENDATION_ONLY",
        "shadow_decision_effect": "NONE",
        "verifier_decisions_mutated": False,
        "pair_aggregate": pair_aggregate,
        "array_aggregate": array_aggregate,
        "pair_by_game": pair_by_game,
        "array_by_game": array_by_game,
        "array_by_size": array_by_size,
        "monitoring_checks": monitoring_checks,
        "monitoring_passed": all(monitoring_checks.values()),
        "review_queue_count": len(review_rows),
        "review_queue_contains_verifier_labels": False,
        "training_from_disagreements_authorized": False,
        "corpus_promotion_authorized": False,
        "larger_model_training_authorized": False,
        "universal_scalar_champion": None,
        "claim_boundary": contract["claim_scope"],
    }
    if supersedes_comparison_path is not None:
        superseded = json.loads(
            supersedes_comparison_path.read_text(encoding="utf-8")
        )
        binding_fields = (
            "contract_sha256",
            "shadow_manifest_sha256",
            "model_sha256",
            "trainer_summary_sha256",
            "shadow_run_receipt_sha256",
            "pair_recommendations_sha256",
            "array_recommendations_sha256",
            "verifier_pair_labels_sha256",
            "verifier_array_labels_sha256",
        )
        if superseded["schema_version"] != report["schema_version"] or any(
            superseded[field] != report[field] for field in binding_fields
        ):
            raise ValueError("superseded shadow comparison bindings differ")
        report["comparison_revision"] = int(
            superseded.get("comparison_revision", 1)
        ) + 1
        report["supersedes_comparison_sha256"] = sha256_file(
            supersedes_comparison_path
        )
        report["logic_only_correction"] = (
            "Macro F1 slices exclude classes with zero verifier support."
        )
    out_dir.mkdir(parents=True, exist_ok=False)
    report_path = out_dir / "shadow_comparison.json"
    review_path = out_dir / "hash_only_review_queue.jsonl"
    _write_json(report_path, report)
    _write_jsonl(
        review_path,
        sorted(review_rows, key=lambda row: row["review_key_sha256"]),
    )
    _write_json(
        out_dir / "comparison_receipt.json",
        {
            "schema_version": "ldt-shadow-comparison-receipt.v4",
            "shadow_comparison_sha256": sha256_file(report_path),
            "comparison_revision": report["comparison_revision"],
            "verifier_join_count": 1,
            "supersedes_comparison_sha256": report[
                "supersedes_comparison_sha256"
            ],
            "logic_only_correction": report["logic_only_correction"],
            "models_predictions_labels_or_thresholds_changed": False,
            "hash_only_review_queue_sha256": sha256_file(review_path),
            "review_queue_contains_verifier_labels": False,
            "shadow_decision_effect": "NONE",
            "verifier_decisions_mutated": False,
            "post_comparison_training_authorized": False,
        },
    )
    return report
