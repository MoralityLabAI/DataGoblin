#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from datagoblin.experiments.ldt_array_confirmation import (
    CONFIRMATION_FEATURE_WHITELIST,
    ConfirmationPairEncoder,
    load_confirmation_dataset,
)
from datagoblin.experiments.ldt_array_sorter import sha256_file
from datagoblin.experiments.ldt_edge_set_array_v10 import (
    EDGE_STATISTICS,
    EdgeDifferenceCandidateEncoder,
)
from datagoblin.experiments.ldt_listwise_array_v9 import (
    IndependentCandidateEncoder,
)
from datagoblin.experiments.ldt_target_independent_v7 import (
    INDEPENDENT_FEATURE_FORBIDDEN_FRAGMENTS,
)
from datagoblin.serialization import dump_structured, load_structured


def prepare(
    preregistration_path: Path,
    dataset_manifest_path: Path,
    out_path: Path,
) -> dict[str, object]:
    preregistration = load_structured(preregistration_path)
    if preregistration["status"] != "frozen_before_dataset_generation":
        raise ValueError("confirmation preregistration is not frozen")
    (
        manifest,
        train_pairs,
        heldout_pairs,
        train_arrays,
        heldout_arrays,
        train_reports,
        heldout_reports,
    ) = load_confirmation_dataset(dataset_manifest_path)
    contract = preregistration["dataset_contract"]
    observed = {
        "train_pairs": len(train_pairs),
        "heldout_pairs": len(heldout_pairs),
        "train_arrays": len(train_arrays),
        "heldout_arrays": len(heldout_arrays),
        "train_reports": len(train_reports),
        "heldout_reports": len(heldout_reports),
    }
    expected = {
        name: int(contract[f"required_{name}"])
        for name in observed
    }
    if observed != expected:
        raise ValueError("confirmation counts differ from preregistration")
    if manifest["feature_whitelist"] != list(CONFIRMATION_FEATURE_WHITELIST):
        raise ValueError("confirmation feature whitelist differs")
    feature_independence_audit: dict[str, object] | None = None
    schema_version = str(preregistration.get("schema_version"))
    if schema_version in {
        "7.0",
        "8.0",
        "9.0",
        "10.0",
        "11.0",
        "12.0",
        "13.0",
        "14.0",
    }:
        feature_views = {
            str(model["feature_view"])
            for model in preregistration["models"]["candidates"]
        }
        if feature_views != {"independent"}:
            raise ValueError(
                "target-independent candidates include a derived feature view"
            )
        independent_names = ConfirmationPairEncoder.fit(
            train_pairs
        ).feature_names("independent")
        forbidden_hits = [
            name
            for name in independent_names
            if any(
                fragment in name
                for fragment in INDEPENDENT_FEATURE_FORBIDDEN_FRAGMENTS
            )
        ]
        if forbidden_hits:
            raise ValueError(
                "independent feature view contains forbidden proxies"
            )
        feature_independence_audit = {
            "candidate_feature_views": sorted(feature_views),
            "independent_feature_count": len(independent_names),
            "forbidden_feature_hits": forbidden_hits,
            "typed_or_lattice_views_present": False,
            "audit_split": "development_only",
        }
        if schema_version in {
            "8.0",
            "9.0",
            "10.0",
            "11.0",
            "12.0",
            "13.0",
            "14.0",
        }:
            support = manifest.get("development_support_audit")
            if not support or not all(
                int(count) >= int(
                    preregistration["dataset_contract"][
                        "minimum_lineages_per_critical_class"
                    ]
                )
                for axis in support["lineage_support"].values()
                for count in axis.values()
            ):
                raise ValueError(
                    "v8 development critical-class support is incomplete"
                )
            objectives = [
                {
                    "candidate_id": str(model["model_id"]),
                    "paired_swap_batches": bool(
                        model["paired_swap_batches"]
                    ),
                    "antisymmetric_inference": bool(
                        model["antisymmetric_inference"]
                    ),
                    "array_consistency_loss_weight": float(
                        model["array_consistency_loss_weight"]
                    ),
                }
                for model in preregistration["models"]["candidates"]
            ]
            if any(
                row["array_consistency_loss_weight"] < 0.0
                or row["array_consistency_loss_weight"] > 2.0
                for row in objectives
            ):
                raise ValueError(
                    "array objective weight is out of bounds"
                )
            if schema_version == "8.0" and not any(
                row["paired_swap_batches"]
                and row["antisymmetric_inference"]
                and row["array_consistency_loss_weight"] > 0.0
                for row in objectives
            ):
                raise ValueError(
                    "v8 lacks a complete structured objective"
                )
            feature_independence_audit[
                "structured_objectives"
            ] = objectives
        if schema_version == "9.0":
            listwise = [
                {
                    "candidate_id": str(model["model_id"]),
                    **dict(model["listwise_status"]),
                }
                for model in preregistration["models"]["candidates"]
            ]
            enabled = [
                row for row in listwise if bool(row["enabled"])
            ]
            if (
                len(enabled) < 2
                or not any(
                    not bool(row["enabled"])
                    and row["family"]
                    == "deterministic_pair_reducer"
                    for row in listwise
                )
                or any(
                    row["family"]
                    != "looped_permutation_equivariant_status"
                    or not 1 <= int(row["hidden_width"]) <= 16
                    or not 1 <= int(row["loop_count"]) <= 4
                    or not 1 <= int(row["epochs"]) <= 600
                    or not 0.0
                    < float(row["learning_rate"])
                    <= 0.1
                    for row in enabled
                )
                or any(
                    float(model["array_consistency_loss_weight"])
                    != 0.0
                    for model in preregistration["models"]["candidates"]
                )
                or any(
                    not bool(model["paired_swap_batches"])
                    or not bool(model["antisymmetric_inference"])
                    for model in preregistration["models"]["candidates"]
                )
            ):
                raise ValueError(
                    "v9 listwise ablation contract is incomplete"
                )
            candidate_encoder = IndependentCandidateEncoder()
            feature_independence_audit["listwise_status"] = {
                "candidates": listwise,
                "candidate_encoder": candidate_encoder.receipt(),
                "direct_array_targets": "development_only",
                "heldout_array_targets_available": False,
            }
        if schema_version == "10.0":
            models = preregistration["models"]["candidates"]
            listwise = [
                {
                    "candidate_id": str(model["model_id"]),
                    **dict(model["listwise_status"]),
                }
                for model in models
            ]
            reducers = [row for row in listwise if not bool(row["enabled"])]
            recurrent = [
                row
                for row in listwise
                if row["family"]
                == "looped_permutation_equivariant_status"
            ]
            edge = [
                row
                for row in listwise
                if row["family"]
                == "edge_difference_permutation_equivariant_status"
            ]
            if (
                len(reducers) != 1
                or reducers[0]["family"] != "deterministic_pair_reducer"
                or len(recurrent) != 1
                or len(edge) != 2
                or any(
                    not bool(row["enabled"])
                    or not 1 <= int(row["hidden_width"]) <= 16
                    or not 1 <= int(row["loop_count"]) <= 4
                    or not 1 <= int(row["epochs"]) <= 600
                    or not 0.0 < float(row["learning_rate"]) <= 0.1
                    for row in recurrent + edge
                )
                or recurrent[0]["candidate_encoder"]
                != "independent_candidate"
                or int(recurrent[0]["message_steps"]) != 0
                or recurrent[0]["set_coupled_loss"]["family"] != "none"
                or float(recurrent[0]["set_coupled_loss"]["weight"]) != 0.0
                or any(
                    row["candidate_encoder"] != "edge_difference_message"
                    or int(row["message_steps"]) != 1
                    or tuple(row["edge_statistics"]) != EDGE_STATISTICS
                    for row in edge
                )
            ):
                raise ValueError("v10 listwise architecture matrix is incomplete")
            factorized = [
                row
                for row in edge
                if row["set_coupled_loss"]["family"] == "none"
                and float(row["set_coupled_loss"]["weight"]) == 0.0
            ]
            coupled = [
                row
                for row in edge
                if row["set_coupled_loss"]["family"]
                == "soft_dice_selected_set"
                and 0.0 < float(row["set_coupled_loss"]["weight"]) <= 2.0
            ]
            comparable_fields = (
                "family",
                "candidate_encoder",
                "message_steps",
                "edge_statistics",
                "hidden_width",
                "loop_count",
                "epochs",
                "learning_rate",
            )
            if (
                len(factorized) != 1
                or len(coupled) != 1
                or any(
                    factorized[0][field] != coupled[0][field]
                    for field in comparable_fields
                )
                or any(
                    float(model["array_consistency_loss_weight"]) != 0.0
                    or not bool(model["paired_swap_batches"])
                    or not bool(model["antisymmetric_inference"])
                    for model in models
                )
            ):
                raise ValueError("v10 set-coupled ablation is not matched")
            metric_gate = preregistration.get("metric_gate", {})
            if (
                metric_gate.get("heldout_status")
                != "supported_on_declared_holdout"
                or set(metric_gate.get("accepted_metrics", ()))
                != {
                    "exact_status_vector_accuracy",
                    "exact_selected_set_accuracy",
                }
            ):
                raise ValueError("v10 metric robustness gate is incomplete")
            feature_independence_audit["listwise_status"] = {
                "candidates": listwise,
                "candidate_encoders": {
                    "independent_candidate": (
                        IndependentCandidateEncoder().receipt()
                    ),
                    "edge_difference_message": (
                        EdgeDifferenceCandidateEncoder().receipt()
                    ),
                },
                "set_coupled_ablation_matched": True,
                "metric_gate": metric_gate,
                "direct_array_targets": "development_only",
                "heldout_array_targets_available": False,
            }
        if schema_version == "11.0":
            models = preregistration["models"]["candidates"]
            listwise = [
                {
                    "candidate_id": str(model["model_id"]),
                    **dict(model["listwise_status"]),
                }
                for model in models
            ]
            reducers = [row for row in listwise if not bool(row["enabled"])]
            recurrent = [
                row
                for row in listwise
                if row["family"]
                == "looped_permutation_equivariant_status"
            ]
            cardinality = [
                row
                for row in listwise
                if row["family"]
                == "looped_cardinality_permutation_equivariant_status"
            ]
            enabled = recurrent + cardinality
            if (
                len(reducers) != 1
                or reducers[0]["family"] != "deterministic_pair_reducer"
                or len(recurrent) != 1
                or len(cardinality) != 2
                or any(
                    not bool(row["enabled"])
                    or row["candidate_encoder"] != "independent_candidate"
                    or int(row["message_steps"]) != 0
                    or not 1 <= int(row["hidden_width"]) <= 16
                    or not 1 <= int(row["loop_count"]) <= 4
                    or not 1 <= int(row["epochs"]) <= 600
                    or not 0.0 < float(row["learning_rate"]) <= 0.1
                    for row in enabled
                )
            ):
                raise ValueError(
                    "v11 cardinality architecture matrix is incomplete"
                )
            cardinality_specs = [
                dict(row["selected_cardinality"])
                for row in cardinality
            ]
            comparable_fields = (
                "family",
                "candidate_encoder",
                "message_steps",
                "hidden_width",
                "loop_count",
                "epochs",
                "learning_rate",
            )
            decoders = {
                str(spec["decoder"])
                for spec in cardinality_specs
            }
            if (
                any(
                    cardinality[0][field] != cardinality[1][field]
                    for field in comparable_fields
                )
                or decoders != {"factorized", "joint_selected_count_map"}
                or any(
                    int(spec["max_candidates"]) != 6
                    or spec["loss_family"]
                    != "categorical_selected_count_cross_entropy"
                    or not 0.0 < float(spec["loss_weight"]) <= 2.0
                    or bool(spec["shared_state_gradient"])
                    for spec in cardinality_specs
                )
                or any(
                    cardinality_specs[0][field]
                    != cardinality_specs[1][field]
                    for field in (
                        "max_candidates",
                        "loss_family",
                        "loss_weight",
                        "shared_state_gradient",
                    )
                )
                or any(
                    float(model["array_consistency_loss_weight"]) != 0.0
                    or not bool(model["paired_swap_batches"])
                    or not bool(model["antisymmetric_inference"])
                    for model in models
                )
            ):
                raise ValueError("v11 decoder ablation is not matched")
            metric_gate = preregistration.get("metric_gate", {})
            if (
                metric_gate.get("heldout_status")
                != "supported_on_declared_holdout"
                or set(metric_gate.get("accepted_metrics", ()))
                != {
                    "exact_status_vector_accuracy",
                    "exact_selected_set_accuracy",
                }
            ):
                raise ValueError("v11 metric robustness gate is incomplete")
            feature_independence_audit["listwise_status"] = {
                "candidates": listwise,
                "candidate_encoder": (
                    IndependentCandidateEncoder().receipt()
                ),
                "cardinality_decoder_ablation_matched": True,
                "cardinality_shared_state_gradient": False,
                "metric_gate": metric_gate,
                "direct_array_targets": "development_only",
                "heldout_array_targets_available": False,
            }
        if schema_version in {"12.0", "14.0"}:
            models = preregistration["models"]["candidates"]
            listwise = [
                {
                    "candidate_id": str(model["model_id"]),
                    **dict(model["listwise_status"]),
                }
                for model in models
            ]
            reducers = [row for row in listwise if not bool(row["enabled"])]
            recurrent = [
                row
                for row in listwise
                if row["family"]
                == "looped_permutation_equivariant_status"
            ]
            exact = [
                row
                for row in listwise
                if row["family"]
                == "exact_globally_normalized_array_status"
            ]
            enabled = recurrent + exact
            if (
                len(reducers) != 1
                or reducers[0]["family"] != "deterministic_pair_reducer"
                or len(recurrent) != 1
                or len(exact) != 2
                or any(
                    not bool(row["enabled"])
                    or row["candidate_encoder"] != "independent_candidate"
                    or int(row["message_steps"]) != 0
                    or not 1 <= int(row["hidden_width"]) <= 16
                    or not 1 <= int(row["loop_count"]) <= 4
                    or not 1 <= int(row["epochs"]) <= 600
                    or not 0.0 < float(row["learning_rate"]) <= 0.1
                    for row in enabled
                )
            ):
                raise ValueError(
                    "v12 exact-array architecture matrix is incomplete"
                )
            exact_specs = [
                dict(row["exact_array_likelihood"])
                for row in exact
            ]
            comparable_fields = (
                "family",
                "candidate_encoder",
                "message_steps",
                "hidden_width",
                "loop_count",
                "epochs",
                "learning_rate",
            )
            decoders = {str(spec["decoder"]) for spec in exact_specs}
            if (
                any(
                    exact[0][field] != exact[1][field]
                    for field in comparable_fields
                )
                or decoders != {"factorized", "global_selected_count_map"}
                or any(
                    int(spec["max_candidates"]) != 6
                    or spec["loss_family"]
                    != "exact_globally_normalized_status_vector_nll"
                    or float(spec["loss_weight"]) != 1.0
                    or spec["normalization"]
                    != "exact_selected_subset_marginalization"
                    or spec["represented_status_vectors"]
                    != "all_four_status_assignments"
                    or not bool(spec["shared_state_gradient"])
                    or spec["class_balancing"]
                    != "none_proper_likelihood"
                    for spec in exact_specs
                )
                or any(
                    exact_specs[0][field] != exact_specs[1][field]
                    for field in (
                        "max_candidates",
                        "loss_family",
                        "loss_weight",
                        "normalization",
                        "represented_status_vectors",
                        "shared_state_gradient",
                        "class_balancing",
                    )
                )
                or any(
                    float(model["array_consistency_loss_weight"]) != 0.0
                    or not bool(model["paired_swap_batches"])
                    or not bool(model["antisymmetric_inference"])
                    for model in models
                )
            ):
                raise ValueError("v12 exact-array decoder ablation is not matched")
            metric_gate = preregistration.get("metric_gate", {})
            if (
                metric_gate.get("heldout_status")
                != "supported_on_declared_holdout"
                or set(metric_gate.get("accepted_metrics", ()))
                != {
                    "exact_status_vector_accuracy",
                    "exact_selected_set_accuracy",
                }
            ):
                raise ValueError("v12 metric robustness gate is incomplete")
            feature_independence_audit["listwise_status"] = {
                "candidates": listwise,
                "candidate_encoder": (
                    IndependentCandidateEncoder().receipt()
                ),
                "exact_array_decoder_ablation_matched": True,
                "exact_array_shared_state_gradient": True,
                "metric_gate": metric_gate,
                "direct_array_targets": "development_only",
                "heldout_array_targets_available": False,
            }
            if schema_version == "14.0":
                selection = preregistration["development_selection"]
                heldout = preregistration["heldout_evaluation"]
                exact_global_ids = {
                    str(row["candidate_id"])
                    for row in exact
                    if row["exact_array_likelihood"]["decoder"]
                    == "global_selected_count_map"
                }
                if (
                    selection.get("selection_semantics")
                    != "gate_then_rank_per_architecture"
                    or not bool(
                        selection.get(
                            "retain_rank_only_diagnostic_if_none"
                        )
                    )
                    or set(selection.get("primary_candidate_ids", ()))
                    != exact_global_ids
                    or not bool(
                        heldout.get("primary_only_confirmation_packet")
                    )
                    or float(
                        heldout[
                            "minimum_exact_status_vector_accuracy"
                        ]
                    )
                    != float(
                        selection["minimum_mean_exact_array_accuracy"]
                    )
                    or float(
                        heldout["minimum_selected_set_exact_accuracy"]
                    )
                    != float(
                        selection[
                            "minimum_mean_selected_set_exact_accuracy"
                        ]
                    )
                    or float(heldout["minimum_safety_gate_accuracy"])
                    != float(
                        selection["required_mean_safety_gate_accuracy"]
                    )
                    or float(heldout["minimum_swap_consistency"])
                    != float(selection["required_mean_swap_consistency"])
                    or float(heldout["minimum_preference_macro_f1"])
                    != float(
                        selection["minimum_mean_preference_macro_f1"]
                    )
                    or float(heldout["minimum_abstain_recall"])
                    != float(selection["minimum_mean_abstain_recall"])
                    or float(heldout["minimum_status_macro_f1"])
                    != float(selection["minimum_mean_status_macro_f1"])
                ):
                    raise ValueError(
                        "v14 gate-first confirmation contract is incomplete"
                    )
                feature_independence_audit["selection_policy"] = {
                    "semantics": (
                        "gate_then_rank_per_architecture"
                    ),
                    "primary_candidate_ids": sorted(exact_global_ids),
                    "primary_only_confirmation_packet": True,
                    "exact_status_vector_gate": float(
                        heldout[
                            "minimum_exact_status_vector_accuracy"
                        ]
                    ),
                    "exact_selected_set_gate": float(
                        heldout["minimum_selected_set_exact_accuracy"]
                    ),
                    "safety_gate": float(
                        heldout["minimum_safety_gate_accuracy"]
                    ),
                    "swap_gate": float(
                        heldout["minimum_swap_consistency"]
                    ),
                    "preference_macro_f1_gate": float(
                        heldout["minimum_preference_macro_f1"]
                    ),
                    "abstain_recall_gate": float(
                        heldout["minimum_abstain_recall"]
                    ),
                    "status_macro_f1_gate": float(
                        heldout["minimum_status_macro_f1"]
                    ),
                }
        if schema_version == "13.0":
            models = preregistration["models"]["candidates"]
            listwise = [
                {
                    "candidate_id": str(model["model_id"]),
                    **dict(model["listwise_status"]),
                }
                for model in models
            ]
            reducers = [row for row in listwise if not bool(row["enabled"])]
            recurrent = [
                row
                for row in listwise
                if row["family"]
                == "looped_permutation_equivariant_status"
            ]
            selected_count = [
                row
                for row in listwise
                if row["family"]
                == "exact_globally_normalized_array_status"
            ]
            composition = [
                row
                for row in listwise
                if row["family"]
                == "exact_composition_normalized_array_status"
            ]
            enabled = recurrent + selected_count + composition
            if (
                len(reducers) != 1
                or reducers[0]["family"] != "deterministic_pair_reducer"
                or len(recurrent) != 1
                or len(selected_count) != 1
                or len(composition) != 2
                or any(
                    not bool(row["enabled"])
                    or row["candidate_encoder"] != "independent_candidate"
                    or int(row["message_steps"]) != 0
                    or not 1 <= int(row["hidden_width"]) <= 16
                    or not 1 <= int(row["loop_count"]) <= 4
                    or not 1 <= int(row["epochs"]) <= 600
                    or not 0.0 < float(row["learning_rate"]) <= 0.1
                    for row in enabled
                )
            ):
                raise ValueError(
                    "v13 composition architecture matrix is incomplete"
                )
            selected_spec = dict(
                selected_count[0]["exact_array_likelihood"]
            )
            if (
                int(selected_spec["max_candidates"]) != 6
                or selected_spec["loss_family"]
                != "exact_globally_normalized_status_vector_nll"
                or selected_spec["decoder"]
                != "global_selected_count_map"
                or selected_spec["normalization"]
                != "exact_selected_subset_marginalization"
                or not bool(selected_spec["shared_state_gradient"])
            ):
                raise ValueError("v13 selected-count control is invalid")
            composition_specs = [
                dict(row["composition_likelihood"])
                for row in composition
            ]
            comparable_fields = (
                "family",
                "candidate_encoder",
                "message_steps",
                "hidden_width",
                "loop_count",
                "epochs",
                "learning_rate",
            )
            decoders = {
                str(spec["decoder"])
                for spec in composition_specs
            }
            if (
                any(
                    composition[0][field] != composition[1][field]
                    for field in comparable_fields
                )
                or decoders != {"factorized", "global_composition_map"}
                or any(
                    int(spec["max_candidates"]) != 6
                    or int(spec["composition_parameter_count"]) != 210
                    or spec["loss_family"]
                    != "exact_status_composition_vector_nll"
                    or float(spec["loss_weight"]) != 1.0
                    or spec["normalization"]
                    != "exact_all_status_assignment_enumeration"
                    or spec["represented_status_vectors"]
                    != "all_four_status_assignments"
                    or not bool(spec["shared_state_gradient"])
                    or spec["class_balancing"]
                    != "none_proper_likelihood"
                    for spec in composition_specs
                )
                or any(
                    composition_specs[0][field]
                    != composition_specs[1][field]
                    for field in (
                        "max_candidates",
                        "composition_parameter_count",
                        "loss_family",
                        "loss_weight",
                        "normalization",
                        "represented_status_vectors",
                        "shared_state_gradient",
                        "class_balancing",
                    )
                )
                or any(
                    float(model["array_consistency_loss_weight"]) != 0.0
                    or not bool(model["paired_swap_batches"])
                    or not bool(model["antisymmetric_inference"])
                    for model in models
                )
            ):
                raise ValueError("v13 composition decoder ablation is not matched")
            metric_gate = preregistration.get("metric_gate", {})
            if (
                metric_gate.get("heldout_status")
                != "supported_on_declared_holdout"
                or set(metric_gate.get("accepted_metrics", ()))
                != {
                    "exact_status_vector_accuracy",
                    "exact_selected_set_accuracy",
                }
            ):
                raise ValueError("v13 metric robustness gate is incomplete")
            feature_independence_audit["listwise_status"] = {
                "candidates": listwise,
                "candidate_encoder": (
                    IndependentCandidateEncoder().receipt()
                ),
                "selected_count_control_present": True,
                "composition_decoder_ablation_matched": True,
                "composition_shared_state_gradient": True,
                "metric_gate": metric_gate,
                "direct_array_targets": "development_only",
                "heldout_array_targets_available": False,
            }
    hard_caps = preregistration["hard_caps"]
    if hard_caps != {
        "wrapper": "scripts/run_training_gym_capped_wsl.sh",
        "ram_mb": 2048,
        "cpu_percent": 50,
        "io_mb_per_second": 50,
        "wall_clock_seconds": 900,
        "swap_bytes": 0,
        "fallback_if_unavailable": "abort_without_training",
    }:
        raise ValueError("confirmation hard caps differ from the approved contract")
    plan: dict[str, object] = {
        "schema_version": "ldt-array-confirmation-trainer-plan.v3",
        "preregistration_schema_version": schema_version,
        "experiment_id": preregistration["experiment_id"],
        "training_task_id": preregistration["training_task_id"],
        "operator_approval": "user instruction: continue",
        "preregistration_sha256": sha256_file(preregistration_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "dataset_counts": observed,
        "preference_labels": manifest["preference_labels"],
        "rationale_labels": manifest["rationale_labels"],
        "array_status_labels": manifest["array_status_labels"],
        "candidate_models": preregistration["models"]["candidates"],
        "controls": preregistration["models"]["controls"],
        "training": preregistration["training"],
        "confidence_calibration": preregistration["confidence_calibration"],
        "hard_caps": hard_caps,
        "metric_equivalence_tolerance": manifest["metric_equivalence_tolerance"],
        "heldout_pair_label_access": False,
        "heldout_array_label_access": False,
        "external_checkpoint": None,
        "network": False,
        "sorter_authority": "MODEL_RECOMMENDATION_ONLY",
        "larger_model_training_authorized": False,
    }
    if feature_independence_audit is not None:
        plan["feature_independence_audit"] = feature_independence_audit
    dump_structured(out_path, plan)
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a public-only LDT confirmation trainer plan"
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    prepare(arguments.preregistration, arguments.dataset_manifest, arguments.out)
    print(arguments.out)
