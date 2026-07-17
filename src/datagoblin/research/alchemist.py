from __future__ import annotations

from datagoblin.domain.evidence import EvidenceClaim, EvidenceType
from datagoblin.domain.hypotheses import ResearchHypothesis
from datagoblin.domain.learner import LearnerSnapshot
from datagoblin.domain.recipes import DataRecipe, PredictedOutcome, TrajectorySource
from datagoblin.hashing import sha256_json

from .scout import ScoutFindings


class Alchemist:
    def propose(
        self,
        snapshot: LearnerSnapshot,
        findings: ScoutFindings,
        hypotheses: tuple[ResearchHypothesis, ...],
    ) -> DataRecipe:
        source_ref = f"sha256:{sha256_json(snapshot.to_jsonable())}"
        return DataRecipe(
            recipe_id="recipe_causal_switches_balanced_v1",
            schema_version="1.0",
            title="Role permutation and delayed-intervention curriculum",
            hypothesis_ids=tuple(item.hypothesis_id for item in hypotheses),
            target_capabilities=(
                "delayed_causality",
                "role_invariance",
                "counterfactual_consistency",
            ),
            environment_specification={
                "name": "causal_switches_v1",
                "task_kind": "executable",
                "harmless": True,
                "delays": [1, 2, 3, 4],
            },
            generator_specification={
                "component": "causal_switches_generator",
                "role_permutation": True,
                "counterfactual_pairs": True,
                "train_size": 512,
            },
            trajectory_source_mixture=(
                TrajectorySource("role_permuted", 0.55, EvidenceType.ENVIRONMENT_VERIFIED),
                TrajectorySource("counterfactual_pairs", 0.25, EvidenceType.ENVIRONMENT_VERIFIED),
                TrajectorySource("biased_control", 0.20, EvidenceType.ENVIRONMENT_VERIFIED),
            ),
            supervision_objectives=(
                {
                    "name": "delayed_effect_binary",
                    "weight": 0.8,
                    "label_provenance": "exact_solver",
                },
                {
                    "name": "counterfactual_consistency",
                    "weight": 0.2,
                    "label_provenance": "exact_solver",
                },
            ),
            curriculum_controller={
                "kind": "stratified",
                "axes": ["delay", "role_color", "intervention"],
                "unique_fraction": 1.0,
            },
            split_contract={
                "train_seed_namespace": "causal-switches/train/v1",
                "evaluation_seed_namespace": "causal-switches/eval-hidden/v1",
                "train_generator_family": "causal_switches_train_balanced_v1",
                "evaluation_generator_family": "causal_switches_holdout_permuted_v1",
                "held_out": True,
                "metric": "heldout_accuracy",
                "metric_frozen": True,
                "builder_exposed_fields": ["public_split_contract", "train_seed_namespace"],
                "evaluation_seeds_exposed": False,
                "exclude_failed_seeds": False,
            },
            counterfactual_and_symmetry_transforms=(
                "role_color_permutation",
                "toggle_intervention_pair",
            ),
            reference_solver_requirements={
                "required": True,
                "component": "causal_switches_exact_solver",
            },
            independent_verifier_requirements={
                "required": True,
                "component": "causal_switches_independent_verifier",
            },
            builder_permissions={
                "component": "template_builder",
                "network": True,
                "network_justification": "",
                "subprocess": False,
                "may_modify_ledger": False,
                "may_modify_evaluator": False,
            },
            resource_budget={
                "cpu_seconds": 120,
                "memory_mb": 512,
                "storage_mb": 128,
                "wall_clock_seconds": 120,
                "process_count": 1,
            },
            ablations=(
                {"name": "remove_role_permutation", "changed_variables": ["role_permutation"]},
                {"name": "narrow_delay_curriculum", "changed_variables": ["delay_width"]},
            ),
            controls=({"name": "biased_baseline", "compute_matched": True},),
            success_criteria=("Improved recipe exceeds baseline held-out accuracy by at least 0.05.",),
            falsification_criteria=(
                "No held-out gain and no reduction in color sensitivity under matched compute.",
                "Balanced data helps only when an explicit history feature is added.",
            ),
            predicted_outcomes=(
                PredictedOutcome(
                    "heldout_accuracy_delta",
                    0.02,
                    0.35,
                    0.55,
                    EvidenceType.MODEL_INFERRED,
                    (source_ref,),
                ),
            ),
            uncertainty={"epistemic": "high", "aleatoric": "low", "reason": "toy smoke prediction"},
            provenance=EvidenceClaim(
                EvidenceType.MODEL_INFERRED,
                "heuristic_alchemist",
                (source_ref,),
                "heuristic_alchemist",
            ),
            constitution_version="0.1.0",
            claim_boundary=(
                "A toy smoke-test recipe; it makes no claim about real learners, HRM/TRM/LDT, or a future Goblin model."
            ),
            metadata={
                "information_gain_rationale": (
                    "The two ablations distinguish shortcut learning from missing temporal representation."
                )
            },
        )
