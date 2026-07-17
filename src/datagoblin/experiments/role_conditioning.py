from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datagoblin.domain.recipes import DataRecipe
from datagoblin.hrm_text.instrumented_forward import TraceOptions, run_instrumented_forward
from datagoblin.hrm_text.prompting import ROLE_CONDITIONS, build_verdict_prompt, schema_complexity
from datagoblin.hrm_text.trace_protocol import TraceableRecurrentModel
from datagoblin.serialization import dump_structured

from .recipe_verdict_dataset import authoritative_label


def run_role_conditioning_ablation(
    model: TraceableRecurrentModel,
    dataset_rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
    *,
    maximum_examples: int = 12,
    trace_options: TraceOptions | None = None,
) -> dict[str, Any]:
    selected = list(dataset_rows)[:maximum_examples]
    by_condition: dict[str, list[bool]] = defaultdict(list)
    for condition in ROLE_CONDITIONS:
        for row in selected:
            recipe = DataRecipe.from_jsonable(row["recipe"])
            label = authoritative_label(recipe)
            prompt = build_verdict_prompt(recipe, condition=condition)
            summary = run_instrumented_forward(
                model,
                prompt,
                authoritative_decision=label["decision"],
                authoritative_violations=label["violations"],
                schema_complexity=schema_complexity(recipe),
                artifact_hashes=(str(row["recipe_hash"]),),
                options=trace_options,
            )
            by_condition[condition].append(bool(summary.correctness_label))
    report = {
        "schema_version": "1.0",
        "model_id": model.model_id,
        "evidence_status": (
            "pipeline_smoke_only_not_hrm_text_evidence"
            if model.model_id.startswith("deterministic-mock")
            else "optional_checkpoint_role_ablation"
        ),
        "conditions": list(ROLE_CONDITIONS),
        "new_tokenizer_tokens_added": False,
        "examples_per_condition": len(selected),
        "exact_verdict_and_mask_accuracy": {
            condition: sum(values) / len(values) if values else 0.0 for condition, values in by_condition.items()
        },
        "interpretation": (
            "Textual role effects are prompt-conditioning observations, not evidence of distinct internal modules."
        ),
        "claim_boundary": [
            "Convergence does not imply correctness.",
            "This experiment does not establish alignment.",
            "Recurrent signals may merely track prompt complexity.",
            "HRM-specific value requires comparison with recurrent baselines.",
            "A null result is valid.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_structured(out_dir / "role_ablation_report.json", report)
    return report
