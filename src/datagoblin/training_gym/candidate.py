from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from datagoblin.domain.recipes import DataRecipe
from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured


def write_mock_candidate(
    rows: Iterable[Mapping[str, Any]],
    out_dir: Path,
    *,
    candidate_id: str,
    kind: str,
    dataset_manifest_path: Path,
) -> tuple[Path, Path]:
    if kind not in {"constant_accept", "structural_smoke"}:
        raise ValueError(f"unknown mock candidate: {kind}")
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            public_input = {
                "example_id": row["example_id"],
                "recipe": row["recipe"],
            }
            prediction = _mock_prediction(public_input, kind)
            handle.write(json.dumps(prediction, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "architecture_family": "negative_control" if kind == "constant_accept" else "structural_control",
        "mock": True,
        "weights_loaded": False,
        "training_splits": [],
        "training_split_hashes": {},
        "selection_splits": [],
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "trainable_parameters": 0,
        "total_parameters": 0,
        "maximum_loops": 1 if kind == "constant_accept" else 2,
        "inference_update_budget": 1 if kind == "constant_accept" else 2,
        "resource_summary": {
            "caps_enforced": False,
            "caps_not_required_reason": "deterministic mock performs no training and loads no weights",
            "peak_ram_mb": 0.0,
            "avg_ram_mb": 0.0,
            "peak_io_mb_s": 0.0,
            "cpu_pct": 0.0,
            "steps_completed": 0,
            "cleanup_passed": True,
        },
        "claim_boundary": "Pipeline control only; this candidate is not evidence for any trainable architecture.",
    }
    manifest_path = out_dir / "candidate_manifest.json"
    dump_structured(manifest_path, manifest)
    return manifest_path, predictions_path


def _mock_prediction(public_input: Mapping[str, Any], kind: str) -> dict[str, Any]:
    if kind == "constant_accept":
        return {
            "example_id": public_input["example_id"],
            "parsed": True,
            "decision": "ACCEPT",
            "violations": [],
            "confidence": 0.55,
            "abstained": False,
            "halted": True,
            "loop_trace": [{"loop": 1, "decision": "ACCEPT", "confidence": 0.55}],
        }
    recipe = DataRecipe.from_jsonable(public_input["recipe"])
    violations: list[str] = []
    split = recipe.split_contract
    exposed = {str(item).lower() for item in split.get("builder_exposed_fields", [])}
    if split.get("evaluation_seeds_exposed") or any("evaluation_seed" in item for item in exposed):
        violations.append("evaluation_leakage")
    verifier = recipe.independent_verifier_requirements
    if not verifier.get("required") or not verifier.get("component"):
        violations.append("missing_independent_verifier")
    if any(float(recipe.resource_budget.get(name, 0)) <= 0 for name in recipe.resource_budget):
        violations.append("unbounded_budget")
    if not recipe.falsification_criteria:
        violations.append("missing_falsifier")
    critical = {"evaluation_leakage", "missing_independent_verifier"}
    decision = "REJECT" if any(item in critical for item in violations) else "REVISE" if violations else "ACCEPT"
    confidence = 0.82 if violations else 0.65
    return {
        "example_id": public_input["example_id"],
        "parsed": True,
        "decision": decision,
        "violations": sorted(violations),
        "confidence": confidence,
        "abstained": False,
        "halted": True,
        "loop_trace": [
            {"loop": 1, "decision": "ACCEPT", "confidence": 0.51},
            {"loop": 2, "decision": decision, "confidence": confidence},
        ],
    }
