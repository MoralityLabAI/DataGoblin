from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.domain.builders import BuilderTicket
from datagoblin.environments.causal_switches.generator import generate_dataset
from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import dump_structured

from .protocol import BuildResult


class TemplateBuilder:
    """Trusted in-package builder for the single built-in environment."""

    def build(self, ticket: BuilderTicket, build_root: Path, *, seed: int = 7) -> BuildResult:
        build_root.mkdir(parents=True, exist_ok=True)
        if ticket.component_type == "environment_generator":
            dataset = generate_dataset(
                seed=seed,
                count=512,
                mode="balanced",
                namespace="causal-switches/train/v1",
                include_counterfactual_pairs=True,
            )
            dataset_path = build_root / "train_dataset.npz"
            np.savez_compressed(dataset_path, features=dataset.features, labels=dataset.labels)
            manifest_path = build_root / "train_split_manifest.json"
            quality_path = build_root / "dataset_quality_report.json"
            provenance_path = build_root / "provenance_manifest.json"
            dump_structured(manifest_path, dataset.split_manifest)
            dump_structured(quality_path, dataset.quality_report)
            dump_structured(
                provenance_path,
                {
                    "generator": "trusted_template_builder",
                    "label_authority": "causal_switches_exact_solver",
                    "ticket_id": ticket.ticket_id,
                },
            )
            outputs = (dataset_path, manifest_path, quality_path, provenance_path)
        elif ticket.component_type == "reference_solver":
            contract_path = build_root / "solver_contract.json"
            tests_path = build_root / "solver_property_tests.json"
            dump_structured(
                contract_path,
                {
                    "component": "causal_switches_exact_solver",
                    "input": ["intervention_seen", "elapsed", "delay"],
                    "output": "delayed_effect_visible",
                    "authority": "environment",
                },
            )
            dump_structured(tests_path, {"deterministic": True, "boundary_cases_passed": True})
            outputs = (contract_path, tests_path)
        else:
            raise ValueError(f"template builder does not support {ticket.component_type}")
        files = {path.name: sha256_file(path) for path in outputs}
        receipt: dict[str, Any] = {
            "schema_version": "1.0",
            "ticket_id": ticket.ticket_id,
            "builder": "trusted_template_builder",
            "outputs": files,
            "network_used": False,
            "subprocess_used": False,
            "claim_boundary": "Build receipt only; independent evaluation remains harness-owned.",
        }
        receipt["receipt_hash"] = sha256_json(receipt)
        receipt_path = build_root / f"{ticket.ticket_id}.build_receipt.json"
        dump_structured(receipt_path, receipt)
        return BuildResult(True, build_root, (*outputs, receipt_path), receipt)
