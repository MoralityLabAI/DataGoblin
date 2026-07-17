from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


class SmallControlHarnessAdapter:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def capabilities(self) -> dict[str, list[str]]:
        patterns = {
            "experiment_cards": "results/cards/*.md",
            "run_manifests": "results/**/run_manifest.json",
            "receipts": "results/**/*.receipt.json",
            "result_jsonl": "results/**/*.jsonl",
            "harness_entry_points": "scripts/run_*harness*.py",
        }
        return {
            name: [path.relative_to(self.root).as_posix() for path in sorted(self.root.glob(pattern)) if path.is_file()]
            for name, pattern in patterns.items()
        }

    def normalize_reference(self, path: Path, claim_boundary: str) -> dict[str, Any]:
        resolved = path.resolve()
        resolved.relative_to(self.root)
        return {
            "upstream": "small_control_harness",
            "relative_path": resolved.relative_to(self.root).as_posix(),
            "evidence_type": "UNKNOWN",
            "claim_boundary": claim_boundary,
        }

    def construct_command(self, argv: Sequence[str]) -> list[str]:
        if not argv:
            raise ValueError("argv must not be empty")
        return [str(item) for item in argv]
