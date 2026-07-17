from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from datagoblin.domain.hypotheses import ResearchHypothesis
from datagoblin.domain.recipes import DataRecipe

from .scout import ScoutFindings
from .state import ResearchState


@dataclass(frozen=True)
class BackendProposal:
    scout_findings: ScoutFindings
    hypotheses: tuple[ResearchHypothesis, ...]
    recipe: DataRecipe


class GoblinBackend(Protocol):
    def propose(self, state: ResearchState) -> BackendProposal: ...


class CommandGoblinBackend:
    """Opt-in local model seam. This is process isolation, not a perfect sandbox."""

    def __init__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int = 30,
        maximum_output_bytes: int = 1_048_576,
        environment_allowlist: Sequence[str] = (),
    ) -> None:
        if not argv:
            raise ValueError("command backend requires a non-empty argv list")
        self.argv = tuple(argv)
        self.timeout_seconds = timeout_seconds
        self.maximum_output_bytes = maximum_output_bytes
        self.environment_allowlist = tuple(environment_allowlist)

    def propose(self, state: ResearchState) -> BackendProposal:
        allowed_env = {name: os.environ[name] for name in self.environment_allowlist if name in os.environ}
        with tempfile.TemporaryDirectory(prefix="datagoblin-model-") as directory:
            completed = subprocess.run(
                self.argv,
                input=json.dumps(state.to_jsonable()),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                cwd=Path(directory),
                env=allowed_env,
                shell=False,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"model command failed with exit code {completed.returncode}")
        if len(completed.stdout.encode("utf-8")) > self.maximum_output_bytes:
            raise ValueError("model command output exceeded configured cap")
        value = json.loads(completed.stdout)
        if not isinstance(value, Mapping):
            raise ValueError("model command must emit one JSON object")
        return BackendProposal(
            ScoutFindings(**value["scout_findings"]),
            tuple(ResearchHypothesis.from_jsonable(item) for item in value["hypotheses"]),
            DataRecipe.from_jsonable(value["recipe"]),
        )
