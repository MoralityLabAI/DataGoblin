from __future__ import annotations

from pathlib import Path

import pytest

from datagoblin.domain.learner import LearnerSnapshot
from datagoblin.domain.recipes import DataRecipe
from datagoblin.research.heuristic_backend import HeuristicGoblinBackend
from datagoblin.research.orchestrator import ResearchOrchestrator
from datagoblin.serialization import load_structured

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def snapshot() -> LearnerSnapshot:
    return LearnerSnapshot.from_jsonable(load_structured(ROOT / "examples" / "learner_snapshot.delayed_causality.json"))


@pytest.fixture
def good_recipe() -> DataRecipe:
    return DataRecipe.from_jsonable(load_structured(ROOT / "examples" / "recipe.good.yaml"))


@pytest.fixture
def orchestration(snapshot: LearnerSnapshot):
    return ResearchOrchestrator(HeuristicGoblinBackend()).propose(snapshot)
