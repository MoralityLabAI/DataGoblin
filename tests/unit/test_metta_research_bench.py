import asyncio
import sys

import pytest

from MeTTa.ram_trajectory import STOP, RepairVerifier, training_trajectory
from MeTTa.sft_oracle import first_state, make_task, replay

vf = pytest.importorskip("verifiers.v1")


@pytest.mark.skipif(sys.platform == "win32", reason="Pinned Verifiers runtime requires POSIX file locking")
def test_registered_taskset_scores_legal_repairs_through_prime_verifiers():
    from metta_research_bench.experiment import score_submission
    from metta_research_bench.taskset import BenchConfig, submitted_actions

    taskset = vf.load_taskset(BenchConfig(roots_per_family=1))
    task = next(iter(taskset))
    world = make_task(task.data.family, task.data.source_seed)
    target = RepairVerifier(world).optimum["selected"]
    selected = first_state(world, task.data.source_seed)
    actions = [row["action"] for row in replay(world, selected, target)]
    good = asyncio.run(score_submission(task, actions, "test/gold"))
    assert good.reward == 1.0
    assert good.metrics["unsafe_additions"] == 0
    assert good.metrics["format_valid"] == 1
    assert good.rewards["verified_safe_completion"].score == 1.0
    malformed = asyncio.run(score_submission(task, [STOP, STOP], "test/malformed"))
    assert malformed.reward == 0.0
    assert malformed.metrics["format_valid"] == 0
    assert submitted_actions('[{"op":"stop","id":"none","reward":1}]') is None


def test_metta_curation_preserves_per_root_quota_and_terminal_label():
    from metta_research_bench.curation import curate

    pytest.importorskip("hyperon")
    rows = [training_trajectory("coverage", 21000), training_trajectory("revision", 21008)]
    datasets, receipts = curate(rows)
    baseline, treatment = datasets["chronological"], datasets["metta_curated"]
    assert any(row["selected_steps"]["chronological"] != row["selected_steps"]["metta_curated"] for row in receipts)
    for first, second in zip(baseline, treatment, strict=True):
        assert first["root"] == second["root"]
        assert first["source_sha256"] == second["source_sha256"]
        assert len(first["transitions"]) == len(second["transitions"])
        assert first["transitions"][-1] == second["transitions"][-1]
        assert first["transitions"][-1]["next_edit"] == STOP
