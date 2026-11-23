from pathlib import Path

import pytest


def stop_module():
    pytest.importorskip("torch")
    pytest.importorskip("research_gym.neural.looped_decoder")
    pytest.importorskip("verifiers.v1")
    from metta_research_bench.looped import stop_policy

    return stop_policy


def test_stop_witness_respects_scope_feasibility_and_rank_lexicographically():
    pytest.importorskip("hyperon")
    module = stop_module()
    path = Path(module.__file__).with_name("stop_witness.metta")
    witness = module.StopWitness(path.read_text(encoding="utf-8"))
    facts = """
    (PublicRank CurrentState (Rank 3 18 22 500))
    (PublicRank LowerCost (Rank 3 18 20 900))
    (PublicRank Earlier (Rank 3 18 22 100))
    (PublicRank Shrunk (Rank 2 21 13 200))
    (CheckedStopNeighbor CurrentState Cost Accepted Feasible LowerCost)
    (CheckedStopNeighbor CurrentState Tie Accepted Feasible Earlier)
    (CheckedStopNeighbor CurrentState Bad Accepted Feasible Shrunk)
    (CheckedStopNeighbor CurrentState Blocked Vetoed Feasible Earlier)
    (CheckedStopNeighbor CurrentState Unsafe Accepted Invalid Earlier)
    (CheckedStopNeighbor OtherState Unscoped Accepted Feasible Earlier)
    """.splitlines()
    (result,) = witness.native.evaluate(facts, (module.QUERY,))
    assert {str(value) for value in result} == {"(ContinueWith Cost)", "(ContinueWith Tie)"}


def test_stop_witness_shares_remaining_proposals_and_declines_to_veto_the_exact_goal():
    pytest.importorskip("hyperon")
    module = stop_module()
    from MeTTa.ram_trajectory import RepairVerifier
    from MeTTa.sft_oracle import make_task

    world = make_task("coverage", 16)
    verifier = RepairVerifier(world)
    path = Path(module.__file__).with_name("stop_witness.metta")
    witness = module.StopWitness(path.read_text(encoding="utf-8"))
    blocked = witness(world, [], {()}, set(), 0, 0, verifier.refine)
    assert blocked["checks"] == [] and blocked["rank_probes"] == 0
    allowed = witness(world, [], {()}, set(), 1, 0, verifier.refine)
    assert len(allowed["checks"]) == verifier.checks == 1
    reused = witness(world, [], {()}, set(), 3, 2, verifier.refine)
    assert reused["checks"] == []
    optimum = verifier.optimum["selected"]
    assert witness(world, optimum, {tuple(optimum)}, set(), 4, 0, verifier.refine)["chosen"] is None


def test_stop_calibration_selects_completions_under_the_dev_call_cap():
    module = stop_module()

    def summary(completed, calls):
        return {"safe_completions": completed, "policy_verifier_calls": calls, "unsafe_additions": 0,
                "premature_stops": 10 - completed}

    metrics = {"raw": summary(4, 100), "mild": summary(6, 115), "strong": summary(7, 116)}
    assert module.select_offset(metrics, {"raw": 0, "mild": -0.25, "strong": -1}) == ("mild", ["raw", "mild"])
