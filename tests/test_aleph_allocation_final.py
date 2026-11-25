"""Proof, matched-budget, calibration and custody regressions for external allocation."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "aleph_next" / "state01"))
from allocation_replay import accepted  # noqa: E402
from allocation_routes import acquisition, route  # noqa: E402
from allocation_tasks import Leaf, declaration, equal, inside, pair_key, solve, valid  # noqa: E402


def test_four_solvers_require_replay_and_mixed_congruence():
    source = (Leaf("nat", ("+", "x", "0"), "x"), Leaf("list", ("++", "xs", "nil"), "xs"))
    wrappers = (("+", "HOLE", "z"), ("mf", "HOLE"))
    goals = tuple(Leaf(v.domain, inside(w, v.left), inside(w, v.right)) for v, w in zip(source, wrappers, strict=True))
    anchor, *_ = solve(source, "frontiers")
    assert valid(source, anchor)
    for action in ("normal", "congruence", "frontiers", "feedback"):
        cert, *_ = solve(goals, action, source, wrappers, anchor)
        assert valid(goals, cert)
        assert "Prod.ext" in declaration("demo", goals, cert)
    alternate = (Leaf("nat", goals[0].left, ("+", "0", goals[0].right)), goals[1])
    repaired, *_ = solve(alternate, "feedback", source, wrappers, anchor)
    assert valid(alternate, repaired)
    assert not valid(alternate, cert)
    false = (Leaf("nat", goals[0].left, ("+", goals[0].right, "1")), goals[1])
    assert not equal(false)


def test_pair_identity_treats_exact_reverse_as_same_and_keeps_carrier():
    goal = (Leaf("nat", "x", "y"), Leaf("list", "xs", "ys"))
    reverse = tuple(Leaf(v.domain, v.right, v.left) for v in goal)
    assert pair_key(goal) == pair_key(reverse)
    assert pair_key(goal) != pair_key(goal[:1])


def test_failed_acquisition_and_overrunning_verification_are_charged():
    def act(ok, key, wall):
        return {"ok": ok, "solver_seconds": 0.01, "steps": 1, "work": 2, "kernel": {"source": key, "seconds": wall}}

    acquire = act(False, "bad", 0.4)
    acquire2 = act(True, "source", 0.3)
    normal = act(True, "target", 0.5)
    row = {
        "gate": True,
        "gate_s": 0.02,
        "features_s": 0.01,
        "lineage": "l",
        "source": [acquire, acquire2],
        "actions": {"feedback": normal, "normal": normal},
        "stale": None,
    }
    r = route(row, ["feedback", "normal"], budget=1.0)
    assert not r["ok"] and r["kernels"] == 3 and r["seconds"] == 1 and r["uncensored"] > 1.2
    assert route(row, ["normal"], budget=1.0)["ok"]
    acq = acquisition([row, row])
    assert acq["unique_kernel_calls"] == 3 and acq["kernel_seconds"] == pytest.approx(1.2)
    gated = route(dict(row, gate=False), ["feedback"])
    assert not gated["ok"] and gated["kernels"] == 0


def test_positive_lean_gate_requires_the_expected_singleton_and_allowed_axioms():
    assert accepted(0, "'goal' does not depend on any axioms\n", "goal")
    assert not accepted(0, "'goal' depends on axioms: [sorryAx]\n", "goal")
    assert not accepted(0, "'decoy' does not depend on any axioms\n", "goal")
    assert not accepted(1, "'goal' does not depend on any axioms\n", "goal")


def test_measured_evidence_roundtrip_replays_policy_decisions(tmp_path):
    from allocation_evidence import checksum, pack, text_json
    from allocation_replay import MEDIAN

    def write(k, v):
        (tmp_path / k).write_bytes(text_json(v))

    source = Leaf("nat", "x", "x")
    train = {
        "rows": [
            {
                "key": pair_key((source,)),
                "lineage": "source",
                "gate": True,
                "public": [["nat", "x", "x"]],
                "source": [],
                "actions": {},
                "stale": None,
            }
        ]
    }
    write("train.json", train)
    train_hash = checksum((tmp_path / "train.json").read_bytes())
    write("FROZEN.json", {"train_sha256": train_hash, "fits": []})
    frozen = checksum((tmp_path / "FROZEN.json").read_bytes())
    row = {
        "key": pair_key((Leaf("nat", "x", ("+", "x", "1")),)),
        "public": [["nat", "x", ["+", "x", "1"]]],
        "source": [],
        "actions": {},
        "stale": None,
        "lineage": "target",
        "intervention": "invalid",
        "gate": False,
        "gate_s": 0.001,
        "features_s": 0.0,
    }
    write("held.json", {"freeze_sha256": frozen, "rows": [row]})
    held = checksum((tmp_path / "held.json").read_bytes())
    write(
        "POLICIES.json",
        {
            "fit_sha256": frozen,
            "held_sha256": held,
            "choices": {"normal": [["normal"]]},
            "inference_seconds_by_row": {"normal": [0.02]},
            "details": {"normal": [route(row, ["normal"], 4 * MEDIAN, 0.02)]},
            "passes": {"mlp": False},
        },
    )
    (tmp_path / "train-only.jsonl").write_bytes(b"")
    (tmp_path / "singleton").mkdir()
    result = pack(tmp_path, tmp_path / "test.zip")
    assert result["primary_decisions"] == 1 and result["held_lineages"] == 1
