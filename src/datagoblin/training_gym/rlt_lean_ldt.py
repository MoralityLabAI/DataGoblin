"""Compare live LDT-certified Lean proposals against a native tactic portfolio."""

from __future__ import annotations

import re
import secrets
import subprocess
import time
from collections import Counter
from pathlib import Path

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import dump_structured, load_structured

BUDGETS = (4, 8)
AXIOMS = frozenset(("propext", "Quot.sound", "Classical.choice"))


def fixture(root):
    path = root / "fixture.json"
    if path.exists():
        return load_structured(path)
    examples = []
    constants = set()
    for _ in range(8):
        base = 113 + secrets.randbelow(3871)
        while base in constants:
            base = 113 + secrets.randbelow(3871)
        constants.add(base)
        rows = {
            "definition": f"∀ (n : Nat), (fun x : Nat => x + {base}) n = n + {base}",
            "list_rewrite": f"∀ (xs : List Nat), [] ++ (xs ++ [{base}]) = xs ++ [{base}]",
            "linear_bound": f"∀ (a b : Nat), a + {base} ≤ b → a + {base - 2} < b",
            "add_commute": f"∀ (a b : Nat), (a + b) + {base} = (b + a) + {base}",
            "hypothesis_rewrite": f"∀ (f : Nat → Nat) (a b : Nat), a = b → f (a + {base}) = f (b + {base})",
            "branch_logic": f"∀ (p q : Prop), Or p q → Or (And p ({base} = {base})) q",
            "closed_division": f"{base} % 17 = {base % 17} ∧ {base} / 17 = {base // 17}",
            "contradiction": f"∀ (a : Nat), a = {base} → a ≠ {base} → False",
        }
        examples.extend({"family": family, "statement": statement, "id": sha256_json(statement)}
                        for family, statement in rows.items())
    result = {"schema": "datagoblin.lean-core-stopping-fixture.v1", "goals": examples}
    dump_structured(path, result)
    return result


def lean_env(config):
    env = config.lean_env()
    env["LEAN_SYSROOT"] = str(config.LEAN_HOME)
    env.pop("LEAN_PATH", None)
    return env


class LeanGate:
    """Own the current Lean goal, tactic check and explicit kernel audit."""

    def __init__(self, hybrid, lean, root_sid, goal_id, tactics):
        self.hybrid, self.lean = hybrid, lean
        self.root_sid, self.goal_id, self.tactics = root_sid, goal_id, tactics
        self.state = hybrid.CandidateState({"proof": frozenset(("open", "certified"))})
        self.closed = hybrid.CandidateState({"proof": frozenset(("certified",))})
        self.attempts = []

    def verify(self, proposal, _context):
        soundness = self.hybrid.SoundnessType
        meta = proposal.metadata
        if meta["goal_id"] != self.goal_id or meta["root_sid"] != self.root_sid or meta["tactic"] not in self.tactics:
            return soundness.UNKNOWN
        row = {"tactic": meta["tactic"], "closed": False, "kernel_calls": 0, "kernel_wall_s": 0.0}
        started = time.perf_counter()
        outcome = self.lean.run(self.root_sid, meta["tactic"])
        row["tactic_wall_s"] = time.perf_counter() - started
        if outcome["ok"] and not outcome["state"]["goals"] and outcome["state"]["closed"]:
            row["closed"] = True
            started = time.perf_counter()
            kernel = self.lean.kernel(outcome["state"]["sid"])
            row["kernel_calls"] = 1
            row["kernel_wall_s"] = time.perf_counter() - started
            row["kernel_ok"] = kernel["ok"]
            row["axioms"] = kernel.get("axioms", [])
        elif not outcome["ok"]:
            row["error"] = outcome["error"]
        self.attempts.append(row)
        if row.get("kernel_ok") and set(row["axioms"]).issubset(AXIOMS):
            return soundness.ENV_SOUND_DEAD
        return soundness.UNKNOWN

    def check(self, tactic, *, goal_id=None):
        proposal = self.hybrid.LatticeProposal(
            self.closed, self.hybrid.SoundnessType.MODEL_SOUND_DEAD,
            metadata={"goal_id": self.goal_id if goal_id is None else goal_id,
                      "root_sid": self.root_sid, "tactic": tactic},
        )
        result = self.hybrid.certify_and_apply_verified(self.state, proposal, provenance_verifier=self.verify)
        self.state = result.after
        return result.accepted


def trial(lean, hybrid, goal, state, arm, budget, rankers, tactics, guard):
    gate = LeanGate(hybrid, lean, state["sid"], goal["id"], tactics)
    model = rankers.get(arm)
    proposals = model.predict(state["goals"][0]["pp"], arm, budget) if model else iter(tactics[:budget])
    started = time.perf_counter()
    accepted = None
    already = set()
    ranking_s = 0.0
    for _ in range(budget):
        guard()
        ts = time.perf_counter()
        proposal = next(proposals)
        if model:
            ranking = sorted(range(len(tactics)), key=lambda i: -proposal[i])
            tactic = next(tactics[i] for i in ranking if tactics[i] not in already)
        else:
            tactic = proposal
        ranking_s += time.perf_counter() - ts
        already.add(tactic)
        if gate.check(tactic):
            accepted = tactic
            break
    return {
        "arm": arm, "budget": budget, "accepted": accepted, "attempts": gate.attempts,
        "encoder_visits": len(already) if model else 0, "tactic_checks": len(already),
        "kernel_calls": sum(row["kernel_calls"] for row in gate.attempts), "rank_wall_s": ranking_s,
        "tactic_wall_s": sum(row["tactic_wall_s"] for row in gate.attempts),
        "kernel_wall_s": sum(row["kernel_wall_s"] for row in gate.attempts),
        "total_wall_s": time.perf_counter() - started,
    }


def evaluate(root, guard):
    import numpy as np
    from lcgrt import config
    from lcgrt.feedback import advance
    from lcgrt.lean_repl import LeanRepl
    from research_gym.core import hybrid

    from . import rlt_lean_rank as rank

    parity_path = root / "model-parity.json"
    models = load_structured(parity_path)
    if models["ranker_source_sha256"] != sha256_file(Path(rank.__file__)):
        raise ValueError("the tested model source changed")
    rankers = {}
    for mode in rank.MODES:
        path = root / f"{mode}.npz"
        expected = models["models"][mode]
        if sha256_file(path) != expected["array_sha256"] or expected["rank_checks"] != 24:
            raise ValueError("unexpected model weights or incomplete numerical reference")
        with np.load(path, allow_pickle=False) as tensors:
            rankers[mode] = rank.LoopTacticRanker(dict(tensors), advance)
    goals = fixture(root)["goals"]
    arms = ("native", *rank.MODES)
    protocol = {
        "arms": arms, "budgets": BUDGETS, "native_order": rank.TACTICS,
        "axioms_allowed": sorted(AXIOMS), "fixture_sha256": sha256_file(root / "fixture.json"),
        "parity_sha256": sha256_file(parity_path), "evaluator_source_sha256": sha256_file(Path(__file__)),
        "repl_sha256": sha256_file(config.REPL_EXE), "lean_sha256": sha256_file(config.LEAN_EXE),
        "membrane_sha256": sha256_file(Path(hybrid.__file__)), "imports": ["Init"],
    }
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and sha256_json(load_structured(protocol_path)) != sha256_json(protocol):
        raise ValueError("frozen Lean stopping protocol changed")
    dump_structured(protocol_path, protocol)
    traces = root / "traces"
    traces.mkdir(exist_ok=True)
    with LeanRepl(timeout=30, restart_on_timeout=False, env=lean_env(config), cwd=root) as lean:
        if lean.version != config.LEAN_VERSION:
            raise ValueError("unexpected Lean toolchain")
        audit = lean.init("False", pp=True)
        gate = LeanGate(hybrid, lean, audit["sid"], "negative-audit", rank.TACTICS)
        if gate.check("rfl", goal_id="stale") or gate.attempts or gate.check("simp") or gate.check("decide"):
            raise ValueError("the membrane accepted a stale or invalid proof")
        dump_structured(root / "gate-audit.json", {"stale_rejected_before_lean": True, "false_attempts": gate.attempts})
        lean.reset()
        for i, goal in enumerate(goals):
            guard()
            path = traces / f"{goal['id']}.json"
            if path.exists():
                continue
            state = lean.init(goal["statement"], pp=True)
            if len(state["goals"]) != 1 or not state["goals"][0].get("pp"):
                raise ValueError("expected one initial Lean goal")
            rotated = arms[i % len(arms) :] + arms[: i % len(arms)]
            rows = [trial(lean, hybrid, goal, state, arm, budget, rankers, rank.TACTICS, guard)
                    for budget in BUDGETS for arm in rotated]
            lean.reset()
            pending = path.with_suffix(".part.json")
            dump_structured(pending, {"goal": goal, "goal_pp": state["goals"][0]["pp"], "results": rows})
            pending.replace(path)
            if i % 16 == 15:
                print(f"Verified Lean coverage logged for {i + 1}/{len(goals)} goals", flush=True)


def certify(root, guard):
    from lcgrt import config

    protocol = load_structured(root / "protocol.json")
    fixture_path = root / "fixture.json"
    if (sha256_file(fixture_path) != protocol["fixture_sha256"]
            or sha256_file(config.LEAN_EXE) != protocol["lean_sha256"]):
        raise ValueError("certificate input or Lean executable changed")
    goals = load_structured(fixture_path)["goals"]
    rows = [load_structured(root / "traces" / f"{goal['id']}.json") for goal in goals]
    unique = {}
    for goal, row in zip(goals, rows, strict=True):
        if row["goal"] != goal:
            raise ValueError("trace and frozen goal disagree")
        for result in row["results"]:
            tactic = result["accepted"]
            if tactic is not None:
                if tactic not in protocol["native_order"] or tactic != result["attempts"][-1]["tactic"]:
                    raise ValueError("unregistered winning tactic")
                unique[goal["id"], tactic] = goal["statement"]
    replays = []
    certificates = root / "certificates"
    certificates.mkdir(exist_ok=True)
    examples = sorted(unique.items())
    for offset in range(0, len(examples), 16):
        guard()
        batch = examples[offset : offset + 16]
        path = certificates / f"Cert_{offset // 16:02d}.lean"
        names = [f"dgCert{idx}" for idx in range(len(batch))]
        statements = [f"theorem {name} : {stmt} := by\n  repeat' intro\n  {tactic}"
                      for name, ((_, tactic), stmt) in zip(names, batch, strict=True)]
        path.write_text("\n\n".join(statements + [f"#print axioms {n}" for n in names]) + "\n", encoding="utf-8")
        started = time.perf_counter()
        p = subprocess.run([str(config.LEAN_EXE), "-j", "1", str(path)], cwd=root, env=lean_env(config),
                           timeout=30, capture_output=True, encoding="utf-8")
        text = p.stdout + p.stderr
        path.with_suffix(".log").write_text(text, encoding="utf-8")
        axiom_lines = {}
        audit_pattern = r"^'(dgCert\d+)' (?:does not depend on any axioms|depends on axioms: \[(.*)\])$"
        for name, payload in re.findall(audit_pattern, text, re.M):
            axiom_lines[name] = [part.strip() for part in payload.split(",")] if payload else []
        if p.returncode or set(axiom_lines) != set(names) or any(set(xs) - AXIOMS for xs in axiom_lines.values()):
            raise ValueError(f"standalone certificate replay failed: {path.name}")
        replays.append({"file": path.name, "wall_s": time.perf_counter() - started, "declarations": len(batch),
                        "source_sha256": sha256_file(path), "output_sha256": sha256_file(path.with_suffix(".log"))})
    cells = []
    for budget in protocol["budgets"]:
        for arm in protocol["arms"]:
            totals = Counter()
            for row in rows:
                match, = (r for r in row["results"] if r["budget"] == budget and r["arm"] == arm)
                totals["certified"] += match["accepted"] is not None
                for field in ("encoder_visits", "tactic_checks", "kernel_calls", "tactic_wall_s",
                              "kernel_wall_s", "rank_wall_s", "total_wall_s"):
                    totals[field] += match[field]
            cells.append({"budget": budget, "arm": arm, "goals": len(goals), **totals})
    dump_structured(root / "evidence.json", {
        "schema": "datagoblin.rlt-lean-ldt-stopping.v1", "protocol": protocol, "cells": cells,
        "standalone_replay": replays, "unique_proofs_replayed": len(unique), "model_parity": load_structured(
            root / "model-parity.json"
        ),
    })
