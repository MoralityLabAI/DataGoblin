"""DG-AGN-TINKER-01: independently scored curriculum, routing and repair gym.

No provider import or network access in the fixture path. All observations are
original synthetic AGN-02 worlds. Student uplift refers to the AGN-02 linear
student, not Aleph or a Tinker adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Agonist.agn02 import (
    _evaluation_rate,
    construct_roots,
    exact_actions,
    fit_student,
    make_tastes,
)

ROLES = ("curriculum", "route", "diagnose", "repair")
SCHEMA = "dg-agn-tinker-01.v1"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Task:
    root: str
    split: str
    role: str
    public: dict[str, Any]
    scores: dict[str, float]

    @property
    def key(self) -> str:
        return f"{self.root}:{self.role}"

    @property
    def gold(self) -> str:
        return min(self.scores, key=lambda k: (-self.scores[k], k))

    def prompt(self, feedback: list[dict[str, Any]] | None = None) -> str:
        return canonical({
            "instruction": "Return a JSON object with action (one allowed ID) and a brief decision note.",
            "role": self.role,
            "public": self.public,
            "allowed_actions": sorted(self.scores),
            "previous_attempts": feedback or [],
        })

    def grade(self, action: str | None) -> dict[str, Any]:
        valid = action in self.scores
        reward = self.scores[action] if valid else None
        best = self.scores[self.gold]
        return {
            "valid": valid,
            "success": valid and reward == best,
            "reward": reward,
            "regret": best - reward if reward is not None else None,
            "code": "ok" if valid and reward == best else "suboptimal" if valid else "invalid_action",
        }

    def demonstration(self, feedback: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if self.split != "train":
            raise ValueError("only train roots can produce SFT labels")
        return {
            "task_id": self.key,
            "root": self.root,
            "role": self.role,
            "messages": [
                {"role": "user", "content": self.prompt(feedback)},
                {"role": "assistant", "content": canonical({"action": self.gold, "note": "Verified action."})},
            ],
            "provenance": "independent_original_fixture_oracle",
            "feedback_count": len(feedback or []),
        }


def make_tasks(seed: int, groups: int, split: str) -> list[Task]:
    """Each independently generated group has four presentations maximum.

    Curriculum labels are measured gains after training a separate small student
    on an equal-size candidate augmentation. Labels and calibration outcomes are
    kept out of actor prompts. The train/dev namespaces cannot overlap.
    """
    if split not in ("train", "dev") or not 1 <= groups <= 256 or seed < 0:
        raise ValueError("invalid fixture namespace or size")
    tasks = []
    for index in range(groups):
        root = digest([SCHEMA, split, seed, index])[:32]
        local_seed = int(root[:15], 16)
        rng = random.Random(local_seed)
        roots = construct_roots(local_seed, make_tastes(local_seed))
        anchors = [r for r in roots if r["cohort"] == "anchor"]
        calibration = [r for r in roots if r["cohort"] == "calibration"]
        pool = [r for r in roots if r["cohort"] == "pool"]
        candidates = rng.sample(pool, 6)
        # Same seed and update budget for all candidate fits and baseline.
        student_seed = 41
        updates = 240
        baseline = _evaluation_rate(fit_student(anchors, student_seed, updates), calibration)
        choices, scores = [], {}
        for j, candidate in enumerate(candidates):
            action = f"candidate-{j}"
            measured = _evaluation_rate(
                fit_student([*anchors, candidate], student_seed, updates), calibration
            )
            scores[action] = round(measured - baseline, 12)
            choices.append({
                "id": action,
                "initial": candidate["world_a"]["state"],
                "revised": candidate["world_b"]["state"],
                "surface_cue": candidate["world_a"]["surface_cue"],
                "check_case": candidate["check_case"],
                "verified_train_actions": exact_actions(candidate),
            })
        tasks.append(Task(root, split, "curriculum", {
            "goal": "Select one additional train root to maximize held-out linear student decision accuracy.",
            "student": "AGN-02 linear policy, fixed 240 updates",
            "anchor_baseline": baseline,
            "anchors": [
                {"initial": a["world_a"]["state"], "revised": a["world_b"]["state"],
                 "surface_cue": a["world_a"]["surface_cue"], "check_case": a["check_case"],
                 "verified_train_actions": exact_actions(a)}
                for a in anchors
            ],
            "candidates": choices,
            "cost": "All options use one root and identical student update budget.",
        }, scores))
        target = rng.choice(("curriculum", "diagnose", "repair"))
        available = {r: bool(rng.getrandbits(1)) for r in ("curriculum", "diagnose", "repair")}
        authorized = bool(rng.getrandbits(1))
        correct = target if authorized and available[target] else "STOP"
        descriptions = {
            "curriculum": "Choose additional training examples after a student accuracy plateau.",
            "diagnose": "Identify an independently observable failure prerequisite.",
            "repair": "Revise a rejected decision using the current public evidence.",
        }
        tasks.append(Task(root, split, "route", {
            "request": descriptions[target], "available": available, "authorized": authorized,
            "rule": "Choose the matching available specialist only if authorized; otherwise STOP.",
        }, {a: float(a == correct) for a in (*available, "STOP")}))
        world = rng.choice(candidates)
        state = world["world_a"]["state"]
        failures = [k for k, failed in (
            ("available", state["available"] < state["required"]),
            ("safety_approved", not state["safety_approved"]),
            ("source_fresh", not state["source_fresh"]),
        ) if failed]
        diagnosis = failures[0] if failures else "none"
        tasks.append(Task(root, split, "diagnose", {
            "state": state, "request": "Name the unmet prerequisite, or none.",
        }, {a: float(a == diagnosis) for a in ("available", "safety_approved", "source_fresh", "none")}))
        gold = exact_actions(world)[1]
        tasks.append(Task(root, split, "repair", {
            "current_state": world["world_b"]["state"],
            "previous_action": exact_actions(world)[0],
            "evidence": "One causal prerequisite changed. Recompute the deployment decision.",
            "rule": "Deploy iff available >= required, safety approved, and source fresh.",
        }, {a: float(a == gold) for a in ("deploy", "hold")}))
    return tasks


def parse_action(text: str) -> str | None:
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("action"), str):
        return None
    return obj["action"]


class Journal:
    """Durable single-worker ledger. Unknown remote outcomes block all new calls.

    Caps are microdollar estimates, conservative for declared token prices only.
    Provider account caps remain necessary for storage and provider-side retries.
    """
    def __init__(self, directory: Path, mode: str):
        if mode not in ("clean", "advisory"):
            raise ValueError("invalid journal mode")
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.db = sqlite3.connect(directory / "journal.sqlite", timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS calls (
                id TEXT PRIMARY KEY, provider TEXT NOT NULL, fingerprint TEXT NOT NULL,
                reserved INTEGER NOT NULL, state TEXT NOT NULL, result TEXT
            );
            CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, body TEXT NOT NULL);
        """)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('mode', ?)", (mode,))
        if self.db.execute("SELECT value FROM metadata WHERE key='mode'").fetchone()[0] != mode:
            raise ValueError("cannot mix advisory and clean journals")
        self.mode = mode

    def event(self, key: str, body: dict[str, Any]) -> None:
        raw = canonical(body)
        with self.db:
            old = self.db.execute("SELECT body FROM events WHERE id=?", (key,)).fetchone()
            if old is not None and old[0] != raw:
                raise ValueError("immutable event conflict: " + key)
            self.db.execute("INSERT OR IGNORE INTO events VALUES (?,?)", (key, raw))

    def get_event(self, key: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT body FROM events WHERE id=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def events(self) -> list[dict[str, Any]]:
        return [json.loads(r[0]) for r in self.db.execute("SELECT body FROM events ORDER BY rowid")]

    def call(
        self, key: str, provider: str, inputs: Any, reserve: int, cap: int,
        invoke: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        if reserve < 0 or cap < 0:
            raise ValueError("negative spending limits")
        if provider == "fable" and self.mode != "advisory":
            raise ValueError("Fable only allowed in segregated advisory journal")
        fingerprint = digest(inputs)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            old = self.db.execute(
                "SELECT provider,fingerprint,state,result FROM calls WHERE id=?", (key,)
            ).fetchone()
            if old:
                if old[:2] != (provider, fingerprint):
                    raise ValueError("request identity conflict")
                if old[2] != "complete":
                    raise RuntimeError("unknown outcome; reconcile provider request before continuing")
                self.db.commit()
                return json.loads(old[3])
            if (self.directory / "STOP").exists():
                raise RuntimeError("operator STOP file")
            if self.db.execute("SELECT 1 FROM calls WHERE state='pending' LIMIT 1").fetchone():
                raise RuntimeError("unresolved remote call; no automatic resubmission")
            spent = self.db.execute(
                "SELECT COALESCE(SUM(reserved),0) FROM calls WHERE provider=?", (provider,)
            ).fetchone()[0]
            if spent + reserve > cap:
                raise RuntimeError(f"{provider} reservation cap exceeded")
            self.db.execute(
                "INSERT INTO calls VALUES (?,?,?,?,?,NULL)",
                (key, provider, fingerprint, reserve, "pending"),
            )
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        # On exception or process death, intentionally retain the pending reservation.
        result = invoke()
        with self.db:
            self.db.execute(
                "UPDATE calls SET state='complete',result=? WHERE id=?", (canonical(result), key)
            )
        return result

    def receipts(self) -> list[dict[str, Any]]:
        return [
            {"type": "provider_receipt", "call_id": k, "provider": p, "fingerprint": f,
             "reserved_usd": u / 1_000_000, "state": s, "result": json.loads(r) if r else None}
            for k, p, f, u, s, r in self.db.execute(
                "SELECT id,provider,fingerprint,reserved,state,result FROM calls ORDER BY rowid"
            )
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reservations": [
                {"provider": p, "state": s, "calls": n, "reserved_usd": u / 1_000_000}
                for p, s, n, u in self.db.execute(
                    "SELECT provider,state,COUNT(*),SUM(reserved) FROM calls GROUP BY provider,state"
                )
            ],
            "events": len(self.events()),
        }


def training_example(task: Task, journal: Journal, round_id: int) -> dict[str, Any]:
    """Construct a single independently labeled correction example from a clean rollout.

    Only allowlisted action IDs and recomputed failure codes cross into SFT. No
    free-form actor reply, note, advisory content or development result is copied.
    A fresh root without an observed error uses the ordinary demonstration.
    """
    if journal.mode != "clean" or task.split != "train":
        raise ValueError("SFT requires a clean train root")
    run = f"rollout-{round_id}-pre"
    key = f"episode:{run}:{task.key}"
    contract = journal.get_event(key + ":contract")
    feedback = []
    if contract is not None:
        if contract["advisory"] or contract["task_hash"] != digest([task.public, task.scores]):
            raise ValueError("incompatible or advisory episode contract")
        for i in range(min(contract["attempt_limit"], 8)):
            record = journal.get_event(f"{key}:{i}")
            if record is None:
                break
            # An arbitrary generated string is not an admissible action ID.
            action = record["action"] if record["action"] in task.scores else None
            grade = task.grade(action)
            if grade["code"] != record["code"] or grade["success"] != record["success"]:
                raise ValueError("attempt verifier receipt mismatch")
            if grade["success"]:
                break
            feedback.append({"action": action, "code": grade["code"]})
    return task.demonstration(feedback)


def episode(
    task: Task, journal: Journal, run: str,
    propose: Callable[[str, Task, str, list[dict[str, Any]]], dict[str, Any]],
    attempts: int = 3,
    consult: Callable[[str, Task, list[dict[str, Any]]], str] | None = None,
) -> dict[str, Any]:
    """Bounded proposal/verification/revision; only visible decision notes are retained."""
    if not 1 <= attempts <= 8:
        raise ValueError("attempt limit outside 1..8")
    if consult and journal.mode != "advisory":
        raise ValueError("cannot consult from a clean episode")
    key = f"episode:{run}:{task.key}"
    journal.event(key + ":contract", {"type": "episode_contract", "task_id": task.key,
                  "task_hash": digest([task.public, task.scores]), "attempt_limit": attempts,
                  "advisory": consult is not None})
    existing = journal.get_event(key)
    if existing is not None:
        return existing
    history: list[dict[str, Any]] = []
    for i in range(attempts):
        attempt_key = f"{key}:{i}"
        record = journal.get_event(attempt_key)
        if record is None:
            provider_result = propose(attempt_key, task, task.prompt(history), history)
            action = parse_action(provider_result.get("text", ""))
            result = task.grade(action)
            record = {
                "type": "attempt", "task_id": task.key, "run": run, "role": task.role,
                "attempt": i, "prompt": task.prompt(history),
                "action": action, "reply": provider_result.get("text", ""),
                "actor": provider_result.get("actor", "unknown"),
                "usage": provider_result.get("usage", {}),
                "code": result["code"], "success": result["success"],
                "reward": result["reward"], "regret": result["regret"],
            }
            # Keep verifier labels out of actor-visible feedback; only failure codes.
            journal.event(attempt_key, record)
        history.append({"action": record["action"], "code": record["code"]})
        if record["success"]:
            break
        if consult and i == attempts - 2:
            correction = consult(f"{key}:consult", task, history)
            history.append({"advisory_note": correction})
    result = {
        "type": "episode", "run": run, "task_id": task.key, "root": task.root,
        "split": task.split, "role": task.role, "mode": journal.mode,
        "public_hash": digest(task.public), "public": task.public,
        "success": record["success"], "attempts": i + 1,
        "reward": record["reward"], "regret": record["regret"],
        "training_ready": False,
    }
    journal.event(key, result)
    return result


def fixture(directory: Path, seed: int, groups: int) -> dict[str, Any]:
    journal = Journal(directory, "clean")
    tasks = make_tasks(seed, groups, "train")

    def actor(key: str, task: Task, prompt: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        # A deterministic plumbing fixture deliberately tries a wrong answer first.
        del key, prompt
        wrong = next((a for a in sorted(task.scores) if not task.grade(a)["success"]), task.gold)
        action = task.gold if history else wrong
        return {"text": canonical({"action": action}), "actor": "fixture_oracle"}
    rows = [episode(t, journal, f"fixture-{seed}", actor) for t in tasks]
    return {**journal.summary(), "episodes": len(rows), "passed": sum(r["success"] for r in rows),
            "claim": "offline plumbing only; zero model calls"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fixture", "status"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--mode", choices=("clean", "advisory"), default="clean")
        p.add_argument("--seed", type=int, default=17)
        p.add_argument("--groups", type=int, default=4)
    args = parser.parse_args()
    if args.command == "fixture":
        if args.mode != "clean":
            parser.error("fixture uses clean journal")
        result = fixture(args.out, args.seed, args.groups)
    else:
        result = Journal(args.out, args.mode).summary()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
