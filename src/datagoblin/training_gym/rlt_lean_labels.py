"""Freeze separated Lean goal cohorts and label grounded candidates with the kernel."""

from __future__ import annotations

import re
import secrets
import time
from collections import Counter
from pathlib import Path

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import dump_structured, load_structured

from . import rlt_lean_grounded as grounded
from .rlt_lean_ldt import LeanGate, lean_env

TRAIN_SALTS = 12
SAME_TEMPLATE_SALTS = 4
HELD_TEMPLATE_SALTS = 6
BUDGET = 6
INCONCLUSIVE = re.compile(r"heartbeat|timed? out|out of memory|maximum recursion depth", re.I)


def held_templates(n):
    prefix = "∀ (f : Nat → Nat → Nat) (g : Nat → Nat) (xs ys : List Nat) (k : Nat), "
    yield from (
        (
            "reverse_take",
            f"∀ (xs : List Nat) (k : Nat), ((xs ++ [{n}]).take k).reverse = "
            f"([{n}].reverse ++ xs.reverse).drop ((xs ++ [{n}]).length - k)",
            "simp only [List.reverse_take, List.reverse_append]",
        ),
        (
            "drop_after_map_zip",
            prefix + f"((List.zipWith f (xs ++ [{n}]) ys).map g).drop k = "
            f"List.zipWith (fun a b => g (f a b)) ((xs ++ [{n}]).drop k) (ys.drop k)",
            "simp only [List.map_zipWith, List.drop_zipWith]",
        ),
        (
            "map_twice_filter_map",
            f"∀ (xs : List Nat) (f : Nat → Option Nat) (g h : Nat → Nat), "
            f"(((xs ++ [{n}]).filterMap f).map g).map h = "
            f"(xs ++ [{n}]).filterMap (fun x => ((f x).map g).map h)",
            "simp only [List.map_filterMap]",
        ),
        (
            "drop_then_take_while_map",
            f"∀ (xs : List Nat) (f : Nat → Nat) (p q : Nat → Bool), "
            f"(((xs ++ [{n}]).map f).dropWhile p).takeWhile q = "
            f"(((xs ++ [{n}]).dropWhile (p ∘ f)).takeWhile (q ∘ f)).map f",
            "simp only [List.dropWhile_map, List.takeWhile_map]",
        ),
    )


def freeze(root, previous, entries, config, hybrid):
    old = load_structured(previous)
    used = {g["salt"] for g in old["goals"]}
    splits = {}
    for name, size, generate in (
        ("train", TRAIN_SALTS, grounded.propositions),
        ("same_templates", SAME_TEMPLATE_SALTS, grounded.propositions),
        ("held_templates", HELD_TEMPLATE_SALTS, held_templates),
    ):
        salts = []
        while len(salts) < size:
            n = 4117 + secrets.randbelow(23500)
            if n not in used:
                used.add(n)
                salts.append(n)
        splits[name] = [{"id": sha256_json(stmt), "family": family, "statement": stmt, "salt": n}
                        for n in sorted(salts) for family, stmt, _ in generate(n)]
    dump_structured(root / "fixtures.json", {"schema": "datagoblin.lean-grounded-split.v1", "splits": splits})
    dump_structured(root / "library.json", {"entries": entries})
    return {
        "previous_fixture_sha256": sha256_file(previous), "fixtures_sha256": sha256_file(root / "fixtures.json"),
        "library_sha256": sha256_file(root / "library.json"), "label_source_sha256": sha256_file(Path(__file__)),
        "grounding_source_sha256": sha256_file(Path(grounded.__file__)),
        "gate_source_sha256": sha256_file(Path(grounded.__file__).with_name("rlt_lean_ldt.py")),
        "membrane_sha256": sha256_file(Path(hybrid.__file__)), "lean_sha256": sha256_file(config.LEAN_EXE),
        "repl_sha256": sha256_file(config.REPL_EXE), "budget": BUDGET,
        "fit": {"steps": 400, "depth": 6, "batch": 10, "width": 20, "lr": 0.005, "seeds": [131, 149]},
    }


def atomic_record(path, row):
    pending = path.with_suffix(".part.json")
    dump_structured(pending, row)
    pending.replace(path)


def collect(root, previous, guard):
    from lcgrt import config
    from lcgrt.lean_repl import LeanRepl
    from research_gym.core import hybrid

    indexed = grounded.library(config.LEAN_HOME)
    frozen = root / "protocol.json"
    protocol = load_structured(frozen) if frozen.exists() else freeze(root, previous, indexed, config, hybrid)
    if not frozen.exists():
        atomic_record(frozen, protocol)
    verify_sources(root, protocol, config, hybrid)
    if (
        sha256_file(previous) != protocol["previous_fixture_sha256"]
        or sha256_file(Path(__file__)) != protocol["label_source_sha256"]
    ):
        raise ValueError("previous holdout or frozen label collector changed")
    goals = load_structured(root / "fixtures.json")["splits"]["train"]
    labels = root / "train-labels"
    labels.mkdir(exist_ok=True)
    with LeanRepl(timeout=30, restart_on_timeout=False, env=lean_env(config), cwd=root) as lean:
        if lean.version != config.LEAN_VERSION:
            raise ValueError("unexpected Lean version")
        for index, goal in enumerate(goals, 1):
            guard()
            path = labels / f"{goal['id']}.json"
            if path.exists():
                continue
            state = lean.init(goal["statement"], pp=True)
            pp = state["goals"][0]["pp"]
            started = time.perf_counter()
            hits, pool = grounded.candidates(pp, indexed)
            retrieval_s = time.perf_counter() - started
            rows = []
            for tactic in pool:
                guard()
                gate = LeanGate(hybrid, lean, state["sid"], goal["id"], pool)
                accepted = gate.check(tactic)
                attempt, = gate.attempts
                label = None if INCONCLUSIVE.search(attempt.get("error", "")) else int(accepted)
                rows.append({**attempt, "label": label})
            lean.reset()
            atomic_record(path, {
                "goal": goal, "goal_pp": pp, "hits": hits, "outcomes": rows, "retrieval_s": retrieval_s,
            })
            if index % 36 == 0:
                print(f"Labeled all candidates on {index}/{len(goals)} training goals", flush=True)
    manifest = {g["id"]: sha256_file(labels / f"{g['id']}.json") for g in goals}
    atomic_record(root / "label-manifest.json", {"protocol_sha256": sha256_file(frozen), "rows": manifest})


def verify_sources(root, protocol, config, hybrid):
    expected = {
        "fixtures_sha256": root / "fixtures.json", "library_sha256": root / "library.json",
        "grounding_source_sha256": Path(grounded.__file__),
        "gate_source_sha256": Path(grounded.__file__).with_name("rlt_lean_ldt.py"),
        "membrane_sha256": Path(hybrid.__file__), "lean_sha256": config.LEAN_EXE, "repl_sha256": config.REPL_EXE,
    }
    if any(sha256_file(path) != protocol[key] for key, path in expected.items()):
        raise ValueError("frozen fixture, verifier or source changed")


def training_labels(root):
    manifest_path = root / "label-manifest.json"
    manifest = load_structured(manifest_path)
    if manifest["protocol_sha256"] != sha256_file(root / "protocol.json"):
        raise ValueError("training labels belong to a different protocol")
    goals = load_structured(root / "fixtures.json")["splits"]["train"]
    if {g["id"] for g in goals} != manifest["rows"].keys():
        raise ValueError("training label manifest does not cover its frozen split")
    rows = []
    for goal in goals:
        path = root / "train-labels" / f"{goal['id']}.json"
        if manifest["rows"][goal["id"]] != sha256_file(path):
            raise ValueError("label row changed after freezing")
        row = load_structured(path)
        if row["goal"] != goal:
            raise ValueError("labeled goal does not match the fixture")
        rows.append(row)
    return rows


def label_counts(rows):
    counts = Counter()
    for row in rows:
        attempts = row["outcomes"]
        counts["tactic_checks"] += len(attempts)
        counts["kernel_calls"] += sum(a["kernel_calls"] for a in attempts)
        counts["positive"] += sum(a["label"] == 1 for a in attempts)
        counts["negative"] += sum(a["label"] == 0 for a in attempts)
        counts["unknown"] += sum(a["label"] is None for a in attempts)
        counts["positive_goals"] += any(a["label"] == 1 for a in attempts)
        for field in ("tactic_wall_s", "kernel_wall_s"):
            counts[field] += sum(a[field] for a in attempts)
    return {"goals": len(rows), "retrieval_wall_s": sum(r["retrieval_s"] for r in rows), **counts}
