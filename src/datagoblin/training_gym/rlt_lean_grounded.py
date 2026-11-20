"""Grounded, LDT-verified fallback on harder Lean core goals."""

from __future__ import annotations

import itertools
import re
import secrets
import time
from collections import Counter
from pathlib import Path

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import dump_structured, load_structured

from .rlt_lean_ldt import certify_proofs, lean_env, trial

FALLBACK_BUDGET = 6
COUNTS = ("encoder_visits", "tactic_checks", "kernel_calls")
TIMES = ("rank_wall_s", "tactic_wall_s", "kernel_wall_s", "total_wall_s")
LIBRARY = {
    "List/Basic": ("append_assoc",),
    "List/Lemmas": (
        "filterMap_append", "filterMap_filterMap", "filterMap_map", "filterMap_reverse",
        "flatMap_append", "flatMap_def", "flatten_append", "flatten_flatten", "flatten_reverse",
        "map_append", "map_filterMap", "map_flatten", "map_reverse", "reverse_append",
        "reverse_flatMap", "reverse_flatten", "reverse_reverse",
    ),
    "List/Count": ("count_append", "count_flatten", "count_reverse", "count_le_length", "countP_flatMap"),
    "List/Zip": ("drop_zipWith", "map_zipWith", "take_zipWith", "zipWith_map", "zipWith_self"),
    "List/Nat/TakeDrop": ("drop_append", "length_take", "length_zipWith",
                         "reverse_drop", "reverse_take", "reverse_zipWith", "take_append"),
    "List/TakeDrop": ("dropWhile_map", "dropWhile_filter", "length_drop", "map_drop", "map_take",
                      "takeWhile_append", "takeWhile_map", "takeWhile_filter"),
    "Nat/Gcd": ("gcd_assoc", "gcd_comm", "gcd_dvd_left", "gcd_dvd_right"),
}


def propositions(n):
    zip_binders = "∀ (f : Nat → Nat → Nat) (g : Nat → Nat) (xs ys : List Nat) (k : Nat), "
    zip_inputs = f"List.zipWith f (xs ++ [{n}]) ys"
    yield from (
        (
            "zip_take_map",
            zip_binders + f"(({zip_inputs}).take k).map g = "
            f"List.zipWith (fun a b => g (f a b)) ((xs ++ [{n}]).take k) (ys.take k)",
            "simp only [List.take_zipWith, List.map_zipWith]",
        ),
        (
            "zip_drop_map",
            zip_binders + f"(({zip_inputs}).drop k).map g = "
            f"List.zipWith (fun a b => g (f a b)) ((xs ++ [{n}]).drop k) (ys.drop k)",
            "simp only [List.drop_zipWith, List.map_zipWith]",
        ),
        (
            "reverse_drop",
            f"∀ (xs : List Nat) (k : Nat), ((xs ++ [{n}]).drop k).reverse = "
            f"([{n}].reverse ++ xs.reverse).take ((xs ++ [{n}]).length - k)",
            "simp only [List.reverse_drop, List.reverse_append]",
        ),
        (
            "reverse_flatten_map",
            f"∀ (xss : List (List Nat)) (f : Nat → Nat), ((xss ++ [[{n}]]).flatten.reverse).map f = "
            f"(((xss ++ [[{n}]]).map List.reverse).reverse.map (List.map f)).flatten",
            "simp only [List.reverse_flatten, List.map_flatten]",
        ),
        (
            "append_filter_map",
            f"∀ (xs ys : List Nat) (f : Nat → Option Nat) (g : Nat → Nat), "
            f"(((xs ++ [{n}]) ++ ys).filterMap f).map g = "
            f"(xs ++ [{n}]).filterMap (fun x => (f x).map g) ++ ys.filterMap (fun x => (f x).map g)",
            "simp only [List.filterMap_append, List.map_append, List.map_filterMap]",
        ),
        (
            "map_take_drop_while",
            f"∀ (xs : List Nat) (f : Nat → Nat) (p q : Nat → Bool), "
            f"(((xs ++ [{n}]).map f).takeWhile p).dropWhile q = "
            f"(((xs ++ [{n}]).takeWhile (p ∘ f)).dropWhile (q ∘ f)).map f",
            "simp only [List.takeWhile_map, List.dropWhile_map]",
        ),
        (
            "count_reverse_flatten",
            f"∀ (xss : List (List Nat)), ((xss ++ [[{n}]]).flatten.reverse).count {n} = "
            f"((xss ++ [[{n}]]).map (List.count {n})).sum",
            "simp only [List.count_reverse, List.count_flatten]",
        ),
        (
            "gcd_commute_reassociate",
            f"∀ (a b c : Nat), Nat.gcd (Nat.gcd (a + {n}) b) c = Nat.gcd (a + {n}) (Nat.gcd c b)",
            "rw [Nat.gcd_assoc] <;> exact congrArg (Nat.gcd _) (Nat.gcd_comm _ _)",
        ),
        (
            "reverse_flat_map",
            f"∀ (xs : List Nat) (f : Nat → List Nat), ((xs ++ [{n}]).flatMap f).reverse = "
            f"([{n}] ++ xs.reverse).flatMap (List.reverse ∘ f)",
            "simp [List.reverse_flatMap]",
        ),
    )


def freeze(root):
    path = root / "fixture.json"
    if path.exists():
        return load_structured(path)["goals"]
    salts = set()
    while len(salts) < 6:
        salts.add(307 + secrets.randbelow(3600))
    goals = [{"id": sha256_json(stmt), "family": family, "statement": stmt, "salt": n}
             for n in sorted(salts) for family, stmt, _ in propositions(n)]
    dump_structured(path, {"schema": "datagoblin.lean-grounded-holdout.v1", "goals": goals})
    return goals


def library(lean_home):
    rows = []
    for module, names in LIBRARY.items():
        path = lean_home / "src/lean/Init/Data" / (module + ".lean")
        source = path.read_text(encoding="utf-8")
        source_hash = sha256_file(path)
        namespace = module.split("/")[0]
        for name in names:
            match = re.search(r"\btheorem " + re.escape(name) + r"(?=\s|:)", source)
            if match is None:
                raise ValueError(f"missing pinned core declaration {namespace}.{name}")
            rows.append({"name": f"{namespace}.{name}", "module": module,
                         "line": source.count("\n", 0, match.start()) + 1, "source_sha256": source_hash})
    return rows


def words(text):
    text = text.replace("++", " append ")
    return set(re.findall(r"[A-Za-z][A-Za-z0-9]*", text.lower())) - {"list", "nat", "fun"}


def candidates(query, entries):
    terms = words(query.split("⊢", 1)[-1])
    scored = []
    for entry in entries:
        name = entry["name"]
        operations = words(name.split(".", 1)[1])
        match = terms & operations
        if match:
            value = 2 * len(match) / (len(operations) + len(match))
            scored.append((value, name))
    hits = [name for _, name in sorted(scored, key=lambda row: (-row[0], row[1]))[:6]]
    # Each composite grows from the same lexical matches as the single-lemma proposals.
    grouped = [tuple(hits[:3])] if len(hits) >= 3 else []
    grouped.extend(itertools.combinations(hits[:5], 2))
    proposed = [f"simp [{', '.join(names)}]" for names in grouped]
    proposed[:0] = [f"simp [{name}]" for name in hits[:2]]
    proposed.extend(f"{form} {name}" for name in hits for form in ("exact", "apply"))
    proposed.extend(f"rw [{name}]" for name in hits)
    proposed.extend(f"simp [{name}]" for name in hits[2:])
    return hits, tuple(dict.fromkeys(t for t in proposed if len(t.encode("utf-8")) < 96))


def evaluate(root, guard):
    import numpy as np
    from lcgrt import config
    from lcgrt.feedback import advance
    from lcgrt.lean_repl import LeanRepl
    from research_gym.core import hybrid

    from . import rlt_lean_rank as rank

    parity_path = root / "model-parity.json"
    parity = load_structured(parity_path)
    if parity["ranker_source_sha256"] != sha256_file(Path(rank.__file__)):
        raise ValueError("the validated ranker source changed")
    rankers = {}
    for mode in rank.MODES:
        path = root / f"{mode}.npz"
        model = parity["models"][mode]
        if sha256_file(path) != model["array_sha256"] or not model.get("grounded_rank_checks"):
            raise ValueError("missing grounded candidate parity or expected weights")
        with np.load(path, allow_pickle=False) as tensors:
            rankers[mode] = rank.LoopTacticRanker(dict(tensors), advance)
    goals = freeze(root)
    entries = library(config.LEAN_HOME)
    arms = ("lexical", *rank.MODES)
    protocol = {
        "fixture_sha256": sha256_file(root / "fixture.json"), "library_sha256": sha256_json(entries),
        "native_order": rank.TACTICS, "fallback_budget": FALLBACK_BUDGET, "fallback_arms": arms,
        "evaluator_sha256": sha256_file(Path(__file__)), "parity_sha256": sha256_file(parity_path),
        "gate_source_sha256": sha256_file(Path(__file__).with_name("rlt_lean_ldt.py")),
        "lean_sha256": sha256_file(config.LEAN_EXE), "repl_sha256": sha256_file(config.REPL_EXE),
        "membrane_sha256": sha256_file(Path(hybrid.__file__)),
    }
    frozen = root / "protocol.json"
    if frozen.exists() and sha256_json(load_structured(frozen)) != sha256_json(protocol):
        raise ValueError("the frozen grounded protocol changed")
    dump_structured(frozen, protocol)
    dump_structured(root / "library.json", {"entries": entries})
    traces = root / "traces"
    traces.mkdir(exist_ok=True)
    with LeanRepl(timeout=30, restart_on_timeout=False, env=lean_env(config), cwd=root) as lean:
        if lean.version != config.LEAN_VERSION:
            raise ValueError("unexpected Lean toolchain")
        for index, goal in enumerate(goals):
            guard()
            path = traces / f"{goal['id']}.json"
            if path.exists():
                continue
            state = lean.init(goal["statement"], pp=True)
            pp = state["goals"][0]["pp"]
            native = trial(lean, hybrid, goal, state, "native", len(rank.TACTICS), {}, rank.TACTICS, guard)
            record = {"goal": goal, "goal_pp": pp, "native": native}
            if native["accepted"] is None:
                started = time.perf_counter()
                hits, pool = candidates(pp, entries)
                elapsed = time.perf_counter() - started
                if len(pool) < FALLBACK_BUDGET:
                    raise ValueError("the goal's grounded pool is smaller than the fallback budget")
                rotated = arms[index % len(arms) :] + arms[: index % len(arms)]
                fallback = [trial(lean, hybrid, goal, state, arm, FALLBACK_BUDGET,
                                  rankers, pool, guard, grounded=True) for arm in rotated]
                coverage = trial(lean, hybrid, goal, state, "pool_coverage", len(pool), {}, pool, guard)
                record.update({"retrieval": {"lemmas_scored": len(entries), "hits": hits, "wall_s": elapsed},
                               "pool": pool, "fallback": fallback, "pool_coverage": coverage})
            lean.reset()
            pending = path.with_suffix(".part.json")
            dump_structured(pending, record)
            pending.replace(path)
            if (index + 1) % 18 == 0:
                print(f"Recorded paired Lean outcomes for {index + 1}/{len(goals)} goals", flush=True)


def counts(results):
    values = Counter()
    for result in results:
        values["visited_goals"] += 1
        values["certified"] += result["accepted"] is not None
        for field in (*COUNTS, *TIMES):
            values[field] += result[field]
    return values


def certify(root, guard):
    from lcgrt import config

    protocol = load_structured(root / "protocol.json")
    path = root / "fixture.json"
    if sha256_file(path) != protocol["fixture_sha256"] or sha256_file(config.LEAN_EXE) != protocol["lean_sha256"]:
        raise ValueError("certificate inputs changed")
    goals = load_structured(path)["goals"]
    winning = {}
    planted = {}
    rows = []
    for goal in goals:
        row = load_structured(root / "traces" / f"{goal['id']}.json")
        if row["goal"] != goal:
            raise ValueError("frozen goal and run receipt disagree")
        rows.append(row)
        _, stmt, tactic = next(item for item in propositions(goal["salt"]) if item[0] == goal["family"])
        if stmt != goal["statement"]:
            raise ValueError("planted proof does not match the frozen goal")
        planted[goal["id"], tactic] = stmt
        for result in (row["native"], *row.get("fallback", ()), *([row["pool_coverage"]] if "pool" in row else [])):
            accepted = result["accepted"]
            pool = protocol["native_order"] if result["arm"] == "native" else row["pool"]
            if accepted is None:
                continue
            if accepted not in pool or result["attempts"][-1]["tactic"] != accepted:
                raise ValueError("accepted tactic did not come from the frozen candidate pool")
            winning[goal["id"], accepted] = stmt
    replays = certify_proofs(root, planted | winning, config, guard, batch_size=8)
    stalled = [row for row in rows if row["native"]["accepted"] is None]
    cells = [{"arm": "native", **counts(row["native"] for row in rows)}]
    for arm in protocol["fallback_arms"]:
        cell = counts(next(item for item in row["fallback"] if item["arm"] == arm) for row in stalled)
        total = cell.copy()
        total.update(counts(row["native"] for row in rows))
        total["visited_goals"] = len(rows)
        cells.append({"arm": arm, "fallback_only": dict(cell), **total})
    dump_structured(root / "evidence.json", {
        "schema": "datagoblin.rlt-lean-grounded-fallback.v1", "protocol": protocol,
        "cells": cells, "native_stalls": len(stalled),
        "retrieval_calls": len(stalled),
        "retrieval_lemmas_scored": sum(row["retrieval"]["lemmas_scored"] for row in stalled),
        "retrieval_wall_s": sum(row["retrieval"]["wall_s"] for row in stalled),
        "pool_coverage": counts(row["pool_coverage"] for row in stalled),
        "standalone_replay": replays, "distinct_winning_proofs": len(winning), "planted_proofs": len(planted),
        "certificate_evaluator_sha256": sha256_file(Path(__file__)),
    })
