"""Collect paired Lean evidence for widening and composing lemma proposals."""

from __future__ import annotations

import re
import secrets
import time
from collections import Counter
from pathlib import Path

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import load_structured

from . import rlt_lean_composition as composition
from . import rlt_lean_grounded as grounded
from .rlt_lean_labels import INCONCLUSIVE, atomic_record, held_templates
from .rlt_lean_ldt import LeanGate, certify_proofs, lean_env, trial
from .rlt_lean_rank import TACTICS

TRAIN_FAMILIES = frozenset((
    "append_filter_map", "count_reverse_flatten", "gcd_commute_reassociate", "map_take_drop_while",
    "reverse_flatten_map", "reverse_take", "drop_then_take_while_map", "zip_drop_map", "reverse_flat_map",
))
SPLITS = ("train", "held_structures")
BUDGETS = (6, 12, 24)
LIBRARY_NAMES = re.compile(r"\b(?:List|Nat)\.[A-Za-z_][A-Za-z0-9_]*")


def train_templates(n):
    for episode in (*grounded.propositions(n), *held_templates(n)):
        if episode[0] in TRAIN_FAMILIES:
            yield episode


def test_templates(n):
    yield from (
        (
            "map_after_reverse_take",
            f"∀ (xs : List Nat) (f : Nat → Nat) (k : Nat), (((xs ++ [{n}]).take k).reverse).map f = "
            f"(([{n}].reverse ++ xs.reverse).map f).drop ((xs ++ [{n}]).length - k)",
            "simp only [List.reverse_take, List.reverse_append, List.map_drop]",
        ),
        (
            "take_map_after_reverse_flatten",
            f"∀ (xss : List (List Nat)) (f : Nat → Nat) (k : Nat), "
            f"(((xss ++ [[{n}]]).flatten.reverse).take k).map f = "
            f"((((xss ++ [[{n}]]).map List.reverse).reverse.map (List.map f)).flatten).take k",
            "simp only [List.reverse_flatten, List.map_take, List.map_flatten]",
        ),
        (
            "filter_map_reverse_append",
            f"∀ (xs ys : List Nat) (f : Nat → Option Nat) (g : Nat → Nat), "
            f"(((xs ++ [{n}]) ++ ys).filterMap f).reverse.map g = "
            f"(ys.filterMap (fun x => (f x).map g)).reverse ++ "
            f"((xs ++ [{n}]).filterMap (fun x => (f x).map g)).reverse",
            "simp only [List.filterMap_append, List.reverse_append, List.map_reverse, "
            "List.map_append, List.map_filterMap]",
        ),
        (
            "count_partitioned_reverse_flatten",
            f"∀ (xss yss : List (List Nat)), (((xss ++ [[{n}]]) ++ yss).flatten.reverse).count {n} = "
            f"(xss.map (List.count {n})).sum + 1 + (yss.map (List.count {n})).sum",
            "simp only [List.count_reverse, List.flatten_append, List.count_append, "
            "List.flatten_singleton, List.count_cons_self, List.count_nil, Nat.zero_add, List.count_flatten]",
        ),
        (
            "drop_after_predicate_pair",
            f"∀ (xs : List Nat) (f : Nat → Nat) (p q : Nat → Bool) (k : Nat), "
            f"((((xs ++ [{n}]).map f).dropWhile p).takeWhile q).drop k = "
            f"((((xs ++ [{n}]).dropWhile (p ∘ f)).takeWhile (q ∘ f)).drop k).map f",
            "simp only [List.takeWhile_map, List.dropWhile_map, List.map_drop]",
        ),
        (
            "gcd_reassociate_under_add",
            f"∀ (a b c : Nat), Nat.gcd (Nat.gcd (a + {n}) b) c + {n} = "
            f"Nat.gcd (a + {n}) (Nat.gcd c b) + {n}",
            "rw [Nat.gcd_assoc] <;> congr 2 <;> apply Nat.gcd_comm",
        ),
        (
            "take_after_map_zip",
            f"∀ (f : Nat → Nat → Nat) (g : Nat → Nat) (xs ys : List Nat) (k : Nat), "
            f"((List.zipWith f (xs ++ [{n}]) ys).map g).take k = "
            f"List.zipWith (fun a b => g (f a b)) ((xs ++ [{n}]).take k) (ys.take k)",
            "simp only [List.map_zipWith, List.take_zipWith]",
        ),
    )


GENERATORS = {"train": train_templates, "held_structures": test_templates}


def fixture(root, previous):
    path = root / "fixture.json"
    if path.exists():
        return load_structured(path)["splits"]
    excluded = {g["salt"] for group in previous["splits"].values() for g in group}
    splits = {}
    for name, count in (("train", 4), ("held_structures", 4)):
        salts = []
        while len(salts) < count:
            salt = 30000 + secrets.randbelow(45000)
            if salt not in excluded:
                excluded.add(salt)
                salts.append(salt)
        splits[name] = [
            {"id": sha256_json(stmt), "family": family, "lineage": sha256_json(["lean-pool-v1", family]),
             "statement": stmt, "salt": salt}
            for salt in sorted(salts) for family, stmt, _ in GENERATORS[name](salt)
        ]
    atomic_record(path, {"schema": "datagoblin.lean-pool-audit-fixture.v1", "splits": splits})
    return splits


def register(root, previous_path, config, hybrid):
    splits = fixture(root, load_structured(previous_path))
    entries = grounded.library(config.LEAN_HOME)
    protocol = {
        "fixture_sha256": sha256_file(root / "fixture.json"),
        "prior_fixture_sha256": sha256_file(previous_path),
        "retrieval_index_sha256": sha256_json(entries),
        "learner_label_semantics": "1=verified, 0=failed, null=unknown",
        "source_sha256": sha256_file(Path(__file__)),
        "generator_sha256": sha256_file(Path(composition.__file__)),
        "gate_sha256": sha256_file(Path(__file__).with_name("rlt_lean_ldt.py")),
        "membrane_sha256": sha256_file(Path(hybrid.__file__)),
        "lean_sha256": sha256_file(config.LEAN_EXE),
        "repl_sha256": sha256_file(config.REPL_EXE),
        "methods": composition.METHODS, "budgets": BUDGETS,
        "pool_limit": composition.POOL_LIMIT, "native_tactics": TACTICS,
    }
    path = root / "protocol.json"
    if path.exists() and sha256_json(protocol) != sha256_json(load_structured(path)):
        raise ValueError("registered Lean pool audit changed")
    atomic_record(path, protocol)
    atomic_record(root / "library.json", {"entries": entries})
    return splits, entries


def pool_repl_class(base, lean_error):
    class PoolRepl(base):
        def _reader(self):
            # A retiring reader must only signal the queue of its own process.
            events, stdout = self._q, self.proc.stdout
            for line in stdout:
                events.put(line)
            events.put(None)

        def init(self, stmt, pp=False):
            state = super().init(stmt, pp)
            self.root_query = (stmt, pp)
            self.logical_sid = self.current_sid = state["sid"]
            return state

        def request(self, **request):
            operation = request["op"]
            if operation not in ("run", "kernel"):
                return super().request(**request)
            if operation == "run":
                if request["sid"] != self.logical_sid:
                    raise ValueError("pool tactic does not reference the current root goal")
                request["sid"] = self.current_sid
            try:
                return super().request(**request)
            except lean_error as error:
                if error.code == "lean_timeout":
                    self.restart()
                    restored = super().init(*self.root_query)
                    self.current_sid = restored["sid"]
                    return {"ok": False, "error": f"Lean timed out: {error.message}"}
                if error.code != "lean_request_error" or not INCONCLUSIVE.search(error.message):
                    raise
                return {"ok": False, "error": error.message}

    return PoolRepl


def evaluate(root, prior, guard):
    from lcgrt import config
    from lcgrt.errors import LeanError
    from lcgrt.lean_repl import LeanRepl
    from research_gym.core import hybrid

    splits, entries = register(root, prior, config, hybrid)
    preflight = root / "witness-preflight.json"
    receipts = root / "traces"
    receipts.mkdir(exist_ok=True)
    connection = pool_repl_class(LeanRepl, LeanError)
    with connection(timeout=30, restart_on_timeout=False, env=lean_env(config), cwd=root) as lean:
        if lean.version != config.LEAN_VERSION:
            raise ValueError("unexpected Lean version")
        if not preflight.exists():
            dev = []
            for family, stmt, tactic in test_templates(827):
                guard()
                state = lean.init(stmt, pp=True)
                gate = LeanGate(hybrid, lean, state["sid"], sha256_json(stmt), (tactic,))
                if not gate.check(tactic):
                    raise ValueError(f"construction proof failed for {family}: {gate.attempts}")
                dev.append({"family": family, "attempt": gate.attempts[-1]})
                lean.reset()
            atomic_record(preflight, {"checked": dev})
            print(f"Lean checked {len(dev)} development witnesses", flush=True)
        since_restart = 0
        for split in SPLITS:
            for index, goal in enumerate(splits[split], 1):
                guard()
                path = receipts / f"{goal['id']}.json"
                if path.exists():
                    continue
                if since_restart == 8:
                    lean.restart()
                    since_restart = 0
                state = lean.init(goal["statement"], pp=True)
                pp = state["goals"][0]["pp"]
                native = None
                if split == "held_structures":
                    native = trial(lean, hybrid, goal, state, "native", len(TACTICS), {}, TACTICS, guard)
                start = time.perf_counter()
                hits, pools = composition.expanded_candidates(pp, entries)
                generated_wall_s = time.perf_counter() - start
                union = tuple(dict.fromkeys(tactic for pool in pools.values() for tactic in pool))
                results = []
                for tactic in union:
                    guard()
                    gate = LeanGate(hybrid, lean, state["sid"], goal["id"], union)
                    passed = gate.check(tactic)
                    result, = gate.attempts
                    label = None if INCONCLUSIVE.search(result.get("error", "")) else int(passed)
                    results.append(dict(result, label=label))
                lean.reset()
                atomic_record(path, {
                    "goal": goal, "split": split, "goal_pp": pp, "native": native,
                    "retrieved": hits, "pools": pools, "outcomes": results,
                    "retrieval_comparisons": 2 * len(entries),
                    "generation_wall_s": generated_wall_s,
                })
                since_restart += 1
                if index % 8 == 0:
                    print(f"Recorded {index}/{len(splits[split])} {split} pools", flush=True)


def summary(rows, method, budget):
    counts = Counter()
    for row in rows:
        indexed = {r["tactic"]: r for r in row["outcomes"]}
        pool = row["pools"][method]
        counts["goals"] += 1
        counts["pool_candidates"] += len(pool)
        counts["any_verified_in_pool"] += any(indexed[t]["label"] == 1 for t in pool)
        for tactic in pool[:budget]:
            result = indexed[tactic]
            counts["tactic_requests"] += 1
            counts["kernel_requests"] += result["kernel_calls"]
            counts["tactic_wall_s"] += result["tactic_wall_s"]
            counts["kernel_wall_s"] += result["kernel_wall_s"]
            counts["inconclusive"] += result["label"] is None
            if result["label"] == 1:
                counts["verified"] += 1
                break
    return dict(counts)


def retrieval_coverage(rows, witnesses, indexed, method):
    counts = Counter()
    for row in rows:
        names = set(LIBRARY_NAMES.findall(witnesses[row["goal"]["id"]]))
        retrieved = set(row["retrieved"][method])
        counts["witness_lemmas"] += len(names)
        counts["indexed_witness_lemmas"] += len(names & indexed)
        counts["retrieved_witness_lemmas"] += len(names & retrieved)
        counts["all_witness_lemmas_retrieved"] += bool(names) and names <= retrieved
    return dict(counts)


def certify(root, guard):
    from lcgrt import config

    protocol = load_structured(root / "protocol.json")
    if (
        sha256_file(root / "fixture.json") != protocol["fixture_sha256"]
        or sha256_file(Path(__file__)) != protocol["source_sha256"]
        or sha256_file(Path(composition.__file__)) != protocol["generator_sha256"]
        or sha256_file(config.LEAN_EXE) != protocol["lean_sha256"]
    ):
        raise ValueError("certificate inputs differ from the registered run")
    goals = load_structured(root / "fixture.json")["splits"]
    indexed = {entry["name"] for entry in load_structured(root / "library.json")["entries"]}
    certificates = {}
    overview = {}
    hashes = {}
    for split, group in goals.items():
        checked = []
        counts = Counter()
        witnesses = {}
        for goal in group:
            path = root / "traces" / f"{goal['id']}.json"
            trace = load_structured(path)
            if trace["goal"] != goal or trace["split"] != split:
                raise ValueError("trace and fixture disagree")
            hashes[goal["id"]] = sha256_file(path)
            _, stmt, witness = next(item for item in GENERATORS[split](goal["salt"]) if item[0] == goal["family"])
            if stmt != goal["statement"]:
                raise ValueError("witness and frozen statement disagree")
            witnesses[goal["id"]] = witness
            certificates[goal["id"], witness] = stmt
            for row in trace["outcomes"]:
                counts["actual_kernel_requests"] += row["kernel_calls"]
                counts["actual_tactic_requests"] += 1
                counts["actual_tactic_wall_s"] += row["tactic_wall_s"]
                counts["actual_kernel_wall_s"] += row["kernel_wall_s"]
                counts["positive" if row["label"] == 1 else "unknown" if row["label"] is None else "negative"] += 1
                if row["label"] == 1:
                    certificates[goal["id"], row["tactic"]] = stmt
            if trace["native"]:
                counts["native_tactic_requests"] += trace["native"]["tactic_checks"]
                counts["native_kernel_requests"] += trace["native"]["kernel_calls"]
                counts["native_inconclusive_tactic_requests"] += sum(
                    bool(INCONCLUSIVE.search(row.get("error", ""))) for row in trace["native"]["attempts"]
                )
                if trace["native"]["accepted"]:
                    certificates[goal["id"], trace["native"]["accepted"]] = stmt
            checked.append(trace)
        comparison = [r for r in checked if r["native"] is None or not r["native"]["accepted"]]
        overview[split] = {
            "goals": len(group), "comparison_goals": len(comparison), "measured_work": counts,
            "generation_wall_s": sum(r["generation_wall_s"] for r in checked),
            "retrieval_comparisons": sum(r["retrieval_comparisons"] for r in checked),
            "witness_retrieval": {
                method: retrieval_coverage(comparison, witnesses, indexed, method) for method in composition.METHODS
            },
            "policies": {method: {str(k): summary(comparison, method, k)
                                  for k in (*BUDGETS, composition.POOL_LIMIT)} for method in composition.METHODS},
        }
    replays = certify_proofs(root, certificates, config, guard, batch_size=8)
    atomic_record(root / "evidence.json", {
        "schema": "datagoblin.lean-pool-audit.v1", "overview": overview,
        "protocol_sha256": sha256_file(root / "protocol.json"), "traces_sha256": hashes,
        "standalone_replay": replays,
    })
