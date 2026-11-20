"""Run the trained grounded rankers and certify the held-out proof results."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import load_structured

from . import rlt_lean_grounded as grounded
from . import rlt_lean_labels
from . import rlt_lean_train_rank as rank
from .rlt_lean_labels import atomic_record, held_templates, verify_sources
from .rlt_lean_ldt import certify_proofs, lean_env, trial
from .rlt_lean_rank import TACTICS

HOLDOUTS = ("same_templates", "held_templates")


def load_fits(root, protocol):
    path = root / "fit-protocol.json"
    fit = load_structured(path)
    receipts = load_structured(root / "fit-receipts.json")
    expected = {(seed, mode) for seed in protocol["fit"]["seeds"] for mode in fit["modes"]}
    if (
        receipts["fit_protocol_sha256"] != sha256_file(path)
        or fit["source_sha256"] != sha256_file(Path(rank.__file__))
        or fit["protocol_sha256"] != sha256_file(root / "protocol.json")
        or fit["label_manifest_sha256"] != sha256_file(root / "label-manifest.json")
        or {(r["seed"], r["mode"]) for r in receipts["arms"]} != expected
        or len(receipts["arms"]) != len(expected)
    ):
        raise ValueError("fitted rankers do not match the frozen experiment")
    models = {}
    for arm in receipts["arms"]:
        seed, mode = arm["seed"], arm["mode"]
        weight_path = root / "fits" / str(seed) / mode / "weights.npz"
        if sha256_file(weight_path) != arm["weights_sha256"]:
            raise ValueError("fitted ranker changed")
        with np.load(weight_path, allow_pickle=False) as weights:
            models[f"{seed}/{mode}"] = (mode, dict(weights))
    return fit, models


def evaluate(root, guard):
    from lcgrt import config
    from lcgrt.lean_repl import LeanRepl
    from research_gym.core import hybrid

    protocol = load_structured(root / "protocol.json")
    verify_sources(root, protocol, config, hybrid)
    fit, models = load_fits(root, protocol)
    budget = protocol["budget"]
    fixtures = load_structured(root / "fixtures.json")["splits"]
    entries = load_structured(root / "library.json")["entries"]
    words = fit["feature_vocabulary"]
    registration = {
        "collect_protocol_sha256": sha256_file(root / "protocol.json"),
        "fit_receipts_sha256": sha256_file(root / "fit-receipts.json"),
        "source_sha256": sha256_file(Path(__file__)), "native": TACTICS, "budget": budget,
        "holdout_source_sha256": sha256_file(Path(rlt_lean_labels.__file__)),
        "policies": ["lexical", *models], "holdouts": HOLDOUTS,
    }
    path = root / "eval-protocol.json"
    if path.exists() and sha256_json(load_structured(path)) != sha256_json(registration):
        raise ValueError("the frozen evaluation changed")
    atomic_record(path, registration)
    traces = root / "eval-traces"
    traces.mkdir(exist_ok=True)
    with LeanRepl(timeout=30, restart_on_timeout=False, env=lean_env(config), cwd=root) as lean:
        if lean.version != config.LEAN_VERSION:
            raise ValueError("unexpected Lean version")
        for split in HOLDOUTS:
            for i, goal in enumerate(fixtures[split], 1):
                guard()
                output = traces / f"{goal['id']}.json"
                if output.exists():
                    continue
                state = lean.init(goal["statement"], pp=True)
                pp = state["goals"][0]["pp"]
                native = trial(lean, hybrid, goal, state, "native", len(TACTICS), {}, TACTICS, guard)
                row = {"split": split, "goal": goal, "goal_pp": pp, "native": native}
                if native["accepted"] is None:
                    started = time.perf_counter()
                    hits, pool = grounded.candidates(pp, entries)
                    seconds = time.perf_counter() - started
                    if len(pool) < budget:
                        raise ValueError("frozen candidate pool is smaller than the matched budget")
                    policies = registration["policies"]
                    offset = i % len(policies)
                    fallback = {}
                    for policy in policies[offset:] + policies[:offset]:
                        if policy == "lexical":
                            mode, rankers = policy, {}
                        else:
                            mode, weights = models[policy]
                            rankers = {mode: rank.CheckedRanker(weights, words, hits)}
                        fallback[policy] = trial(
                            lean, hybrid, goal, state, mode, budget, rankers, pool, guard, grounded=True,
                        )
                    coverage = trial(lean, hybrid, goal, state, "pool_coverage", len(pool), {}, pool, guard)
                    row.update({
                        "fallback": fallback, "pool": pool, "pool_coverage": coverage,
                        "retrieval": {"hits": hits, "wall_s": seconds, "lemmas_scored": len(entries)},
                    })
                lean.reset()
                atomic_record(output, row)
                if i % 12 == 0:
                    print(f"Verified {i}/{len(fixtures[split])} {split} goals", flush=True)


def certify(root, guard):
    from lcgrt import config

    protocol = load_structured(root / "protocol.json")
    evaluation = load_structured(root / "eval-protocol.json")
    if (
        sha256_file(root / "fixtures.json") != protocol["fixtures_sha256"]
        or sha256_file(config.LEAN_EXE) != protocol["lean_sha256"]
        or sha256_file(Path(__file__)) != evaluation["source_sha256"]
        or sha256_file(Path(rlt_lean_labels.__file__)) != evaluation["holdout_source_sha256"]
    ):
        raise ValueError("the frozen certificate inputs changed")
    fixtures = load_structured(root / "fixtures.json")["splits"]
    winning = {}
    planted = {}
    summaries = {}
    hashes = {}
    for split in HOLDOUTS:
        rows = []
        for goal in fixtures[split]:
            path = root / "eval-traces" / f"{goal['id']}.json"
            row = load_structured(path)
            if row["goal"] != goal or row["split"] != split:
                raise ValueError("scored goal does not match its frozen split")
            hashes[goal["id"]] = sha256_file(path)
            rows.append(row)
            make = held_templates if split == "held_templates" else grounded.propositions
            _, stmt, construction = next(r for r in make(goal["salt"]) if r[0] == goal["family"])
            if stmt != goal["statement"]:
                raise ValueError("withheld proof statement changed")
            planted[goal["id"], construction] = stmt
            arms = [row["native"]]
            if "fallback" in row:
                arms.extend([*row["fallback"].values(), row["pool_coverage"]])
            for result in arms:
                tactic = result["accepted"]
                pool = evaluation["native"] if result is row["native"] else row["pool"]
                if tactic is not None:
                    if tactic not in pool or result["attempts"][-1]["tactic"] != tactic:
                        raise ValueError("committed tactic did not come from the frozen pool")
                    winning[goal["id"], tactic] = stmt
        stalled = [r for r in rows if r["native"]["accepted"] is None]
        native = grounded.counts(r["native"] for r in rows)
        cells = []
        for policy in evaluation["policies"]:
            fallback = grounded.counts(r["fallback"][policy] for r in stalled)
            total = native.copy()
            total.update(fallback)
            total["visited_goals"] = len(rows)
            cells.append({"policy": policy, "end_to_end": total, "fallback_only": fallback})
        summaries[split] = {
            "native": native, "native_stalls": len(stalled), "policies": cells,
            "pool_coverage": grounded.counts(r["pool_coverage"] for r in stalled),
            "retrieval_calls": len(stalled), "lemma_comparisons": sum(r["retrieval"]["lemmas_scored"] for r in stalled),
            "retrieval_wall_s": sum(r["retrieval"]["wall_s"] for r in stalled),
        }
    replays = certify_proofs(root, planted | winning, config, guard, batch_size=8)
    atomic_record(root / "evidence.json", {
        "schema": "datagoblin.lean-checked-ranker.v1", "splits": summaries,
        "standalone_replay": replays, "withheld_proofs": len(planted), "winning_scripts": len(winning),
        "trace_sha256": hashes, "fit_protocol": load_structured(root / "fit-protocol.json"),
        "fit_receipts": load_structured(root / "fit-receipts.json"), "eval_protocol": evaluation,
    })
