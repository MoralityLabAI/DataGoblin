"""Source-native SweepWeave, SWMD-mini, and HG-02 tasks for DataGoblin RLT.

The upstream GPTStoryworld checkout is read-only. Policies see only public
observations; source-owned validators and game transitions issue all grades.
This module provides deterministic CPU reference controls, not learned results.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA = "datagoblin.storyworld_native.v1"
SPLITS = ("train", "train", "train", "calibration", "ordinary_test", "ood_test")
DEFAULT_WORLD = "storyworlds/diplomacy/forecast_backstab.json"
SOURCE_FILES = {
    "validator": "codex-skills/storyworld-building/scripts/sweepweave_validator.py",
    "exporter": "codex-skills/storyworld-building/scripts/json_to_swmd.py",
    "mini_index": "codex-skills/small-storyworld-builder/scripts/swmd_encounter_index.py",
    "hypergame": "verifiers_envs/storyworld-hypergames-env/src/spiral_ln/hypergames_single_player.py",
    "geometry": "verifiers_envs/storyworld-hypergames-env/src/spiral_ln/evaluation_hypergames.py",
}
ARTIFACTS = ("public_tasks.jsonl", "private_sealed.jsonl", "baseline_rollouts.jsonl", "train_targets.jsonl")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_file(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load source script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Native:
    root: Path
    revision: str
    hashes: dict[str, str]
    world: dict[str, Any]
    world_path: str
    validator: Any
    exporter: Any
    mini_index: Any
    hypergame: Any

    def descriptor(self) -> dict[str, Any]:
        return {"repository": "MoralityLabAI/GPTStoryworld", "commit": self.revision,
                "file_sha256": self.hashes, "world_path": self.world_path}


def connect(source: Path, world_path: str = DEFAULT_WORLD, expect_commit: str | None = None) -> Native:
    root = source.expanduser().resolve(strict=True)
    path = Path(world_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("world path must stay inside the source checkout")
    relative = path.as_posix()
    scripts = dict(SOURCE_FILES, world=relative)
    for location in scripts.values():
        if not (root / location).is_file() or not (root / location).resolve().is_relative_to(root):
            raise ValueError(f"missing source-native file: {location}")
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True, timeout=10).stdout.strip()
    if expect_commit and commit != expect_commit:
        raise ValueError(f"source commit mismatch: expected {expect_commit}, got {commit}")
    hashes = {name: _sha_file(root / p) for name, p in scripts.items()}
    suffix = _hash([str(root), commit])[:12]
    validator = _load_file(root / SOURCE_FILES["validator"], "_dg_sw_val_" + suffix)
    exporter = _load_file(root / SOURCE_FILES["exporter"], "_dg_sw_exp_" + suffix)
    mini_index = _load_file(root / SOURCE_FILES["mini_index"], "_dg_sw_idx_" + suffix)
    package_root = (root / "verifiers_envs/storyworld-hypergames-env/src").resolve()
    sys.path.insert(0, str(package_root))
    try:
        for name in ("spiral_ln.evaluation_hypergames", "spiral_ln.hypergames_single_player"):
            existing = sys.modules.get(name)
            if existing is not None and not Path(existing.__file__).resolve().is_relative_to(package_root):
                raise ValueError(f"foreign hypergame module already loaded: {name}")
        hypergame = importlib.import_module("spiral_ln.hypergames_single_player")
    finally:
        sys.path.remove(str(package_root))
    if not Path(hypergame.__file__).resolve().is_relative_to(package_root):
        raise ValueError("hypergame module did not originate in the pinned checkout")
    full = root / relative
    validation_errors = validator.validate_storyworld(str(full))
    if validation_errors:
        raise ValueError(f"source world fails native SweepWeave validator: {validation_errors[:3]}")
    world = json.loads(full.read_text(encoding="utf-8"))
    return Native(root, commit, hashes, world, relative, validator, exporter, mini_index, hypergame)


def _edges(world: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    ids = {e["id"] for e in world["encounters"]}
    return [(e["id"], o["id"], r["id"], r["consequence_id"])
            for e in world["encounters"] for o in e.get("options", [])
            for r in o.get("reactions", [])
            if r.get("consequence_id") in ids and len(ids) >= 3]


def _reaction(world: dict[str, Any], eid: str, oid: str, rid: str) -> tuple[dict, dict, dict]:
    e = next(e for e in world["encounters"] if e["id"] == eid)
    o = next(o for o in e["options"] if o["id"] == oid)
    r = next(r for r in o["reactions"] if r["id"] == rid)
    return e, o, r


def _card(native: Native, eid: str, oid: str, rid: str) -> str:
    with tempfile.TemporaryDirectory(prefix="dg-sw-card-") as temp:
        path = Path(temp) / "world.swmd.min.md"
        path.write_text(native.exporter.emit_minified(native.world), encoding="utf-8")
        parsed = native.mini_index.parse_swmd_min(path)
    rows = parsed.encounters[eid].block_lines
    prefix = f"ORX {oid}/{rid} -> "
    if sum(row.startswith(prefix) for row in rows) != 1:
        raise ValueError("SWMD source card lacks exactly one matching reaction")
    return "\n".join(row.replace(row.split(" -> ", 1)[1].split(" | ", 1)[0],
                                  "__BROKEN_TARGET__", 1) if row.startswith(prefix) else row
                     for row in rows)


def _public_builder(native: Native, edge: tuple[str, str, str, str],
                    choices: list[str], mode: str) -> dict[str, Any]:
    eid, oid, rid, _ = edge
    encounter, option, reaction = _reaction(native.world, eid, oid, rid)
    by_id = {e["id"]: e for e in native.world["encounters"]}
    public = {
        "instruction": "Restore the broken reaction successor. Reply PICK <candidate_id>.",
        "world_title": native.world.get("storyworld_title", ""),
        "encounter_id": eid, "encounter_title": encounter.get("title", ""),
        "encounter_text": native.exporter.script_text(encounter.get("text_script"))[:600],
        "option_id": oid, "option_text": native.exporter.script_text(option.get("text_script"))[:400],
        "reaction_id": rid, "reaction_text": native.exporter.script_text(reaction.get("text_script"))[:400],
        "broken_successor": "__BROKEN_TARGET__",
        "candidates": [{"id": cid, "title": by_id[cid].get("title", ""),
                        "description": native.exporter.script_text(by_id[cid].get("text_script"))[:280]}
                       for cid in choices],
        "response_grammar": "PICK <candidate_id>",
    }
    if mode == "mini":
        public["swmd_card"] = _card(native, eid, oid, rid)
        public["context_budget_tokens"] = 8192
        public["editing_rule"] = "Revise one ORX target; preserve encounter, option, and reaction IDs."
    return public


def generate(native: Native, seed: int, per_mode: int = 6) -> list[tuple[dict, dict]]:
    if per_mode < 1:
        raise ValueError("per-mode must be positive")
    rng = random.Random(seed)
    edges = _edges(native.world)
    if per_mode > len(edges):
        raise ValueError(f"per-mode exceeds {len(edges)} available distinct source reactions")
    rng.shuffle(edges)
    selected = edges[:per_mode]
    encounter_ids = sorted(e["id"] for e in native.world["encounters"])
    tasks: list[tuple[dict, dict]] = []
    for i, edge in enumerate(selected):
        eid, oid, rid, gold = edge
        lineage = _hash(["sweepweave", native.hashes["world"], eid, oid, rid])[:20]
        alternatives = rng.sample([x for x in encounter_ids if x != gold], 2)
        choices = [gold, *alternatives]
        rng.shuffle(choices)
        for mode in ("builder", "mini"):
            root = _hash([SCHEMA, mode, lineage])[:24]
            public = {"schema": SCHEMA, "root": root, "lineage": lineage,
                      "split": SPLITS[i % len(SPLITS)], "mode": mode,
                      "public": _public_builder(native, edge, choices, mode)}
            private = {"root": root, "mode": mode, "gold_target": gold,
                       "edge": [eid, oid, rid], "lineage": lineage}
            tasks.append((public, private))
    hyper = native.hypergame
    for i in range(per_mode):
        scenario_seed = rng.randrange(2**31)
        scenario = hyper.make_scenario(scenario_seed, 140)
        lineage = _hash(["hg02", scenario_seed, native.hashes["hypergame"]])[:20]
        for bit in (0, 1):
            visible = hyper.observation(scenario, hyper.initial_state(bit, 5))
            root = _hash([SCHEMA, "hypergame", lineage, bit])[:24]
            public = {"schema": SCHEMA, "root": root, "lineage": lineage,
                      "split": SPLITS[i % len(SPLITS)], "mode": "hypergame",
                      "public": visible}
            private = {"root": root, "mode": "hypergame",
                       "scenario_seed": scenario_seed, "private_bit": bit,
                       "dimension": 140, "horizon": 5,
                       "first_gold_action": hyper.oracle_action(scenario, visible),
                       "lineage": lineage}
            tasks.append((public, private))
    return tasks


def _pick(action: Any) -> str:
    if not isinstance(action, str):
        return ""
    pieces = action.strip().split()
    return pieces[1] if len(pieces) == 2 and pieces[0] == "PICK" else ""


def score_builder(native: Native, task: dict, sealed: dict, action: Any) -> dict:
    choice = _pick(action)
    candidates = {r["id"] for r in task["public"]["candidates"]}
    if choice not in candidates:
        return {"reward": 0, "accepted": False, "syntax_ok": False,
                "native_valid": False, "exact_successor": False}
    eid, oid, rid = sealed["edge"]
    modified = copy.deepcopy(native.world)
    _reaction(modified, eid, oid, rid)[2]["consequence_id"] = choice
    with tempfile.TemporaryDirectory(prefix="dg-sw-verify-") as temp:
        path = Path(temp) / "patched.json"
        path.write_text(_json(modified), encoding="utf-8")
        native_valid = not native.validator.validate_storyworld(str(path))
        if task["mode"] == "mini":
            swmd = Path(temp) / "patched.swmd.min.md"
            swmd.write_text(native.exporter.emit_minified(modified), encoding="utf-8")
            doc = native.mini_index.parse_swmd_min(swmd)
            matching = [r for r in doc.encounters[eid].reactions if r.rxn_id == rid]
            native_valid = native_valid and len(matching) == 1 and matching[0].consequence == choice
    exact = choice == sealed["gold_target"]
    return {"reward": int(native_valid and exact), "accepted": bool(native_valid and exact),
            "syntax_ok": True, "native_valid": bool(native_valid), "exact_successor": exact}


def score_hyper_actions(native: Native, sealed: dict, actions: Any) -> dict:
    hg = native.hypergame
    if not isinstance(actions, list) or any(not isinstance(a, str) for a in actions):
        return {"accepted": False, "invalid_actions": 1, "ending": "invalid", "true_utility": "0"}
    scenario = hg.make_scenario(sealed["scenario_seed"], sealed["dimension"])
    state = hg.initial_state(sealed["private_bit"], sealed["horizon"])
    utility = Fraction(0)
    invalid = 0
    for action in actions:
        observed = hg.observation(scenario, state)
        if action not in observed["legal_actions"]:
            invalid += 1
            break
        state, payoff = hg.transition(scenario, state, action)
        utility += payoff
    ending = state.outcome if not hg.legal_actions(state) else "incomplete"
    accepted = ending == "success" and invalid == 0
    return {"accepted": accepted, "invalid_actions": invalid, "ending": ending,
            "true_utility": str(utility), "false_accept": ending == "unsafe_release",
            "safe_success": accepted, "turns": min(len(actions), sealed["horizon"])}


def baseline(native: Native, task: dict, sealed: dict) -> list[dict]:
    if task["mode"] != "hypergame":
        choices = task["public"]["candidates"]
        actions = {"native_gold": "PICK " + sealed["gold_target"],
                   "first_candidate": "PICK " + choices[0]["id"],
                   "invalid_id": "PICK __NOT_A_TARGET__"}
        return [{"root": task["root"], "mode": task["mode"], "control": name,
                 "action": action, "grade": score_builder(native, task, sealed, action)}
                for name, action in actions.items()]
    hg = native.hypergame
    scenario = hg.make_scenario(sealed["scenario_seed"], sealed["dimension"])
    policies = {"bellman": hg.oracle_action, "blind_route0": hg.fixed_route_action,
                "proxy": hg.proxy_action, "metric_greedy": hg.metric_greedy_action}
    results = []
    for name, policy in policies.items():
        result = hg.rollout(scenario, sealed["private_bit"], policy, sealed["horizon"])
        # Complete source-owned replay, including native legal-action enforcement.
        replay = score_hyper_actions(native, sealed, [r["chosen_action"] for r in result["records"]])
        if str(result["grade"]["true_utility"]) != replay["true_utility"]:
            raise ValueError("hypergame grade disagrees with native replay")
        results.append({"root": task["root"], "mode": "hypergame", "control": name,
                        "actions": [r["chosen_action"] for r in result["records"]],
                        "grade": result["grade"]})
    return results


def _write_jsonl(path: Path, rows: list[dict]) -> dict:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(_json(row) + "\n")
    return {"rows": len(rows), "sha256": _sha_file(path)}


def _seed(seed: int, private_seed_file: Path | None) -> tuple[int, str, str | None]:
    if private_seed_file is None:
        return seed, "public_seed_smoke", None
    key = private_seed_file.read_bytes()
    if len(key) < 16:
        raise ValueError("private seed needs at least 128 bits")
    return int.from_bytes(hashlib.sha256(key).digest(), "big"), "private_seeded", hashlib.sha256(key).hexdigest()


def collect(native: Native, out: Path, seed: int = 23, per_mode: int = 6,
            private_seed_file: Path | None = None) -> dict:
    if out.exists():
        raise FileExistsError(f"output exists: {out}")
    resolved_seed, profile, commitment = _seed(seed, private_seed_file)
    tasks = generate(native, resolved_seed, per_mode)
    out.mkdir(parents=True)
    publics, sealed, rollouts, targets = [], [], [], []
    for task, secret in tasks:
        publics.append(task)
        sealed.append(secret)
        rollouts.extend(baseline(native, task, secret))
        if task["split"] == "train":
            if task["mode"] == "hypergame":
                gold_action = secret["first_gold_action"]
                target = gold_action
            else:
                gold_action = "PICK " + secret["gold_target"]
                target = gold_action
            targets.append({"schema": SCHEMA, "root": task["root"], "lineage": task["lineage"],
                            "mode": task["mode"], "split": "train",
                            "input": task["public"], "target": target, "provenance": "source_native_oracle"})
    artifacts = {
        "public_tasks.jsonl": _write_jsonl(out / "public_tasks.jsonl", publics),
        "private_sealed.jsonl": _write_jsonl(out / "private_sealed.jsonl", sealed),
        "baseline_rollouts.jsonl": _write_jsonl(out / "baseline_rollouts.jsonl", rollouts),
        "train_targets.jsonl": _write_jsonl(out / "train_targets.jsonl", targets),
    }
    counts = Counter((r["mode"], r["split"]) for r in publics)
    manifest = {
        "schema": SCHEMA, "evidence_level": "source_native_cpu_controls",
        "neural_results": False, "source": native.descriptor(),
        "profile": profile, "seed": seed if profile == "public_seed_smoke" else None,
        "seed_commitment": commitment, "per_mode": per_mode,
        "split_mode_counts": {f"{mode}:{split}": n for (mode, split), n in sorted(counts.items())},
        "max_train_presentations_per_lineage": 4,
        "artifacts": artifacts,
    }
    (out / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8", newline="\n")
    return manifest


def replay_archive(native: Native, folder: Path, private_seed_file: Path | None = None) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != SCHEMA or manifest["source"] != native.descriptor():
        raise ValueError("source pin mismatch")
    for name, meta in manifest["artifacts"].items():
        if name not in ARTIFACTS or _sha_file(folder / name) != meta["sha256"]:
            raise ValueError(f"archive hash mismatch: {name}")
    if set(manifest["artifacts"]) != set(ARTIFACTS):
        raise ValueError("artifact inventory mismatch")
    if manifest["profile"] == "private_seeded" and private_seed_file is None:
        raise ValueError("private seed file required for reproducibility")
    if (private_seed_file is not None
            and hashlib.sha256(private_seed_file.read_bytes()).hexdigest() != manifest["seed_commitment"]):
        raise ValueError("private seed commitment mismatch")
    with tempfile.TemporaryDirectory(prefix="dg-sw-replay-") as temp:
        fresh = collect(native, Path(temp) / "regenerated", seed=manifest["seed"] or 0,
                        per_mode=manifest["per_mode"], private_seed_file=private_seed_file)
        if fresh != manifest:
            raise ValueError("regenerated manifest mismatch")
        for name in ARTIFACTS:
            if (folder / name).read_bytes() != (Path(temp) / "regenerated" / name).read_bytes():
                raise ValueError(f"archive content mismatch: {name}")
    return {"status": "verified", "tasks": manifest["artifacts"]["public_tasks.jsonl"]["rows"],
            "rollouts": manifest["artifacts"]["baseline_rollouts.jsonl"]["rows"]}


def grade_submissions(native: Native, folder: Path, submissions: Path, output: Path) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source"] != native.descriptor():
        raise ValueError("source pin mismatch")
    for name in ("public_tasks.jsonl", "private_sealed.jsonl"):
        if _sha_file(folder / name) != manifest["artifacts"][name]["sha256"]:
            raise ValueError(f"archive hash mismatch: {name}")
    def rows(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    public = {r["root"]: r for r in rows(folder / "public_tasks.jsonl")}
    private = {r["root"]: r for r in rows(folder / "private_sealed.jsonl")}
    if public.keys() != private.keys():
        raise ValueError("public/sealed root mismatch")
    results = []
    seen = set()
    for proposed in rows(submissions):
        root = proposed.get("root")
        if root not in public or root in seen:
            raise ValueError("unknown or duplicate submission root")
        seen.add(root)
        task = public[root]
        if task["mode"] == "hypergame":
            grade = score_hyper_actions(native, private[root], proposed.get("actions"))
        else:
            grade = score_builder(native, task, private[root], proposed.get("action"))
        results.append({"root": root, "split": task["split"], "mode": task["mode"], "grade": grade})
    stats = Counter((r["mode"], r["split"], bool(r["grade"]["accepted"])) for r in results)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, results)
    return {"graded": len(results), "missing": len(public) - len(seen),
            "counts": {f"{m}:{s}:{'accepted' if ok else 'failed'}": n for (m, s, ok), n in sorted(stats.items())},
            "output_sha256": _sha_file(output)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../GPTStoryworld"))
    parser.add_argument("--world", default=DEFAULT_WORLD)
    parser.add_argument("--expect-source-commit", default=None)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--private-seed-file", type=Path)
    parser.add_argument("--per-mode", type=int, default=6)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--out", type=Path)
    operation.add_argument("--replay", type=Path)
    operation.add_argument("--grade", type=Path, help="Path to collected gym archive")
    parser.add_argument("--submissions", type=Path)
    parser.add_argument("--graded-out", type=Path)
    args = parser.parse_args(argv)
    native = connect(args.source, args.world, args.expect_source_commit)
    if args.out:
        result = collect(native, args.out, args.seed, args.per_mode, args.private_seed_file)
    elif args.replay:
        result = replay_archive(native, args.replay, args.private_seed_file)
    else:
        if args.submissions is None or args.graded_out is None:
            parser.error("--grade requires --submissions and --graded-out")
        result = grade_submissions(native, args.grade, args.submissions, args.graded_out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
