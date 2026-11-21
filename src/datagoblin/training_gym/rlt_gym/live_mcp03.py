"""Live MCP-03 repository-tool evaluation with independent gold-symbol state judge.

Requires a separately checked-out, commit-pinned HybridTRMLDT MCP-03 package.
A fresh CodeEnv is created for *each* candidate, tool calls read actual source
files/symbol tables, and the private judge evaluates packet state effects.
Synthetic new repositories make this an instrumentation positive control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .external_checked_exit import SCHEMA, digest, replay_corpus
from .external_native_adapters import MCPStateChecker

HYBRID_SHA = "7e90a2d446dbb2f994a33632350e7219bdee1ba4"
SEED = 20261122
SPLITS = ("train", "calibration", "ordinary_test", "ood_test")
MODEL_SHA = hashlib.sha256(b"static_mcp03_tool_menu_v1_untrained").hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_digest(snapshot: Any) -> str:
    return digest({
        "repository": snapshot.repo,
        "files": sorted(snapshot.sources.items()),
        "symbols": sorted(s.uri for s in snapshot.all_symbols()),
    })


def _create_root(path: Path, index: int) -> tuple[dict[str, Any], str]:
    """Generate independent executable Python source and a private edit target."""
    suffix = f"{index:08x}"
    path.mkdir(parents=True)
    for folder in ("src", "tests"):
        (path / folder).mkdir()
    selected = ("normalize_payload", "resolve_endpoint", "repair_index")[index % 3]
    filename = "src/handlers.py"
    target = f"{selected}_{suffix}"
    (path / filename).write_text(
        f'''"""Runtime handlers for record decoding and identifier routing."""


def {target}(raw: str) -> str:
    """Canonical payload normalization for client route and repair request."""
    return raw.strip().lower()


def fallback_codec_{suffix}(raw: str) -> str:
    """Backward compatible ASCII payload representation."""
    return raw.upper()


def index_hint_{suffix}(raw: str) -> int:
    """Legacy cache sizing helper."""
    return len(raw)
''', encoding="utf-8",
    )
    (path / "src" / "logging.py").write_text(
        f'''"""Logging and telemetry, distinct from handler semantics."""


def log_event_{suffix}(message: str) -> str:
    """Format timestamped debug information."""
    return "log:" + message
''', encoding="utf-8",
    )
    (path / "tests" / "test_handlers.py").write_text(
        f'''"""Test scaffolding and a distractor function."""


def test_handler_{suffix}() -> None:
    """Check canonical normalization in a unit test."""
    assert "{suffix}".strip() == "{suffix}"
''', encoding="utf-8",
    )
    public = {
        "kind": "code_repository_symbol_retrieval",
        "issue": f"Locate the {selected} canonical payload handler for request {suffix}.",
        "expected_symbol_name": target,
        "hint_module": filename,
        "repo_generation_id": suffix,
    }
    return public, f"{filename}::{target}"


def _tool(env: Any, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    value = env.call(name, arguments)
    if value.get("isError") is not False or not isinstance(value.get("structuredContent"), dict):
        raise ValueError(f"actual MCP tool {name} failed: {value}")
    return value["structuredContent"]


def run(per_split: int, out: Path) -> dict[str, Any]:
    try:
        import MCP.mcp03.env as upstream_env
        import MCP.mcp03.repo as upstream_repo
        from MCP.mcp03.env import CodeEnv
        from MCP.mcp03.repo import snapshot_from_dir
    except ImportError as error:
        raise RuntimeError("pinned HybridTRMLDT MCP-03 source is required on PYTHONPATH") from error
    if out.exists():
        raise FileExistsError("MCP output must be fresh")
    if type(per_split) is not int or not 1 <= per_split <= 32:
        raise ValueError("per_split must be 1..32")
    env_source_sha = _sha(Path(upstream_env.__file__))
    repo_source_sha = _sha(Path(upstream_repo.__file__))
    driver_sha = _sha(Path(__file__))
    verifier_sha = digest({
        "hybrid_commit": HYBRID_SHA, "environment_source": env_source_sha,
        "snapshot_source": repo_source_sha, "driver_source": driver_sha,
    })
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dg-mcp03-live-") as tmp:
        base = Path(tmp)
        registry: dict[str, dict[str, Any]] = {}
        tasks: list[dict[str, Any]] = []
        for split_num, split in enumerate(SPLITS):
            for index in range(per_split):
                global_index = SEED + split_num * 1000 + index
                root_dir = base / f"repo_{split_num}_{index}"
                task, gold_symbol = _create_root(root_dir, global_index)
                snap = snapshot_from_dir(root_dir)
                snapshot_sha = _snapshot_digest(snap)
                gold = {"files": ["src/handlers.py"], "symbols": [gold_symbol]}
                key = digest(task)
                registry[key] = {
                    "path": root_dir, "snapshot_sha": snapshot_sha,
                    "menu": {
                        0: "wrong_distractor_symbol",
                        1: "semantic_code_search",
                        2: "outline_target_symbol",
                    },
                    "gold": gold, "public_task": task,
                    "wrong_symbol": f"src/logging.py::log_event_{global_index:08x}",
                }
                tasks.append({"task": task, "split": split, "key": key})
        source_revision = digest({
            "registry": sorted((k, v["snapshot_sha"]) for k, v in registry.items()),
            "driver_sha": driver_sha, "hybrid_commit": HYBRID_SHA, "suite_seed": SEED,
        })
        calls = Counter()

        def execute_fresh(key: str, action_id: int) -> dict[str, Any]:
            spec = registry[key]
            snapshot = snapshot_from_dir(spec["path"])
            computed_sha = _snapshot_digest(snapshot)
            if computed_sha != spec["snapshot_sha"]:
                raise ValueError("working-tree source changed after registration")
            snapshot.commit = computed_sha
            env = CodeEnv(snapshot, spec["public_task"], spec["gold"], arm="layered_ldt", page_limit=8)
            listing = _tool(env, "index_children", {"node_id": "d:", "limit": 16})
            # The actual LDT environment owns the safe elimination check.
            children = {row["node_id"] for row in listing["children"]}
            if "d:tests" in children:
                excluded = _tool(env, "candidate_refine", {
                    "node_id": "d:", "exclude": ["d:tests"], "predicate": "test_path",
                })
                if "d:tests" not in excluded["excluded"]:
                    raise ValueError("independently verified LDT exclusion did not apply")
            if action_id == 0:
                symbol = spec["wrong_symbol"]
            elif action_id == 1:
                hits = _tool(env, "code_search", {
                    "query": spec["public_task"]["issue"], "limit": 8,
                })
                symbols = [row for row in hits["results"] if row["kind"] == "symbol"]
                symbol = symbols[0]["uri"] if symbols else spec["wrong_symbol"]
            elif action_id == 2:
                outline = _tool(env, "read_outline", {
                    "path": spec["public_task"]["hint_module"],
                })
                selected = [row for row in outline["symbols"]
                            if row["uri"].endswith("::" + spec["public_task"]["expected_symbol_name"])]
                if len(selected) != 1:
                    raise ValueError("public outline does not expose unique requested symbol")
                symbol = selected[0]["uri"]
            else:
                raise ValueError("unregistered MCP action")
            read = _tool(env, "read_symbol", {"uri": symbol})
            if read["uri"] != symbol or not read["source"]:
                raise ValueError("MCP returned an empty/mismatched symbol")
            packet = {
                "map": [read["path"]], "snippets": [{"uri": read["uri"]}],
                "ranked_files": [read["path"]],
            }
            calls["actual_tool_calls"] += env.stats["tool_calls"]
            calls["ldt_checked"] += env.stats["ldt_checks"]
            calls["source_reads"] += env.stats["read_calls"]
            calls["environment_resets"] += 1
            return {
                "snapshot_sha": computed_sha, "action_id": action_id,
                "effect": {"packet": packet, "source_sha256": digest(read["source"]),
                           "read_uri": symbol, "tool_calls": env.stats["tool_calls"],
                           "ldt_excluded": env.stats["ldt_excluded"]},
            }

        def independent_judge(key: str, result: dict[str, Any]) -> tuple[bool, str]:
            spec = registry[key]
            snap = snapshot_from_dir(spec["path"])
            snapshot_sha = _snapshot_digest(snap)
            if snapshot_sha != spec["snapshot_sha"]:
                raise ValueError("judge observed different root snapshot")
            snap.commit = snapshot_sha
            env = CodeEnv(snap, spec["public_task"], spec["gold"], arm="hybrid")
            effect = result.get("effect", {})
            packet = effect.get("packet")
            if not isinstance(packet, dict):
                raise ValueError("MCP effect packet missing")
            measure = env.grade(packet)
            passed = bool(measure["all_files"] and measure["all_symbols"])
            calls["independent_private_judgments"] += 1
            return passed, "full_gold_symbol_recall" if passed else "gold_symbol_missing"

        checker = MCPStateChecker(
            cases=registry, verifier_sha=verifier_sha,
            execute_fresh=execute_fresh, judge=independent_judge,
        )
        episodes = []
        for record in tasks:
            task = record["task"]
            split = record["split"]
            ep = {
                "schema_version": SCHEMA, "kind": "mcp", "split": split,
                "root_id": f"mcp:{SEED}:{split}:{task['repo_generation_id']}",
                "source_revision": source_revision, "model_sha256": MODEL_SHA,
                "verifier_sha256": verifier_sha, "source_exposure": "fresh",
                "evidence_kind": "native_operator_verified",
                "public_task": task, "task_digest": digest(task),
                "planned_visits": len(registry[record["key"]]["menu"]),
                "verified_stop": False, "attempts": [],
            }
            for i in range(ep["planned_visits"]):
                candidate = {"id": i}
                ep["attempts"].append({
                    "visit": i, "candidate": candidate,
                    "verification": checker(task, candidate),
                })
            episodes.append(ep)
        first_run_resets = calls["environment_resets"]
        approved = frozenset(x["root_id"] for x in episodes if x["split"] == "train")
        verified = replay_corpus(
            episodes, checkers={"mcp": checker}, trusted_shas={"mcp": verifier_sha},
            trusted_source_revisions={"mcp": source_revision},
            approved_train_roots=approved,
        )
        if calls["environment_resets"] != first_run_resets * 2:
            raise ValueError("full MCP external tool replay was omitted")
    by_split = []
    for split in SPLITS:
        cohort = [x for x in (*verified["train"], *verified["evaluation_only"])
                  if x["split"] == split]
        by_split.append({
            "split": split, "roots": len(cohort),
            "first_valid": sum(row["first_passed"] for row in cohort),
            "future_rescued": sum(row["future_rescue"] for row in cohort),
            "ever_verified": sum(row["first_verified_visit"] is not None for row in cohort),
            "source_tool_calls_per_round": calls["actual_tool_calls"] // (len(episodes) * 2),
        })
    out.mkdir(parents=True)
    (out / "source_episodes.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in episodes), encoding="utf-8",
    )
    (out / "verified_rows.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n"
                for x in (*verified["train"], *verified["evaluation_only"])), encoding="utf-8",
    )
    report = {
        "schema_version": "rlt_gym.live_mcp03_retrieval.v1",
        "evidence_class": "real_mcp03_tools_and_independent_gold_symbol_judge",
        "hybridtrmldt_commit_sha": HYBRID_SHA,
        "upstream_env_sha256": env_source_sha,
        "upstream_repo_sha256": repo_source_sha,
        "driver_sha256": driver_sha,
        "verifier_sha256": verifier_sha, "source_revision_sha256": source_revision,
        "model_sha256": MODEL_SHA,
        "root_count": len(episodes), "by_split": by_split,
        "native_tool_calls": calls["actual_tool_calls"],
        "independent_private_judgments": calls["independent_private_judgments"],
        "source_reads": calls["source_reads"], "ldt_checks": calls["ldt_checked"],
        "fresh_environment_resets": calls["environment_resets"],
        "training_eligible_rows": len(verified["train"]),
        "evaluation_only_rows": len(verified["evaluation_only"]),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "files_sha256": {
            name: _sha(out / name) for name in ("source_episodes.jsonl", "verified_rows.jsonl")
        },
        "claim_boundary": (
            "Real MCP-03 code_search/index_children/candidate_refine/read_symbol calls "
            "against new generated source trees, with independent hidden gold-file and "
            "symbol recall grader. The target module/name are public hints in a "
            "contrived untrained menu; not a learned tool-use agent or SWE-bench pass."
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-split", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.per_split, args.out)
    print(json.dumps({
        "roots": report["root_count"],
        "by_split": report["by_split"],
        "tool_calls": report["native_tool_calls"],
        "replay_resets": report["fresh_environment_resets"],
        "ldt_checks": report["ldt_checks"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
