"""Real MCP-03 stdio tool calls on fresh, frozen repository snapshots.

Run against an exact SHA-pinned HybridTRMLDT source checkout. The external
MCP server is restarted for every candidate and its returned source is
checked against repository bytes by a separate evaluator. This establishes
native tool-result coverage, not learned agent planning or code edit success.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .external_checked_exit import digest, replay_corpus
from .external_native_adapters import MCPStateChecker

EXPECTED_HYBRID_COMMIT = "7e90a2d446dbb2f994a33632350e7219bdee1ba4"
SPLITS = ("train", "calibration", "ordinary_test", "ood_test")
SPLIT_SEEDS = {"train": 202611221, "calibration": 202611222,
               "ordinary_test": 202611223, "ood_test": 202611224}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tree_sha(root: Path) -> str:
    files = sorted(root.rglob("*.py"))
    if not files or len(files) > 32:
        raise ValueError("invalid frozen MCP source tree")
    return digest([
        {"path": p.relative_to(root).as_posix(), "sha256": _sha(p.read_bytes())}
        for p in files
    ])


def _source_for_symbol(root: Path, uri: str) -> str:
    from_name, qualname = uri.split("::", 1)
    file = (root / from_name).resolve()
    if not file.is_relative_to(root.resolve()) or not file.is_file():
        raise ValueError("symbol escapes frozen source")
    source = file.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == qualname
    ]
    if len(nodes) != 1:
        raise ValueError("source symbol not found uniquely")
    n = nodes[0]
    return "\n".join(source.splitlines()[n.lineno - 1:n.end_lineno])


def _write_world(root: Path, *, split: str, index: int) -> dict[str, Any]:
    seed = SPLIT_SEEDS[split] + 101 * index
    rng = random.Random(seed)
    suffix = f"{rng.randrange(16**8):08x}"
    additive = rng.randint(7, 1700)
    paths = ("src/transform.py", "src/decoy.py")
    script = (
        f"def transform_{suffix}(value: int) -> int:\n"
        f"    \"\"\"A deterministic source transform for {suffix}.\"\"\"\n"
        f"    return value + {additive}\n"
    )
    decoy = (
        f"def decoy_{suffix}(value: int) -> int:\n"
        f"    \"\"\"A differently named function for {suffix}.\"\"\"\n"
        f"    return value - {additive}\n"
    )
    for name, body in zip(paths, (script, decoy), strict=True):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    good = f"{paths[0]}::transform_{suffix}"
    bad = f"{paths[1]}::decoy_{suffix}"
    return {
        "root_id": f"mcp03-{split}-{index}-{suffix}",
        "split": split,
        "seed": seed,
        "repository_sha": _tree_sha(root),
        "public_task": {
            "issue": f"Find the function in this repository that adds {additive} to an integer.",
            "snapshot_kind": "frozen_python_worktree",
            "snapshot_sha": _tree_sha(root),
        },
        "menu": {1: bad, 2: good},
        "target_uri": good,
        "target_source_sha256": _sha(_source_for_symbol(root, good).encode()),
    }


def evaluate(*, out: Path, hybrid_root: Path, roots_per_split: int = 3) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("output exists")
    if type(roots_per_split) is not int or not 1 <= roots_per_split <= 8:
        raise ValueError("roots_per_split must be 1..8")
    hybrid_root = hybrid_root.resolve()
    actual_head = subprocess.check_output(
        ["git", "-C", str(hybrid_root), "rev-parse", "HEAD"], text=True, timeout=10,
    ).strip()
    if actual_head != EXPECTED_HYBRID_COMMIT:
        raise ValueError("HybridTRMLDT source revision differs from frozen registration")
    if not (hybrid_root / "MCP/mcp03/goblin_server.py").is_file():
        raise ValueError("MCP-03 server missing")
    # Import the public MCP transport from this exact registered code revision.
    sys.path.insert(0, str(hybrid_root))
    from MCP.transport import StdioClient

    source = {
        "hybrid_commit": actual_head,
        "mcp_server_sha256": _sha((hybrid_root / "MCP/mcp03/goblin_server.py").read_bytes()),
        "mcp_transport_sha256": _sha((hybrid_root / "MCP/transport.py").read_bytes()),
        "bridge_sha256": _sha(Path(__file__).read_bytes()),
    }
    verifier_sha = digest(source)
    root_path = out / "worlds"
    root_path.mkdir(parents=True)
    case_reg: dict[str, dict[str, Any]] = {}
    episodes = []
    native_receipts = []
    for split in SPLITS:
        for i in range(roots_per_split):
            world_root = root_path / split / f"root-{i}"
            world_root.mkdir(parents=True)
            item = _write_world(world_root, split=split, index=i)
            task = item["public_task"]
            task_digest = digest(task)
            if task_digest in case_reg:
                raise ValueError("duplicate source task")
            case_reg[task_digest] = {
                "menu": item["menu"],
                "snapshot_sha": item["repository_sha"],
                "repo_path": str(world_root.resolve()),
                "target_uri": item["target_uri"],
                "target_source_sha256": item["target_source_sha256"],
            }
            sequence = (2, 1) if i % 4 == 0 else (1, 2)
            episodes.append({
                "schema_version": "rlt_gym.external_checked_episode.v1",
                "kind": "mcp",
                "root_id": item["root_id"], "split": split,
                "source_exposure": "fresh",
                "evidence_kind": "native_operator_verified",
                "source_revision": verifier_sha,
                "model_sha256": _sha(b"deterministic_mcp_menu_control_v1"),
                "verifier_sha256": verifier_sha,
                "public_task": task, "task_digest": task_digest,
                "planned_visits": len(sequence), "verified_stop": False,
                "attempts": [{"visit": j, "candidate": {"id": choice}} for j, choice in enumerate(sequence)],
            })
    counter = {"tool_calls": 0, "handshakes": 0}
    def execute_fresh(task_digest: str, choice: int) -> dict[str, Any]:
        spec = case_reg[task_digest]
        root = Path(spec["repo_path"])
        if _tree_sha(root) != spec["snapshot_sha"]:
            raise ValueError("source changed after snapshot registration")
        uri = spec["menu"][choice]
        with StdioClient(
            command=[sys.executable, "-u", "-m", "MCP.mcp03.goblin_server"],
            cwd=str(hybrid_root), timeout=12,
        ) as client:
            init = client.initialize()
            tools = client.tools()
            if init["protocolVersion"] != "2025-06-18" or not {
                t["name"] for t in tools
            } >= {"read_symbol", "compact_context"}:
                raise ValueError("MCP-03 stdio handshake/tools catalog changed")
            counter["handshakes"] += 1
            result = client.call("read_symbol", {
                "repo_path": str(root), "uri": uri,
            })
            counter["tool_calls"] += 1
            tool_stats = dict(client.last_exchange or {})
        effect = dict(result.get("structuredContent", {}))
        effect["mcp_is_error"] = result.get("isError", False)
        receipt = {"root_digest": task_digest, "candidate_id": choice,
                   "tool": "read_symbol", "mcp_exchange": tool_stats,
                   "response_sha256": digest(effect)}
        native_receipts.append(receipt)
        if _tree_sha(root) != spec["snapshot_sha"]:
            raise ValueError("source mutated during MCP tool execution")
        return {
            "action_id": choice,
            "snapshot_sha": spec["snapshot_sha"],
            "effect": effect,
        }

    def judge(task_digest: str, result: dict[str, Any]) -> tuple[bool, str]:
        spec = case_reg[task_digest]
        effect = result["effect"]
        if effect.get("mcp_is_error") or not isinstance(effect.get("source"), str):
            return False, "symbol_not_read"
        uri = spec["menu"][result["action_id"]]
        if effect.get("uri") != uri:
            raise ValueError("MCP result has wrong URI")
        expected = _source_for_symbol(Path(spec["repo_path"]), uri)
        if effect["source"] != expected:
            raise ValueError("MCP result differs from independently read source lines")
        passed = uri == spec["target_uri"] and _sha(effect["source"].encode()) == spec["target_source_sha256"]
        return passed, "target_symbol_covered" if passed else "non_target_symbol_read"

    checker = MCPStateChecker(
        cases=case_reg, verifier_sha=verifier_sha,
        execute_fresh=execute_fresh, judge=judge,
    )
    for episode in episodes:
        for step in episode["attempts"]:
            step["verification"] = checker(episode["public_task"], step["candidate"])
    approved = frozenset(e["root_id"] for e in episodes if e["split"] == "train")
    corpus = replay_corpus(
        episodes, checkers={"mcp": checker}, trusted_shas={"mcp": verifier_sha},
        trusted_source_revisions={"mcp": verifier_sha},
        approved_train_roots=approved,
    )
    count = len(episodes)
    if counter["tool_calls"] != 4 * count or corpus["verified_candidate_checks"] != 2 * count:
        raise ValueError("missing native MCP tool calls / independent replay")
    if len(corpus["train"]) != roots_per_split or len(corpus["evaluation_only"]) != 3 * roots_per_split:
        raise ValueError("training split firewall failed")
    summary = {
        "schema_version": "rlt_gym.live_mcp03.v1",
        "evidence_class": "real_mcp03_stdio_read_symbol_verified_fresh_worktrees",
        "producer": source,
        "roots": count, "training_roots": len(corpus["train"]),
        "evaluation_only_roots": len(corpus["evaluation_only"]),
        "first_covered": sum(x["first_passed"] for x in (*corpus["train"], *corpus["evaluation_only"])),
        "rescues": sum(x["future_rescue"] for x in (*corpus["train"], *corpus["evaluation_only"])),
        "live_stdio_tool_calls_including_replay": counter["tool_calls"],
        "stdio_handshakes": counter["handshakes"],
        "original_candidate_checks": 2 * count,
        "independent_replay_candidate_checks": 2 * count,
        "limitations": [
            "worlds_are_synthetic_new_python_repositories",
            "algorithmic_menu_control_not_learned_mcp_router",
            "native_tool_effect_is_symbol_read_coverage_not_code_edit",
            "three_nontrain_splits_preserved",
            "server_uses_pinned_subset_of_mcp_stdio_protocol",
        ],
    }
    for name, value in (("episodes.json", episodes), ("corpus.json", corpus),
                        ("mcp_receipts.json", native_receipts), ("report.json", summary)):
        (out / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["files_sha256"] = {n: _sha((out / n).read_bytes()) for n in (
        "episodes.json", "corpus.json", "mcp_receipts.json",
    )}
    (out / "report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hybrid-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--roots-per-split", type=int, default=3)
    args = p.parse_args()
    result = evaluate(out=args.out, hybrid_root=args.hybrid_root, roots_per_split=args.roots_per_split)
    print(json.dumps({k: result[k] for k in (
        "roots", "first_covered", "rescues", "live_stdio_tool_calls_including_replay",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
