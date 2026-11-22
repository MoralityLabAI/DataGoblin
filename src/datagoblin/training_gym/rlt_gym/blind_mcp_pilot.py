"""TRAIN-only learned stopping over actual MCP-03 reads of generated source roots.

The policy is evaluated on fully collected source traces. Its prefix cost is
counterfactual; live online latency requires a separate experiment.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .blind_mcp_sources import SPLITS, PrivateCase, audit_source_roots, build_source_roots, judge


def _tool(env: Any, name: str, args: dict[str, Any]) -> dict[str, Any]:
    response = env.call(name, args)
    if response.get("isError") is not False or not isinstance(response.get("structuredContent"), dict):
        raise ValueError(f"MCP tool error on {name}: {response}")
    return response["structuredContent"]


def _safe_arithmetic(node: ast.AST, value: int) -> int:
    if isinstance(node, ast.Constant) and type(node.value) is int and 0 <= node.value < 10000:
        return node.value
    if isinstance(node, ast.Name) and node.id == "value":
        return value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
        left, right = _safe_arithmetic(node.left, value), _safe_arithmetic(node.right, value)
        return left + right if isinstance(node.op, ast.Add) else left * right
    raise ValueError("unsupported or unsafe expression in synthetic source")


def _source_predictions(source: str, inputs: tuple[int, int]) -> tuple[int, int]:
    tree = ast.parse(source)
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(funcs) != 1:
        raise ValueError("symbol read did not return one function")
    final = funcs[0].body[-1]
    if not isinstance(final, ast.Return):
        raise ValueError("no return in source snippet")
    return (_safe_arithmetic(final.value, inputs[0]), _safe_arithmetic(final.value, inputs[1]))


def _features(source: str, examples: tuple[tuple[int, int], ...]) -> np.ndarray:
    expected = tuple(y for _, y in examples)
    predicted = _source_predictions(source, (examples[0][0], examples[1][0]))
    error = tuple(abs(a - b) for a, b in zip(predicted, expected, strict=True))
    return np.array([1.0, *[math.log1p(v) for v in error],
                     float(error[0] == 0), float(error[1] == 0),
                     float(error[0] == 0 and error[1] == 0)], dtype=np.float64)


@dataclass
class Trace:
    case: PrivateCase
    symbols: list[str]
    features: np.ndarray
    actual_calls: int


def collect(case: PrivateCase) -> Trace:
    try:
        from MCP.mcp03.env import CodeEnv
        from MCP.mcp03.repo import snapshot_from_dir
    except ImportError as e:
        raise RuntimeError("requires pinned HybridTRMLDT MCP-03 on PYTHONPATH") from e
    snap = snapshot_from_dir(case.directory)
    # Even the upstream environment receives an empty oracle. Only the
    # evaluator's independent judge can inspect case.gold_uri.
    env = CodeEnv(snap, case.public_task, {"files": [], "symbols": []}, arm="hybrid", page_limit=16)
    _tool(env, "index_children", {"node_id": "d:", "limit": 16})
    _tool(env, "code_search", {"query": case.public_task["issue"], "limit": 8})
    symbols = []
    for file_path in sorted(snap.sources):
        outline = _tool(env, "read_outline", {"path": file_path})
        symbols.extend(row["uri"] for row in outline["symbols"] if row.get("type") == "function")
    symbols = sorted(set(symbols))
    if set(symbols) != set(case.candidate_uris):
        raise ValueError("MCP discovered a different symbol set from frozen source")
    observations = []
    for uri in symbols:
        record = _tool(env, "read_symbol", {"uri": uri})
        if record["uri"] != uri or record.get("truncated"):
            raise ValueError("MCP symbol identity/length changed")
        observations.append(_features(record["source"], case.target_examples))
    return Trace(case=case, symbols=symbols,
                 features=np.stack(observations), actual_calls=env.stats["tool_calls"])


def fit_logistic(traces: list[Trace], *, steps: int = 1200, lr: float = 0.05) -> np.ndarray:
    if any(t.case.split != "train" for t in traces):
        raise ValueError("training model received non-TRAIN roots")
    x = np.concatenate([t.features for t in traces])
    y = np.concatenate([np.array([float(uri == t.case.gold_uri) for uri in t.symbols]) for t in traces])
    if int(y.sum()) == 0 or int(y.sum()) == len(y):
        raise ValueError("training data has no positive and negative symbols")
    w = np.zeros(x.shape[1], dtype=np.float64)
    weight = np.where(y == 1, min(60.0, float(len(y) / y.sum())), 1.0)
    for _ in range(steps):
        logits = np.clip(x @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-logits))
        gradient = x.T @ ((p - y) * weight) / len(y)
        gradient[1:] += 0.025 * w[1:]
        w -= lr * gradient
    if not np.isfinite(w).all():
        raise FloatingPointError("fit diverged")
    return w


def _choose(trace: Trace, probabilities: np.ndarray, threshold: float) -> tuple[str | None, int]:
    for i, (uri, probability) in enumerate(zip(trace.symbols, probabilities, strict=True), 1):
        if probability >= threshold:
            return uri, i
    return None, len(trace.symbols)


def _score_policy(traces: list[Trace], weights: np.ndarray, threshold: float) -> dict[str, Any]:
    correct = 0
    reads = 0
    answerable_correct = 0
    abstention_correct = 0
    for trace in traces:
        ps = 1 / (1 + np.exp(-np.clip(trace.features @ weights, -30, 30)))
        proposal, spent = _choose(trace, ps, threshold)
        scored = judge(trace.case, proposal)
        correct += scored["correct"]
        answerable_correct += scored["found"]
        abstention_correct += scored["correct_abstention"]
        reads += spent
    return {"roots": len(traces), "correct": int(correct), "found": int(answerable_correct),
            "correct_abstentions": int(abstention_correct), "counterfactual_symbol_reads": reads,
            "counterfactual_avg_symbol_reads": round(reads / len(traces), 4)}


def _rule_baseline(traces: list[Trace]) -> dict[str, Any]:
    correct = 0
    reads = 0
    for t in traces:
        match = [int(f[-1] == 1.0) for f in t.features]
        ix = match.index(1) if 1 in match else None
        proposal = t.symbols[ix] if ix is not None else None
        reads += ix + 1 if ix is not None else len(t.symbols)
        correct += judge(t.case, proposal)["correct"]
    return {"correct": correct, "roots": len(traces), "counterfactual_symbol_reads": reads}


def run(per_split: int = 8) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="dg-blind-mcp-") as tmp:
        cases = build_source_roots(Path(tmp), per_split=per_split)
        traces = [collect(c) for c in cases]
        by_split = {s: [t for t in traces if t.case.split == s] for s in SPLITS}
        weights = fit_logistic(by_split["train"])
        # Calibrate on CAL only, then freeze before touching held-out labels.
        scored = []
        for threshold in np.linspace(0.05, 0.95, 19):
            result = _score_policy(by_split["calibration"], weights, float(threshold))
            utility = result["correct"] - 0.015 * result["counterfactual_symbol_reads"]
            scored.append((utility, -float(threshold), float(threshold)))
        frozen_threshold = max(scored)[2]
        result: dict[str, Any] = {
            "schema": "dg_blind_mcp_learned_stop_development_v1",
            "source_custody": audit_source_roots(cases),
            "training_roots": len(by_split["train"]),
            "calibration_roots": len(by_split["calibration"]),
            "frozen_threshold": round(frozen_threshold, 5),
            "model_weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
            "model_features": ["bias", "log_error_first", "log_error_second",
                               "first_exact", "second_exact", "both_exact"],
            "native_tool_calls_full_trace": sum(t.actual_calls for t in traces),
            "slices": {},
            "limitations": [
                "generated_affine_code_behavior_tasks",
                "symbol_read_order_is_sorted_public_symbol_uri",
                "policy_cost_from_exhaustively_collected_native_tool_traces_not_online_timed",
                "feature_engineer_evaluates_small_trusted_ast_grammar",
                "tool_server_has_empty_private_oracle",
                "development_heldout_roots_not_sealed_confirmation",
                "no_pretrained_agent_or_SWE_bench_claim",
            ],
        }
        for split in SPLITS:
            result["slices"][split] = {
                "learned": _score_policy(by_split[split], weights, frozen_threshold),
                "exact_behavior_rule": _rule_baseline(by_split[split]),
            }
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-split", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("create-only report")
    result = run(args.per_split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
