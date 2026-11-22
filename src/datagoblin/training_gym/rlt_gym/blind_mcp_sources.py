"""Build source-disjoint behavior-only MCP roots with evaluator-owned gold.

The exported public task contains behavioral examples, never a target symbol,
module, or filename. This generator is a development fixture, not sealed data.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPLITS = ("train", "calibration", "ordinary_test", "ood_test")
N_FILES = 6
PER_FILE = 6
SEED = 20261122


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class PrivateCase:
    root_id: str
    split: str
    directory: Path
    public_task: dict[str, Any]
    source_sha256: str
    gold_uri: str | None
    candidate_uris: tuple[str, ...]
    target_examples: tuple[tuple[int, int], ...]


def _emit_function(name: str, factor: int, bias: int, *, ood: bool) -> str:
    expression = f"{bias} + (value * {factor})" if ood else f"(value * {factor}) + {bias}"
    return (
        f"def {name}(value: int) -> int:\n"
        f'    """Legacy scalar adapter used by the routing layer."""\n'
        f"    return {expression}\n"
    )


def _source_digest(directory: Path) -> str:
    rows = []
    for file in sorted(directory.rglob("*.py")):
        rows.append([file.relative_to(directory).as_posix(), _digest(file.read_bytes())])
    return _digest(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())


def create_case(directory: Path, *, split: str, index: int, seed: int = SEED) -> PrivateCase:
    if split not in SPLITS or type(index) is not int or index < 0:
        raise ValueError("invalid split/index")
    if directory.exists():
        raise FileExistsError("source roots must be create-only")
    directory.mkdir(parents=True)
    stable_seed = int.from_bytes(hashlib.sha256(f"{seed}/{split}/{index}".encode()).digest()[:8], "big")
    rng = random.Random(stable_seed)
    x0, x1 = rng.sample(range(1, 40), 2)
    baseline_factor = rng.randrange(3, 31)
    baseline_bias = rng.randrange(40, 500)
    has_target = index % 4 != 3
    decoy_factor = baseline_factor + rng.choice((-2, -1, 1, 2))
    decoy_bias = baseline_bias + x0 * (baseline_factor - decoy_factor)
    pairs = [(baseline_factor, baseline_bias), (decoy_factor, decoy_bias)]
    for _ in range(N_FILES * PER_FILE - 2):
        while True:
            pair = (rng.randrange(1, 44), rng.randrange(1, 700))
            if pair not in pairs:
                pairs.append(pair)
                break
    target = pairs[0] if has_target else (baseline_factor + 49, baseline_bias)
    rng.shuffle(pairs)
    public = {
        "kind": "behavior_only_symbol_retrieval",
        "issue": (
            "Find the production function that maps these two integer requests "
            f"to their returned payloads: {x0} -> {target[0] * x0 + target[1]} "
            f"and {x1} -> {target[0] * x1 + target[1]}. "
            "Return the symbol URI, or abstain if the implementation is absent."
        ),
        "examples": [[x0, target[0] * x0 + target[1]], [x1, target[0] * x1 + target[1]]],
        "repository_kind": "fresh_generated_python_worktree",
    }
    uris = []
    by_pair = {}
    for file_index in range(N_FILES):
        filename = f"src/module_{rng.getrandbits(32):08x}.py"
        content = ['"""Runtime scalar handlers and compatibility adapters."""\n']
        for function_index in range(PER_FILE):
            factor, bias = pairs[file_index * PER_FILE + function_index]
            sym = f"op_{rng.getrandbits(48):012x}"
            uri = f"{filename}::{sym}"
            by_pair[(factor, bias)] = uri
            uris.append(uri)
            content.append(_emit_function(sym, factor, bias, ood=split == "ood_test"))
        file = directory / filename
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("\n".join(content), encoding="utf-8")
        ast.parse(file.read_text(encoding="utf-8"))
    gold_uri = by_pair.get(target) if has_target else None
    if has_target != (gold_uri is not None):
        raise AssertionError("private task registration failed")
    public_serialized = json.dumps(public, sort_keys=True)
    if any(sym in public_serialized for sym in ("op_", "module_", "gold", "hint_module", "target_uri")):
        raise AssertionError("target location leaked into public packet")
    return PrivateCase(
        root_id=f"blind-mcp/{split}/{_digest(str(stable_seed).encode())[:16]}",
        split=split, directory=directory, public_task=public,
        source_sha256=_source_digest(directory), gold_uri=gold_uri,
        candidate_uris=tuple(uris), target_examples=tuple(tuple(x) for x in public["examples"]),
    )


def judge(case: PrivateCase, proposed_uri: str | None) -> dict[str, Any]:
    """Check frozen source bytes, then compare proposal with evaluator-owned gold."""
    if _source_digest(case.directory) != case.source_sha256:
        raise ValueError("repository snapshot mutated")
    if proposed_uri is not None:
        if not isinstance(proposed_uri, str) or proposed_uri not in case.candidate_uris:
            raise ValueError("unregistered symbol or invalid abstention")
        file_part, symbol_part = proposed_uri.split("::", 1)
        file_path = (case.directory / file_part).resolve()
        if not file_path.is_relative_to(case.directory.resolve()):
            raise ValueError("symbol path escapes frozen source")
        definitions = [
            n for n in ast.parse(file_path.read_text(encoding="utf-8")).body
            if isinstance(n, ast.FunctionDef) and n.name == symbol_part
        ]
        if len(definitions) != 1:
            raise ValueError("symbol is absent from source")
    hit = proposed_uri == case.gold_uri
    return {"correct": hit, "found": case.gold_uri is not None and hit,
            "correct_abstention": case.gold_uri is None and proposed_uri is None}


def build_source_roots(base: Path, *, per_split: int = 8) -> list[PrivateCase]:
    if not 4 <= per_split <= 32 or per_split % 4:
        raise ValueError("per_split must be a multiple of four in 4..32")
    cases = []
    for split in SPLITS:
        for i in range(per_split):
            cases.append(create_case(base / split / f"case_{i:03d}", split=split, index=i))
    if len({c.source_sha256 for c in cases}) != len(cases):
        raise ValueError("source roots have overlapping snapshots")
    if len({c.root_id for c in cases}) != len(cases):
        raise ValueError("root IDs overlap")
    return cases


def public_record(case: PrivateCase) -> dict[str, Any]:
    return {"root_id": case.root_id, "split": case.split,
            "public_task": case.public_task, "snapshot_sha256": case.source_sha256}


def audit_source_roots(cases: list[PrivateCase]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in SPLITS:
        relevant = [c for c in cases if c.split == split]
        positives = sum(c.gold_uri is not None for c in relevant)
        negatives = len(relevant) - positives
        leaked = sum(bool(re.search(r"op_[0-9a-f]{12}|module_[0-9a-f]{8}",
                                    json.dumps(c.public_task))) for c in relevant)
        result[split] = {"roots": len(relevant), "answerable": positives,
                         "unanswerable": negatives, "public_symbol_path_leaks": leaked}
    return result
