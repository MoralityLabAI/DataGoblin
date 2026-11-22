"""Blind behavioral MCP roots and TRAIN-only stopping policy contracts."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from datagoblin.training_gym.rlt_gym.blind_mcp_pilot import (
    Trace,
    _features,
    _rule_baseline,
    _score_policy,
    _source_predictions,
    fit_logistic,
)
from datagoblin.training_gym.rlt_gym.blind_mcp_sources import (
    SPLITS,
    audit_source_roots,
    build_source_roots,
    create_case,
    judge,
    public_record,
)


def _local_trace(case):
    observations = []
    for uri in sorted(case.candidate_uris):
        filename, name = uri.split("::")
        source = (case.directory / filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        snippet = "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])
        observations.append(_features(snippet, case.target_examples))
    return Trace(case=case, symbols=sorted(case.candidate_uris),
                 features=np.stack(observations), actual_calls=0)


def test_source_disjoint_and_private_custody(tmp_path: Path):
    cases = build_source_roots(tmp_path, per_split=8)
    assert len(cases) == 32
    audits = audit_source_roots(cases)
    assert all(audits[split]["answerable"] == 6 for split in SPLITS)
    assert all(audits[split]["unanswerable"] == 2 for split in SPLITS)
    assert all(audits[split]["public_symbol_path_leaks"] == 0 for split in SPLITS)
    assert all("gold_uri" not in public_record(case) for case in cases)
    assert len({case.source_sha256 for case in cases}) == 32
    for case in cases:
        assert case.gold_uri is None or case.gold_uri in case.candidate_uris
        assert judge(case, case.gold_uri)["correct"]
        if case.gold_uri:
            assert not judge(case, None)["correct"]
        else:
            assert not judge(case, case.candidate_uris[0])["correct"]


def test_identical_root_replays_and_mutation_detection(tmp_path: Path):
    a = create_case(tmp_path / "a", split="train", index=0)
    b = create_case(tmp_path / "b", split="train", index=0)
    assert a.public_task == b.public_task
    assert a.source_sha256 == b.source_sha256
    assert a.gold_uri == b.gold_uri
    selected = next(a.directory.rglob("*.py"))
    selected.write_text(selected.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mutated"):
        judge(a, a.gold_uri)


def test_reject_unregistered_symbol_and_existing_root(tmp_path: Path):
    case = create_case(tmp_path / "root", split="ood_test", index=3)
    with pytest.raises(ValueError, match="unregistered"):
        judge(case, "src/fake.py::fake")
    with pytest.raises(FileExistsError):
        create_case(tmp_path / "root", split="ood_test", index=3)


def test_grammar_and_ood_normalization():
    a = "def fn(value: int) -> int:\n    return (value * 3) + 8\n"
    b = "def fn(value: int) -> int:\n    return 8 + (value * 3)\n"
    assert _source_predictions(a, (2, 7)) == (14, 29)
    assert _source_predictions(b, (2, 7)) == (14, 29)
    assert _features(a, ((2, 14), (7, 29)))[-1] == 1
    with pytest.raises(ValueError, match="unsupported"):
        _source_predictions('def fn(value):\n    return __import__("os")\n', (2, 7))


def test_fit_train_only(tmp_path: Path):
    roots = [create_case(tmp_path / f"root_{i}", split="train", index=i) for i in range(8)]
    traces = [_local_trace(root) for root in roots]
    weights = fit_logistic(traces)
    assert weights.shape == (6,)
    assert np.isfinite(weights).all()
    assert _rule_baseline(traces)["correct"] == 8
    assert _score_policy(traces, weights, 0.5)["roots"] == 8
    held = create_case(tmp_path / "cal", split="calibration", index=0)
    with pytest.raises(ValueError, match="non-TRAIN"):
        fit_logistic([*traces, _local_trace(held)])
