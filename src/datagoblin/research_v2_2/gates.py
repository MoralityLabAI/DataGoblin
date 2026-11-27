"""Fail-closed checks for private evaluation and Baby Goblin continuation."""

from __future__ import annotations

import json
import math
from pathlib import Path

from datagoblin.research_v2_1.suite import load_suite


def _outside_git(path: Path) -> None:
    path = path.resolve()
    if any((p / ".git").exists() for p in (path, *path.parents)):
        raise ValueError("private input must be outside Git")


def validate_panel(cases, suite_path: Path, exclusion_registry: Path | None) -> None:
    if not cases:
        raise ValueError("empty evaluation")
    if all(c["split"] == "development" for c in cases):
        return
    _outside_git(suite_path)
    if exclusion_registry is None:
        raise ValueError("non-development evaluation requires private exclusion registry")
    _outside_git(exclusion_registry)
    exclusion = json.loads(exclusion_registry.read_text())
    required = {"v2.0-held", "v2.1-prospective32", "training-and-calibration"}
    sources = exclusion["sources"]
    if not required <= set(sources):
        raise ValueError("incomplete exclusion provenance")
    excluded = set()
    for name in required:
        rows = sources[name]
        if not rows or not all(isinstance(r, str) and len(r) == 64 for r in rows):
            raise ValueError(f"missing or malformed root inventory: {name}")
        excluded.update(rows)
    # Check the entire panel, not just the balanced prefix selected by the runner.
    roots = [c["root_id"] for c in load_suite(suite_path)]
    if len(roots) != len(set(roots)) or excluded.intersection(roots):
        raise ValueError("duplicate or previously exposed evaluation root")


def validate_continuation(receipt: dict) -> None:
    """Required operator-attested evidence before a stage can be launched.

    The runner must independently verify the named artifacts; this validator
    checks the receipt contract and does not certify the underlying bytes.
    """
    if receipt["stage_positions"] not in (20_000_000, 50_000_000, 100_000_000, 200_000_000):
        raise ValueError("unregistered stage")
    if receipt["context_tokens"] != 2048 or receipt["truncated_prompts"] != 0:
        raise ValueError("context or truncation gate")
    for field in ("weights_sha256", "optimizer_sha256", "scheduler_sha256",
                  "rng_sha256", "cursor_sha256", "corpus_sha256", "tokenizer_sha256"):
        value = receipt[field]
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError(f"missing digest: {field}")
    if receipt["resume_positions"] >= receipt["stage_positions"] or receipt["resume_positions"] < 10_000_000:
        raise ValueError("invalid cursor position")
    expected, observed = receipt["scheduler_resume_lr"], receipt["optimizer_resume_lr"]
    if (not all(math.isfinite(v) and v > 0 for v in (expected, observed))
            or not math.isclose(expected, observed, rel_tol=1e-6)):
        raise ValueError("learning-rate discontinuity")
    if receipt["optimizer_state_restored"] is not True or receipt["rng_cursor_restored"] is not True:
        raise ValueError("weights-only or discontinuous resume")
    if receipt["excluded_root_overlap"] != 0:
        raise ValueError("held root overlap")


def validate_specialist_accounting(receipt: dict) -> None:
    if receipt["truncated_prompts"] != 0 or receipt["missing_eos_targets"] != 0:
        raise ValueError("prompt or EOS supervision defect")
    if receipt["specialist_loss_tokens"] < 423_812 or receipt["pool_epochs_complete"] < 2:
        raise ValueError("insufficient specialist exposure")
    for left, right in (
        ("specialist_forward_tokens", "forward_control_tokens"),
        ("specialist_loss_tokens_total", "loss_control_tokens"),
    ):
        a, b = receipt[left], receipt[right]
        if min(a, b) <= 0 or abs(a - b) / max(a, b) > 0.01:
            raise ValueError(f"unmatched control: {left}")
