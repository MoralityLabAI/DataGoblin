"""Permutation-invariant, source-observable features for Mathlib tactic choices.

Unlike the older grounded ranker's first-command/qualified-lemma vocabulary,
this representation reads the full candidate script. It excludes candidate
position, the synthetic witness flag, kernel outcomes and evaluation labels.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

TOKENIZER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\s]")
DIMENSION = 208
BUCKETS = 64
EXPLICIT = 16
FEATURE_VERSION = "rlt_hard02_lexical_hash_v1"


def _hash(value: str) -> tuple[int, int]:
    hashed = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(hashed[:8], "big") % BUCKETS, 1 if hashed[8] & 1 else -1


def _tokens(source: str) -> list[str]:
    return [t.lower() for t in TOKENIZER.findall(source.strip())]


def _candidate(goal_pp: str, script: str) -> np.ndarray:
    script = script.strip()
    if not script:
        raise ValueError("registered candidate tactic is empty")
    tokens = _tokens(script)
    goal = set(_tokens(goal_pp))
    identifiers = {token for token in tokens if token.isidentifier()}
    local = Counter(tokens)
    goal_overlap = len(identifiers & goal) / max(1, len(identifiers))
    lines = script.splitlines()
    parts = [
        1.0,
        min(len(lines), 30) / 30,
        min(len(tokens), 120) / 120,
        min(len(identifiers), 64) / 64,
        min(len(local), 120) / 120,
        goal_overlap,
        float("exact" in local),
        float("apply" in local),
        float("simp" in local or "simpa" in local),
        float("rw" in local or "rewrite" in local),
        float("rcases" in local),
        float("constructor" in local),
        float("intro" in local),
        float("dvd_add" in local or "dvd_mul" in local),
        float("congrarg" in local),
        float("false" in local or "true" in local),
    ]
    if len(parts) != EXPLICIT:
        raise AssertionError("feature width drift")
    x = np.zeros(DIMENSION, dtype=np.float64)
    x[:EXPLICIT] = parts
    # Signed hashing keeps stable feature width across TRAIN/CAL/holdouts
    # without building a vocabulary using evaluation source text.
    for stream, offset in (
        (tokens, EXPLICIT),
        ([" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)], EXPLICIT + BUCKETS),
        ([" ".join(tokens[i:i + 3]) for i in range(len(tokens) - 2)], EXPLICIT + 2 * BUCKETS),
    ):
        scale = math.sqrt(max(1, len(stream)))
        for key in stream:
            index, sign = _hash(key)
            x[offset + index] += sign / scale
    return x


def features(goal_pp: str, candidates: tuple[str, ...] | list[str]) -> np.ndarray:
    if not goal_pp or not candidates or len(candidates) > 32:
        raise ValueError("nonempty registered goal and bounded candidate menu required")
    rows = np.stack([_candidate(goal_pp, script) for script in candidates])
    if rows.shape != (len(candidates), DIMENSION) or not np.isfinite(rows).all():
        raise ValueError("nonfinite or mismatched feature bank")
    return rows
