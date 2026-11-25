"""Environment-owned replay for COL arithmetic certificates.

This checker uses explicit full-map updates and repeated division. It never
calls the generator's accelerated step or accepts its evidence label.
"""
from __future__ import annotations

import hashlib
import json
from math import gcd
from typing import Any

CERT_SCHEMA = "DataGoblin/COL-CERT/v1"
SPLIT_SALT = "COL-parameter-split-v1"


def parameter_split(d: int) -> str:
    if type(d) is not int or d < 1 or d % 4 != 1 or gcd(d, 6) != 1:
        raise ValueError("d must be positive, 1 modulo 4, and coprime to 6")
    bucket = int(hashlib.sha256(f"{SPLIT_SALT}:{d}".encode()).hexdigest(), 16) % 10
    return "train" if bucket < 6 else "dev" if bucket < 8 else "holdout"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _integer(value: Any, name: str, *, minimum: int = 0, max_bits: int = 4096) -> int:
    if type(value) is not int or value < minimum or value.bit_length() > max_bits:
        raise ValueError(f"{name} must be an integer in the declared resource bound")
    return value


def verify_certificate(candidate: dict, *, max_edges: int = 100_000) -> dict:
    """Return a fresh checked receipt, or reject any malformed or false claim."""
    if type(candidate) is not dict or candidate.get("schema") != CERT_SCHEMA:
        raise ValueError("unrecognized certificate schema")
    _integer(max_edges, "max_edges", minimum=1)
    d = _integer(candidate.get("d"), "d", minimum=1)
    split = parameter_split(d)
    seed = _integer(candidate.get("seed"), "seed", minimum=1)
    if seed % 2 != 1:
        raise ValueError("seed must be odd")
    if candidate.get("root_id") != f"d:{d}" or candidate.get("split") != split:
        raise ValueError("root identity or parameter split mismatch")
    states, edges = candidate.get("odd_states"), candidate.get("edges")
    if type(states) is not list or type(edges) is not list or not 1 <= len(edges) <= max_edges:
        raise ValueError("certificate edges exceed their declared bound")
    if len(states) != len(edges) + 1 or states[0] != seed:
        raise ValueError("state and edge lengths or seed disagree")
    for state in states:
        _integer(state, "state", minimum=1)
        if state % 2 != 1:
            raise ValueError("odd trace contains an even state")
    if len(set(states[:-1])) != len(states) - 1:
        raise ValueError("trace repeats before its claimed first closure")
    start = _integer(candidate.get("cycle_start"), "cycle_start")
    if start >= len(edges) or states[-1] != states[start]:
        raise ValueError("claimed cycle does not close at its entry")
    if states[-1] not in states[:-1]:
        raise ValueError("terminal state does not close the trace")
    triples: list[int] = []
    halvings: list[int] = []
    microsteps = 0
    g = gcd(seed, d)
    for i, edge in enumerate(edges):
        if type(edge) is not dict:
            raise ValueError("edge must be an object")
        claimed_a = _integer(edge.get("halvings"), "halvings", minimum=1)
        claimed_u = _integer(edge.get("triple"), "triple")
        n = states[i]
        triple = 1 if n % 4 == 1 else 0
        x = 3 * n + d if triple else n + d
        a = 0
        while x % 2 == 0:
            x //= 2
            a += 1
        if (x, a, triple) != (states[i + 1], claimed_a, claimed_u):
            raise ValueError(f"full-map replay rejected edge {i}")
        if gcd(x, d) != g:
            raise ValueError("gcd conservation failed")
        triples.append(triple)
        halvings.append(a)
        microsteps += a + 1
    r = len(edges) - start
    a, u = sum(halvings[start:]), sum(triples[start:])
    expected = {"cycle_minimum": min(states[start:-1]), "odd_steps": r,
                "halvings": a, "triple_branches": u, "period": r + a,
                "gcd_with_d": g}
    claim = candidate.get("claim")
    if type(claim) is not dict or set(claim) != set(expected):
        raise ValueError("cycle claim fields differ from the schema")
    for key, value in expected.items():
        if _integer(claim.get(key), key) != value:
            raise ValueError(f"cycle claim mismatch: {key}")
    modulus = d // g
    if pow(2, a, modulus) != pow(3, u, modulus):
        raise ValueError("cycle product congruence failed")
    return {"status": "verified", "scope": "finite_unaccelerated_replay",
            "certificate_sha256": canonical_hash(candidate), "root_id": f"d:{d}",
            "split": split, "case_id": f"COL:{d}:{seed}",
            "cycle_minimum": expected["cycle_minimum"], "checked_odd_edges": len(edges),
            "checked_full_map_steps": microsteps}
