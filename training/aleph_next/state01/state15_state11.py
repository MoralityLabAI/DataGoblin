"""Restore the registered STATE-11 proof bytes without depending on floating-point ranker ties."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import state11


def held_rows():
    rng = random.Random(state11.SEED)
    seen = set()
    for domain, group in [
        ("nat", {"struct", "id"}),
        ("nat", {"dist", "id"}),
        ("list", {"append", "rev"}),
        ("list", {"append", "map"}),
    ]:
        state11.generate(rng, domain, group, 120, False, seen)
    held = []
    for domain, group in [("nat", {"struct", "id", "dist"}), ("list", {"append", "rev", "map"})]:
        held.extend(state11.generate(rng, domain, group, 32, True, seen))
    return held


def restore(extra_steps, row):
    domain, current, target, _ = row
    path = []
    for rule, *position in extra_steps:
        hit = next((move for move in state11.moves(current, domain) if move[1:] == (rule, tuple(position))), None)
        if hit is None:
            raise ValueError("registered STATE-11 move is not admissible")
        path.append(hit)
        current = hit[0]
    if current != target:
        raise ValueError("registered STATE-11 path ended at the wrong target")
    return path


def regenerate(out):
    witness = json.loads(Path(__file__).with_name("STATE15_REWRITE_WITNESSES.json").read_bytes())
    held = held_rows()
    if hashlib.sha256(repr(held).encode()).hexdigest() != witness["held_sha256"]:
        raise ValueError("held STATE-11 obligations changed")
    by_arm = {"gold": [row[3] for row in held]}
    for arm in ("bfs", "symbolic"):
        by_arm[arm] = [state11.search(d, x, y, arm)[0] for d, x, y, _ in held]
    for seed in state11.SEEDS:
        choices = witness["learned_routes"][str(seed)]
        if len(choices) != len(held):
            raise ValueError("incomplete STATE-11 witness")
        by_arm[f"learned_{seed}"] = [
            by_arm[c][i] if isinstance(c, str) else None if c is None else restore(c, held[i])
            for i, c in enumerate(choices)
        ]
    terms = {}
    for arm, paths in by_arm.items():
        for i, (row, path) in enumerate(zip(held, paths, strict=True)):
            if path is None:
                continue
            if not state11.check(*row[:3], path):
                raise ValueError("STATE-11 certificate did not replay")
            name = f"gold_{i}" if arm == "gold" else f"found_{arm}_{i}"
            terms[name] = state11.declaration(name, row, path)
    source = (
        "set_option autoImplicit false\n\n"
        + "\n".join(terms.values())
        + "\n"
        + "\n".join("#print axioms " + k for k in terms)
        + "\n"
    )
    if hashlib.sha256(source.encode()).hexdigest() != witness["archive_sha256"]:
        raise ValueError("restoration changed archived theorem bytes")
    controls = []
    for i, (domain, x, _, _) in enumerate(held):
        args, vals = (
            ("(x y z : Nat)", "x y z")
            if domain == "nat"
            else ("(xs ys zs : List Nat) (f g : Nat → Nat)", "xs ys zs f g")
        )
        expr = state11.lean(x)
        bad = f"({expr} + 1)" if domain == "nat" else f"({expr} ++ [0])"
        controls.append(
            {
                "id": i,
                "source": terms[f"gold_{i}"],
                "invalid": f"theorem invalid_{i} {args} : {expr} = {bad} := gold_{i} {vals}\n",
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "STATE11_POSITIVE.lean").write_bytes(source.encode())
    (out / "STATE11_NEGATIVE.json").write_bytes(json.dumps(controls, indent=2).encode())
