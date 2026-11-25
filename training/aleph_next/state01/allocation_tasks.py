"""Public generators and certificate solvers for the final external allocator study."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass

import state13 as prior

rw = prior.s
ast = rw.s
TRAIN = ("nat_distribute", "nat_exchange", "list_reverse_append", "list_map_append")
HELD = ("nat_crossed_expansion", "list_antihom_nested_map", "mixed_nat_list_product", "mixed_nested_product")
ACTIONS = ("normal", "congruence", "frontiers", "feedback")
SEEDS = (401, 409, 419)
TRAIN_SEED, HELD_SEED = 2026112701, 2026112702


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Leaf:
    domain: str
    left: object
    right: object


@dataclass(frozen=True)
class Task:
    split: str
    family: str
    lineage: str
    intervention: str
    source: tuple
    wrappers: tuple
    target: tuple

    def public(self):
        return [[leaf.domain, leaf.left, leaf.right] for leaf in self.target]


def node_values(rng, domain, n):
    result = []
    for _ in range(n):
        v = rng.choice(("x", "y", "z") if domain == "nat" else ("xs", "ys", "zs"))
        choice = rng.randrange(6)
        if domain == "nat":
            v = (("+", v, "1"), ("+", "1", v), ("*", v, "1"), ("+", v, "0"), v, v)[choice]
        else:
            v = (("rev", v), ("mf", v), ("mg", v), ("++", v, "nil"), v, v)[choice]
        result.append(v)
    return result


def law(rng, family):
    if family.startswith("nat"):
        a, b, c, d = node_values(rng, "nat", 4)
        if family == "nat_distribute":
            left, right = ("*", a, ("+", b, c)), ("+", ("*", a, b), ("*", a, c))
        elif family == "nat_exchange":
            left, right = ("+", a, ("+", b, c)), ("+", c, ("+", a, b))
        else:
            left = ("*", ("+", a, b), ("+", c, d))
            terms = [("*", r, s) for r in (a, b) for s in (c, d)]
            rng.shuffle(terms)
            right = ("+", ("+", terms[0], terms[1]), ("+", terms[2], terms[3]))
        domain = "nat"
    else:
        a, b, c = node_values(rng, "list", 3)
        if family == "list_reverse_append":
            left, right = ("rev", ("++", a, b)), ("++", ("rev", b), ("rev", a))
        elif family == "list_map_append":
            f = rng.choice(("mf", "mg"))
            left, right = (f, ("++", a, b)), ("++", (f, a), (f, b))
        else:
            left = ("rev", ("mf", ("++", a, ("mg", ("++", b, c)))))
            right = ("++", ("mf", ("mg", ("rev", c))), ("++", ("mf", ("mg", ("rev", b))), ("mf", ("rev", a))))
        domain = "list"
    if rng.randrange(2):
        left, right = right, left
    return Leaf(domain, left, right)


def wrapping(rng, d):
    if d == "nat":
        (arg,) = node_values(rng, d, 1)
        return (rng.choice(("+", "*")), "HOLE", arg)
    return rng.choice((("mf", "HOLE"), ("mg", "HOLE"), ("++", "HOLE", rng.choice(("xs", "ys", "zs")))))


def inside(wrapper, x):
    return ast.replace(wrapper, (1,), x)


def changed(term, domain, invalid):
    if domain == "nat":
        return ("+", term, "1") if invalid else ("+", "0", term)
    return ("++", term, "xs") if invalid else ("rev", ("rev", term))


def pair_key(leaves):
    sides = [[(v.domain, getattr(v, side)) for v in leaves] for side in ("left", "right")]
    return digest(sorted(sides, key=repr))


def historical():
    import state14

    seen = set()
    rng = random.Random(ast.SEED)
    for d, group in (
        ("nat", {"struct", "id"}),
        ("nat", {"dist", "id"}),
        ("list", {"append", "rev"}),
        ("list", {"append", "map"}),
    ):
        ast.generate(rng, d, group, 120, False, seen)
    for d, group in (("nat", {"struct", "id", "dist"}), ("list", {"append", "rev", "map"})):
        ast.generate(rng, d, group, 32, True, seen)
    rows = [(d, x, y) for d, x, y, *_ in rw.panel()]
    rows.extend((r["domain"], r["x"], r["y"]) for r in [*prior.generate(), *state14.panel()])
    return {pair_key((Leaf(d, x, y),)) for d, x, y in [*seen, *rows]}


def generate(split, lineages=6, banned=()):
    if split not in ("train", "held"):
        raise ValueError("unknown split")
    rng = random.Random(TRAIN_SEED if split == "train" else HELD_SEED)
    used, tasks, keys = set(), [], set(banned)
    for family in TRAIN if split == "train" else HELD:
        for _ in range(lineages):
            for _retry in range(200):
                if family == "mixed_nat_list_product":
                    parts = ("nat_distribute", "list_antihom_nested_map")
                elif family == "mixed_nested_product":
                    parts = ("nat_crossed_expansion", "list_reverse_append", "nat_exchange")
                else:
                    parts = (family,)
                source = tuple(law(rng, p) for p in parts)
                wrappers = tuple(wrapping(rng, v.domain) for v in source)
                base = tuple(
                    Leaf(leaf.domain, inside(w, leaf.left), inside(w, leaf.right))
                    for w, leaf in zip(wrappers, source, strict=True)
                )
                lineage = digest(
                    [(leaf.domain, leaf.left, leaf.right, w) for leaf, w in zip(source, wrappers, strict=True)]
                )
                affected = rng.randrange(len(base))
                trial = []
                for mode in ("unchanged", "edited", "invalid"):
                    target = list(base)
                    if mode != "unchanged":
                        v = target[affected]
                        target[affected] = Leaf(v.domain, v.left, changed(v.right, v.domain, mode == "invalid"))
                    trial.append(Task(split, family, lineage, mode, source, wrappers, tuple(target)))
                proposed = {pair_key(t.target) for t in trial}
                if (
                    lineage not in used
                    and len(proposed) == 3
                    and not proposed & keys
                    and all(leaf.left != leaf.right for leaf in source)
                ):
                    used.add(lineage)
                    keys.update(proposed)
                    tasks.extend(trial)
                    break
            else:
                raise RuntimeError("lineage sampler exhausted")
    return tasks


def equal(leaves):
    return all(prior.nf(v.domain, v.left) == prior.nf(v.domain, v.right) for v in leaves)


def valid(leaves, paths):
    if paths is None or len(paths) != len(leaves):
        return False
    for v, path in zip(leaves, paths, strict=True):
        base = prior.nf(v.domain, v.left)
        if not rw.verify(v.domain, v.left, v.right, path) or any(prior.nf(v.domain, z) != base for z, *_ in path):
            return False
    return True


def solve(leaves, action, source=None, wrappers=None, anchor=None):
    """Return only complete checked paths. All attempts share one total search allowance."""
    start = time.perf_counter()
    rw.moves.cache_clear()
    ast.distance.cache_clear()
    paths, work, expansions, remaining = [], 0, 0, 64
    for i, leaf in enumerate(leaves):
        d, x, y = leaf.domain, leaf.left, leaf.right
        if action == "normal":
            path, steps, visits, tests, _ = rw.normal(d, x, y, 128)
            work += steps + visits + tests
            n = 0
        elif action == "congruence":
            old = prior.CAP
            try:
                prior.CAP = remaining
                path, n, edges, tests, visits = prior.decompose(d, x, y)
            finally:
                prior.CAP = old
            work += n + edges + tests + visits
        elif action == "frontiers":
            path, n, edges, tests = prior.search(d, x, y, remaining)
            work += n + edges + tests
        elif action == "feedback":
            if anchor is None:
                return None, work, expansions, time.perf_counter() - start
            core, wrapper = source[i], wrappers[i]
            path = [(inside(wrapper, z), rule, (1, *pos), rev) for z, rule, pos, rev in anchor[i]]
            midpoint = inside(wrapper, core.right)
            if inside(wrapper, core.left) != x or not rw.verify(d, x, midpoint, path):
                return None, work, expansions, time.perf_counter() - start
            work += len(path) + ast.size(midpoint)
            tail, n, edges, tests = prior.search(d, midpoint, y, min(remaining, 8))
            work += n + edges + tests
            path = None if tail is None else path + tail
        else:
            raise ValueError(action)
        remaining -= n
        expansions += n
        if path is None:
            return None, work, expansions, time.perf_counter() - start
        paths.append(path)
    if not valid(leaves, paths):
        raise ValueError("invalid claimed path")
    return paths, work, expansions, time.perf_counter() - start


def features(task):
    vals = [prior.features(v.domain, v.left, v.right) for v in task.target]
    means = [sum(x) / len(vals) for x in zip(*vals, strict=True)]
    return [
        *means,
        len(vals) / 3,
        max(r[1] for r in vals),
        max(r[2] for r in vals),
        sum(
            inside(w, leaf.right) == target.right
            for w, leaf, target in zip(task.wrappers, task.source, task.target, strict=True)
        )
        / len(vals),
    ]


def pair(terms):
    return terms[0] if len(terms) == 1 else f"({terms[0]}, {pair(terms[1:])})"


ARGS = "(x y z : Nat) (xs ys zs : List Nat) (f g : Nat -> Nat)"


def declaration(name, leaves, paths):
    left = pair([ast.lean(v.left) for v in leaves])
    right = pair([ast.lean(v.right) for v in leaves])
    # A one-step nested calc can consume a following tactic in Lean's layout parser.
    # Top-level independent declarations close each component's proof unambiguously.
    lines = []
    refs = []
    for i, (leaf, path) in enumerate(zip(leaves, paths, strict=True)):
        proof = rw.declaration("leaf", leaf.domain, leaf.left, leaf.right, path).split(" := by\n", 1)[1]
        subname = f"{name}_component_{i}"
        lines.append(
            f"private theorem {subname} {ARGS} : {ast.lean(leaf.left)} = {ast.lean(leaf.right)} := by\n" + proof
        )
        refs.append(f"({subname} x y z xs ys zs f g)")
    expression = refs[-1]
    for ref in reversed(refs[:-1]):
        expression = f"(Prod.ext {ref} {expression})"
    lines.append(f"theorem {name} {ARGS} : {left} = {right} := {expression}\n")
    return "\n".join(lines)


def invalid_declaration(leaves):
    left = pair([ast.lean(v.left) for v in leaves])
    right = pair([ast.lean(v.right) for v in leaves])
    return f"theorem stale {ARGS} : {left} = {right} := witness x y z xs ys zs f g\n"
