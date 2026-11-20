"""Derive bounded, multi-lemma Lean tactics from the visible target."""

from __future__ import annotations

import itertools

from .rlt_lean_grounded import candidates, words

POOL_LIMIT = 36
RETRIEVAL_WIDTH = 10
METHODS = ("legacy", "wider", "compose")


def tactic_list(scripts):
    return tuple(dict.fromkeys(script for script in scripts if len(script.encode("utf-8")) < 96))[:POOL_LIMIT]


def name_terms(name):
    return words(name.split(".", 1)[1])


def retrieve(pp, entries):
    target = pp.rsplit("⊢", 1)[-1]
    lhs, _, rhs = target.partition("=")
    left, right = words(lhs), words(rhs)
    visible = left | right
    rows = []
    for item in entries:
        name = item["name"]
        ops = name_terms(name)
        common = ops & visible
        if common:
            head = name.split(".", 1)[1].split("_", 1)[0].lower()
            # Prefer operators on the rewrite side and declarations that connect both sides.
            score = 3 * len(common) / len(ops) + 0.45 * (head in left) + 0.10 * (head in right)
            rows.append((-score, name))
    return visible, tuple(name for _, name in sorted(rows)[:RETRIEVAL_WIDTH])


def ranked_groups(visible, hits, width):
    operations = {name: name_terms(name) for name in hits}
    positions = {name: i for i, name in enumerate(hits)}

    def key(group):
        union = set().union(*(operations[name] for name in group))
        overlap = union & visible
        # Reward complementary operations over another lemma on the same set of operators.
        return (-len(overlap), len(union - visible), sum(positions[name] for name in group), group)

    return sorted(itertools.combinations(hits, width), key=key)


def algebraic_actions(hits):
    hit_set = set(hits)
    for assoc in hits:
        head, sep, suffix = assoc.rpartition("_")
        comm = head + "_comm"
        if sep and suffix == "assoc" and comm in hit_set:
            for depth in (1, 2, 3):
                yield f"rw [{assoc}] <;> congr {depth} <;> apply {comm}"


def expanded_candidates(pp, entries):
    legacy_hits, legacy = candidates(pp, entries)
    visible, hits = retrieve(pp, entries)
    pairs = ranked_groups(visible, hits, 2)
    triples = ranked_groups(visible, hits, 3)
    singles = [f"simp [{name}]" for name in hits]
    wider = tactic_list([
        *singles[:2],
        *(f"simp [{', '.join(group)}]" for group in triples[:4]),
        *(f"simp [{', '.join(group)}]" for group in pairs[:12]),
        *singles[2:],
        *(f"rw [{name}]" for name in hits),
        *(f"exact {name}" for name in hits),
        *legacy,
    ])

    combined = list(algebraic_actions(hits))
    for i in range(max(min(10, len(pairs)), min(6, len(triples)))):
        if i < len(pairs):
            group = ", ".join(pairs[i])
            combined.extend((f"simp only [{group}]", f"simp [{group}]"))
        if i < min(6, len(triples)):
            group = ", ".join(triples[i])
            combined.extend((f"simp only [{group}]", f"simp [{group}]"))
    composed = tactic_list([*combined, *wider])
    return {"legacy": tuple(legacy_hits), "wider": hits, "compose": hits}, {
        "legacy": legacy, "wider": wider, "compose": composed,
    }
