"""Recompile archived explicit rewrite certificates using typed steps and a balanced Eq.trans tree."""

import re

from state15_kernel import compile_query, digest, proof_axioms
from state15_replay import HEADER

ARGUMENTS = {
    "(x y z : Nat)": "x y z",
    "(xs ys zs : List Nat) (f g : Nat → Nat)": "xs ys zs f g",
}


def build(name, term):
    """Check every original endpoint and preserve each source proof RHS byte-for-byte."""
    lines = term.rstrip("\n").splitlines()
    original_sha = digest(term.encode())
    if len(lines) < 3 or lines[1] != "  calc":
        return (
            HEADER + term + "\n#print axioms " + name + "\n",
            [name],
            {
                "variant": "raw_singleton",
                "archived_term_sha256": original_sha,
            },
        )
    first = lines[0]
    for binders in ARGUMENTS:
        prefix = f"theorem {name} {binders} : "
        if first.startswith(prefix):
            break
    else:
        raise ValueError("unrecognized rewrite theorem binders")
    arguments = ARGUMENTS[binders]
    if not first.endswith(" := by") or not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError("unrecognized rewrite theorem statement")
    signature = first.removesuffix(" := by")
    start, end = signature.removeprefix(prefix).split(" = ")
    original_steps = lines[2:]
    helpers = []
    names = []
    links = []
    current = start
    for i, line in enumerate(original_steps):
        if not line.startswith("    "):
            raise ValueError("unexpected syntax in archived calculation")
        equality, rhs = line[4:].split(" := ", 1)
        left, right = equality.split(" = ")
        if left != current:
            raise ValueError("archived calculation has disconnected steps")
        leaf = f"{name}_state15_step_{i}"
        helpers.append(f"theorem {leaf} {binders} : {equality} := {rhs}\n")
        names.append(leaf)
        links.append((left, right, leaf))
        current = right
    if current != end or not links:
        raise ValueError("archived calculation ends at a different goal")
    level = 0
    while len(links) > 1:
        joined = []
        for i in range(0, len(links), 2):
            left = links[i]
            if i + 1 == len(links):
                joined.append(left)
                continue
            right = links[i + 1]
            if left[1] != right[0]:
                raise ValueError("balanced composition would change an endpoint")
            node = f"{name}_state15_join_{level}_{i // 2}"
            helpers.append(
                f"theorem {node} {binders} : {left[0]} = {right[1]} := "
                f"Eq.trans ({left[2]} {arguments}) ({right[2]} {arguments})\n"
            )
            names.append(node)
            joined.append((left[0], right[1], node))
        links = joined
        level += 1
    helpers.append(signature + " := " + links[0][2] + " " + arguments + "\n")
    names.append(name)
    source = HEADER + "set_option linter.unusedVariables false\n" + "\n".join(helpers) + "\n"
    source += "\n".join("#print axioms " + key for key in names) + "\n"
    meta = {
        "variant": "typed_steps_balanced_trans",
        "archived_term_sha256": original_sha,
        "statement_sha256": digest(signature.encode()),
        "original_step_lines_sha256": digest("\n".join(original_steps).encode()),
        "original_steps": len(original_steps),
        "kernel_declarations": len(names),
    }
    return source, names, meta


def check(lean, out, stage, name, term, timeout):
    source, names, meta = build(name, term)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"s{stage}_{name}.lean"
    path.write_bytes(source.encode())
    cost, log = compile_query(lean, path, timeout)
    audited, ax = proof_axioms(log, names)
    return {
        "stage": stage,
        "name": name,
        **meta,
        **cost,
        "passed": cost["exit_code"] == 0 and not cost["timed_out"] and audited,
        "axioms": sorted({a for group in ax.values() for a in group}),
        "audited_declarations": len(ax),
    }
