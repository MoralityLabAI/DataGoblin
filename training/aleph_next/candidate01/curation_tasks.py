"""Train-only generators. Public premises are serialized independently of the response."""

import hashlib
import itertools
import json
import random
import re
from collections import deque
from pathlib import Path

SEED = 2026112539
SKILLS = ("state", "operator", "composition", "repair", "lean", "allocation")
NATIVE = {
    "lean-train.public.jsonl": "abc7ae1a15e61bc678e4ecafddac428d73905a42e895e0fcd1ef218e391a88f9",
    "lean-train.receipts.jsonl": "b00ccc793e6cb66588829da59de4ff5ea0512075fd333d3369cb19ebc6022178",
}
NATIVE_FAMILIES = {
    "zip_drop_map",
    "reverse_flatten_map",
    "append_filter_map",
    "map_take_drop_while",
    "count_reverse_flatten",
    "gcd_commute_reassociate",
    "reverse_flat_map",
    "reverse_take",
    "drop_then_take_while_map",
}
NATIVE_HELD = {
    "count_partitioned_reverse_flatten",
    "drop_after_predicate_pair",
    "filter_map_reverse_append",
    "gcd_reassociate_under_add",
    "map_after_reverse_take",
    "take_after_map_zip",
    "take_map_after_reverse_flatten",
}
RESERVED = {
    "state": ["branching-register", "double-partial-erasure"],
    "operator": ["piecewise-threshold", "cubic-mixed"],
    "composition": ["shared-subproof-dag", "cyclic-horn"],
    "repair": ["two-fault-machine", "transition-corruption"],
    "lean": sorted(NATIVE_HELD | {"dependent-existential", "case-analysis"}),
    "allocation": ["cyclic-directed", "weighted-directed"],
    "external_tournament": [
        "nat_crossed_expansion",
        "list_antihom_nested_map",
        "mixed_nat_list_product",
        "mixed_nested_product",
    ],
}
INSTRUCTIONS = {
    "state": (
        "Track two residues modulo p. U(k) adds k times y to x, L(k) adds k times x to y, and X "
        "swaps them. Return every state after an instruction, in order, as a JSON list."
    ),
    "operator": (
        "Find the unique coefficients, in the printed basis order, over the field of residues "
        "modulo p from the observations [x,y,value]. Also predict the value at query. Return JSON "
        "with coefficients and value."
    ),
    "composition": (
        "Facts are proved propositions. Each named rule derives its conclusion from proofs of its "
        "ordered premises. Build a proof of goal as [rule_name, premise_proof, ...], using a fact "
        "name as an atomic proof. Return the proof as JSON."
    ),
    "repair": (
        "Machines read bits starting at start. Bit b chooses column b of next. Accept when the "
        "final state belongs to accept. Find the shortest then lexicographically first word on "
        "which reference and proposal differ. Return its bits as a string and the one proposal "
        "acceptance-bit toggle that restores equivalence. If equivalent return null for both. "
        "Return JSON with witness and toggle."
    ),
    "lean": (
        "Close this Lean 4 theorem in an environment importing Std. Supply one complete tactic "
        "proof beginning with by."
    ),
    "allocation": (
        "Choose forward or reverse breadth-first search from start to goal on this directed graph. "
        "Search an entire frontier at a time. Charge all edges leaving that frontier, including "
        "edges to visited states. Reverse search uses reversed edges. Stop after the first full "
        "layer reaching the opposite endpoint. Return JSON with distance, forward cost, reverse "
        "cost, and solver (forward, reverse or tie)."
    ),
}
DELIMITER = "\nAnswer\n"


def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(x):
    return hashlib.sha256(canon(x).encode()).hexdigest()


def record(skill, family, public, answer, **source):
    return dict(
        skill=skill, family=family, public=public, answer=answer, origin=source or {"kind": "fresh-first-party"}
    )


def state(r, i):
    p = r.choice((5, 7, 11))
    x, y = r.randrange(p), r.randrange(p)
    length = r.choice((3, 5, 8, 12))
    ops = [[r.choice(("U", "L")), r.randrange(1, p)] for _ in range(length)]
    family = "register-shear"
    if i % 2:
        ops[r.randrange(length)] = ["X", 0]
        family = "register-shear-swap"
    public = dict(p=p, start=[x, y], ops=ops)
    result = []
    for op, k in ops:
        if op == "U":
            x = (x + k * y) % p
        elif op == "L":
            y = (y + k * x) % p
        else:
            x, y = y, x
        result.append([x, y])
    return record("state", family, public, result)


def operator(r, i):
    p = r.choice((5, 7, 11))
    multi = i % 2
    basis = ["1", "x", "y", "xy"] if multi else ["1", "x", "xx"]
    a = [r.randrange(p) for _ in basis]
    a[-1] = r.randrange(1, p)

    def f(x, y):
        vals = (1, x, y, x * y) if multi else (1, x, x * x)
        return sum(v * c for v, c in zip(vals, a, strict=True)) % p

    xs = r.sample(range(p), 2 if multi else 3)
    ys = r.sample(range(p), 2)
    points = list(itertools.product(xs, ys)) if multi else [(x, 0) for x in xs]
    r.shuffle(points)
    others = [(x, y) for x in range(p) for y in (range(p) if multi else [0]) if (x, y) not in points]
    x, y = r.choice(others)
    q = dict(p=p, basis=basis, observations=[[u, v, f(u, v)] for u, v in points], query=[x, y])
    return record("operator", "bilinear-grid" if multi else "quadratic-line", q, dict(coefficients=a, value=f(x, y)))


def composition(r, i):
    # Disjoint subtrees give a unique Horn proof. Decorations randomize all atom/rule names.
    leaves = r.randint(4, 12)
    pool = [None] * leaves
    while len(pool) > 1:
        r.shuffle(pool)
        a, b = pool.pop(), pool.pop()
        node = (a, b)
        if r.random() < 0.55:
            node = (node,)
        pool.append(node)
    tree = (pool[0],) if i % 2 else (pool[0], None)
    ids = list(range(80))
    r.shuffle(ids)
    facts = []
    rules = []

    def build(node):
        symbol = "p" + str(ids.pop())
        if node is None:
            facts.append(symbol)
            return symbol, symbol
        parts = [build(x) for x in node]
        rule = "r" + str(ids.pop())
        rules.append([rule, [p[0] for p in parts], symbol])
        return symbol, [rule, *[p[1] for p in parts]]

    goal, answer = build(tree)
    r.shuffle(facts)
    r.shuffle(rules)
    return record(
        "composition",
        "tree-top-unary" if i % 2 else "tree-top-binary",
        dict(facts=facts, rules=rules, goal=goal),
        answer,
    )


def accepts(machine, word):
    pos = machine["start"]
    for bit in word:
        pos = machine["next"][pos][int(bit)]
    return pos in machine["accept"]


def repair(r, i):
    n = r.randint(4, 8)
    edges = [[r.randrange(n) for _ in range(2)] for _ in range(n)]
    ref = dict(start=0, next=edges, accept=sorted(r.sample(range(n), r.randrange(1, n))))
    perm = list(range(n))
    r.shuffle(perm)
    nxt = [None] * n
    for x in range(n):
        nxt[perm[x]] = [perm[v] for v in edges[x]]
    accept = {perm[x] for x in ref["accept"]}
    toggle = None
    if i % 4:
        # Pick a reachable state requiring at least two input bits.
        path = {0: ""}
        todo = deque([0])
        while todo:
            x = todo.popleft()
            for bit, v in enumerate(edges[x]):
                if v not in path:
                    path[v] = path[x] + str(bit)
                    todo.append(v)
        choices = [x for x in path if len(path[x]) >= 2]
        if not choices:
            return None
        bad = r.choice(choices)
        toggle = perm[bad]
        witness = path[bad]
        accept ^= {toggle}
    else:
        witness = None
    prop = dict(start=perm[0], next=nxt, accept=sorted(accept))
    return record(
        "repair",
        "acceptance-fault" if toggle is not None else "equivalent-relabel",
        dict(reference=ref, proposal=prop),
        dict(witness=witness, toggle=toggle),
    )


def allocation(r, i):
    k = r.randint(2, 4) if i % 2 else r.randint(5, 8)
    forward = [r.randint(1, 7) for _ in range(r.randrange(1, 6))]
    reverse = [r.randint(1, 7) for _ in range(r.randrange(1, 6))]
    edges = [(j, j + 1) for j in range(k)]
    node = k + 1
    for side, branches in (("out", forward), ("in", reverse)):
        for length in branches:
            previous = 0 if side == "out" else k
            for _ in range(length):
                edges.append((previous, node) if side == "out" else (node, previous))
                previous = node
                node += 1
    perm = list(range(node))
    r.shuffle(perm)
    edges = [[perm[a], perm[b]] for a, b in edges]
    r.shuffle(edges)
    a = k + sum(min(k, t) for t in forward)
    b = k + sum(min(k, t) for t in reverse)
    return record(
        "allocation",
        "short-spine" if k <= 4 else "long-spine",
        dict(start=perm[0], goal=perm[k], edges=edges),
        dict(distance=k, forward=a, reverse=b, solver="forward" if a < b else "reverse" if b < a else "tie"),
    )


def lean(r, i):
    k, m = r.sample(range(2, 48), 2)
    if i % 2:
        src = f"(x + {k})"
        dst = "y"
        for name in r.choices(("f", "g"), k=r.randint(1, 5)):
            src = f"({name} {src})"
            dst = f"({name} {dst})"
        stmt = f"∀ (P : Nat → Prop) (f g : Nat → Nat) (x y : Nat), x + {k} = y → P ({dst} * {m}) → P ({src} * {m})"
        proof = "by\n  intro P f g x y h hp\n  simpa only [h] using hp"
        family = "predicate-congruence"
    else:
        variables = ["x", *["v" + str(n) for n in range(r.randint(1, 7))], "z"]
        terms = [f"x + {k}", *variables[1:-1], f"z + {m}"]
        bounds = [f"{a} ≤ {b}" for a, b in itertools.pairwise(terms)]
        evidence = ["h" + str(j) for j in range(len(bounds))]
        term = evidence[-1]
        for hyp in reversed(evidence[:-1]):
            term = f"(Nat.le_trans {hyp} {term})"
        stmt = "∀ (" + " ".join(variables) + " : Nat), " + " → ".join([*bounds, f"x + {k} ≤ z + {m}"])
        proof = "by\n  intro " + " ".join([*variables, *evidence]) + "\n  exact " + term
        family = "ordered-transitivity"
    return record("lean", family, dict(statement=stmt), proof)


def native_lean(pack):
    checked = []
    for name, sha in NATIVE.items():
        content = (Path(pack) / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != sha:
            raise ValueError("Native training shard identity changed")
        checked.append([json.loads(x) for x in content.splitlines()])
    for public, grades in zip(*checked, strict=True):
        family = public["family"]
        if public["id"] != grades["id"] or public["epoch"] != grades["epoch"] or family not in NATIVE_FAMILIES:
            raise ValueError("Invalid native train lineage")
        winners = []
        for c in public["candidates"]:
            o = grades["outcomes"].get(c["id"], {})
            if (
                o.get("label") == 1
                and o.get("closed")
                and o.get("kernel_ok")
                and o["kernel_calls"] == 1
                and set(o["axioms"]) <= {"propext", "Quot.sound", "Classical.choice"}
            ):
                winners.append((len(c["tactic"]), c["tactic"], c["id"], o))
        if not winners:
            continue
        _, tactic, action, old = min(winners)
        q = record(
            "lean",
            "native." + family,
            dict(statement=public["statement"]),
            "by\n  intros\n  " + tactic,
            kind="native-train",
            root=public["id"],
            lineage=public["lineage"],
            action=action,
            previous_grade_sha256=digest(old),
            native_train_sha256=NATIVE,
        )
        yield q


def library_lean(lean_binary):
    root = Path(lean_binary).resolve().parents[1]
    license_data = (root / "LICENSE").read_bytes()
    if b"Apache License" not in license_data or b"Version 2.0" not in license_data:
        raise ValueError("Missing upstream license")

    def source(family, paths):
        files = {}
        for p, lemmas in paths.items():
            data = (root / p).read_bytes()
            if any(not re.search(rb"\btheorem\s+" + name.encode() + rb"\b", data) for name in lemmas):
                raise ValueError(f"Missing reference declarations in {p}")
            files[p] = dict(sha256=hashlib.sha256(data).hexdigest(), lemmas=lemmas)
        return dict(
            kind="lean-core",
            repository="leanprover/lean4",
            revision="7e01a1bf5c70fc6167d49c345d3bf80596e9a79b",
            upstream_files=files,
            license="Apache-2.0",
            license_sha256=hashlib.sha256(license_data).hexdigest(),
            source_family=family,
        )

    array = source(
        "array-stack",
        {"src/lean/Init/Data/Array/Basic.lean": ["size_push"], "src/lean/Init/Data/Array/Lemmas.lean": ["pop_push"]},
    )
    lists = source("replicated-list", {"src/lean/Init/Data/List/Lemmas.lean": ["reverse_replicate"]})
    bit = source("bitvector-cancel", {"src/lean/Init/Data/BitVec/Lemmas.lean": ["sub_add_cancel"]})
    for depth in range(1, 8):
        for pops in range(depth + 1):
            xs = "xs"
            for i in range(depth):
                xs = f"({xs}.push {1 + i})"
            xs += ".pop" * pops
            if pops:
                statement = f"∀ (xs : Array Nat), {xs}.size = xs.size + {depth - pops}"
                proof = "by\n  intro xs\n  simp only [Array.pop_push, Array.size_push]\n  omega"
            else:
                statement = f"∀ (xs : Array Nat), {xs}.size = xs.size + {depth}"
                proof = "by\n  intro xs\n  simp only [Array.size_push]\n  omega"
            # The permissive final tactic handles goals that simp has already closed.
            proof = proof.replace("\n  omega", " <;> omega")
            yield record("lean", "array-stack", dict(statement=statement), proof, **array)
        expr = "(List.replicate n x)" + ".reverse" * depth
        stmt = f"∀ (n x : Nat), {expr} = List.replicate n x"
        yield record(
            "lean",
            "replicated-list",
            dict(statement=stmt),
            "by\n  intro n x\n  simp only [List.reverse_replicate]",
            **lists,
        )
        expr = "x"
        for j in range(depth):
            var = "y" if j % 2 == 0 else "z"
            expr = f"(({expr} - {var}) + {var})"
        stmt = f"∀ (x y z : BitVec 16), {expr} = x"
        yield record(
            "lean",
            "bitvector-cancel",
            dict(statement=stmt),
            "by\n  intro x y z\n  simp only [BitVec.sub_add_cancel]",
            **bit,
        )


def candidates(count, pack):
    makers = (state, operator, composition, repair, lean, allocation)
    for i in range(count):
        for kind, gen in zip(SKILLS, makers, strict=True):
            row = gen(random.Random(f"{SEED}/{kind}/{i}"), i)
            if row is not None:
                yield row
    yield from native_lean(pack)
