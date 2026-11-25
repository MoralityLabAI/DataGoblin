"""Independent solvers of public prompts. No generator answers are inputs to these algorithms."""

import hashlib
import json
import re
import subprocess
from collections import defaultdict, deque
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def key(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def state(q):
    p = q["p"]
    start = q["start"]
    matrix = (1, 0, 0, 1)
    maps = []
    trace = []
    swaps = 0
    if p not in (5, 7, 11) or len(start) != 2 or any(v not in range(p) for v in start):
        raise ValueError("register domain")
    for op, k in q["ops"]:
        a, b, c, d = matrix
        if op == "U" and k in range(1, p):
            matrix = (a + k * c, b + k * d, c, d)
        elif op == "L" and k in range(1, p):
            matrix = (a, b, c + k * a, d + k * b)
        elif op == "X" and k == 0:
            matrix = (c, d, a, b)
            swaps += 1
        else:
            raise ValueError("unknown register operator")
        matrix = tuple(v % p for v in matrix)
        maps.append(matrix)
        a, b, c, d = matrix
        x, y = start
        trace.append([(a * x + b * y) % p, (c * x + d * y) % p])
    if len(trace) not in (3, 5, 8, 12) or swaps > 1:
        raise ValueError("reserved state grammar")
    return (
        trace,
        "register-shear-swap" if swaps else "register-shear",
        (p, start, maps),
        dict(instructions=len(trace), domain_states=p * p),
    )


def operator(q):
    p = q["p"]
    basis = q["basis"]
    n = len(basis)
    if p not in (5, 7, 11) or basis not in (["1", "x", "y", "xy"], ["1", "x", "xx"]):
        raise ValueError("reserved operator grammar")

    def features(x, y):
        vals = {"1": 1, "x": x, "y": y, "xy": x * y, "xx": x * x}
        return [vals[b] % p for b in basis]

    pairs = q["observations"]
    x, y = q["query"]
    if (
        len(pairs) != n
        or any(v not in range(p) for row in pairs for v in row)
        or (x, y) in [(u, v) for u, v, _ in pairs]
    ):
        raise ValueError("invalid public evidence")
    work = [[*features(u, v), w] for u, v, w in pairs]
    for j in range(n):
        pivot = next((i for i in range(j, n) if work[i][j] % p), None)
        if pivot is None:
            raise ValueError("operator unidentifiable from observations")
        work[pivot], work[j] = work[j], work[pivot]
        inv = pow(work[j][j], -1, p)
        work[j] = [v * inv % p for v in work[j]]
        for i in range(n):
            if i != j:
                factor = work[i][j]
                work[i] = [(a - factor * b) % p for a, b in zip(work[i], work[j], strict=True)]
    a = [row[-1] for row in work]
    if not a[-1] or x not in range(p) or y not in range(p):
        raise ValueError("degenerate operator or invalid query")

    def value(x, y):
        return sum(t * c for t, c in zip(features(x, y), a, strict=True)) % p

    table = [value(u, v) for u in range(p) for v in range(p)]
    return (
        dict(coefficients=a, value=value(x, y)),
        ("bilinear-grid" if n == 4 else "quadratic-line"),
        (p, table),
        dict(rank=n, domain_points=p * p),
    )


def composition(q):
    facts = set(q["facts"])
    rules = q["rules"]
    if (
        len(facts) != len(q["facts"])
        or len({r[2] for r in rules} | facts) != len(rules) + len(facts)
        or len({r[0] for r in rules}) != len(rules)
    ):
        raise ValueError("ambiguous proof context")
    table = {p: (p, "*", 1) for p in facts}
    parents = defaultdict(int)
    for _, ps, _ in rules:
        if len(ps) not in (1, 2):
            raise ValueError("proof arity")
        for p in ps:
            parents[p] += 1
    if any(v != 1 for v in parents.values()) or q["goal"] in parents:
        raise ValueError("held shared-subproof shape")
    todo = list(rules)
    while todo:
        later = []
        for r, ps, y in todo:
            if any(p not in table for p in ps):
                later.append((r, ps, y))
                continue
            vals = [table[p] for p in ps]
            shape = "(" + ",".join(sorted(row[1] for row in vals)) + ")"
            table[y] = [r, *[row[0] for row in vals]], shape, 1 + max(row[2] for row in vals)
        if len(later) == len(todo):
            raise ValueError("unproved premises or cycle")
        todo = later
    if set(table) - set(parents) != {q["goal"]}:
        raise ValueError("disconnected proof")
    proof, shape, depth = table[q["goal"]]
    return (
        proof,
        "tree-top-unary" if len(proof) == 2 else "tree-top-binary",
        shape,
        dict(rules=len(rules), proof_depth=depth),
    )


def machine_form(m):
    """Canonical minimal DFA of the accepted language, including unreachable-state removal."""
    edges = m["next"]
    accepting = set(m["accept"])
    start = m["start"]
    states = set(range(len(edges)))
    if start not in states or not accepting <= states or any(len(row) != 2 or not set(row) <= states for row in edges):
        raise ValueError("malformed finite automaton")
    reachable = {start}
    frontier = {start}
    while frontier:
        frontier = {v for u in frontier for v in edges[u]} - reachable
        reachable.update(frontier)
    part = {i: int(i in accepting) for i in reachable}
    while True:
        sig = {i: (i in accepting, *(part[t] for t in edges[i])) for i in reachable}
        classes = {v: j for j, v in enumerate(sorted(set(sig.values())))}
        new = {i: classes[sig[i]] for i in reachable}
        stable = len(set(new.values())) == len(set(part.values()))
        part = new
        if stable:
            break
    lookup = {v: k for k, v in part.items()}
    queue = deque([part[start]])
    order = {part[start]: 0}
    form = []
    while queue:
        cls = queue.popleft()
        node = lookup[cls]
        row = []
        for t in edges[node]:
            c = part[t]
            if c not in order:
                order[c] = len(order)
                queue.append(c)
            row.append(order[c])
        form.append([int(node in accepting), *row])
    return form


def machine_witness(a, b):
    queue = deque([(a["start"], b["start"], "")])
    seen = {(a["start"], b["start"])}
    while queue:
        x, y, w = queue.popleft()
        if (x in a["accept"]) != (y in b["accept"]):
            return w, y
        for bit in (0, 1):
            pair = a["next"][x][bit], b["next"][y][bit]
            if pair not in seen:
                seen.add(pair)
                queue.append((*pair, w + str(bit)))
    return None, None


def repair(q):
    a, b = q["reference"], q["proposal"]
    old, new = machine_form(a), machine_form(b)
    witness, toggle = machine_witness(a, b)
    if witness is not None:
        fixed = {**b, "accept": sorted(set(b["accept"]) ^ {toggle})}
        if len(witness) < 2 or machine_form(fixed) != old:
            raise ValueError("invalid single-toggle repair")
    elif old != new:
        raise ValueError("bisimulation discrepancy")
    return (
        dict(witness=witness, toggle=toggle),
        "equivalent-relabel" if witness is None else "acceptance-fault",
        (old, new),
        dict(states=len(b["next"]), witness_length=None if witness is None else len(witness)),
    )


def allocation(q):
    edges = {tuple(t) for t in q["edges"]}
    start, goal = q["start"], q["goal"]
    if len(edges) != len(q["edges"]) or start == goal:
        raise ValueError("malformed graph")
    out = defaultdict(set)
    inc = defaultdict(set)
    for a, b in edges:
        out[a].add(b)
        inc[b].add(a)

    def search(s, t, graph):
        seen = {s}
        layer = {s}
        work = 0
        depth = 0
        while t not in layer:
            if not layer:
                raise ValueError("unreachable target")
            work += sum(len(graph[u]) for u in layer)
            layer = set().union(*(graph[u] for u in layer)) - seen
            seen.update(layer)
            depth += 1
        return depth, work

    dist, cf = search(start, goal, out)
    d2, cr = search(goal, start, inc)
    if dist != d2 or not 2 <= dist <= 8:
        raise ValueError("unexpected graph distance")

    # Prove the admissible structural class and canonicalize it under vertex relabeling.
    def chains(origin, end, graph):
        shapes = []
        used = set()
        spine = []
        for nxt in graph[origin]:
            v = nxt
            chain = []
            while True:
                if v in used or v == origin:
                    raise ValueError("cycle or shared branch")
                used.add(v)
                chain.append(v)
                if v == end or not graph[v]:
                    break
                if len(graph[v]) != 1:
                    raise ValueError("not a single branch")
                v = next(iter(graph[v]))
            if v == end:
                spine = chain
            else:
                shapes.append(len(chain))
        if len(spine) != dist:
            raise ValueError("unexpected backbone")
        return sorted(shapes), used

    f, forward = chains(start, goal, out)
    b, back = chains(goal, start, inc)
    if len(edges) != dist + sum(f) + sum(b) or (forward | back) != {v for edge in edges for v in edge}:
        raise ValueError("unused or cross-linked graph")
    ans = dict(distance=dist, forward=cf, reverse=cr, solver="forward" if cf < cr else "reverse" if cr < cf else "tie")
    return (
        ans,
        "short-spine" if dist <= 4 else "long-spine",
        (dist, f, b),
        dict(edges=len(edges), optimal_cost=min(cf, cr), regret=abs(cf - cr)),
    )


SOLVE = {"state": state, "operator": operator, "composition": composition, "repair": repair, "allocation": allocation}
ALLOWED_AXIOMS = {"propext", "Quot.sound", "Classical.choice"}
AX = re.compile(r"^'(\w+)' (?:does not depend on any axioms|depends on axioms: \[([^\]]*)\])$", re.M)
FORBIDDEN = re.compile(
    r"\b(?:sorry|admit|axiom|unsafe|native_decide|run_tac|elab|macro|set_option|import|theorem|def)\b|[#;]", re.I
)


def lean_template(stmt):
    return re.sub(r"\b\d+\b", "NUM", " ".join(stmt.split()))


def kernel_batch(rows, lean_binary, directory):
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)
    binary = Path(lean_binary).resolve()
    exe_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    version = subprocess.check_output([str(binary), "--version"], text=True, encoding="utf8").strip()
    if "4.28.0" not in version:
        raise ValueError("Lean 4.28.0 is required")
    results = {}
    counter = 0

    def invoke(source):
        nonlocal counter
        counter += 1
        name = f"batch-{counter:04d}.lean"
        f = directory / name
        f.write_bytes(source.encode())
        run = subprocess.run(
            [str(binary), "-j", "1", name],
            cwd=directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=90,
        )
        (directory / (name + ".log")).write_bytes(run.stdout)
        return (
            run.returncode,
            run.stdout.decode("utf8", errors="replace").replace("\r\n", "\n"),
            dict(
                file=name,
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                compiler_output_sha256=hashlib.sha256(run.stdout).hexdigest(),
                compiler_sha256=exe_sha,
                version=version,
            ),
        )

    code, log, neg = invoke("import Std\ntheorem negative_control : False := by exact True.intro\n")
    if code == 0 or "type mismatch" not in log.lower():
        raise ValueError("Lean did not reject negative control")

    def check(block):
        declarations = []
        for rid, stmt, proof in block:
            if FORBIDDEN.search(stmt) or FORBIDDEN.search(proof.replace("<;>", "")) or not proof.startswith("by\n"):
                raise ValueError("unsupported tactic source")
            name = "candidate_" + rid
            declarations.append(f"theorem {name} : {stmt} := {proof}\n#print axioms {name}\n")
        code, log, base = invoke("import Std\nset_option autoImplicit false\n" + "\n".join(declarations))
        if code != 0 and len(block) > 1:
            j = len(block) // 2
            check(block[:j])
            check(block[j:])
            return
        audits = {name[10:]: sorted(t.strip() for t in vals.split(",") if t.strip()) for name, vals in AX.findall(log)}
        for rid, stmt, proof in block:
            axioms = audits.get(rid)
            accepted = code == 0 and axioms is not None and set(axioms) <= ALLOWED_AXIOMS
            results[rid] = dict(
                **base, passed=accepted, axioms=axioms, statement_sha256=key(stmt), proof_sha256=key(proof)
            )

    for i in range(0, len(rows), 32):
        check(rows[i : i + 32])
    return results, neg
