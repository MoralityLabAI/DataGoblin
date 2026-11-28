"""Plan-specialized DAG circuit: one fixed-shape folding step per worker.

Public register file; constraint oracle only, not a recursive proof system.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from datagoblin.zk_replay import Plan, clamp, digest, require, validate
from datagoblin.zk_sigma import Circuit, Expr, const, neg

# Pasta Fp and Fq. All bounded integer relations fit well below either field.
PALLAS_BASE = 0x40000000000000000000000000000000224698FC094CF91B992D30ED00000001
PALLAS_SCALAR = 0x40000000000000000000000000000000224698FC0994A8DD8C46EB2100000001


def times(e: Expr, k: int) -> Expr:
    return tuple((i, a * k) for i, a in e)


def total(es: list[Expr]) -> Expr:
    return tuple(term for e in es for term in e)


def product(c: Circuit, a: Expr, b: Expr) -> Expr:
    z = c.alloc(c.value(a) * c.value(b))
    c.products.append((a, b, z))
    return z


def equal(c: Circuit, a: Expr, b: Expr) -> None:
    c.linear.append(a + neg(b))


def context(plan: Plan) -> tuple[int, int]:
    h = digest("folded-graph-plan", asdict(plan))
    return int(h[:32], 16), int(h[32:], 16)


def initial(plan: Plan) -> tuple[int, ...]:
    width = validate(plan)
    require(width <= 2 and len(plan.jobs) <= 32 and max(j.steps for j in plan.jobs) <= 32, "folded graph limits")
    return (0, *context(plan), *((0,) * (width * len(plan.jobs))))


def compile_step(plan: Plan, state: tuple[int, ...]) -> dict[str, Any]:
    """Compile from trusted plan; advice from state is never used to choose shape."""
    init = initial(plan)
    n = len(plan.jobs)
    w = len(plan.jobs[0].source)
    require(len(state) == len(init) and all(type(v) is int for v in state), "state shape")
    require(0 <= state[0] < n and state[1:3] == init[1:3], "state context or counter")
    require(all(-127 <= v <= 127 for v in state[3:]), "register range")
    c = Circuit()
    wires = [c.alloc(v, True) for v in state]
    pc, ctx1, ctx2 = wires[:3]
    regs = [wires[3+i*w:3+(i+1)*w] for i in range(n)]
    equal(c, ctx1, const(init[1]))
    equal(c, ctx2, const(init[2]))
    for e in wires[3:]:
        c.range(e + const(127), 8)
        c.range(const(127) + neg(e), 8)
    # One-hot authorization of worker index. The plan is topologically ordered.
    sels = [c.alloc(int(state[0] == j)) for j in range(n)]
    for s in sels:
        c.products.append((s, s, s))
    equal(c, total(sels), const(1))
    equal(c, pc, total([times(s, j) for j, s in enumerate(sels)]))
    indices = {job.name: j for j, job in enumerate(plan.jobs)}
    x = []
    for i in range(w):
        joined = [const(job.source[i]) + total([regs[indices[p]][i] for p in job.parents])
                  for job in plan.jobs]
        q = total([product(c, s, z) for s, z in zip(sels, joined, strict=True)])
        y = c.alloc(clamp(c.value(q)))
        c.clip(q, y, 19)
        x.append(y)
    # Select coefficients from the authorized plan, not from the witness.
    weights = [[total([times(s, job.weights[i][k]) for s, job in zip(sels, plan.jobs, strict=True)])
                for k in range(w)] for i in range(w)]
    biases = [total([times(s, job.bias[i]) for s, job in zip(sels, plan.jobs, strict=True)]) for i in range(w)]
    h = x
    for t in range(max(job.steps for job in plan.jobs)):
        active = total([s for s, job in zip(sels, plan.jobs, strict=True) if t < job.steps])
        nxt = []
        for i in range(w):
            q = total([product(c, a, z) for a, z in zip(weights[i], h, strict=True)]) + biases[i] + x[i]
            y = c.alloc(clamp(c.value(q)))
            c.clip(q, y, 19)
            delta = product(c, active, y + neg(h[i]))
            z = c.alloc(c.value(h[i]) + c.value(delta))
            equal(c, z, h[i] + delta)
            nxt.append(z)
        h = nxt
    pcn = c.alloc(state[0] + 1, True)
    equal(c, pcn, pc + const(1))
    out = [pcn, ctx1, ctx2]
    for j in range(n):
        for i in range(w):
            delta = product(c, sels[j], h[i] + neg(regs[j][i]))
            z = c.alloc(c.value(regs[j][i]) + c.value(delta), True)
            equal(c, z, regs[j][i] + delta)
            out.append(z)
    return {"format": "zk-dg-scheduled-r1cs-v1", "state_in": [e[0][0] for e in wires],
            "state_out": [e[0][0] for e in out],
            "constraints": [(e, const(1), ()) for e in c.linear] + c.products,
            "witness": c.values}


def next_state(export: dict[str, Any]) -> tuple[int, ...]:
    return tuple(export["witness"][i] for i in export["state_out"])


def schedule(plan: Plan) -> list[dict[str, Any]]:
    state = initial(plan)
    result = []
    for _ in plan.jobs:
        item = compile_step(plan, state)
        result.append(item)
        state = next_state(item)
    return result


def native_bundle(plan: Plan) -> tuple[dict[str, Any], list[list[str]]]:
    """Compile a relying-party-owned statement and separate prover witness file.

    The statement includes the expected public terminal registers. Never use
    a worker-provided statement as the source of authorization.
    """
    items = schedule(plan)
    shape = items[0]
    require(all(e['constraints'] == shape['constraints'] and e['state_in'] == shape['state_in']
                and e['state_out'] == shape['state_out'] for e in items), 'shape mismatch')
    return ({'format': 'zk-dg-nova-statement-v1', 'steps': len(items), 'wires': len(shape['witness']),
             'state_in': shape['state_in'], 'state_out': shape['state_out'],
             'constraints': [[[[i, str(a)] for i,a in expr] for expr in row] for row in shape['constraints']],
             'z0': [str(v) for v in initial(plan)], 'zn': [str(v) for v in next_state(items[-1])]},
            [[str(v) for v in e['witness']] for e in items])
