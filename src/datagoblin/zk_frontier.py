"""Plan-specialized IVC with reusable live-frontier registers.

The terminal state exposes only the sink output and zeroed retired slots.
This is a constraint compiler, not a recursive proof implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from datagoblin.zk_replay import Plan, clamp, digest, require, validate
from datagoblin.zk_schedule import equal, next_state, product, times, total
from datagoblin.zk_sigma import Circuit, Expr, const, neg


@dataclass(frozen=True)
class Layout:
    slots: int
    before: tuple[tuple[int, ...], ...]
    after: tuple[tuple[int, ...], ...]
    destination: tuple[int, ...]


def layout(plan: Plan) -> Layout:
    """Optimal slot reuse for this fixed order, with reads before writes."""
    width = validate(plan)
    n = len(plan.jobs)
    require(width <= 2 and n <= 32 and max(j.steps for j in plan.jobs) <= 32, "frontier limits")
    index = {job.name: i for i, job in enumerate(plan.jobs)}
    last = list(range(n))
    for j, job in enumerate(plan.jobs):
        for p in job.parents:
            last[index[p]] = j
    last[-1] = n  # Keep the terminal output.
    k = max(sum(i <= t < last[i] for i in range(n)) for t in range(n))
    owners = [-1] * k
    before, after, destination = [], [], []
    for j, job in enumerate(plan.jobs):
        before.append(tuple(owners))
        require(all(index[p] in owners for p in job.parents), "missing live parent")
        for slot, old in enumerate(owners):
            if old >= 0 and last[old] == j:
                owners[slot] = -1
        slot = owners.index(-1)
        owners[slot] = j
        destination.append(slot)
        after.append(tuple(owners))
    require(sum(v >= 0 for v in owners) == 1 and n - 1 in owners, "terminal layout")
    return Layout(k, tuple(before), tuple(after), tuple(destination))


def frontier_context(plan: Plan) -> tuple[int, int]:
    raw = digest("frontier-graph-plan-v1", asdict(plan))
    return int(raw[:32], 16), int(raw[32:], 16)


def initial(plan: Plan) -> tuple[int, ...]:
    k = layout(plan).slots
    return (0, *frontier_context(plan), *((0,) * (k * len(plan.jobs[0].source))))


def compile_step(plan: Plan, state: tuple[int, ...]) -> dict[str, Any]:
    """Use the plan, never a witness-selected layout, to construct every row."""
    alloc = layout(plan)
    n, w, k = len(plan.jobs), len(plan.jobs[0].source), alloc.slots
    ctx = frontier_context(plan)
    require(len(state) == 3 + k*w and all(type(v) is int for v in state), "state shape")
    require(0 <= state[0] < n and state[1:3] == ctx, "state context or counter")
    require(all(-127 <= v <= 127 for v in state[3:]), "register range")
    c = Circuit()
    wires = [c.alloc(v, True) for v in state]
    pc, ctx1, ctx2 = wires[:3]
    regs = [wires[3+i*w:3+(i+1)*w] for i in range(k)]
    equal(c, ctx1, const(ctx[0]))
    equal(c, ctx2, const(ctx[1]))
    for e in wires[3:]:
        c.range(e + const(127), 8)
        c.range(const(127) + neg(e), 8)
    sels = [c.alloc(int(state[0] == j)) for j in range(n)]
    for s in sels:
        c.products.append((s, s, s))
    equal(c, total(sels), const(1))
    equal(c, pc, total([times(s, j) for j, s in enumerate(sels)]))
    indices = {job.name: j for j, job in enumerate(plan.jobs)}
    x: list[Expr] = []
    for i in range(w):
        joined = [
            const(job.source[i]) + total([regs[alloc.before[j].index(indices[p])][i] for p in job.parents])
            for j, job in enumerate(plan.jobs)
        ]
        q = total([product(c, s, z) for s, z in zip(sels, joined, strict=True)])
        y = c.alloc(clamp(c.value(q)))
        c.clip(q, y, 19)
        x.append(y)
    weights = [
        [total([times(s, job.weights[i][m]) for s, job in zip(sels, plan.jobs, strict=True)])
         for m in range(w)]
        for i in range(w)
    ]
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
    for slot in range(k):
        write = total([s for j, s in enumerate(sels) if alloc.destination[j] == slot])
        keep = total([s for j, s in enumerate(sels)
                      if alloc.before[j][slot] >= 0 and alloc.before[j][slot] == alloc.after[j][slot]])
        for i in range(w):
            val = product(c, keep, regs[slot][i]) + product(c, write, h[i])
            z = c.alloc(c.value(val), True)
            equal(c, z, val)
            out.append(z)
    return {"format": "zk-dg-frontier-r1cs-v1", "state_in": [e[0][0] for e in wires],
            "state_out": [e[0][0] for e in out],
            "constraints": [(e, const(1), ()) for e in c.linear] + c.products,
            "witness": c.values, "worker_selectors": sels, "worker_output": h}


def schedule(plan: Plan) -> list[dict[str, Any]]:
    state = initial(plan)
    result = []
    for _ in plan.jobs:
        item = compile_step(plan, state)
        result.append(item)
        state = next_state(item)
    return result


def terminal(plan: Plan, state: tuple[int, ...]) -> tuple[int, ...]:
    alloc = layout(plan)
    w = len(plan.jobs[0].source)
    require(len(state) == 3 + alloc.slots*w and state[:3] == (len(plan.jobs), *frontier_context(plan)),
            "terminal state mismatch")
    require(all(type(v) is int and -127 <= v <= 127 for v in state[3:]), "terminal register range")
    slot = alloc.destination[-1]
    require(all(v == 0 for j in range(alloc.slots) if j != slot for v in state[3+j*w:3+(j+1)*w]),
            "retired register not cleared")
    return state[3+slot*w:3+(slot+1)*w]


def native_bundle(plan: Plan) -> tuple[dict[str, Any], list[list[str]]]:
    items = schedule(plan)
    shape = items[0]
    require(all(e["constraints"] == shape["constraints"] and e["state_in"] == shape["state_in"]
                and e["state_out"] == shape["state_out"] for e in items), "shape mismatch")
    zn = next_state(items[-1])
    terminal(plan, zn)
    return ({"format": "zk-dg-nova-statement-v1", "steps": len(items), "wires": len(shape["witness"]),
             "state_in": shape["state_in"], "state_out": shape["state_out"],
             "constraints": [[[[i, str(a)] for i, a in expr] for expr in row] for row in shape["constraints"]],
             "z0": [str(v) for v in initial(plan)], "zn": [str(v) for v in zn]},
            [[str(v) for v in e["witness"]] for e in items])


def optimize_order(plan: Plan, exact_limit: int = 16) -> Plan:
    """Return a proposed plan. Reordering requires independent reauthorization."""
    layout(plan)
    require(type(exact_limit) is int and 0 <= exact_limit <= 16, "exact DP limit")
    n = len(plan.jobs)
    index = {j.name: i for i, j in enumerate(plan.jobs)}
    parents = [sum(1 << index[p] for p in j.parents) for j in plan.jobs]
    children = [sum(1 << j for j, bits in enumerate(parents) if bits & (1 << i)) for i in range(n)]
    full = (1 << n) - 1

    def live(mask: int) -> int:
        return sum(bool(mask & (1 << i)) and (bool(bits & ~mask) or i == n - 1)
                   for i, bits in enumerate(children))

    def ready(mask: int) -> list[int]:
        return [i for i in range(n) if not mask & (1 << i) and not parents[i] & ~mask]

    order = []
    if n <= exact_limit:
        dp = [n+1] * (full+1)
        prev = [-1] * (full+1)
        dp[0] = 0
        for mask in range(full):
            if dp[mask] > n:
                continue
            for i in ready(mask):
                nxt = mask | (1 << i)
                cost = max(dp[mask], live(nxt))
                if cost < dp[nxt]:
                    dp[nxt], prev[nxt] = cost, i
        mask = full
        while mask:
            i = prev[mask]
            require(i >= 0, "no topological order")
            order.append(i)
            mask ^= 1 << i
        order.reverse()
    else:
        mask = 0
        while mask != full:
            i = min(ready(mask), key=lambda j: (live(mask | (1 << j)), j))
            order.append(i)
            mask |= 1 << i
    proposed = replace(plan, jobs=tuple(plan.jobs[i] for i in order))
    return proposed if layout(proposed).slots < layout(plan).slots else plan
