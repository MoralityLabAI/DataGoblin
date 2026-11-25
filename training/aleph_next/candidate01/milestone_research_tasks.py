"""First-party train-only research decisions with original mathematical premises."""

import itertools
import random

from datagoblin.hashing import sha256_json
from datagoblin.training_gym.rlt_gym import tasks as rlt

SEED = 2026112737
FAMILIES = (
    "minimax-probe-purchase",
    "counterfactual-sensor-choice",
    "gcd-solver-tariff",
    "polarity-cube-witness-cost",
    "native-experiment-plan-audit",
)
INSTRUCTIONS = {
    FAMILIES[0]: (
        "Each integer hypothesis stores a Boolean outcome in bit j for probe j (low bit first). Filter on "
        "the observed [j,bit] pairs. Stop costs verify_fee times the surviving count. An unobserved probe "
        "costs its fee plus verify_fee times the larger remaining branch. Minimize worst bill; ties prefer "
        "stop, then smaller j. Return survivors in ascending order, probe (null for stop), branch counts "
        "[zero,one] (null for stop), and bill."
    ),
    FAMILIES[1]: (
        "Run the baseline and substituted state through the same maps modulo p. A map [a,b,c,d,e,f] sends "
        "(x,y) to (ax+by+e,cx+dy+f). A sensor [t,u,v,fee] reads ux+vy modulo p after t maps (0 means input). "
        "Choose the cheapest distinguishing sensor, ties by list index. Return terminal pair, first equal "
        "boundary or null, sensor index or null, its two readings or null, and its fee or null."
    ),
    FAMILIES[2]: (
        "Compare Euclidean remainder iteration starting in the given order to binary GCD. Binary GCD strips "
        "common twos from both inputs, makes the first odd, then repeatedly makes the second odd, orders the "
        "pair ascending and subtracts the first from the second until zero. Charge right shifts and final "
        "restoration left shifts individually. Comparisons and swaps are free. Return gcd, counts "
        "[remainders,shifts,subtractions], bills [Euclid,binary] using the three prices in that order, and "
        "winner (euclid, binary or tie)."
    ),
    FAMILIES[3]: (
        "An XOR program consists of cubes [ones,zeros]. A cube is true if all ones bits are set and zeros "
        "bits cleared. Find the least unsigned input where the programs differ, or null. Scan visits inputs "
        "in ascending order and charges scan_fee per cube tested on both programs through the witness or "
        "exhaustion. Algebra expands each negated literal as 1+x over GF(2), charges emit_fee per emitted "
        "monomial and scan_support_fee per surviving monomial of the difference. Return witness, truth pair "
        "or null, charges [scan,algebra] and winner (scan, algebra or tie)."
    ),
    FAMILIES[4]: (
        "Audit this proposed DataGoblin experiment plan against its charter. It must cover exactly the "
        "required arms and metrics, meet the replicate minimum and both caps, and train on train, select on "
        "calibration and confirm on ordinary_test with an independent verifier. Return valid, the proposed "
        "runs and tokens, and a reference plan using minimum replicates and the full per-run budget with "
        "sorted arms and metrics. Report its reference_runs and reference_tokens as well. Runs equal arms "
        "times replicates. Charge every proposed run even if the proposal is invalid."
    ),
}


def probes(rng, i):
    n = rng.randint(7, 12)
    catalog = rng.sample(range(1 << n), rng.randint(8, 18))
    known = rng.choice(catalog)
    obs = sorted((j, (known >> j) & 1) for j in rng.sample(range(n), rng.choice((1, 1, 2, 2, 3))))
    fees = [rng.randint(1, 11) for _ in range(n)]
    v = rng.randint(1, 6)
    q = dict(catalog=catalog, observed=obs, fees=fees, verify_fee=v)
    surviving = sorted(h for h in catalog if all((h >> j) & 1 == bit for j, bit in obs))
    best = (len(surviving) * v, -1, None)
    for j, price in enumerate(fees):
        if j in dict(obs):
            continue
        ones = sum((h >> j) & 1 for h in surviving)
        counts = [len(surviving) - ones, ones]
        cand = (price + v * max(counts), j, counts)
        if cand[:2] < best[:2]:
            best = cand
    return q, dict(survivors=surviving, probe=None if best[1] < 0 else best[1], branches=best[2], bill=best[0])


def sensors(rng, i):
    p = rng.choice((13, 17, 19))
    start = [rng.randrange(p), rng.randrange(p)]
    alt = [(start[0] + rng.randrange(1, p)) % p, (start[1] + rng.randrange(p)) % p]
    n = rng.randint(3, 7)
    steps = []
    for _ in range(n):
        while True:
            a, b, c, d = [rng.randrange(p) for _ in range(4)]
            if (a * d - b * c) % p:
                break
        steps.append([a, b, c, d, rng.randrange(p), rng.randrange(p)])
    if i % 3 == 0:
        j = rng.randrange(n)
        steps[j][:4] = [0, 0, 0, 0]
    elif i % 3 == 1:
        j = rng.randrange(n)
        a, b = rng.randrange(1, p), rng.randrange(p)
        m = rng.randrange(p)
        steps[j][:4] = [a, b, m * a % p, m * b % p]
    sensors = []
    for _ in range(rng.randint(4, 9)):
        u, v = rng.randrange(p), rng.randrange(p)
        if u == 0 and v == 0:
            u = 1
        sensors.append([rng.randrange(n + 1), u, v, rng.randint(1, 14)])
    q = dict(p=p, baseline=start, substitution=alt, maps=steps, sensors=sensors)
    paths = []
    for state in (start, alt):
        trace = [state]
        for a, b, c, d, e, f in steps:
            x, y = trace[-1]
            trace.append([(a * x + b * y + e) % p, (c * x + d * y + f) % p])
        paths.append(trace)
    first = next((t for t, (x, y) in enumerate(zip(*paths, strict=True)) if x == y), None)
    options = []
    for idx, (t, u, v, fee) in enumerate(sensors):
        readings = [(u * path[t][0] + v * path[t][1]) % p for path in paths]
        if readings[0] != readings[1]:
            options.append((fee, idx, readings))
    fee, idx, readings = min(options, default=(None, None, None))
    return q, dict(terminal=[x[-1] for x in paths], first_equal=first, sensor=idx, readings=readings, fee=fee)


def gcd_route(rng, i):
    if i % 4 == 0:
        a, b = 1, 1
        for _ in range(rng.randint(4, 17)):
            a, b = b, a + b
        common = rng.randint(2, 89)
        a, b = a * common, b * common
    elif i % 4 == 1:
        a, b = (rng.randrange(3, 300, 2) << rng.randrange(3, 10) for _ in range(2))
    elif i % 4 == 2:
        a = rng.randint(9, 15000)
        b = a + rng.choice((-3, -1, 1, 3))
    else:
        common = rng.randint(1, 53)
        a, b = rng.randint(4, 4096) * common, rng.randint(4, 4096) * common
    if rng.randrange(2):
        a, b = b, a
    prices = [rng.choice((1, 2, 4, 8, 16, 32)), rng.randint(1, 4), rng.randint(1, 5)]
    q = dict(pair=[a, b], prices=prices)
    x, y = a, b
    rem = 0
    while y:
        x, y = y, x % y
        rem += 1
    x, y = a, b
    shifts = subs = common = 0
    while x % 2 == 0 and y % 2 == 0:
        x //= 2
        y //= 2
        shifts += 2
        common += 1
    while x % 2 == 0:
        x //= 2
        shifts += 1
    while y:
        while y % 2 == 0:
            y //= 2
            shifts += 1
        if x > y:
            x, y = y, x
        y -= x
        subs += 1
    shifts += common
    bills = [rem * prices[0], shifts * prices[1] + subs * prices[2]]
    winner = "euclid" if bills[0] < bills[1] else "binary" if bills[1] < bills[0] else "tie"
    return q, dict(gcd=x << common, counts=[rem, shifts, subs], bills=bills, winner=winner)


def cubes(rng, i):
    dim = rng.randint(4, 7)

    def cube():
        bits = rng.sample(range(dim), rng.randint(1, min(dim - 1, 4)))
        on = off = 0
        for b in bits:
            if rng.randrange(2):
                on |= 1 << b
            else:
                off |= 1 << b
        return (on, off)

    original = []
    size = rng.randint(3, 7)
    while len(original) < size:
        c = cube()
        if c not in original:
            original.append(c)
    dest = original.copy()
    k = rng.randrange(len(dest))
    pos, neg = dest.pop(k)
    choices = [1 << j for j in range(dim) if not ((pos | neg) >> j) & 1]
    j = rng.choice(choices)
    dest.extend([(pos | j, neg), (pos, neg | j)])
    if i % 3:
        if i % 3 == 1:
            dest.pop(rng.choice((-1, -2)))
        else:
            c = cube()
            dest.append(c)
    rng.shuffle(original)
    rng.shuffle(dest)
    scan_fee, emit_fee, read_fee = rng.randint(1, 5), rng.choice((1, 2, 3, 6, 12)), rng.randint(1, 4)
    q = dict(
        bits=dim, reference=original, proposal=dest, scan_fee=scan_fee, emit_fee=emit_fee, scan_support_fee=read_fee
    )

    def value(cs, x):
        return sum((x & on) == on and (x & off) == 0 for on, off in cs) % 2

    witness = None
    pair = None
    for x in range(1 << dim):
        pair = [value(original, x), value(dest, x)]
        if pair[0] != pair[1]:
            witness = x
            break
    if witness is None:
        pair = None
    support = set()
    emitted = 0
    for on, off in itertools.chain(original, dest):
        sub = off
        while True:
            term = on | sub
            support.symmetric_difference_update([term])
            emitted += 1
            if not sub:
                break
            sub = (sub - 1) & off
    visits = (1 << dim) if witness is None else witness + 1
    bills = [visits * (len(original) + len(dest)) * scan_fee, emitted * emit_fee + len(support) * read_fee]
    winner = "scan" if bills[0] < bills[1] else "algebra" if bills[1] < bills[0] else "tie"
    return q, dict(witness=witness, truth=pair, charges=bills, winner=winner)


def native_task(world):
    lineage = sha256_json({"family": "experiment_plan", "input": world})
    task = dict(
        schema_version=rlt.SCHEMA,
        task_id="experiment_plan:" + lineage[:20],
        family="experiment_plan",
        generator_id=rlt.GENERATOR_VERSION + "/experiment_plan/feedback_control_plan",
        lineage_id="lineage:" + lineage,
        split="train",
        prompt=INSTRUCTIONS[FAMILIES[4]],
        input=world,
        candidates=rlt._pool("experiment_plan", world, random.Random(0)),
        evidence_epoch=1,
        provenance=dict(
            kind="synthetic_environment",
            source="DataGoblin/rlt_gym",
            generator_version=rlt.GENERATOR_VERSION,
            rights="project_generated",
        ),
    )
    task["task_digest"] = sha256_json(task)
    return task


def experiment(rng, i):
    world, _, variant = rlt._world("experiment_plan", rng, False, 1)
    if variant != "feedback_control_plan":
        raise ValueError("unexpected native train generator")
    task = native_task(world)
    ref = rlt.reference_answer(task)
    proposal = {**ref, "arms": ref["arms"].copy(), "metrics": ref["metrics"].copy()}
    if i % 5 == 0:
        proposal["tokens_per_run"] = rng.randint(1, world["max_tokens_per_run"])
    else:
        for op in rng.sample(range(7), rng.choice((1, 1, 2))):
            if op == 0:
                proposal["independent_verifier"] = False
            elif op == 1:
                proposal["selection_split"] = "ordinary_test"
            elif op == 2:
                proposal["replicates"] += rng.randint(1, 4)
            elif op == 3:
                proposal["tokens_per_run"] += rng.randint(1, 64)
            elif op == 4:
                proposal["metrics"].pop(rng.randrange(len(proposal["metrics"])))
            elif op == 5:
                proposal["arms"].pop(rng.randrange(len(proposal["arms"])))
            else:
                proposal["replicates"] = world["minimum_replicates"] - 1

    def charge(plan):
        runs = len(plan["arms"]) * plan["replicates"]
        return runs, runs * plan["tokens_per_run"]

    runs, bill = charge(proposal)
    ref_runs, ref_bill = charge(ref)
    if not rlt.verify(task, ref)["passed"]:
        raise ValueError("native reference failed")
    ok = rlt.verify(task, proposal)["passed"]
    return dict(charter=world, proposal=proposal), dict(
        valid=ok,
        proposed_runs=runs,
        proposed_tokens=bill,
        reference=ref,
        reference_runs=ref_runs,
        reference_tokens=ref_bill,
    )


BUILD = (probes, sensors, gcd_route, cubes, experiment)


def candidate(serial):
    family_idx = serial % len(FAMILIES)
    i = serial // len(FAMILIES)
    rng = random.Random(f"datagoblin-research-train:{SEED}:{family_idx}:{i}")
    q, answer = BUILD[family_idx](rng, i)
    return FAMILIES[family_idx], q, answer
