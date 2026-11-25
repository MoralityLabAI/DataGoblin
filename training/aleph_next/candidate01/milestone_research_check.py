"""Independent public-input solvers and mathematical identities for research admissions."""

import math

from curation_check import machine_witness


def minimax(q):
    catalog = sorted(q["catalog"])
    prices = q["fees"]
    n = len(prices)
    observed = dict(q["observed"])
    if len(set(catalog)) != len(catalog) or any(not 0 <= h < 2**n for h in catalog):
        raise ValueError("invalid finite hypothesis catalog")
    if len(observed) != len(q["observed"]) or any(not 0 <= j < n or y not in (0, 1) for j, y in observed.items()):
        raise ValueError("invalid observations")
    if any(v <= 0 for v in [*prices, q["verify_fee"]]):
        raise ValueError("invalid purchase tariff")
    columns = [{h for h in catalog if h & (1 << j)} for j in range(n)]
    available = set(catalog)
    for j, y in observed.items():
        available &= columns[j] if y else set(catalog) - columns[j]
    if not available:
        raise ValueError("empty version space")
    viable = sorted(available)
    # State the two adversarial branches explicitly as sets and charge every surviving check.
    quotes = [(len(available) * q["verify_fee"], -1, None)]
    options = []
    for j, fee in enumerate(prices):
        if j in observed:
            continue
        yes = available & columns[j]
        no = available - columns[j]
        branches = [len(no), len(yes)]
        quotes.append((max(fee + len(side) * q["verify_fee"] for side in (yes, no)), j, branches))
        options.append([j, sorted(yes), fee])
    bill, j, branches = min(quotes, key=lambda v: v[:2])
    label = dict(survivors=viable, probe=None if j == -1 else j, branches=branches, bill=bill)
    semantic = ["finite-binary-acquisition", viable, options, q["verify_fee"]]
    lineage = ["finite-binary-hypothesis-book", n, catalog]
    return (
        label,
        semantic,
        lineage,
        {"survivors": len(viable), "options": len(quotes) - 1, "choice": "stop" if j == -1 else "probe"},
    )


def matmul(a, b, p):
    return [[sum(x * y for x, y in zip(row, column, strict=True)) % p for column in zip(*b, strict=True)] for row in a]


def counterfactual(q):
    p = q["p"]
    if p not in (13, 17, 19) or q["baseline"] == q["substitution"]:
        raise ValueError("invalid counterfactual")
    matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    initial = [[q["baseline"][0], q["substitution"][0]], [q["baseline"][1], q["substitution"][1]], [1, 1]]
    history = []
    if sum((a * d - b * c) % p == 0 for a, b, c, d, _, _ in q["maps"]) > 1:
        raise ValueError("repeated noninvertible maps lie outside this training family")
    for step in [None, *q["maps"]]:
        if step is not None:
            if len(step) != 6 or any(not 0 <= v < p for v in step):
                raise ValueError("invalid affine operator")
            a, b, c, d, e, f = step
            matrix = matmul([[a, b, e], [c, d, f], [0, 0, 1]], matrix, p)
        outputs = matmul(matrix, initial, p)
        history.append([list(v) for v in zip(*outputs[:2], strict=True)])
    reads = []
    for t, a, b, fee in q["sensors"]:
        if not 0 <= t < len(history) or not 0 <= a < p or not 0 <= b < p or (a, b) == (0, 0) or fee <= 0:
            raise ValueError("invalid observation")
        reads.append([[(a * x + b * y) % p for x, y in history[t]], fee])
    enumerated = sorted((fee, j, reading) for j, (reading, fee) in enumerate(reads) if len(set(reading)) == 2)
    cost, idx, reading = enumerated[0] if enumerated else (None, None, None)
    merger = next((i for i, (x, y) in enumerate(history) if x == y), None)
    if merger is not None and any(x != y for x, y in history[merger:]):
        raise ValueError("invalid deterministic propagation")
    label = dict(terminal=history[-1], first_equal=merger, sensor=idx, readings=reading, fee=cost)
    sem = ["affine-intervention-sensor", p, history, reads]
    return (
        label,
        sem,
        ["affine-intervention-orbit", p, history],
        {"maps": len(q["maps"]), "merged": merger is not None, "choice": "unobservable" if idx is None else "witness"},
    )


def twos(x):
    return (x & -x).bit_length() - 1


def gcd_cost(q):
    a, b = q["pair"]
    dr, sh, su = q["prices"]
    if min(a, b, dr, sh, su) <= 0:
        raise ValueError("invalid GCD inputs or tariff")
    qseq = []
    x, y = a, b
    while y:
        k, r = divmod(x, y)
        if x != k * y + r or not 0 <= r < y:
            raise ValueError("bad Euclidean quotient")
        qseq.append(k)
        x, y = y, r
    common = min(twos(a), twos(b))
    odd = a >> twos(a)
    other = b >> common
    shift = 2 * common + (twos(a) - common) + common
    sub = 0
    while other:
        k = twos(other)
        shift += k
        left, right = sorted((odd, other >> k))
        odd, other = left, right - left
        sub += 1
    ans = math.gcd(a, b)
    if x != ans or odd * 2**common != ans:
        raise ValueError("GCD certificates differ")
    bills = [len(qseq) * dr, shift * sh + sub * su]
    label = dict(
        gcd=ans,
        counts=[len(qseq), shift, sub],
        bills=bills,
        winner=("tie" if len(set(bills)) == 1 else ("euclid", "binary")[bills.index(min(bills))]),
    )
    return (
        label,
        ["priced-integer-gcd", a, b, [dr, sh, su]],
        ["integer-gcd", sorted((a, b))],
        {"remainders": len(qseq), "subtractions": sub, "choice": label["winner"]},
    )


def cube_cost(q):
    d = q["bits"]
    if not 4 <= d <= 7 or min(q["scan_fee"], q["emit_fee"], q["scan_support_fee"]) < 1:
        raise ValueError("invalid Boolean cube task")
    width = 1 << d
    full = (1 << width) - 1
    columns = [sum(1 << value for value in range(width) if value & (1 << j)) for j in range(d)]
    tables = []
    emissions = 0
    for program in (q["reference"], q["proposal"]):
        table = 0
        for yes, no in program:
            if min(yes, no) < 0 or (yes | no) >= width or yes & no:
                raise ValueError("malformed polarity cube")
            states = full
            for j, col in enumerate(columns):
                if yes & (1 << j):
                    states &= col
                if no & (1 << j):
                    states &= full ^ col
            table ^= states
            emissions += 1 << no.bit_count()
        tables.append(table)
    delta = tables[0] ^ tables[1]
    w = (delta & -delta).bit_length() - 1 if delta else None
    # A complete depth-d binary tree accepts exactly the true bit strings of length d.
    # Most-significant-bit first makes the native BFS order equal unsigned input order.
    sink = 2 * width - 1
    edges = [[2 * i + 1, 2 * i + 2] if i < width - 1 else [sink, sink] for i in range(sink + 1)]
    machines = [
        {"start": 0, "next": edges, "accept": {width - 1 + k for k in range(width) if (table >> k) & 1}}
        for table in tables
    ]
    native, _ = machine_witness(*machines)
    if (None if native is None else int(native, 2)) != w:
        raise ValueError("native DFA witness disagrees")
    coef = [(delta >> i) & 1 for i in range(width)]
    for j in range(d):
        for m in range(width):
            if (m >> j) & 1:
                coef[m] ^= coef[m ^ (1 << j)]
    witness_by_anf = next((i for i, x in enumerate(coef) if x), None)
    if witness_by_anf != w:
        raise ValueError("Boolean inversion and witness search disagree")
    tested = width if w is None else w + 1
    clause_count = len(q["reference"]) + len(q["proposal"])
    charges = [tested * clause_count * q["scan_fee"], emissions * q["emit_fee"] + sum(coef) * q["scan_support_fee"]]
    lab = dict(
        witness=w,
        truth=None if w is None else [(x >> w) & 1 for x in tables],
        charges=charges,
        winner="tie" if len(set(charges)) == 1 else ("scan", "algebra")[charges.index(min(charges))],
    )
    sem = [
        "polarity-program-verification",
        d,
        tables,
        clause_count,
        emissions,
        q["scan_fee"],
        q["emit_fee"],
        q["scan_support_fee"],
    ]
    return (
        lab,
        sem,
        ["polarity-reference-function", d, tables[0]],
        {"inputs": width, "equivalent": w is None, "choice": lab["winner"]},
    )


def experiment(q):
    w = q["charter"]
    proposal = q["proposal"]
    required = w["required_arms"]
    metrics = w["required_metrics"]
    minimum = w["minimum_replicates"]
    cap = w["max_tokens_per_run"]
    if (
        set(required) != {"direct", "answer_rlt", "matched_damping"}
        or set(metrics) != {"verified_success", "calls", "regressions"}
        or not 2 <= minimum <= 5
        or not 0 <= w["max_runs"] - 3 * minimum <= 3
        or cap % 16
        or not 32 * 16 <= cap <= 512 * 16
    ):
        raise ValueError("outside native train feedback-control charter")
    ref = dict(
        arms=sorted(required),
        metrics=sorted(metrics),
        replicates=minimum,
        tokens_per_run=cap,
        training_split="train",
        selection_split="calibration",
        confirmation_split="ordinary_test",
        independent_verifier=True,
    )
    spent = len(proposal["arms"]) * proposal["replicates"]
    bad = []
    if sorted(proposal["arms"]) != ref["arms"]:
        bad.append("arm-coverage")
    if sorted(proposal["metrics"]) != ref["metrics"]:
        bad.append("metric-coverage")
    if not minimum <= proposal["replicates"] <= 100:
        bad.append("replicate-limit")
    if spent > w["max_runs"]:
        bad.append("run-budget")
    if not 0 < proposal["tokens_per_run"] <= cap:
        bad.append("per-run-budget")
    for key in ("training_split", "selection_split", "confirmation_split", "independent_verifier"):
        if proposal[key] != ref[key]:
            bad.append(key)
    runs = minimum * len(required)
    label = dict(
        valid=not bad,
        proposed_runs=spent,
        proposed_tokens=spent * proposal["tokens_per_run"],
        reference=ref,
        reference_runs=runs,
        reference_tokens=runs * cap,
    )
    charter = [sorted(required), sorted(metrics), minimum, w["max_runs"], cap]
    normalized = {**proposal, "arms": sorted(proposal["arms"]), "metrics": sorted(proposal["metrics"])}
    return (
        label,
        ["native-experiment-plan", charter, normalized],
        ["native-feedback-control-charter", charter],
        {"choice": "accept" if not bad else "correct", "violations": bad, "reference_runs": runs},
    )


SOLVE = {
    "minimax-probe-purchase": minimax,
    "counterfactual-sensor-choice": counterfactual,
    "gcd-solver-tariff": gcd_cost,
    "polarity-cube-witness-cost": cube_cost,
    "native-experiment-plan-audit": experiment,
}
