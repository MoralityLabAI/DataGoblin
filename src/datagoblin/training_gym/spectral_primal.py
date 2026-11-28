"""SG-14: feasible dual rounding and exact coordinate repairs."""
from datagoblin.training_gym.spectral_quotient import indices, score, validate
from datagoblin.training_gym.spectral_routing import TaskError


def dual_round(q, eta):
    """Reverse conditional relaxation rounding; exact row budgets throughout."""
    a, b = [0]*q['d'], [0]*q['d']
    tables = []
    for i, row in enumerate(q['u']):
        price = q['lambda']*(q['x'][i+1]-q['x'][0])
        tables.append([v+a[j]-b[j]-price*q['bits'][j] for j, v in enumerate(row)])
        for j, v in enumerate(row):
            z = v-eta[i]
            b[j] = max(b[j], a[j]+z-price*q['bits'][j])
            a[j] += max(0, z)
    s = 0
    result = [0] * q['T']
    for i in range(q['T']-1, -1, -1):
        row = q['u'][i]
        delta = tables[i]
        order = sorted(range(q['d']), key=lambda j: (-(row[j] if s & (1 << j) else delta[j]), j))
        m = sum(1 << j for j in order[:q['k'][i]])
        result[i] = m
        s |= m
    return result


def visit_masks(q, schedule):
    masks = [0] * q['d']
    for i, a in enumerate(schedule):
        for j in indices(a):
            masks[j] |= 1 << i
    return masks


def row_best(q, schedule, i, masks=None):
    """Exact best row with other rows held fixed; retain incumbent on ties."""
    masks = visit_masks(q, schedule) if masks is None else masks
    old = schedule[i]
    bit = 1 << i
    x = q['x']
    values = [u - q['lambda'] * b * max(0, x[i+1]-x[(m & ~bit).bit_length()])
              for u, b, m in zip(q['u'][i], q['bits'], masks, strict=True)]
    order = sorted(range(q['d']), key=lambda j: (-values[j], not bool(old & (1 << j)), j))
    new = sum(1 << j for j in order[:q['k'][i]])
    gain = sum(values[j] for j in indices(new & ~old)) - sum(values[j] for j in indices(old & ~new))
    return new, gain


def row_sweep(q, schedule, reverse=True):
    """One Gauss-Seidel sweep of exact row optimizations; no tie moves."""
    result = list(schedule)
    masks = visit_masks(q, result)
    gain = 0
    for i in (range(q['T']-1, -1, -1) if reverse else range(q['T'])):
        old = result[i]
        new, delta = row_best(q, result, i, masks)
        if delta > 0:
            for j in indices(old ^ new):
                masks[j] ^= 1 << i
            result[i] = new
            gain += delta
    return result, gain


def pair_best(q, schedule, a, b):
    """Exact joint reassignment of two labels, preserving each row's occupancy.

    Four suffix states represent whether each of the two labels has a later use.
    Runtime O(T); row budget and all other labels are unchanged.
    """
    if not 0 <= a < b < q['d']:
        raise TaskError('invalid label pair')
    ma, mb = 1 << a, 1 << b
    states = {0: (0, ())}
    old_value = 0
    later = 0
    for i in range(q['T']-1, -1, -1):
        old = int(bool(schedule[i] & ma)) | (2 * int(bool(schedule[i] & mb)))
        count = old.bit_count()
        choices = (0,) if count == 0 else (3,) if count == 2 else (1, 2)
        price = q['lambda'] * (q['x'][i+1]-q['x'][0])
        ua, ub = q['u'][i][a], q['u'][i][b]
        ca, cb = price*q['bits'][a], price*q['bits'][b]
        def contribution(m, s):
            return (ua - (0 if s & 1 else ca) if m & 1 else 0) + (ub - (0 if s & 2 else cb) if m & 2 else 0)
        old_value += contribution(old, later)
        later |= old
        nxt = {}
        for s, (v, tail) in states.items():
            for m in choices:
                union = s | m
                z = v + contribution(m, s)
                if union not in nxt or z > nxt[union][0]:
                    nxt[union] = z, (m, *tail)
        states = nxt
    best, seq = max(states.values(), key=lambda z: z[0])
    if best <= old_value:
        return list(schedule), 0
    result = [(m & ~(ma | mb)) | (ma if z & 1 else 0) | (mb if z & 2 else 0)
              for m, z in zip(schedule, seq, strict=True)]
    return result, best-old_value


def pair_repair(q, schedule, cap=12):
    """Try at most cap pairs prioritized by utility and cost of label substitution."""
    if type(cap) is not int or not 0 <= cap <= 128:
        raise TaskError('invalid pair cap')
    masks = visit_masks(q, schedule)
    # Only pairs with different occupancy can change a schedule. Rank by
    # the objective of swapping their entire histories (a feasible move).
    x, u, lam, bits = q['x'], q['u'], q['lambda'], q['bits']
    ranks = []
    for a in range(q['d']):
        for b in range(a+1, q['d']):
            diff = masks[a] ^ masks[b]
            if not diff:
                continue
            v = sum((u[i][b]-u[i][a]) if masks[a] & (1 << i) else (u[i][a]-u[i][b])
                    for i in indices(diff))
            v += lam*(bits[a]-bits[b])*(x[masks[a].bit_length()]-x[masks[b].bit_length()])
            ranks.append((-v, a, b))
    result = list(schedule)
    gain = 0
    for _, a, b in sorted(ranks)[:cap]:
        result, delta = pair_best(q, result, a, b)
        gain += delta
    return result, gain, min(cap, len(ranks))


def recover(task, initial, eta, method):
    """Return a verified feasible incumbent; preserve baseline if no gain."""
    q = validate(task)
    base = list(initial)
    objective = score(q, base)['objective']
    result = base
    if method not in ('base', 'row', 'dual', 'dual_row', 'pair'):
        raise TaskError('invalid repair method')
    if method in ('dual', 'dual_row', 'pair'):
        rounded = dual_round(q, eta)
        if score(q, rounded)['objective'] > objective:
            result = rounded
    if method in ('row', 'dual_row', 'pair'):
        result, _ = row_sweep(q, result)
        result, _ = row_sweep(q, result, False)
    if method == 'pair':
        result, _, _ = pair_repair(q, result)
        result, _ = row_sweep(q, result)
    value = score(q, result)['objective']
    if value < objective:
        raise AssertionError('repair worsened incumbent')
    return result, value-objective
