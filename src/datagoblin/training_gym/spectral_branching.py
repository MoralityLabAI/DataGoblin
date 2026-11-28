"""SG-13: exact one-step branching scores and optional learned policies."""
from math import log1p

from datagoblin.training_gym.spectral_quotient import indices


def free_bits(q, b):
    return ((1 << q['d']) - 1) & ~(b.suffix | b.include | b.exclude)


def child_bounds(session, b):
    """Exact conditioned child upper bounds via one replacement per multiplier.

    For a normalized unresolved block, the remaining cardinality lies strictly
    between zero and the number of allowed components. Normalizing either child
    preserves its feasible current actions. The result includes inherited caps.
    """
    js = list(indices(free_bits(session.q, b)))
    if b.t <= 1 or not js:
        return {}
    r, e = b.include, b.exclude
    k = session.q['k'][b.t - 1] - r.bit_count()
    out = {j: [b.upper, b.upper] for j in js}
    for base, delta, order in session.relax.context(b.t, b.suffix):
        allowed = [j for j in order if not (r | e) & (1 << j)]
        assert 0 < k < len(allowed)
        selected = set(allowed[:k])
        value = b.value + base + sum(delta[j] for j in indices(r)) + sum(delta[j] for j in selected)
        last, next_ = delta[allowed[k - 1]], delta[allowed[k]]
        for j in js:
            inc = value if j in selected else value + delta[j] - last
            exc = value - delta[j] + next_ if j in selected else value
            out[j][0] = min(out[j][0], inc)
            out[j][1] = min(out[j][1], exc)
    return {j: tuple(v) for j, v in out.items()}


def gains(session, b, pair):
    return tuple(b.upper - max(session.incumbent, v) for v in pair)


def rank_score(gain):
    a, c = gain
    return (1 + a) * (1 + c)


def feature_rows(session, b):
    """Cheap per-component features using just the final relaxation vector."""
    q = session.q
    r, e = b.include, b.exclude
    k = q['k'][b.t - 1] - r.bit_count()
    _, delta, order = session.relax.context(b.t, b.suffix)[-1]
    allowed = [j for j in order if not (r | e) & (1 << j)]
    gap = max(1, b.upper - session.incumbent)
    last, next_ = delta[allowed[k - 1]], delta[allowed[k]]
    rows = {}
    for rank, j in enumerate(allowed):
        if not free_bits(q, b) & (1 << j):
            continue
        margin = delta[j] - (next_ if rank < k else last)
        rows[j] = (1., rank / q['d'], float(rank < k), min(4., abs(margin) / gap),
                   delta[j] / 32, q['u'][b.t - 1][j] / 16, q['bits'][j] / 3,
                   b.t / q['T'], b.suffix.bit_count() / q['d'], k / len(allowed),
                   min(4., gap / 16), min(4., abs(margin) / 8))
    return rows


class BranchPolicy:
    """Policy scores only select a split; the SG-12 engine proves the result."""
    def __init__(self, kind, beta=()):
        if kind not in ('analytic', 'strong', 'pseudo', 'learned', 'shuffled'):
            raise ValueError('unknown policy')
        if kind in ('learned', 'shuffled') and len(beta) != 12:
            raise ValueError('expected twelve coefficients')
        self.kind, self.beta = kind, tuple(beta)
        self.history = {}
        self.probes = 0

    def choose(self, session, b):
        analytic = session.relax.choose(b)
        if self.kind == 'analytic':
            return analytic
        if self.kind == 'strong':
            bounds = child_bounds(session, b)
            self.probes += 2 * len(bounds)
            return max(bounds, key=lambda j: (rank_score(gains(session, b, bounds[j])), j == analytic, -j))
        rows = feature_rows(session, b)
        if self.kind == 'pseudo':
            def value(j):
                n, a, c = self.history.get(j, (0, 0., 0.))
                # One prior observation per side; a cheap margin guides cold starts.
                margin = min(1., rows[j][3])
                return (1 + (a + margin) / (n + 1)) * (1 + (c + margin) / (n + 1))
        else:
            def value(j):
                return sum(a * x for a, x in zip(self.beta, rows[j], strict=True))
        return max(rows, key=lambda j: (value(j), j == analytic, -j))

    def advance(self, session, ident, j):
        if self.kind != 'pseudo' or j is None:
            session.advance(ident, j)
            return
        b = session.live[ident]
        begin = session.next_id
        gap = max(1, b.upper - session.incumbent)
        session.advance(ident, j)
        # Pruned children have certified upper <= incumbent. Read the two actual
        # child records directly, with no extra relaxation evaluations.
        ub = [session.incumbent, session.incumbent]
        for i in range(begin, session.next_id):
            c = session.live.get(i)
            if c is not None:
                side = int(bool(c.exclude & (1 << j)))
                ub[side] = max(session.incumbent, c.upper)
        n, a, c = self.history.get(j, (0, 0., 0.))
        self.history[j] = (n + 1, a + (b.upper - ub[0]) / gap, c + (b.upper - ub[1]) / gap)


def teacher_rows(session, b):
    bounds = child_bounds(session, b)
    rows = feature_rows(session, b)
    return [(rows[j], log1p(rank_score(gains(session, b, bounds[j])) - 1)) for j in rows]
