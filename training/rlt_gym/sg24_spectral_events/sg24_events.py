"""Exact continuous crossing of a fixed schedule's affine dual-gap envelope."""
from collections import defaultdict
from fractions import Fraction

from sg20_delta import State
from datagoblin.training_gym.spectral_routing import TaskError


def lines(z, dz, p, dp, horizon):
    """Generate sufficient genuine last-use lines on [0,horizon]."""
    breaks = {Fraction(0), Fraction(horizon)}
    for a, b in zip(z, dz, strict=True):
        if b:
            x = Fraction(-a, b)
            if 0 < x < horizon:
                breaks.add(x)
    stops = sorted(breaks)
    out = {(0, 0)}
    for a, b in zip(stops, stops[1:]):
        mid = (a+b)/2
        c = v = 0
        for i, (x, y) in enumerate(zip(z, dz, strict=True)):
            out.add((c+x-p[i+1], v+y-dp[i+1]))
            if x*mid.denominator+y*mid.numerator > 0:
                c += x; v += y
    return out


def hull(candidates):
    """Upper envelope of affine (intercept,slope) pairs; starts may be negative."""
    by_slope = {}
    for a, b in candidates:
        by_slope[b] = max(a, by_slope.get(b, a))
    stack = []
    for b, a in sorted(by_slope.items()):
        while stack:
            x, y, start = stack[-1]
            cross = Fraction(x-a, b-y)
            if start is None or cross > start:
                break
            stack.pop()
        stack.append((a, b, cross if stack else None))
    return stack


def envelope_cross(terms, horizon, threshold):
    """First point where aggregate nonnegative regret reaches threshold.

    Positive threshold uses a strict certified sublevel; threshold zero instead
    returns the last point of the initial exact-face plateau. None means no
    crossing before or at the horizon (for zero, no positive regret there).
    """
    if horizon <= 0 or threshold < 0:
        raise TaskError('invalid crossing domain')
    events = defaultdict(lambda: [0, 0])
    intercept = slope = count = 0
    for candidates in terms:
        old = None
        for a, b, start in hull(candidates):
            if start is None or start <= 0:
                old = (a, b)
                continue
            if start > horizon:
                break
            assert old is not None
            da, db = a-old[0], b-old[1]
            events[start][0] += da; events[start][1] += db
            old = (a, b)
        # Locate the active initial line (right derivative at zero).
        a, b = max(candidates, key=lambda x: (x[0], x[1]))
        intercept += a; slope += b; count += len(candidates)
    if intercept > threshold or (threshold and intercept == threshold):
        raise TaskError('uncertified crossing anchor')
    for end in sorted(set(events) | {Fraction(horizon)}):
        value = intercept+slope*end
        if (value > threshold if threshold == 0 else value >= threshold):
            assert slope > 0
            return Fraction(threshold-intercept, slope), count, len(events)
        da, db = events.get(end, (0, 0))
        intercept += da; slope += db
    return None, count, len(events)


def region(ray, state, start, line):
    """Maximal integer floor certificate, computed from the first exact crossing."""
    if state.q != ray.at(start) or not state.certificate()['certified']:
        raise TaskError('invalid anchor')
    eta, den = line.at(start)
    if len(eta) != state.q.t or any(a*den != b*state.den for a,b in zip(state.eta,eta,strict=True)):
        raise TaskError('line misses anchor')
    line.at(ray.horizon)
    base, rem = divmod(state.bound*den, state.den)
    if rem:
        raise TaskError('nonintegral anchor bound')
    H = ray.horizon-start
    q = state.q
    final = ray.at(ray.horizon)
    # H=0 is already a complete certificate.
    if not H:
        return dict(start=start,end=start,origin=line.origin,eta=list(line.eta),slope=list(line.slope),den=den,
                    schedule=list(state.schedule),bounds=[base,base],objectives=[state.objective]*2,
                    probes=0,crossing=None,lines=0,events=0)
    # One full endpoint evaluation can avoid envelope construction entirely.
    far = State.build(final,line.at(ray.horizon)[0],den,state.schedule)
    probes = 1; cross = None; count = events = 0
    if far.certificate()['certified']:
        end = ray.horizon; last = far
    else:
        terms = []
        for j, mask in enumerate(state.visits):
            old_gap = state.tables[j][0] - (state.den*state.primal[j]-sum(state.eta[i] for i in range(q.t) if mask & (1<<i)))
            new_gap = far.tables[j][0] - (den*far.primal[j]-sum(far.eta[i] for i in range(q.t) if mask & (1<<i)))
            # Convex nonnegative regret with zero at both endpoints is identically zero.
            if old_gap == new_gap == 0:
                continue
            z = [den*u-e for u,e in zip(q.columns[j],eta,strict=True)]
            dz = [den*((v-u)//H)-e for u,v,e in zip(q.columns[j],final.columns[j],line.slope,strict=True)]
            p = [den*q.lam*q.bits[j]*(x-q.x[0]) for x in q.x]
            dp = [(den*final.lam*final.bits[j]*(x-final.x[0])-v)//H for v,x in zip(p,final.x,strict=True)]
            ca = sum(z[i] for i in range(q.t) if mask&(1<<i))-p[mask.bit_length()]
            cb = sum(dz[i] for i in range(q.t) if mask&(1<<i))-dp[mask.bit_length()]
            terms.append({(a-ca,b-cb) for a,b in lines(z,dz,p,dp,H)})
        cross,count,events = envelope_cross(terms,H,den)
        assert cross is not None and 0 < cross <= H
        # Gap equality to 1 is not a floor certificate, even at an integer crossing.
        end = start+(cross.numerator-1)//cross.denominator
        if end==start:
            last = state
        else:
            last = State.build(ray.at(end),line.at(end)[0],den,state.schedule); probes+=1
        assert last.certificate()['certified']
    last_bound, rem = divmod(last.bound*den,last.den)
    assert not rem
    return dict(start=start,end=end,origin=line.origin,eta=list(line.eta),slope=list(line.slope),den=den,
                schedule=list(state.schedule),bounds=[base,last_bound],objectives=[state.objective,last.objective],
                probes=probes,crossing=None if cross is None else [cross.numerator,cross.denominator],lines=count,events=events)
