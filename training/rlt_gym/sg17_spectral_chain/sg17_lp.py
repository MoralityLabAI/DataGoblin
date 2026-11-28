"""Numerical proposal generators for SG-17; never trusted as proofs."""
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from datagoblin.training_gym.spectral_face import exact_bound, rational_vector
from datagoblin.training_gym.spectral_quotient import indices, score, validate
from datagoblin.training_gym.spectral_routing import TaskError


def chain_lp(task):
    """Primal LP: read y <= retained h, with h nonincreasing in visit time."""
    q = validate(task)
    t, d = q['T'], q['d']
    n = t*d
    j = np.arange(n, dtype=np.int64)
    m = n+(t-1)*d
    rows = np.repeat(np.arange(m), 2)
    cols = np.concatenate((np.column_stack((j, n+j)).ravel(),
                           np.column_stack((n+j[d:], n+j[:-d])).ravel()))
    a = coo_matrix((np.tile([1., -1.], m), (rows, cols)), shape=(m, 2*n)).tocsr()
    eq = coo_matrix((np.ones(n), (j//d, j)), shape=(t, 2*n)).tocsr()
    gaps = np.diff(q['x'])
    c = np.concatenate((-np.asarray(q['u'], dtype=float).ravel(),
                        (q['lambda']*gaps[:, None]*np.asarray(q['bits'])).ravel()))
    z = linprog(c, A_ub=a, b_ub=np.zeros(m), A_eq=eq, b_eq=q['k'],
                bounds=(0, 1), method='highs', options={'time_limit': 2.0})
    info = dict(status=int(z.status), variables=2*n, rows=m+t, nnz=a.nnz+eq.nnz,
                iterations=int(z.nit), objective=float(-z.fun) if z.success else None)
    # SciPy solves minimization. Its row-RHS marginals have the opposite sign
    # to the multipliers in the maximization certificate.
    return (rational_vector(-z.eqlin.marginals) if z.success else None), info


def best_mask(q, j, eta, den):
    """Exact separation: maximum adjusted utility over all masks of one label."""
    best = prefix = mask = chosen = 0
    for i, row in enumerate(q['u']):
        v = den*row[j]-eta[i]
        price = den*q['lambda']*q['bits'][j]*(q['x'][i+1]-q['x'][0])
        value = prefix+v-price
        if value > best:
            best, chosen = value, mask | (1 << i)
        if v > 0:
            prefix += v
            mask |= 1 << i
    return chosen, best


def cut_lp(task, initial, root, *, round_cap=8):
    """Restricted dual LP with exact separation; any returned vector is safe."""
    q = validate(task)
    if type(round_cap) is not int or not 1 <= round_cap <= 32:
        raise TaskError('invalid LP round cap')
    incumbent = score(q, initial)['objective']
    t, d = q['T'], q['d']
    if not isinstance(root, list) or len(root) != t or any(type(v) is not int for v in root):
        raise TaskError('invalid seed vector')
    domains = [set() for _ in range(d)]
    # One legal primal schedule ensures the restricted LP is bounded below.
    for j in range(d):
        mask = sum(1 << i for i, m in enumerate(initial) if m & (1 << j))
        domains[j].update((mask, (1 << t)-1, best_mask(q, j, root, 1)[0]))
        domains[j].discard(0)  # w_j >= 0 is the empty-mask constraint.
    c = np.asarray(q['k']+[1]*d, dtype=float)
    best = (list(root), 1)
    best_value = Fraction(exact_bound(q, *best), 1)
    records = []
    stop = 'round_cap'
    for _ in range(round_cap):
        rr, cc, vv, rhs = [], [], [], []
        for j, masks in enumerate(domains):
            for mask in sorted(masks):
                r = len(rhs)
                ids = list(indices(mask))
                rhs.append(q['lambda']*q['bits'][j]*(q['x'][mask.bit_length()]-q['x'][0])
                           -sum(q['u'][i][j] for i in ids))
                rr.extend([r]*(len(ids)+1))
                cc.extend([*ids, t+j])
                vv.extend([-1.]*(len(ids)+1))
        a = coo_matrix((vv, (rr, cc)), shape=(len(rhs), t+d)).tocsr()
        z = linprog(c, A_ub=a, b_ub=rhs, bounds=[(None, None)]*t+[(0, None)]*d,
                    method='highs', options={'time_limit': 2.0})
        ev = dict(status=int(z.status), rows=len(rhs), nnz=a.nnz, iterations=int(z.nit))
        records.append(ev)
        if not z.success:
            stop = 'solver_failure'
            break
        eta, den = rational_vector(z.x[:t])
        value = Fraction(exact_bound(q, eta, den), den)
        if value < best_value:
            best, best_value = (eta, den), value
        added = 0
        for j in range(d):
            mask, _ = best_mask(q, j, eta, den)
            if mask and mask not in domains[j]:
                domains[j].add(mask)
                added += 1
        ev.update(added=added, upper_num=value.numerator, upper_den=value.denominator)
        if best_value.numerator//best_value.denominator == incumbent:
            stop = 'floor'
            break
        if not added:
            stop = 'no_new_mask'
            break
    return best, dict(stop=stop, variables=t+d, rounds=len(records), records=records,
                      rows=max((e['rows'] for e in records), default=0),
                      nnz=max((e['nnz'] for e in records), default=0),
                      iterations=sum(e['iterations'] for e in records))
