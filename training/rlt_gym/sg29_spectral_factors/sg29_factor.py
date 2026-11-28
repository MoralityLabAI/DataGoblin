"""Exact regret screening and separator-conditioned additive certificates."""
from itertools import combinations, product
from math import prod

from datagoblin.training_gym.spectral_quotient import indices, score, validate
from datagoblin.training_gym.spectral_routing import TaskError

CAP = 100_000


def column(q, j, eta, den):
    return ([den * row[j] - e for row, e in zip(q['u'], eta, strict=True)],
            [den * q['lambda'] * q['bits'][j] * (x - q['x'][0]) for x in q['x']])


def value(z, p, m):
    return sum(z[i] for i in indices(m)) - p[m.bit_length()]


def best(z, p, i=None, include=False):
    """Maximum mask value with one optional fixed read, via last-use scan."""
    prefix = 0
    out = None if include else 0
    for l, v in enumerate(z):
        if i != l or include:
            if not include or i is None or l >= i:
                w = prefix + v - p[l + 1]
                out = w if out is None else max(out, w)
        prefix += (v if include else 0) if i == l else max(0, v)
    return out


def ref_best(z, p, i=None, include=False):
    """Independent explicit last-use formula for screening replay."""
    vals = [] if include else [0]
    for l in range(len(z)):
        if (i == l and not include) or (include and l < i):
            continue
        vals.append(z[l] - p[l + 1] + sum(
            (z[h] if include else 0) if h == i else max(0, z[h]) for h in range(l)))
    return max(vals)


def setup(q, initial, eta, den, ref=False):
    if type(den) is not int or not 1 <= den <= 1_000_000 or not isinstance(eta, list) or len(eta) != q['T'] or any(type(e) is not int or abs(e) > 10**15 for e in eta):
        raise TaskError('invalid rational vector')
    L = score(q, initial)['objective']
    cols = [column(q, j, eta, den) for j in range(q['d'])]
    bound = ref_best if ref else best
    maxima = [bound(z, p) for z, p in cols]
    B = sum(e * k for e, k in zip(eta, q['k'], strict=True)) + sum(maxima)
    H = B - den * (L + 1)
    if B < den * L:
        raise TaskError('invalid incumbent bound')
    must, may = [], []
    if H >= 0:
        for (z, p), M in zip(cols, maxima, strict=True):
            a = b = 0
            for i in range(q['T']):
                if M - bound(z, p, i, False) > H:
                    a |= 1 << i
                if M - bound(z, p, i, True) <= H:
                    b |= 1 << i
            assert not a & ~b
            must.append(a); may.append(b)
    residual = [k - sum(bool(a & (1 << i)) for a in must) for i, k in enumerate(q['k'])]
    return L, cols, maxima, B, H, must, may, residual


def partition(must, may, t, sep):
    """Connected components of free-read incidence graph, minus separator rows."""
    d = len(must); remaining = set(range(d + t)) - {d+i for i in sep}
    adj = {v: set() for v in remaining}
    for j, (a, b) in enumerate(zip(must, may, strict=True)):
        for i in indices(b ^ a):
            if i not in sep:
                adj[j].add(d+i); adj[d+i].add(j)
    parts = []
    while remaining:
        stack = [min(remaining)]; seen = set()
        while stack:
            v = stack.pop()
            if v not in seen:
                seen.add(v); stack.extend(adj[v] - seen)
        remaining -= seen
        parts.append((sorted(v for v in seen if v < d), sorted(v-d for v in seen if v >= d)))
    return parts


def estimate(js, rows, sep, must, may, residual):
    axes = sorted(set(rows) | {i for i in sep if any((may[j]^must[j]) & (1<<i) for j in js)})
    return max(1, sum(1 << (may[j]^must[j]).bit_count() for j in js)) * prod(residual[i]+1 for i in axes)


def choose(must, may, residual, max_sep, cap):
    t = len(residual); options = []
    # A conservative operation proxy; enumerate only zero/one/two row separators.
    for n in range(max_sep+1):
        for sep in combinations(range(t), n):
            parts = partition(must, may, t, sep)
            costs = [estimate(js, rows, sep, must, may, residual) for js, rows in parts]
            combine = len(parts) * prod(residual[i]+1 for i in sep)**2
            if combine > cap:
                continue
            options.append((sum(c for c in costs if c > cap) + max(0, sum(costs)+combine-cap),
                            sum(costs)+combine, n, sep, parts, costs, combine))
    return min(options, key=lambda x: x[:4]) if options else None


def table(js, rows, sep, must, may, residual, cols, maxima, H, ref=False):
    axes = sorted(set(rows) | {i for i in sep if any((may[j]^must[j]) & (1<<i) for j in js)})
    limits = tuple(residual[i] for i in axes); pos = {i: n for n, i in enumerate(axes)}
    dp = {tuple(0 for _ in axes): (0, ())}
    for j in reversed(js) if ref else js:
        free = may[j] ^ must[j]; options = []
        if ref:
            bits = list(indices(free))
            subsets = (sum(1<<i for i, on in zip(bits, flags, strict=True) if on) for flags in product((0,1), repeat=len(bits)))
        else:
            def masks():
                s = free
                while True:
                    yield s
                    if not s: break
                    s = (s-1)&free
            subsets = masks()
        z, p = cols[j]
        for s in subsets:
            m = must[j] | s
            r = maxima[j] - (sum(z[i] for i in range(len(z)) if m&(1<<i)) - p[m.bit_length()] if ref else value(z,p,m))
            if r <= H:
                options.append((tuple(int(bool(s&(1<<i))) for i in axes), r, m))
        nxt = {}
        for counts, (v, path) in dp.items():
            for add, r, m in options:
                w = v+r
                if w > H: continue
                c = tuple(a+b for a,b in zip(counts,add,strict=True))
                if any(a>b for a,b in zip(c,limits,strict=True)): continue
                if c not in nxt or w < nxt[c][0]:
                    nxt[c] = (w, (*path,(j,m)))
        dp = nxt
    out = {}
    for counts, (v,path) in dp.items():
        if any(counts[pos[i]] != residual[i] for i in rows): continue
        key = tuple(counts[pos[i]] if i in pos else 0 for i in sep)
        if key not in out or v < out[key][0]: out[key] = (v,path)
    return out


def relaxed(sep, residual):
    return {c: (0, ()) for c in product(*(range(residual[i]+1) for i in sep))}


def combine(tables, sep, residual, H):
    target = tuple(residual[i] for i in sep); dp = {tuple(0 for _ in sep): (0, ())}
    for tab in tables:
        nxt = {}
        for a,(v,path) in dp.items():
            for b,(w,tail) in tab.items():
                c = tuple(x+y for x,y in zip(a,b,strict=True)); cost = v+w
                if cost > H or any(x>y for x,y in zip(c,target,strict=True)): continue
                if c not in nxt or cost<nxt[c][0]: nxt[c]=(cost,(*path,*tail))
        dp = nxt
    return dp.get(target)


def encode(tab):
    return [[list(k),v[0]] for k,v in sorted(tab.items())]


def capcheck(cap, max_sep):
    if type(cap) is not int or not 1 <= cap <= 1_000_000 or type(max_sep) is not int or not 0 <= max_sep <= 2:
        raise TaskError('invalid factor cap')


def solve(task, initial, eta, den, *, cap=CAP, max_sep=1):
    capcheck(cap,max_sep); q=validate(task)
    L,cols,M,B,H,must,may,res=setup(q,initial,eta,den)
    proof=dict(initial=list(initial),eta=list(eta),den=den,bound_num=B,threshold=H,must=must,may=may,separator=[],parts=[])
    schedule=list(initial); upper=B//den; work=0; status='floor'
    if H>=0:
        if any(k<0 or k>sum(bool((a^b)&(1<<i)) for a,b in zip(must,may,strict=True)) for i,k in enumerate(res)):
            upper=L; status='screen_infeasible'
        else:
            choice=choose(must,may,res,max_sep,cap)
            if choice is None:
                return dict(schedule=schedule,objective=L,upper=upper,gap=upper-L,proof=proof,work=0,status='skipped')
            _,_,_,sep,parts,costs,reserve=choice
            proof['separator']=list(sep); work=reserve; tabs=[]; complete=True
            for (js,rows),cost in zip(parts,costs,strict=True):
                if work+cost<=cap:
                    tab=table(js,rows,sep,must,may,res,cols,M,H); work+=cost; encoded=encode(tab)
                else:
                    tab=relaxed(sep,res); complete=False; encoded=None
                proof['parts'].append(dict(labels=js,rows=rows,table=encoded)); tabs.append(tab)
            ans=combine(tabs,sep,res,H)
            if ans is None: upper=L
            else:
                upper=max(L,(B-ans[0])//den)
                if complete:
                    candidate=[0]*q['T']
                    for j,m in ans[1]:
                        for i in indices(m): candidate[i] |= 1<<j
                    obj=score(q,candidate)['objective']; assert den*obj==B-ans[0]
                    if obj>L: schedule=candidate
            status='exact' if complete else 'partial'
    objective=score(q,schedule)['objective']; assert objective<=upper
    return dict(schedule=schedule,objective=objective,upper=upper,gap=upper-objective,proof=proof,work=work,status=status)


def replay(task, result, *, cap=CAP, max_sep=1):
    """Replay all arithmetic; a missing table only relaxes constraints."""
    capcheck(cap,max_sep); q=validate(task); p=result['proof']
    L,cols,M,B,H,must,may,res=setup(q,p['initial'],p['eta'],p['den'],ref=True)
    if (p['bound_num'],p['threshold'],p['must'],p['may'])!=(B,H,must,may): raise TaskError('screen mismatch')
    upper=B//p['den']; sep=p['separator']; parts=p['parts']; work=0
    bad=H>=0 and any(k<0 or k>sum(bool((a^b)&(1<<i)) for a,b in zip(must,may,strict=True)) for i,k in enumerate(res))
    if H<0 or bad:
        if sep or parts: raise TaskError('unexpected factors')
        if bad: upper=L
    elif not parts and not sep:
        # An empty factor proof makes no claim beyond the global bound.
        pass
    else:
        if not isinstance(sep,list) or len(sep)>max_sep or any(type(i) is not int or not 0<=i<q['T'] for i in sep) or sep!=sorted(set(sep)):
            raise TaskError('invalid separator')
        if not isinstance(parts,list) or len(parts)>q['d']+q['T']: raise TaskError('invalid factors')
        seen_j=set();seen_i=set();tabs=[];work=len(parts)*prod(res[i]+1 for i in sep)**2
        if work>cap: raise TaskError('convolution cap')
        for part in parts:
            js,rows,claim=part['labels'],part['rows'],part['table']
            if not isinstance(js,list) or not isinstance(rows,list) or any(type(j) is not int or not 0<=j<q['d'] for j in js) or any(type(i) is not int or not 0<=i<q['T'] for i in rows): raise TaskError('invalid factor indices')
            if len(set(js))!=len(js) or len(set(rows))!=len(rows) or seen_j.intersection(js) or seen_i.intersection(rows) or set(sep).intersection(rows) or not (js or rows): raise TaskError('overlapping factors')
            seen_j.update(js);seen_i.update(rows)
            for j in js:
                if set(indices(may[j]^must[j]))-set(rows)-set(sep): raise TaskError('lost coupling')
            if claim is None: tab=relaxed(sep,res)
            else:
                work+=estimate(js,rows,sep,must,may,res)
                if work>cap: raise TaskError('factor cap')
                tab=table(js,rows,sep,must,may,res,cols,M,H,ref=True)
                if encode(tab)!=claim: raise TaskError('table mismatch')
            tabs.append(tab)
        if seen_j!=set(range(q['d'])) or seen_i!=set(range(q['T']))-set(sep): raise TaskError('incomplete partition')
        ans=combine(tabs,sep,res,H);upper=L if ans is None else max(L,(B-ans[0])//p['den'])
    objective=score(q,result['schedule'])['objective']
    if objective<L or objective>upper or (result['objective'],result['upper'],result['gap'],result['work'])!=(objective,upper,upper-objective,work): raise TaskError('receipt mismatch')
    return dict(objective=objective,upper=upper,gap=upper-objective,work=work)
