"""SG-11: sparse exact union-class search with separate raw and quotient budgets."""
from itertools import combinations
from math import comb

from datagoblin.training_gym.spectral_routing import TaskError
from datagoblin.training_gym.spectral_search import Node


def validate(task):
    def integer(v, lo, hi):
        if type(v) is not int or not lo <= v <= hi:
            raise TaskError('integer outside registered bounds')
        return v

    def vector(v, n, lo, hi):
        if not isinstance(v, list) or len(v) != n:
            raise TaskError('invalid vector')
        return [integer(a, lo, hi) for a in v]

    try:
        d, t = integer(task['d'], 1, 32), integer(task['T'], 1, 16)
        x = vector(task['x'], t + 1, -1000, 1000)
        if x != sorted(x) or not isinstance(task['u'], list) or len(task['u']) != t:
            raise TaskError('invalid coordinates or utility shape')
        return dict(d=d, T=t, x=x, bits=vector(task['bits'], d, 1, 100),
                    k=vector(task['k'], t, 0, d), u=[vector(a, d, -10000, 10000) for a in task['u']],
                    **{'lambda': integer(task['lambda'], 0, 100)})
    except (KeyError, TypeError) as exc:
        raise TaskError('malformed task') from exc


def indices(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def score(q, schedule):
    if not isinstance(schedule, list) or len(schedule) != q['T']:
        raise TaskError('invalid schedule')
    last, utility = [q['x'][0]] * q['d'], 0
    for t, m in enumerate(schedule):
        if type(m) is not int or not 0 <= m < 1 << q['d'] or m.bit_count() != q['k'][t]:
            raise TaskError('invalid action mask')
        for j in indices(m):
            last[j] = q['x'][t + 1]
            utility += q['u'][t][j]
    cost = sum(b * (v - q['x'][0]) for b, v in zip(q['bits'], last, strict=True))
    return dict(utility=utility, cost=cost, objective=utility-q['lambda']*cost)


def quotient_count(d, k, suffix):
    a = suffix.bit_count()
    return sum(comb(d-a, r) for r in range(max(0, k-a), min(k, d-a)+1))


def quotient(row, k, suffix):
    """Yield (union, earliest maximizing action, utility); no raw enumeration."""
    inside = sorted(indices(suffix), key=lambda j: (-row[j], j))
    outside = [j for j in range(len(row)) if not suffix & (1 << j)]
    masks, values = [0], [0]
    for j in inside[:k]:
        masks.append(masks[-1] | (1 << j))
        values.append(values[-1] + row[j])
    for r in range(max(0, k-len(inside)), min(k, len(outside))+1):
        for js in combinations(outside, r):
            extra = sum(1 << j for j in js)
            yield suffix | extra, masks[k-r] | extra, values[k-r] + sum(row[j] for j in js)


def dual(q, t, s, eta):
    value = sum(eta[i]*q['k'][i] for i in range(t))
    for j, bits in enumerate(q['bits']):
        a = b = 0
        for i in range(t):
            z = q['u'][i][j]-eta[i]
            b = max(b, a+z-q['lambda']*bits*(q['x'][i+1]-q['x'][0]))
            a += max(0,z)
        value += a if s & (1 << j) else b
    return value


def coordinate(q,t,s,e,i):
    thresholds = []
    for j,bits in enumerate(q['bits']):
        ui = q['u'][i][j]
        if s & (1 << j):
            thresholds.append(ui)
            continue
        a = b = 0
        include = ui + sum(max(0,q['u'][l][j]-e[l]) for l in range(i))
        include -= q['lambda']*bits*(q['x'][i+1]-q['x'][0])
        for l in range(t):
            if l == i:
                continue
            z = q['u'][l][j]-e[l]
            v = a+z-q['lambda']*bits*(q['x'][l+1]-q['x'][0])
            b = max(b,v)
            if l > i:
                include = max(include,v+ui)
            a += max(0,z)
        thresholds.append(include-b)
    thresholds.sort(reverse=True)
    k = q['k'][i]
    return thresholds[0] if k == 0 else thresholds[-1] if k == q['d'] else (thresholds[k-1]+thresholds[k])//2


class SparseSession:
    """Complete quotient expansions. No dense mask table or action list."""
    def __init__(self, task, initial, *, class_cap=16384, raw_cap=None, proposal_cap=24):
        self.q = validate(task)
        if type(class_cap) is not int or not 1 <= class_cap <= 1_000_000:
            raise TaskError('invalid class cap')
        if raw_cap is not None and (type(raw_cap) is not int or raw_cap < 1):
            raise TaskError('invalid raw cap')
        if type(proposal_cap) is not int or not 0 <= proposal_cap <= 10000:
            raise TaskError('invalid proposal cap')
        self.class_cap,self.raw_cap,self.proposal_cap = class_cap,raw_cap,proposal_cap
        q = self.q
        self.schedule = list(initial)
        self.incumbent = score(q,self.schedule)['objective']
        self.transitions = self.classes = self.proposals = 0
        self.events,self.trace = [],[]
        local = [sorted(r,reverse=True) for r in q['u']]
        cut = [r[k-1] if k else r[0]+1 for r,k in zip(local,q['k'],strict=True)]
        self.weak = [0]
        for row,k in zip(local,q['k'],strict=True):
            self.weak.append(self.weak[-1]+sum(row[:k]))
        self.tables = [[] for _ in range(q['T']+1)]
        for eta in [[0]*q['T'], *[[v+shift for v in cut] for shift in (-6,-3,0,3,6)]]:
            a,b,c = [0]*q['d'],[0]*q['d'],0
            for t in range(q['T']+1):
                if t:
                    i = t-1
                    c += eta[i]*q['k'][i]
                    for j,bit in enumerate(q['bits']):
                        z = q['u'][i][j]-eta[i]
                        b[j] = max(b[j],a[j]+z-q['lambda']*bit*(q['x'][t]-q['x'][0]))
                        a[j] += max(0,z)
                self.tables[t].append((c+sum(b),tuple(v-w for v,w in zip(a,b,strict=True))))
        self.overlay = {}
        self.live = {(q['T'],0): Node(0,(),self.bound(q['T'],0))}
        self.seen = {(q['T'],0):0}
        self._prune()

    def bound(self,t,s):
        if (t,s) in self.overlay:
            return self.overlay[t,s]
        js = tuple(indices(s))
        return min(self.weak[t],*(base+sum(delta[j] for j in js) for base,delta in self.tables[t]))

    def _prune(self):
        self.live = {key:n for key,n in self.live.items() if n.upper > self.incumbent}

    def frontier(self):
        return [(t,s,n.upper) for (t,s),n in sorted(self.live.items())]

    def charges(self,t,s):
        k = self.q['k'][t-1]
        return comb(self.q['d'],k), 1 if t == 1 else quotient_count(self.q['d'],k,s)

    def can_expand(self,t,s):
        raw,classes = self.charges(t,s)
        return self.classes+classes <= self.class_cap and (self.raw_cap is None or self.transitions+raw <= self.raw_cap)

    def advance(self,t,s):
        if type(t) is not int or type(s) is not int or (t,s) not in self.live:
            raise TaskError('state is not live')
        if not self.can_expand(t,s):
            raise TaskError('expansion cap would be exceeded')
        raw,classes = self.charges(t,s)
        self.transitions += raw
        self.classes += classes
        self.trace.append([t,s])
        self.events.append(dict(kind='expand',state=[t,s]))
        node = self.live.pop((t,s))
        q = self.q
        k,row = q['k'][t-1],q['u'][t-1]
        price = q['lambda']*(q['x'][t]-q['x'][0])
        if t == 1:
            adj = [v-(0 if s & (1 << j) else price*q['bits'][j]) for j,v in enumerate(row)]
            js = sorted(range(q['d']),key=lambda j:(-adj[j],j))[:k]
            value = node.value+sum(adj[j] for j in js)
            if value > self.incumbent:
                self.schedule = [sum(1 << j for j in js),*node.tail]
                if score(q,self.schedule)['objective'] != value:
                    raise AssertionError('accounting')
                self.incumbent = value
        else:
            for union,m,u in quotient(row,k,s):
                g = node.value+u-price*sum(q['bits'][j] for j in indices(union & ~s))
                key = (t-1,union)
                if key in self.seen and g <= self.seen[key]:
                    continue
                ub = g+self.bound(*key)
                if ub > self.incumbent:
                    self.seen[key] = g
                    self.live[key] = Node(g,(m,*node.tail),ub)
        self._prune()

    def propose(self,t,s,eta):
        if type(t) is not int or type(s) is not int or (t,s) not in self.live or self.proposals >= self.proposal_cap:
            raise TaskError('invalid state or proposal cap')
        if not isinstance(eta,list) or len(eta) != t or any(type(v) is not int or abs(v)>100_000_000 for v in eta):
            raise TaskError('invalid multipliers')
        n = self.live[t,s]
        b = min(self.bound(t,s),dual(self.q,t,s,eta))
        self.overlay[t,s] = b
        self.live[t,s] = Node(n.value,n.tail,n.value+b)
        self.proposals += 1
        self.events.append(dict(kind='dual',state=[t,s],eta=list(eta)))
        self._prune()

    def receipt(self):
        upper = max([self.incumbent,*[n.upper for n in self.live.values()]])
        return dict(**score(self.q,self.schedule),schedule=list(self.schedule),upper=upper,gap=upper-self.incumbent,
                    transitions=self.transitions,live_states=len(self.live),trace=[list(a) for a in self.trace],
                    proposals=self.proposals,events=[{**e,'state':list(e['state']),**({'eta':list(e['eta'])} if 'eta' in e else {})} for e in self.events],classes=self.classes)


def replay(task,initial,events,**caps):
    s = SparseSession(task,initial,**caps)
    if not isinstance(events,list) or len(events)>s.class_cap+s.proposal_cap:
        raise TaskError('invalid event count')
    for e in events:
        if not isinstance(e,dict) or not isinstance(e.get('state'),list) or len(e['state'])!=2:
            raise TaskError('invalid event')
        if e.get('kind')=='expand' and set(e)=={'kind','state'}:
            s.advance(*e['state'])
        elif e.get('kind')=='dual' and set(e)=={'kind','state','eta'}:
            s.propose(*e['state'],e['eta'])
        else:
            raise TaskError('invalid event')
    return s.receipt()
