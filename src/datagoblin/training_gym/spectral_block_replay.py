"""SG-12 independent structural and arithmetic transcript replay."""
from dataclasses import dataclass
from functools import lru_cache

from datagoblin.training_gym.spectral_quotient import validate
from datagoblin.training_gym.spectral_routing import TaskError


@dataclass(frozen=True)
class Obligation:
    t: int
    s: frozenset
    yes: frozenset
    no: frozenset
    g: int
    tail: tuple
    upper: int


def verify(task, initial, extra, events, *, event_cap=2048):
    q = validate(task)
    d, T = q['d'], q['T']
    universe = frozenset(range(d))
    if type(event_cap) is not int or not 1 <= event_cap <= 100000 or not isinstance(events, list) or len(events) > event_cap:
        raise TaskError('invalid replay cap or events')
    if not isinstance(extra, list) or len(extra) > 4 or any(
        not isinstance(v, list) or len(v) != T or any(type(x) is not int or abs(x) > 100_000_000 for x in v)
        for v in extra
    ):
        raise TaskError('invalid multipliers')
    cut = []
    for row, k in zip(q['u'], q['k'], strict=True):
        ordered = sorted(row, reverse=True)
        cut.append(ordered[k-1] if k else ordered[0]+1)
    vectors = [[0]*T, *[[v+h for v in cut] for h in (-6,-3,0,3,6)], *extra]

    def mask(js):
        return sum(2**j for j in js)

    def scoring(schedule):
        if not isinstance(schedule, list) or len(schedule) != T or any(type(a) is not int or not 0 <= a < 2**d for a in schedule):
            raise TaskError('invalid replay schedule')
        acts = [frozenset(j for j in universe if a//(2**j)%2) for a in schedule]
        if any(len(a) != k for a, k in zip(acts, q['k'], strict=True)):
            raise TaskError('invalid action cardinality')
        utility = sum(q['u'][i][j] for i, a in enumerate(acts) for j in a)
        cost = sum(q['bits'][j]*max([0, *[q['x'][i+1]-q['x'][0] for i, a in enumerate(acts) if j in a]]) for j in universe)
        return dict(utility=utility, cost=cost, objective=utility-q['lambda']*cost)

    # Enumerate possible last uses explicitly, independently of the prefix recurrence.
    tables = []
    for t in range(1,T+1):
        entries = []
        for eta in vectors:
            c = sum(eta[i]*q['k'][i] for i in range(t-1))
            active, inactive = [], []
            for j in range(d):
                z = [q['u'][i][j]-eta[i] for i in range(t-1)]
                active.append(sum(max(0,v) for v in z))
                inactive.append(max([0,*[z[i]+sum(max(0,v) for v in z[:i])-q['lambda']*q['bits'][j]*(q['x'][i+1]-q['x'][0]) for i in range(t-1)]]))
            entries.append((c,active,inactive))
        tables.append(entries)

    @lru_cache(maxsize=256)
    def components(t,s):
        result = []
        row = q['u'][t-1]
        for c,a,b in tables[t-1]:
            absent = [a[j] if j in s else b[j] for j in range(d)]
            present = [a[j]+row[j]-(0 if j in s else q['lambda']*q['bits'][j]*(q['x'][t]-q['x'][0])) for j in range(d)]
            result.append((c+sum(absent),[x-y for x,y in zip(present,absent,strict=True)]))
        return result

    def bound(t,s,yes,no):
        k = q['k'][t-1]-len(yes)
        return min(base+sum(delta[j] for j in yes)+sum(sorted((delta[j] for j in universe-yes-no),reverse=True)[:k])
                   for base,delta in components(t,s))

    schedule = list(initial)
    inc = scoring(schedule)['objective']
    live, seen = {}, {}
    next_id = 0
    counts = {'splits':0,'leaves':0,'terminals':0,'bound_calls':0}

    def add(t,s,yes,no,g,tail,cap,whole=False):
        nonlocal next_id
        k=q['k'][t-1]
        allowed=universe-no
        if yes&no or (yes|no)&s or len(yes)>k or len(allowed)<k:
            return
        if len(yes)==k:
            no=universe-s-yes
        elif len(allowed)==k:
            yes=allowed-s
        key=(t,s)
        if whole and key in seen and g<=seen[key]:
            return
        counts['bound_calls']+=1
        upper=g+bound(t,s,yes,no)
        if cap is not None:
            upper=min(upper,cap)
        if upper<=inc:
            return
        if whole:
            seen[key]=g
        live[next_id]=Obligation(t,s,yes,no,g,tail,upper)
        next_id+=1

    def valid(b):
        return b.upper>inc and b.g==seen.get((b.t,b.s))

    def upper():
        return max([inc,*[b.upper for b in live.values() if valid(b)]])

    empty=frozenset()
    add(T,empty,empty,empty,0,(),None,True)
    previous=upper()
    for event in events:
        if not isinstance(event,dict) or set(event)!={'id','component'} or type(event['id']) is not int:
            raise TaskError('invalid replay event')
        ident,j=event['id'],event['component']
        b=live.get(ident)
        if b is None or not valid(b):
            raise TaskError('replay state not live')
        t,s,yes,no=b.t,b.s,b.yes,b.no
        free=universe-s-yes-no
        if t>1 and free:
            if type(j) is not int or j not in free:
                raise TaskError('invalid replay split')
        elif j is not None:
            raise TaskError('invalid replay leaf')
        del live[ident]
        if t>1 and free:
            counts['splits']+=1
            add(t,s,yes|{j},no,b.g,b.tail,b.upper)
            add(t,s,yes,no|{j},b.g,b.tail,b.upper)
        else:
            row=q['u'][t-1]
            price=q['lambda']*(q['x'][t]-q['x'][0])
            k=q['k'][t-1]-len(yes)
            if t==1:
                pool=universe-yes-no
                counts['terminals']+=1
                values={i:row[i]-(0 if i in s else price*q['bits'][i]) for i in pool}
            else:
                pool=s
                counts['leaves']+=1
                values={i:row[i] for i in pool}
            act=yes|frozenset(sorted(pool,key=lambda i:(-values[i],i))[:k])
            m=mask(act)
            if len(act)!=q['k'][t-1]:
                raise AssertionError('invalid leaf action')
            g=b.g+sum(row[i] for i in act)-price*sum(q['bits'][i] for i in act-s)
            tail=(m,*b.tail)
            if t==1:
                if scoring(list(tail))['objective']!=g or g>b.upper:
                    raise AssertionError('replay terminal')
                if g>inc:
                    inc,schedule=g,list(tail)
            else:
                add(t-1,s|act,empty,empty,g,tail,b.upper,True)
        new=upper()
        if new>previous:
            raise AssertionError('nonmonotone upper bound')
        previous=new
    return dict(**scoring(schedule),schedule=schedule,upper=previous,gap=previous-inc,event_count=len(events),
                **counts,events=[dict(e) for e in events],live_states=sum(valid(b) for b in live.values()))
