"""RV23-B: independent exhaustive completion and certificate audit."""
from random import Random

from datagoblin.research_v2_2.deficit_cover import ARMS, Design, solve


def brute(d):
    items = d['experiments']
    feasible = {}
    for m in range(1 << len(items)):
        selected = [r for i,r in enumerate(items) if m >> i & 1]
        ids = {r['id'] for r in selected}
        cost = sum(r['cost'] for r in selected)
        if cost > d['budget'] or any(not set(r['requires']) <= ids for r in selected):
            continue
        if all(sum(r['predictions'][a]!=r['predictions'][b] for r in selected)>=2*d['error_bits']+1
               for a in range(len(d['mechanisms'])) for b in range(a+1,len(d['mechanisms']))):
            feasible[m] = cost
    return feasible


def test_forced_dependency_and_unresolved():
    d = dict(mechanisms=['a','b'],error_bits=0,budget=3,experiments=[
        dict(id='c',cost=1,requires=[],predictions=[0,0]),
        dict(id='e',cost=2,requires=['c'],predictions=[0,1])])
    answer,stats = solve(d,'forced',4)
    assert answer == dict(status='plan',experiments=['c','e'],cost=3)
    assert stats['certified'] and stats['forced_steps']==1 and stats['inspected']==2
    assert solve(d,'forced',1)[0] is None
    assert not solve(d,'forced',1)[1]['certified']
    d['budget']=2
    answer,stats=solve(d,'forced',4)
    assert answer==dict(status='infeasible') and stats['certified']


def test_exhaustive_bounds_and_native_style_certificates():
    rng = Random(7823)
    for j in range(48):
        n = 4+j%4
        d = dict(mechanisms=list('abcd'[:2+j%3]),error_bits=j%2,budget=3+j%10,experiments=[])
        for i in range(n):
            d['experiments'].append(dict(id=f'e{i}',cost=rng.randrange(1,5),
                requires=[f'e{rng.randrange(i)}'] if i and rng.random()<.4 else [],
                predictions=[rng.randrange(2) for _ in d['mechanisms']]))
        p = Design(d)
        valid = brute(d)
        optimum = min(valid.values(),default=None)
        for m in range(1<<n):
            if p.close(m)!=m:
                continue
            cost,ds = p.record(m)
            lb,_,forced,_ = p.analyze(m,ds)
            supers = [s for s in valid if s&m==m]
            if supers:
                assert lb is not None and cost+lb<=min(valid[s] for s in supers)
                assert all(s&forced==forced for s in supers)
        for arm in ARMS:
            answer,stats = solve(d,arm,1<<n)
            assert stats['inspected']<=1<<n
            if answer is not None and answer['status']=='plan':
                mask = sum(1 << p.ids[x] for x in answer['experiments'])
                assert valid[mask]==answer['cost']
            if arm in ('cover','bound','forced','warm_forced'):
                assert stats['certified']
                assert answer.get('cost')==optimum
                assert (answer['status']=='infeasible')==(optimum is None)
