from itertools import combinations, product
from random import Random

import pytest
import sg07
import sg11
from datagoblin.training_gym.spectral_blocks import BlockSearch, PrefixRelaxation, normalize, reference_bound, root_vector
from datagoblin.training_gym.spectral_block_replay import verify
from datagoblin.training_gym.spectral_quotient import quotient
from datagoblin.training_gym.spectral_routing import TaskError, oracle


def exact_remaining(q,t,s,r,e):
    choices=[]
    for i in range(t):
        a=[sum(1<<j for j in js) for js in combinations(range(q['d']),q['k'][i])]
        if i==t-1:
            a=[m for m in a if m&r==r and not m&e]
        choices.append(a)
    best=None
    for a in product(*choices):
        u=sum(q['u'][i][j] for i,m in enumerate(a) for j in range(q['d']) if m&(1<<j))
        c=sum(q['bits'][j]*max([0,*[q['x'][i+1]-q['x'][0] for i,m in enumerate(a) if m&(1<<j)]])
              for j in range(q['d']) if not s&(1<<j))
        v=u-q['lambda']*c
        best=v if best is None else max(best,v)
    return best


def test_conditioned_bound():
    checks=0
    for seed in range(24):
        q=sg07.task(990000+seed,'mixed',4,3)
        if seed%3==0:q['k'][0]=0
        if seed%5==0:q['k'][1]=4
        if seed%7==0:q['lambda']=0
        extra=[root_vector(q)];bank=PrefixRelaxation(q,extra)
        for t in range(1,4):
            for s in range(16):
                outside=[j for j in range(4) if not s&(1<<j)]
                for pattern in product(range(3),repeat=len(outside)):
                    yes=sum(1<<j for j,v in zip(outside,pattern,strict=True) if v==1)
                    no=sum(1<<j for j,v in zip(outside,pattern,strict=True) if v==2)
                    norm=normalize(q,t,s,yes,no)
                    ex=exact_remaining(q,t,s,yes,no)
                    assert (norm is None)==(ex is None)
                    if norm is None:continue
                    yes,no=norm
                    value=bank.bound(t,s,yes,no)
                    assert ex<=value==reference_bound(q,t,s,yes,no,extra)
                    if t==1:assert value==ex
                    checks+=1
    print('conditioned feasible checks',checks)
    assert checks>5000


def test_binary_partition():
    checks=0
    for seed in range(8):
        q=sg07.task(991000+seed,'mixed',6,3)
        for t in range(1,4):
            for s in range(64):
                all_classes={u for u,_,_ in quotient(q['u'][t-1],q['k'][t-1],s)}
                norm=normalize(q,t,s,0,0);stack=[norm];covered=set()
                while stack:
                    yes,no=stack.pop();free=63&~(s|yes|no)
                    if not free:
                        u=s|yes;assert u in all_classes and u not in covered;covered.add(u);continue
                    bit=free&-free
                    for n in (normalize(q,t,s,yes|bit,no),normalize(q,t,s,yes,no|bit)):
                        if n is not None:stack.append(n)
                assert covered==all_classes;checks+=len(covered)
    print('partition class checks',checks)
    assert checks>5000


def test_capped_and_complete_replay():
    for seed in range(32):
        q=sg07.task(992000+seed,'rotating',5,4);init=sg11.initial(q);ex=oracle(q)['objective'];extra=[root_vector(q)]
        for cap in (1,12,10000):
            s=BlockSearch(q,init,event_cap=cap,extra=extra);r=s.run()
            assert r['objective']<=ex<=r['upper'] and r['event_count']<=cap
            assert verify(q,init,extra,r['events'],event_cap=cap)=={k:v for k,v in r.items() if k!='peak_stored_blocks'}
            if cap==10000:assert r['gap']==0


def test_arbitrary_split_and_cap_atomic():
    q=dict(d=32,T=2,bits=[1]*32,x=[0,1,2],k=[8,8],u=[list(range(32)),list(range(32))],**{'lambda':0})
    s=BlockSearch(q,[255,255],event_cap=3,extra=[]);i=s.best();before=s.receipt()
    for bad in (-1,32,True,None):
        with pytest.raises(TaskError):s.advance(i,bad)
        assert s.receipt()==before
    r=s.run();assert r['event_count']==3 and r['upper']>=2*sum(range(24,32))>=r['objective']
    with pytest.raises(TaskError):s.advance(s.best(),0)
    assert s.receipt()==r
    assert verify(q,[255,255],[],r['events'],event_cap=3)=={k:v for k,v in r.items() if k!='peak_stored_blocks'}
    with pytest.raises(TaskError):verify(q,[255,255],[],[{'id':0,'component':True}],event_cap=3)
