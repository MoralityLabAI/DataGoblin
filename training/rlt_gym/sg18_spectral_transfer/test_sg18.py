from itertools import combinations, product
from random import Random

import sg18
from datagoblin.training_gym.spectral_quotient import score


def fixture(seed):
    r=Random(seed);d,t=4,3
    return dict(d=d,T=t,bits=[r.randrange(1,4) for _ in range(d)],x=sorted([0]+[r.randrange(5) for _ in range(t)]),
                k=[r.randrange(d+1) for _ in range(t)],u=[[r.randrange(-4,9) for _ in range(d)] for _ in range(t)],**{'lambda':r.randrange(4)})


def optimum(q):
    acts=[[sum(1<<j for j in js) for js in combinations(range(q['d']),k)] for k in q['k']]
    return max(score(q,list(a))['objective'] for a in product(*acts))


def test_transfer_and_certificates():
    for seed in range(12):
        q=fixture(1820000+seed);mem=sg18.Memory(2)
        for step in range(4):
            q={**q,'u':[row[:] for row in q['u']]};q['u'][step%3][step%4]+=1
            opt=optimum(q)
            for arm in sg18.ARMS:
                r=sg18.run(q,mem,arm,step,[.1]*7)
                assert r['objective']==r['upper']==opt
                assert r['resident']<=2 and r['probes']<=2


def test_memory_snapshot_and_scope():
    q=fixture(1821000);m=sg18.Memory(1);r=sg18.run(q,m,'identity',0)
    assert len(m.entries)==1
    entry=m.entries[0];sig=entry.sig
    q['u'][0][0]+=1
    assert entry.sig==sig and not sg18.retrieve(q,m,'identity',[0]*q['T'],1)
    assert sg18.retrieve(q,m,'nearest',[0]*q['T'],1)==[entry]
    q['k'][0]=(q['k'][0]+1)%(q['d']+1)
    assert not sg18.retrieve(q,m,'nearest',[0]*q['T'],1)
    sg18.run(q,m,'nearest',2)
    assert len(m.entries)==1 and m.entries[0].sig==sg18.signature(q)


def test_memory_rejects_uncertified_and_zero_cap():
    q=dict(d=3,T=1,bits=[1]*3,x=[0,1],k=[1],u=[[1,5,9]],**{'lambda':0})
    m=sg18.Memory(2)
    assert not m.put(q,[1],[0],1,0) and not m.entries
    r=sg18.run(q,sg18.Memory(0),'recency',0)
    assert r['gap']==0 and r['resident']==0
