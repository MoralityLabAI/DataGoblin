from itertools import combinations, product
from random import Random

import sg18
import sg19
from datagoblin.training_gym.spectral_quotient import score


def test_no_repeat():
    for seed in range(12):
        for drift in (False, True):
            q=[q for _,_,q in sg19.trajectory(1920000+seed,'mixed',drift)]
            assert len(q)==len({sg18.signature(t) for t in q})==12
            assert len({sg18.signature(t)[:-1] for t in q})==1


def test_gating_and_scope():
    q=next(sg19.trajectory(1921000,'mixed',False))[2]
    mem=sg18.Memory();s=sg18.Acquisition(q);s.finish();r=s.receipt()
    assert mem.put(q,r['schedule'],r['eta'],r['den'],0)
    model=dict(learned=[1.]+[0.]*7,shuffled=[-1.]+[0.]*7,threshold=0)
    assert sg19.retrieve(q,mem,'learned',1,3,model)==mem.entries
    assert sg19.retrieve(q,mem,'shuffled',1,3,model)==[]
    q={**q,'u':[row[:] for row in q['u']]};q['u'][0][0]+=17
    assert sg19.retrieve(q,mem,'fixed16',1,3,model)==[]
    assert sg19.retrieve(q,mem,'tuned',1,3,model)==[]
    assert sg19.retrieve(q,mem,'nearest1',1,3,model)==mem.entries
    q={**q,'lambda':q['lambda']+1}
    assert sg19.retrieve(q,mem,'learned',1,3,model)==[]


def test_exact_oracle():
    model=dict(learned=[1.]+[0.]*7,shuffled=[-1.]+[0.]*7,threshold=4)
    for seed in range(8):
        r=Random(1922000+seed);d,t=4,3
        q=dict(d=d,T=t,bits=[r.randrange(1,4) for _ in range(d)],x=sorted([0]+[r.randrange(5) for _ in range(t)]),
               k=[r.randrange(d+1) for _ in range(t)],u=[[r.randrange(-4,9) for _ in range(d)] for _ in range(t)],**{'lambda':r.randrange(4)})
        memories={a:sg18.Memory() for a in sg19.ARMS}
        for i in range(4):
            q['u'][i%t][i%d]+=1
            actions=[[sum(1<<j for j in c) for c in combinations(range(d),k)] for k in q['k']]
            opt=max(score(q,list(a))['objective'] for a in product(*actions))
            for arm in sg19.ARMS:
                z=sg19.run(q,memories[arm],arm,i,model)
                assert z['objective']==z['upper']==opt and z['resident']<=4
                assert z['probes']<=(2 if arm=='nearest2' else 1)
