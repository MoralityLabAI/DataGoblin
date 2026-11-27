from itertools import combinations, product
from random import Random

import sg07
import sg11
from datagoblin.training_gym.spectral_blocks import BlockSearch, root_vector
from datagoblin.training_gym.spectral_block_replay import verify
from datagoblin.training_gym.spectral_primal import pair_best, recover, row_best, row_sweep
from datagoblin.training_gym.spectral_quotient import score
from datagoblin.training_gym.spectral_routing import oracle


def test_row_exact():
    checks = 0
    for seed in range(48):
        q = sg07.task(1400000+seed, 'mixed', 5, 4)
        if seed % 3 == 0: q['k'][0] = 0
        if seed % 5 == 0: q['k'][1] = 5
        if seed % 7 == 0: q['lambda'] = 0
        if seed % 11 == 0: q['x'][1] = q['x'][0]
        s = sg11.initial(q); old = score(q, s)['objective']
        for i,k in enumerate(q['k']):
            m,gain = row_best(q,s,i)
            def value(m):
                a=s.copy();a[i]=m
                return score(q,a)['objective']
            assert value(m)-old == gain >= 0
            assert value(m)==max(value(sum(1<<j for j in c)) for c in combinations(range(q['d']),k))
            checks += 1
        z,gain=row_sweep(q,s)
        assert score(q,z)['objective']-old==gain>=0
    print('row optima',checks)


def test_pair_exact():
    checks = 0
    for seed in range(40):
        q=sg07.task(1410000+seed,'rotating',5,4)
        rng=Random(seed)
        s=[sum(1<<j for j in rng.sample(range(q['d']),k)) for k in q['k']]
        old=score(q,s)['objective']
        for a,b in combinations(range(q['d']),2):
            ma,mb=1<<a,1<<b
            choices=[(m,) if (m&(ma|mb)).bit_count()!=1 else (m,m^ma^mb) for m in s]
            best=max(score(q,list(z))['objective'] for z in product(*choices))
            out,gain=pair_best(q,s,a,b)
            assert score(q,out)['objective']==best==old+gain
            checks+=1
    print('pair optima',checks)


def test_recovery_and_certificates():
    count=0
    for seed in range(20):
        q=sg07.task(1420000+seed,'mixed',6,4)
        init=sg11.initial(q);eta=root_vector(q);opt=oracle(q)['objective']
        for method in ('base','row','dual','dual_row','pair'):
            a,g=recover(q,init,eta,method)
            assert score(q,a)['objective']==score(q,init)['objective']+g<=opt
            s=BlockSearch(q,a,event_cap=2048,extra=[eta]);r=s.run()
            assert r['gap']==0 and r['objective']==opt
            assert verify(q,a,[eta],r['events'],event_cap=2048)=={k:v for k,v in r.items() if k!='peak_stored_blocks'}
            count+=1
    print('exact replays',count)


def test_dual_round_context_parity():
    from datagoblin.training_gym.spectral_blocks import PrefixRelaxation
    from datagoblin.training_gym.spectral_primal import dual_round
    for seed in range(64):
        q=sg07.task(1430000+seed,'mixed',8,6)
        eta=root_vector(q);bank=PrefixRelaxation(q,[eta]);s=0;a=[0]*q['T']
        for t in range(q['T'],0,-1):
            _,_,order=bank.context(t,s)[-1]
            a[t-1]=sum(1<<j for j in order[:q['k'][t-1]])
            s|=a[t-1]
        assert dual_round(q,eta)==a
