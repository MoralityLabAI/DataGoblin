from random import Random

import sg07
import sg11
from datagoblin.training_gym.spectral_blocks import BlockSearch, normalize, reference_bound, root_vector
from datagoblin.training_gym.spectral_block_replay import verify
from datagoblin.training_gym.spectral_branching import BranchPolicy, child_bounds, free_bits
from datagoblin.training_gym.spectral_routing import oracle


def test_exact_all_child_bounds():
    checks = 0
    for seed in range(36):
        q = sg07.task(1290000+seed, 'mixed', 6, 4)
        extra = [root_vector(q)]
        s = BlockSearch(q, sg11.initial(q), event_cap=1000, extra=extra)
        while (i := s.best()) is not None:
            b = s.live[i]
            bounds = child_bounds(s, b)
            for j, pair in bounds.items():
                for (r,e), got in zip(((b.include | (1<<j),b.exclude),(b.include,b.exclude | (1<<j))),pair,strict=True):
                    r,e = normalize(q,b.t,b.suffix,r,e)
                    assert got == min(b.upper,b.value+reference_bound(q,b.t,b.suffix,r,e,extra))
                    checks += 1
            s.advance(i,s.relax.choose(b) if bounds else None)
    print('exact child comparisons',checks)
    assert checks > 100


def test_policy_replay_and_oracle():
    count=0
    for seed in range(24):
        q=sg07.task(1291000+seed,'rotating',6,4)
        initial=sg11.initial(q);extra=[root_vector(q)];opt=oracle(q)['objective']
        for kind in ('analytic','strong','pseudo','learned','shuffled'):
            p=BranchPolicy(kind,[.1]*12)
            s=BlockSearch(q,initial,event_cap=2048,extra=extra)
            while (i:=s.best()) is not None:
                b=s.live[i];old=s.upper()
                j=p.choose(s,b) if b.t>1 and free_bits(q,b) else None
                p.advance(s,i,j)
                assert s.upper()<=old
            r=s.receipt()
            assert r['gap']==0 and r['objective']==opt
            assert verify(q,initial,extra,r['events'],event_cap=2048)=={k:v for k,v in r.items() if k!='peak_stored_blocks'}
            count+=1
    assert count==120
