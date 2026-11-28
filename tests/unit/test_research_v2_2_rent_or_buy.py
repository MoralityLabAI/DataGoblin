"""RV22-X: offline oracle and online payback invariants."""
import sys
from pathlib import Path
from random import Random
from unittest.mock import patch


def module():
    here = Path(__file__).resolve().parent
    scripts = here if (here / 'rent_or_buy.py').exists() else here.parents[1] / 'scripts' / 'research_v2_2'
    with patch.object(sys, 'path', [str(scripts), *sys.path]):
        import rent_or_buy as m
    return m


def brute(requests, cap):
    """Independent forward DP, allowing free eviction even on hits/bypasses."""
    sizes = {k:s for k,s,_,_ in requests}
    front = {frozenset():0}
    for k,_,r,b in requests:
        out = {}
        for resident,cost in front.items():
            options = [(resident, cost+(0 if k in resident else r))]
            if k not in resident:
                options.append((resident | {k}, cost+r+b))
            for universe,value in options:
                ids = list(universe)
                for bits in range(1 << len(ids)):
                    target = frozenset(a for i,a in enumerate(ids) if bits>>i&1)
                    if sum(sizes[a] for a in target) <= cap:
                        out[target] = min(out.get(target,float('inf')),value)
        front = out
    return min(front.values())


def test_offline_oracle_and_competitive_bound():
    m = module()
    rng = Random(801)
    for _ in range(120):
        sizes = [rng.randrange(1,5) for _ in range(4)]
        ds = [(k,sizes[k],rng.randrange(0,8),rng.randrange(0,9))
              for k in (rng.randrange(4) for _ in range(9))]
        cap = rng.randrange(0,10)
        assert m.offline(ds,cap)[0] == brute(ds,cap)
    for r in range(1,10):
        for b in range(1,30):
            j = (b+r-1)//r+1
            for n in range(1,31):
                optimum = min(n*r,r+b)
                actual = n*r if n<j else j*r+b
                assert actual <= 2*optimum
                if n in (1,3,9,30):
                    assert m.offline([(0,1,r,b)]*n,1)[0] == optimum


def test_payback_and_prefix_and_collision():
    m = module()
    rng = Random(802)
    d = m.world(rng,12)
    ds = [m.variant(d,rng) for _ in range(10)]
    ref = m.evaluate(ds,'cold',4096)[1]
    for arm in m.ARMS:
        stats,vals,trace = m.evaluate(ds,arm,4096)
        assert vals == ref and stats['checks'] == stats['admissions']
        assert stats['retained_bytes'] <= 4096
        assert stats['hits']+stats['solves'] == len(ds)
        assert m.evaluate(ds[:4],arm,4096)[2] == trace[:4]
        if arm.startswith('payback'):
            assert trace[:4] == ['cold','cold','buy','hit']
    ds = m.make_stream(803,'cycle12',36)
    ref = m.evaluate(ds,'cold',4096)[1]
    cap = m.Ledger().retained()+m.Arena(101).retained()  # no room for a complete entry
    for arm in m.ARMS:
        with patch.object(m,'tag_of',return_value=1):
            stats,vals,_ = m.evaluate(ds,arm,4096)
        assert vals == ref and stats['checks'] == stats['admissions']
        stats,vals,_ = m.evaluate(ds,arm,cap)
        assert vals == ref and stats['checks'] == stats['hits'] == 0
