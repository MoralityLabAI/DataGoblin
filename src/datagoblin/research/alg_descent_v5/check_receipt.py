"""Check native receipts against the independent reference; summarize paired OOD results."""
import csv
import json
import random
import statistics as stats
import sys
from pathlib import Path


def audit(directory):
    reference = json.loads((directory / 'reference.json').read_text())
    rows = list(csv.DictReader((directory / 'native.csv').open()))
    fields = ('closed', 'work', 'calls', 'rejects', 'hits', 'peak', 'pending_peak')
    key = lambda x: (int(x['seed']), int(x['cap']), x['policy'])
    refs = {key(x): x for x in reference['refs']}
    assert len(rows) == len(refs) == len({key(x) for x in rows}) == 616
    for x in rows:
        y = refs[key(x)]
        assert all(int(x[k]) == y[k] for k in fields), (x, y)
        assert int(x['allocated']) == int(x['slots'])*reference['slot_bytes'] <= int(x['cap'])
        assert int(x['peak']) <= int(x['slots'])
        assert int(x['work']) <= 3*int(x['n'])**2 and int(x['calls']) <= 8*int(x['n'])
    sanitized = directory / 'sanitized.csv'
    if sanitized.exists():
        s = list(csv.DictReader(sanitized.open()))
        assert len(s) == len(rows)
        assert all(all(a[k] == b[k] for k in a if k not in ('median_ns','first_ns')) for a,b in zip(rows,s))
    ood = [x for x in rows if int(x['seed']) >= 160000]
    summaries = []
    for cap in (0,64,128,256,512,1024):
        groups = [{int(x['seed']):x for x in ood if int(x['cap'])==cap and x['policy']==p} for p in ('lru','pending')]
        for p,g in zip(('lru','pending'),groups):
            if not g: continue
            assert len(g)==32
            summaries.append(dict(cap=cap,policy=p,closed=sum(int(x['closed']) for x in g.values()),work=stats.mean(int(x['work']) for x in g.values()),mean_median_ns=stats.mean(int(x['median_ns']) for x in g.values())))
        if all(groups):
            a,b=groups
            ds=[int(b[k]['work'])-int(a[k]['work']) for k in sorted(a)]
            ts=[int(b[k]['median_ns'])-int(a[k]['median_ns']) for k in sorted(a)]
            rng=random.Random(5105); boot=sorted(stats.mean(rng.choices(ds,k=len(ds))) for _ in range(5000))
            summaries[-1].update(delta_work=stats.mean(ds),wins=sum(d<0 for d in ds),ties=sum(d==0 for d in ds),losses=sum(d>0 for d in ds),bootstrap95=[boot[125],boot[4874]],delta_mean_median_ns=stats.mean(ts),delta_paired_median_ns=stats.median(ts))
    return dict(rows=len(rows),matched_fields=len(rows)*len(fields),closed=sum(int(x['closed']) for x in rows),sanitizer_receipt_present=sanitized.exists(),summary=summaries)

if __name__=='__main__':
    print(json.dumps(audit(Path(sys.argv[1] if len(sys.argv)>1 else '.')),sort_keys=True))
