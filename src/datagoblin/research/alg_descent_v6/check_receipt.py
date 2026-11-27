"""Compare all native receipts to independent tuple-key replay; report paired effects."""
import csv,json,random,statistics as st,sys
from pathlib import Path
D=Path(sys.argv[1] if len(sys.argv)>1 else '.')
r=json.loads((D/'reference.json').read_text()); rows=list(csv.DictReader((D/'native.csv').open()))
fields=('closed','work','calls','rejects','hits','peak','pending_peak')
refs={(x['seed'],x['slots'],x['policy']):x for x in r['refs']}; roots={x['seed']:x for x in r['roots']}
lookup={(int(x['seed']),int(x['cap']),x['policy'],x['kind']):x for x in rows}
assert len(rows)==len(lookup)==56*37
matched=0
for x in rows:
    seed,cap,slots,allocated=map(lambda k:int(x[k]),('seed','cap','slots','allocated'))
    y=refs[seed,slots,x['policy']]; assert all(int(x[k])==y[k] for k in fields),(x,y)
    assert int(x['unique'])==roots[seed]['unique'] and int(x['node_capacity'])>=roots[seed]['nodes']
    assert allocated==slots*(36 if x['kind']=='packed' else 12) and allocated<=cap
    assert int(x['work'])<=3*int(x['n'])**2 and int(x['calls'])<=8*int(x['n']) and int(x['peak'])<=slots
    if x['kind']=='matched':
        z=lookup[seed,cap,x['policy'],'packed']; assert all(x[k]==z[k] for k in fields);matched+=1
san=D/'sanitized.csv'
if san.exists():
    ss=list(csv.DictReader(san.open())); assert len(ss)==len(rows)
    for a,b in zip(rows,ss): assert all(a[k]==b[k] for k in a if k not in ('median_ns','first_ns','intern_ns'))
summary=[]
for cap in (0,36,72,108,252,504,1008):
    for p in ('lru','pending') if cap else ('lru',):
        for k in ('packed','interned','matched') if cap else ('packed',):
            a=[x for x in rows if int(x['seed'])>=180000 and (int(x['cap']),x['policy'],x['kind'])==(cap,p,k)]
            assert len(a)==32
            q=dict(cap=cap,policy=p,kind=k,work=st.mean(int(x['work']) for x in a),mean_median_ns=st.mean(int(x['median_ns']) for x in a),closed=sum(int(x['closed']) for x in a))
            if k!='packed':
                ds=[int(x['work'])-int(lookup[int(x['seed']),cap,p,'packed']['work']) for x in a]
                ts=[int(x['median_ns'])-int(lookup[int(x['seed']),cap,p,'packed']['median_ns']) for x in a]
                rng=random.Random(6006); boot=sorted(st.mean(rng.choices(ds,k=len(ds))) for _ in range(5000))
                q.update(delta=st.mean(ds),wins=sum(d<0 for d in ds),ties=sum(d==0 for d in ds),losses=sum(d>0 for d in ds),bootstrap95=[boot[125],boot[4874]],delta_ns=st.mean(ts),paired_median_ns=st.median(ts))
            summary.append(q)
base=[x for x in rows if x['kind']=='packed' and int(x['cap'])==0 and int(x['seed'])>=180000]
report=dict(rows=len(rows),fields=len(rows)*len(fields),matched_slots=matched,closed=sum(int(x['closed']) for x in rows),sanitizer=san.exists(),summary=summary,ood_intern_mean_ns=st.mean(int(x['intern_ns']) for x in base),ood_intern_median_ns=st.median(int(x['intern_ns']) for x in base),ood_scratch_max=max(int(x['scratch_bytes']) for x in base),ood_node_total=sum(int(x['node_capacity']) for x in base),ood_unique_total=sum(int(x['unique']) for x in base),ood_sidecar_min=min(2*int(x['node_capacity']) for x in base),ood_sidecar_max=max(2*int(x['node_capacity']) for x in base))
print(json.dumps(report,sort_keys=True))
