"""Compare native search receipts with independently audited V15 reference."""
import json, random, statistics as st, sys
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
ARMS=('hmax','lazy_step','lazy_index')

def parse(path):
    rows={};layout=None;rss=None;names=None;r=None
    for line in path.read_text().splitlines():
        tag,*parts=line.split()
        if tag=='COUNTERS':names=parts;continue
        v=list(map(int,parts))
        if tag=='LAYOUT':layout=v
        elif tag=='RSS':rss=v[0]
        elif tag=='R':
            key=(v[0],ARMS[v[1]]);assert key not in rows
            r=dict(trace=[],coarse={},cuts=defaultdict(list),snapshots=[]);rows[key]=r
        elif tag=='B':
            s,b=v;assert s not in r['coarse'];r['coarse'][s]=b
        elif tag=='C':
            s,g,old,new,z,d,mask=v
            ids=[i for i in range(64) if mask>>i&1]
            r['trace'].append(dict(event='refine',state=s,g=g,old=old,new=new,rounds=1));r['cuts'][s].append([d,ids,z])
        elif tag=='E':
            s,g,n,*p=v;assert len(p)==n;r['trace'].append(dict(event='expand',state=s,g=g,path=p))
        elif tag=='S':
            cap,L,U,opened=v[:4];c=dict(zip(names,v[4:4+len(names)]));n,*p=v[4+len(names):];assert len(p)==n
            r['snapshots'].append(dict(budget=cap,L=L,U=U,open=opened,proof=p,**c))
        elif tag=='M':r['memory']=dict(zip(('median_ns','peak_requested','live_requested','total_requested','allocations','trace_hash','reps'),v))
        else:raise ValueError(tag)
    return dict(rows=rows,layout=layout,rss=rss)

def main():
    ref=json.loads((HERE/'reference.json').read_text());native=parse(HERE/'native.txt');rows=native['rows'];assert len(rows)==len(ref['runs'])==108
    fields=events=cuts=coarse=0
    for r in ref['runs']:
        n=rows[r['root'],r['arm']];assert n['trace']==r['trace'],(r['root'],r['arm'],'trace');events+=len(n['trace'])
        assert n['coarse']=={int(s):v['coarse'] for s,v in r['bounds'].items()};coarse+=len(n['coarse'])
        for s,b in r['bounds'].items():assert n['cuts'].get(int(s),[])==b['cuts'];cuts+=len(b['cuts'])
        assert len(n['snapshots'])==len(r['snapshots'])==6
        for a,b in zip(n['snapshots'],r['snapshots']):
            for k,v in a.items():assert v==b[k],(r['root'],r['arm'],k,v,b[k]);fields+=1
            assert a['L']<=r['optimum']<=a['U']
        assert n['memory']['reps']==61
    sanitized=HERE/'sanitized.txt'
    if sanitized.exists():
        s=parse(sanitized);assert s['layout']==native['layout'] and s['rows'].keys()==rows.keys()
        for k,n in rows.items():
            q=s['rows'][k];assert all(n[x]==q[x] for x in ('trace','coarse','cuts','snapshots'))
            assert all(n['memory'][x]==q['memory'][x] for x in ('peak_requested','live_requested','total_requested','allocations','trace_hash'))
    summary={};pairs={}
    for arm in ARMS:
        a=[r for (seed,kind),r in rows.items() if seed>=1120000 and kind==arm];assert len(a)==24
        summary[arm]=dict(certified=sum(q['snapshots'][-1]['L']==q['snapshots'][-1]['U'] for q in a),expansions=st.mean(q['snapshots'][-1]['expansions'] for q in a),scans=st.mean(sum(q['snapshots'][-1][k] for k in ('distance_scans','boundary_scans','minimum_scans')) for q in a),**{k:st.mean(q['memory'][k] for q in a) for k in ('median_ns','peak_requested','total_requested','allocations')},max_peak_requested=max(q['memory']['peak_requested'] for q in a))
    for arm,control in [('lazy_index','lazy_step'),('lazy_index','hmax')]:
        d=[rows[seed,arm]['memory']['median_ns']-rows[seed,control]['memory']['median_ns'] for seed in range(1120000,1120024)];rng=random.Random(1616);boot=sorted(st.mean(rng.choices(d,k=len(d))) for _ in range(5000))
        pairs[arm+' vs '+control]=dict(delta_ns=st.mean(d),wins=sum(x<0 for x in d),ties=sum(x==0 for x in d),losses=sum(x>0 for x in d),bootstrap95=[boot[125],boot[4874]])
    out=dict(cases=len(rows),events=events,coarse=coarse,cuts=cuts,fields=fields,layout=native['layout'],rss_kib=native['rss'],sanitized=sanitized.exists(),summary=summary,pairs=pairs,reference_audit={k:sum(q[k] for q in ref['audits']) for k in ref['audits'][0] if k not in ('root','optimum')})
    (HERE/'comparison.json').write_text(json.dumps(out,sort_keys=True,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
