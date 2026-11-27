"""Audit all V17 factorial native arms against the independently audited V15 reference."""
import json, random, statistics as st
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
ARMS=('hmax','hmax_packed','indexed','direct','defer','direct_defer','packed','direct_packed','defer_packed','all')

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
        elif tag=='T':r['times']=v
        elif tag=='M':r['memory']=dict(zip(('median_ns','peak_requested','live_requested','total_requested','allocations','trace_hash','reps'),v))
        else:raise ValueError(tag)
    return dict(rows=rows,layout=layout,rss=rss)

def paired(rows,a,b,k):
    d=[rows[s,a]['memory'][k]-rows[s,b]['memory'][k] for s in range(1220000,1220024)]
    rng=random.Random(1717);boot=sorted(st.mean(rng.choices(d,k=len(d))) for _ in range(5000))
    return dict(delta=st.mean(d),wins=sum(x<0 for x in d),ties=sum(x==0 for x in d),losses=sum(x>0 for x in d),bootstrap95=[boot[125],boot[4874]])

def main():
    ref=json.loads((HERE/'reference.json').read_text());native=parse(HERE/'native.txt');rows=native['rows'];assert len(rows)==360 and len(ref['runs'])==72
    refs={(r['root'],r['arm']):r for r in ref['runs']};fields=events=cuts=coarse=0
    for (seed,arm),n in rows.items():
        r=refs[seed,'hmax' if arm.startswith('hmax') else 'lazy_index']
        iscut=not arm.startswith('hmax');direct=iscut and (arm.startswith('direct') or arm=='all');defer=iscut and ('defer' in arm or arm=='all')
        assert n['trace']==r['trace'],(seed,arm,'trace');events+=len(n['trace'])
        assert n['coarse']=={int(s):v['coarse'] for s,v in r['bounds'].items()};coarse+=len(n['coarse'])
        for s,b in r['bounds'].items():assert n['cuts'].get(int(s),[])==b['cuts'];cuts+=len(b['cuts'])
        assert len(n['snapshots'])==len(r['snapshots'])==6
        # Independently derive storage and direct-union counts from recorded reference cut events.
        seen=set();unions=0;cursor=0;per_state=defaultdict(int)
        for a,b in zip(n['snapshots'],r['snapshots']):
            length=b['expansions']+b['refinements']
            for e in r['trace'][cursor:length]:
                if e['event']=='refine':
                    s=e['state'];seen.add(s);z=r['bounds'][str(s)]['cuts'][per_state[s]][2];per_state[s]+=1;unions+=2*z.bit_count()
            cursor=length
            for k,v in a.items():
                if k=='residual_entries':expected=(len(seen) if defer else b['coarse_calls']) if iscut else 0
                elif k=='recomputed_scans':
                    # Exact recompute count is separately checked by the distance-scan delta.
                    assert v>=0 and (defer or v==0);continue
                elif k=='distance_scans':expected=b[k]+a['recomputed_scans']
                elif direct and k in ('zone_hits','zone_misses','zone_entries','zone_rule_ids'):expected=0
                elif direct and k=='unions':expected=unions
                else:expected=b[k]
                assert v==expected,(seed,arm,k,v,expected);fields+=1
            assert a['L']<=r['optimum']<=a['U']
        m=n['memory'];assert m['reps']==len(n['times'])==101 and m['median_ns']==st.median(n['times'])
        assert m['live_requested']<=m['peak_requested']<=m['total_requested']
        if not defer:assert n['snapshots'][-1]['recomputed_scans']==0
    san=parse(HERE/'sanitized.txt');assert san['layout']==native['layout'] and san['rows'].keys()==rows.keys()
    for k,n in rows.items():
        q=san['rows'][k];assert all(n[x]==q[x] for x in ('trace','coarse','cuts','snapshots'))
        assert all(n['memory'][x]==q['memory'][x] for x in ('peak_requested','live_requested','total_requested','allocations','trace_hash'))
    summary={}
    for arm in ARMS:
        a=[r for (s,k),r in rows.items() if s>=1220000 and k==arm];assert len(a)==24
        summary[arm]=dict(certified=sum(q['snapshots'][-1]['L']==q['snapshots'][-1]['U'] for q in a),expansions=st.mean(q['snapshots'][-1]['expansions'] for q in a),scans=st.mean(sum(q['snapshots'][-1][k] for k in ('distance_scans','boundary_scans','minimum_scans')) for q in a),**{k:st.mean(q['memory'][k] for q in a) for k in ('median_ns','peak_requested','total_requested','allocations')},**{k:st.mean(q['snapshots'][-1][k] for q in a) for k in ('residual_entries','recomputed_scans','unions','zone_entries')},max_peak_requested=max(q['memory']['peak_requested'] for q in a))
    pairs={f'{a} vs {b}: {k}':paired(rows,a,b,k) for a,b in [('all','indexed'),('all','hmax_packed'),('direct','indexed'),('defer','indexed'),('packed','indexed'),('hmax_packed','hmax')] for k in ('median_ns','peak_requested')}
    out=dict(cases=len(rows),events=events,coarse=coarse,cuts=cuts,fields=fields,layout=native['layout'],rss_kib=native['rss'],sanitized=True,summary=summary,pairs=pairs,reference_audit={k:sum(q[k] for q in ref['audits']) for k in ref['audits'][0] if k not in ('root','optimum')})
    (HERE/'comparison.json').write_text(json.dumps(out,sort_keys=True,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
