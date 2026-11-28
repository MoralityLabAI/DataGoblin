"""Independent task, receipt, resident-reference and byte-accounting replay."""
import json
from pathlib import Path

from audit_sg25 import chord, reconstruct
from sg20 import audit
from sg25_regions import Plane, Vertex
from sg26_memory import payload
from datagoblin.training_gym.spectral_face import verify_certificate


def replay(z):
    n=dict(roots=0,anchors=0,queries=0,final=0,closed=0,stages=0,candidates=0,hits=0,
           blocks=0,evictions=0,non_timing_pairs=0,interval_pairs=0,peak_bytes=0)
    seen=set()
    for root in z['roots']:
        p=Plane.build(root['base'],z['horizon'],root['directions'],root['price']);a=root['anchor'];audit(root['base'],a)
        n['roots']+=1;n['anchors']+=1
        tasks=[reconstruct(root,pt) for pt in root['points']]
        for pt,q in zip(root['points'],tasks,strict=True):
            assert p.at(pt).thaw()==q
            sig=json.dumps(q,sort_keys=True);assert sig not in seen;seen.add(sig);n['queries']+=1
        for arm,reps in root['arms'].items():
            for rep,rows in enumerate(reps):
                resident={};cost={}
                if arm!='step' and a['certificate']['certified']:
                    resident[0]=Vertex.checked(p,0,(0,0),a['schedule'],a['eta'],a['den']);cost[0]=1+4*a['lp_calls']
                for ident,(pt,q,r) in enumerate(zip(root['points'],tasks,rows,strict=True),1):
                    assert r['ident']==ident and r['point']==pt and r['gap']>=0
                    c=verify_certificate(q,r['schedule'],r['eta'],r['den']);assert c['objective']==r['objective']
                    if r['hit']:
                        assert c['certified'] and chord(p,pt,r,resident) and not r['evicted'] and r['stage'] is None
                        n['hits']+=1
                    else:
                        s=r['stage'];audit(q,s);n['stages']+=len(s['log']);n['blocks']+=s['block'] is not None
                        assert s['upper']==r['upper'] and s['gap']==r['gap']
                        if arm!='step' and c['certified'] and not s['block']:
                            resident[ident]=Vertex(ident,tuple(pt),tuple(r['schedule']),tuple(r['eta']),r['den'],c['bound_num'],c['objective']);cost[ident]=1+4*r['lp_calls']
                        for i in r['evicted']:
                            assert i in resident;del resident[i];del cost[i];n['evictions']+=1
                    assert list(resident)==r['resident']
                    size=sum(len(payload(v,cost[i])) for i,v in resident.items())
                    assert size==r['bytes']<=z['cap'];n['peak_bytes']=max(n['peak_bytes'],size)
                    for v in r['candidates']:
                        assert verify_certificate(q,v['schedule'],v['eta'],v['den'])==v['certificate'];n['candidates']+=1
                    n['final']+=1;n['closed']+=r['gap']==0
                    if arm!='step':
                        ref=root['arms']['step'][rep][ident-1]
                        assert max(ref['objective'],r['objective'])<=min(ref['upper'],r['upper']);n['interval_pairs']+=1
                    if arm=='area_full':
                        ref=root['arms']['area'][rep][ident-1]
                        assert all(ref[k]==r[k] for k in r if k not in ('cpu_ns','lookup_ns','verify_ns'));n['non_timing_pairs']+=1
        print('audited',root['seed'],flush=True)
    return n


if __name__=='__main__':
    z=replay(json.loads(Path('sg26_results.json').read_text()));Path('audit.json').write_text(json.dumps(z,indent=2)+'\n');print(json.dumps(z,indent=2))
