"""Independent task reconstruction and complete SG-22 certificate replay."""
import json
from pathlib import Path

from sg20 import audit
from sg21_ray import Ray, verify_region as verify_fixed, interval_value
from sg22_move import line_from_proof, value, verify_region
from datagoblin.training_gym.spectral_face import verify_certificate
from datagoblin.training_gym.spectral_quotient import validate


def interpolate(a,b,s,H):
    def f(x,y):
        z,r=divmod((H-s)*x+s*y,H);assert not r;return z
    return validate(dict(d=a['d'],T=a['T'],k=a['k'][:],bits=[f(x,y) for x,y in zip(a['bits'],b['bits'],strict=True)],
                         x=[f(x,y) for x,y in zip(a['x'],b['x'],strict=True)],
                         u=[[f(x,y) for x,y in zip(u,v,strict=True)] for u,v in zip(a['u'],b['u'],strict=True)],
                         **{'lambda':f(a['lambda'],b['lambda'])}))


def replay(z):
    H=z['horizon'];seen=set()
    counts=dict(roots=0,queries=0,anchors=0,final=0,stages=0,candidates=0,regions=0,endpoints=0,
                maximality=0,reused=0,objective_pairs=0,blocks=0)
    for root in z['roots']:
        a,b=root['base'],root['end'];ray=Ray.build(a,b,H)
        tasks=[interpolate(a,b,s,H) for s in range(H+1)]
        for s,q in enumerate(tasks):
            assert q==ray.at(s).thaw()
            if s:
                sig=json.dumps(q,sort_keys=True);assert sig not in seen;seen.add(sig);counts['queries']+=1
        audit(a,root['initial']);counts['anchors']+=1;counts['roots']+=1
        for arm,reps in root['arms'].items():
            for rep,run in enumerate(reps):
                for p in run['proofs']:
                    moving=arm=='moving';assert (verify_region if moving else verify_fixed)(ray,p);counts['regions']+=1
                    for i,s in enumerate((p['start'],p['end'])):
                        e,d=line_from_proof(p).at(s) if moving else (p['eta'],p['den'])
                        c=verify_certificate(tasks[s],p['schedule'],e,d)
                        assert c['certified'] and c['bound_num']==p['bounds'][i] and c['objective']==p['objectives'][i]
                        counts['endpoints']+=1
                    if p['end']<H:
                        s=p['end']+1;e,d=line_from_proof(p).at(s) if moving else (p['eta'],p['den'])
                        assert not verify_certificate(tasks[s],p['schedule'],e,d)['certified'];counts['maximality']+=1
                for s,r in enumerate(run['rows'],1):
                    assert r['step']==s and r['gap']==0
                    c=verify_certificate(tasks[s],r['schedule'],r['eta'],r['den'])
                    assert c['objective']==r['objective']==r['upper']
                    if r['reused']:
                        p=run['proofs'][r['proof']]
                        if arm=='moving':assert all(value(p,s)[k]==r[k] for k in ('objective','upper','gap','schedule','eta','den'))
                        else:assert interval_value(p,s)==(r['objective'],r['upper'])
                        assert c['certified'];counts['reused']+=1
                    else:
                        audit(tasks[s],r['full']);counts['stages']+=len(r['full']['log']);counts['blocks']+=r['full']['block'] is not None
                        assert r['full']['upper']==r['upper'] and r['full']['gap']==r['gap']
                    for p in r.get('candidates',[]):
                        assert verify_certificate(tasks[s],p['schedule'],p['eta'],p['den'])==p['certificate'];counts['candidates']+=1
                    counts['final']+=1
                    if arm!='delta':
                        ref=root['arms']['delta'][rep]['rows'][s-1]
                        assert ref['objective']==r['objective'] and ref['upper']==r['upper'];counts['objective_pairs']+=1
    return counts


if __name__=='__main__':
    out=replay(json.loads(Path('sg22_results.json').read_text()));print(json.dumps(out,indent=2))
    Path('audit.json').write_text(json.dumps(out,indent=2)+'\n')
