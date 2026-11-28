"""Post-run replay, including independent affine task reconstruction."""
import json
from pathlib import Path
from sg20 import audit
from sg21_ray import Ray, interval_value, verify_region
from datagoblin.training_gym.spectral_face import verify_certificate
from datagoblin.training_gym.spectral_quotient import validate


def interpolate(a,b,s,H):
    def f(x,y):
        z,r=divmod((H-s)*x+s*y,H)
        assert not r
        return z
    return validate(dict(d=a['d'],T=a['T'],k=a['k'][:],bits=[f(x,y) for x,y in zip(a['bits'],b['bits'],strict=True)],
                         x=[f(x,y) for x,y in zip(a['x'],b['x'],strict=True)],
                         u=[[f(x,y) for x,y in zip(u,v,strict=True)] for u,v in zip(a['u'],b['u'],strict=True)],
                         **{'lambda':f(a['lambda'],b['lambda'])}))


def replay(path,seen):
    z=json.loads(Path(path).read_text());H=z['horizon']
    counts=dict(roots=0,queries=0,anchors=0,final=0,stages=0,regions=0,endpoints=0,maximality=0,reused=0,pairs=0)
    for root in z['roots']:
        a,b=root['base'],root['end'];ray=Ray.build(a,b,H)
        tasks=[interpolate(a,b,s,H) for s in range(H+1)]
        for s,q in enumerate(tasks):
            assert ray.at(s).thaw()==q
            if s:
                sig=json.dumps(q,sort_keys=True);assert sig not in seen;seen.add(sig);counts['queries']+=1
        audit(a,root['anchor']);counts['anchors']+=1;counts['roots']+=1
        for arm,reps in root['arms'].items():
            for rep,run in enumerate(reps):
                for p in run['proofs']:
                    assert verify_region(ray,p);counts['regions']+=1
                    for i,s in enumerate((p['start'],p['end'])):
                        c=verify_certificate(tasks[s],p['schedule'],p['eta'],p['den'])
                        assert c['certified'] and c['bound_num']==p['bounds'][i] and c['objective']==p['objectives'][i]
                        counts['endpoints']+=1
                    if p['end']<H:
                        assert not verify_certificate(tasks[p['end']+1],p['schedule'],p['eta'],p['den'])['certified']
                        counts['maximality']+=1
                for s,r in enumerate(run['rows'],1):
                    assert r['step']==s and r['gap']==0
                    c=verify_certificate(tasks[s],r['schedule'],r['eta'],r['den'])
                    assert c['objective']==r['objective']
                    if r['reused']:
                        p=run['proofs'][r['proof']]
                        assert all(p[k]==r[k] for k in ('schedule','eta','den'))
                        assert interval_value(p,s)==(r['objective'],r['upper']) and c['certified']
                        counts['reused']+=1
                    else:
                        audit(tasks[s],r['full']);counts['stages']+=len(r['full']['log'])
                        assert r['full']['upper']==r['upper'] and r['full']['gap']==r['gap']
                    counts['final']+=1
                    if arm!='delta':
                        ref=root['arms']['delta'][rep]['rows'][s-1]
                        assert all(ref[k]==r[k] for k in ('schedule','eta','den','objective','upper','gap'))
                        counts['pairs']+=1
    return counts


if __name__=='__main__':
    seen=set();out={name:replay(name,seen) for name in ('sg21_results.json','sg21b_results.json')}
    print(json.dumps(out,indent=2));Path('audit.json').write_text(json.dumps(out,indent=2)+'\n')
