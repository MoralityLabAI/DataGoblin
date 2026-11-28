"""Independent integer/rational replay of SG-24 and SG-24B."""
import json
from fractions import Fraction
from pathlib import Path

from sg20 import audit
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


def rational_gap(a,b,H,p,s):
    """Independent explicit last-use formula at a rational position on the ray."""
    eta=[Fraction(x+(s-p['origin'])*v,p['den']) for x,v in zip(p['eta'],p['slope'],strict=True)]
    total=Fraction(0)
    for j in range(a['d']):
        z=[a['u'][i][j]+(s/H)*(b['u'][i][j]-a['u'][i][j])-eta[i] for i in range(a['T'])]
        prices=[]
        for x,y in zip(a['x'],b['x'],strict=True):
            v=a['lambda']*a['bits'][j]*(x-a['x'][0]);w=b['lambda']*b['bits'][j]*(y-b['x'][0])
            prices.append(v+(s/H)*(w-v))
        best=max([Fraction(0),*[z[i]+sum(max(0,x) for x in z[:i])-prices[i+1] for i in range(a['T'])]])
        selected=[i for i,m in enumerate(p['schedule']) if m&(1<<j)]
        chosen=sum(z[i] for i in selected)-prices[selected[-1]+1 if selected else 0]
        total+=best-chosen
    return total


def replay(z,seen):
    from sg21_ray import Ray
    H=z['horizon'];counts=dict(roots=0,queries=0,anchors=0,final=0,stages=0,candidates=0,regions=0,endpoints=0,
                              maximality=0,continuous=0,reused=0,objective_pairs=0,planner_pairs=0,blocks=0)
    for root in z['roots']:
        a,b=root['base'],root['end'];tasks=[interpolate(a,b,s,H) for s in range(H+1)];ray=Ray.build(a,b,H)
        for q in tasks[1:]:
            sig=json.dumps(q,sort_keys=True);assert sig not in seen;seen.add(sig);counts['queries']+=1
        audit(a,root['initial']);counts['anchors']+=1;counts['roots']+=1
        for arm,reps in root['arms'].items():
            for rep,run in enumerate(reps):
                for p in run['proofs']:
                    assert verify_region(ray,p);counts['regions']+=1
                    for i,s in enumerate((p['start'],p['end'])):
                        e,d=line_from_proof(p).at(s);c=verify_certificate(tasks[s],p['schedule'],e,d)
                        assert c['certified'] and c['bound_num']==p['bounds'][i] and c['objective']==p['objectives'][i]
                        counts['endpoints']+=1
                    if p['end']<H:
                        e,d=line_from_proof(p).at(p['end']+1)
                        assert not verify_certificate(tasks[p['end']+1],p['schedule'],e,d)['certified'];counts['maximality']+=1
                    if p.get('crossing') is not None:
                        x=Fraction(*p['crossing']);assert p['end']-p['start']<x<=p['end']-p['start']+1
                        assert rational_gap(a,b,H,p,Fraction(p['start'])+x)==1;counts['continuous']+=1
                for s,r in enumerate(run['rows'],1):
                    assert r['step']==s and r['gap']==0
                    c=verify_certificate(tasks[s],r['schedule'],r['eta'],r['den'])
                    assert c['objective']==r['objective']==r['upper']
                    if r['reused']:
                        p=run['proofs'][r['proof']];assert all(value(p,s)[k]==r[k] for k in ('objective','upper','gap','schedule','eta','den'))
                        assert c['certified'];counts['reused']+=1
                    else:
                        audit(tasks[s],r['full']);counts['stages']+=len(r['full']['log']);counts['blocks']+=r['full']['block'] is not None
                    for p in r['candidates']:
                        assert verify_certificate(tasks[s],p['schedule'],p['eta'],p['den'])==p['certificate'];counts['candidates']+=1
                    counts['final']+=1
                    if arm!='correct':
                        ref=root['arms']['correct'][rep]['rows'][s-1];assert ref['objective']==r['objective'] and ref['upper']==r['upper'];counts['objective_pairs']+=1
                    if arm in ('events','support'):
                        ref=root['arms']['binary'][rep]['rows'][s-1]
                        assert all(ref[k]==r[k] for k in ('schedule','eta','den','objective','upper','gap','lp_calls','reused','candidates'))
                        counts['planner_pairs']+=1
    return counts


if __name__=='__main__':
    seen=set();out={name:replay(json.loads(Path(name).read_text()),seen) for name in ('sg24_results.json','sg24b_results.json')}
    print(json.dumps(out,indent=2));Path('audit.json').write_text(json.dumps(out,indent=2)+'\n')
