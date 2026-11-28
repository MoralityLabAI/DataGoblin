"""Replay certificates and independently verify directions via last-use enumeration."""
import json
from math import lcm
from pathlib import Path

from sg20 import audit
from datagoblin.training_gym.spectral_face import verify_certificate
from datagoblin.training_gym.spectral_quotient import validate


def interpolate(a,b,s,H):
    def f(x,y):
        z,r=divmod((H-s)*x+s*y,H);assert not r;return z
    return validate(dict(d=a['d'],T=a['T'],k=a['k'][:],bits=[f(x,y) for x,y in zip(a['bits'],b['bits'],strict=True)],
                         x=[f(x,y) for x,y in zip(a['x'],b['x'],strict=True)],
                         u=[[f(x,y) for x,y in zip(u,v,strict=True)] for u,v in zip(a['u'],b['u'],strict=True)],
                         **{'lambda':f(a['lambda'],b['lambda'])}))


def check_direction(q,r,old,v,den):
    """Lexicographic optimum over possible last uses, independent of boxes."""
    T=q['T'];D=old['den']
    assert old['certificate']['exact_face']
    for j in range(q['d']):
        z=[D*q['u'][i][j]-old['eta'][i] for i in range(T)]
        w=[den*(r['u'][i][j]-q['u'][i][j])-v[i] for i in range(T)]
        p=[q['lambda']*q['bits'][j]*(x-q['x'][0]) for x in q['x']]
        dp=[den*(r['lambda']*r['bits'][j]*(x-r['x'][0])-a) for x,a in zip(r['x'],p,strict=True)]
        vals=[(0,0)]
        for i in range(T):
            a=z[i]+sum(max(0,x) for x in z[:i])-D*p[i+1]
            b=w[i]+sum(y if x>0 else max(0,y) if x==0 else 0 for x,y in zip(z[:i],w[:i],strict=True))-dp[i+1]
            vals.append((a,b))
        selected=[i for i,m in enumerate(old['schedule']) if m&(1<<j)];last=selected[-1]+1 if selected else 0
        chosen=(sum(z[i] for i in selected)-D*p[last],sum(w[i] for i in selected)-dp[last])
        assert chosen==max(vals)
    return q['d']


def replay(z,seen):
    H=z['horizon'];counts=dict(roots=0,queries=0,anchors=0,final=0,stages=0,candidates=0,directions=0,
                              label_checks=0,raw_closed=0,accepted=0,objective_pairs=0,receipt_pairs=0,blocks=0,representation_caps=0)
    for root in z['roots']:
        a,b=root['base'],root['end'];tasks=[interpolate(a,b,s,H) for s in range(H+1)]
        for q in tasks[1:]:
            sig=json.dumps(q,sort_keys=True);assert sig not in seen;seen.add(sig);counts['queries']+=1
        audit(a,root['initial']);counts['anchors']+=1;counts['roots']+=1
        for arm,reps in root['arms'].items():
            for rep,rows in enumerate(reps):
                old=root['initial']
                for step,row in enumerate(rows,1):
                    q=tasks[step];r=row['full'];assert row['step']==step and row['gap']==0
                    audit(q,r);counts['stages']+=len(r['log']);counts['blocks']+=r['block'] is not None
                    assert r['certificate']['objective']==row['objective']==row['upper']==r['upper']
                    for c in row['candidates']:
                        assert verify_certificate(q,c['schedule'],c['eta'],c['den'])==c['certificate'];counts['candidates']+=1
                    p=row['proposal']
                    if p is not None:
                        v,den=p['velocity'],p['den'];counts['label_checks']+=check_direction(tasks[step-1],q,old,v,den)
                        D=lcm(old['den'],den)
                        e=[x*(D//old['den'])+y*(D//den) for x,y in zip(old['eta'],v,strict=True)]
                        assert verify_certificate(q,old['schedule'],e,D)==p['raw']
                        counts['directions']+=1;counts['raw_closed']+=p['raw']['certified']
                        c=next(c for c in row['candidates'] if c['kind']=='cone');assert c['eta']==e and c['den']==D
                        assert row['accepted']==c['certificate']['certified']
                    else:assert not row['accepted']
                    counts['accepted']+=row['accepted'];counts['representation_caps']+=(row['cone'] or {}).get('status')=='representation_cap'
                    counts['final']+=1;old=r
                    if arm!='correct':
                        ref=root['arms']['correct'][rep][step-1];assert row['objective']==ref['objective'] and row['upper']==ref['upper'];counts['objective_pairs']+=1
                    if arm=='fast_sweep':
                        ref=root['arms']['sweep'][rep][step-1]
                        assert {k:v for k,v in row.items() if k!='cpu_ns'}=={k:v for k,v in ref.items() if k!='cpu_ns'};counts['receipt_pairs']+=1
    return counts


if __name__=='__main__':
    seen=set();out={name:replay(json.loads(Path(name).read_text()),seen) for name in ('sg23_results.json','sg23b_results.json')}
    print(json.dumps(out,indent=2));Path('audit.json').write_text(json.dumps(out,indent=2)+'\n')
