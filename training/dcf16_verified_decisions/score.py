"""Evaluate a policy by exact-reference expected decision utility."""
import argparse,json,math
from pathlib import Path

def evaluate(rows,policy):
    regret=[];utility=[];acted=correct=opportunities=hits=0
    for r in rows:
        p=r['target']['posterior'];a=policy(r['input']);best=r['target']['action'];u=0 if a is None else 5*p-4 if a==1 else 1-5*p
        utility.append(u);regret.append(r['target']['expected_utility']-u);acted+=a is not None;correct+=a==best;opportunities+=best is not None;hits+=best is not None and a==best
    n=len(rows)
    return dict(n=n,accuracy=correct/n,coverage=acted/n,expected_utility=math.fsum(utility)/n,regret=math.fsum(regret)/n,opportunity_accuracy=hits/opportunities if opportunities else None)

def unary(x):
    a,b=1.,1.
    for s,f in zip(x['scopes'],x['factors']):
        if s==[x['query']]:a*=f[0];b*=f[1]
    p=b/(a+b)
    return 1 if p>=.8 else 0 if p<=.2 else None

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();report={}
    for file in sorted(a.directory.glob('*.jsonl')):
        rows=[json.loads(l) for l in file.read_text().splitlines()]
        report[file.stem]={name:evaluate(rows,policy) for name,policy in [('abstain',lambda x:None),('always0',lambda x:0),('always1',lambda x:1),('unary',unary)]}
        report[file.stem]['reference_expected_utility']=math.fsum(r['target']['expected_utility'] for r in rows)/len(rows)
    print(json.dumps(report,indent=2));(a.directory/'baseline_scores.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
