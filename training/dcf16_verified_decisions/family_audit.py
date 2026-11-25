"""Cross-check fresh holdout against DCF-15 dispatcher and small brute force."""
from dispatcher import compile_plan,execute
from benchmark import brute
import json
from pathlib import Path
import argparse
import numpy as np

a=argparse.ArgumentParser();a.add_argument('directory',type=Path);p=a.parse_args().directory
rows=[json.loads(l) for l in (p/'family_holdout.jsonl').read_text().splitlines()]; mx=0.; mx_brute=0.; cnt=0; scopes=[]
for row in rows:
    x=row['input']; y=row['target']; plan=compile_plan(x['scopes'],x['query']); pred=execute(plan,x['factors']); err=abs(pred-y['posterior']);mx=max(mx,err);assert err<1e-11
    n=max(max(s) for s in x['scopes'])+1
    if n==8:
        pb=brute(x['scopes'],[np.asarray(f) for f in x['factors']],n,x['query']);mx_brute=max(mx_brute,abs(pb-y['posterior']));cnt+=1;assert abs(pb-y['posterior'])<1e-11
    scopes.append(plan.peak_scope)
r=dict(family_holdout_dispatcher_checks=len(rows),max_dispatcher_error=mx,brute_checks=cnt,max_brute_error=mx_brute,max_plan_scope=max(scopes));print(r);(p/'family_audit.json').write_text(json.dumps(r,indent=2)+'\n')
