"""Exposed SG-26/27 regression, separate from the fresh random cohort."""
import json
import statistics
import time
from pathlib import Path

from sg29 import prepare
from sg29_factor import solve, replay
from datagoblin.training_gym.spectral_quotient import score


def campaign():
    rows=[]
    for r in json.loads(Path('sg28_exposed_inputs.json').read_text()):
        q=r['task'];p=r['root'];a=r['initial']
        if p is None:a,p,_=prepare(q)
        tick=time.process_time_ns();z=solve(q,a,*p,max_sep=2);elapsed=time.process_time_ns()-tick
        replay(q,z,max_sep=2)
        if r['prior_upper'] is not None:assert z['objective']<=r['prior_upper']
        rows.append(dict(label=r['label'],task=q,initial=a,eta=p,receipt=z,cpu_ns=elapsed))
        print(r['label'],score(q,a)['objective'],z['objective'],z['upper'],z['status'],flush=True)
    summary={}
    for prefix in ('SG26','SG27'):
        rr=[r for r in rows if r['label'].startswith(prefix)]
        summary[prefix]=dict(tasks=len(rr),closed=sum(r['receipt']['gap']==0 for r in rr),improved=sum(r['receipt']['objective']>score(r['task'],r['initial'])['objective'] for r in rr),mean_gap=statistics.mean(r['receipt']['gap'] for r in rr),cpu_ms=statistics.mean(r['cpu_ns'] for r in rr)/1e6)
    Path('sg29_regression.json').write_text(json.dumps(dict(rows=rows,summary=summary),indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':campaign()
