"""Fresh comparisons and exact hard-family regression; all timing descriptive."""
import json
import random
import statistics
import time
from pathlib import Path

from sg28 import solve
from sg28_integer import conditioned, domains, replay
from sg17_lp import chain_lp
from sg20 import repair
from sg20_delta import State, Task
from test_sg28 import triangle
from sg11 import initial
from datagoblin.training_gym.spectral_quotient import score


def run(q,seed):
    tick=time.process_time_ns();p,_=chain_lp(q);s=repair(State.build(Task.from_task(q),*p,initial(q)))
    a=list(s.schedule);L=s.objective;root=[list(s.eta),s.den];U=conditioned(q,domains(q),*root)
    pre=time.process_time_ns()-tick
    arms={};order=['root','milp','tree','tree_no_milp'];random.Random(seed).shuffle(order)
    for arm in order:
        if arm=='root':
            arms[arm]=dict(schedule=a,objective=L,upper=U,gap=U-L,nodes=1,depth=0,tree=dict(kind='bound',eta=root[0],den=root[1],upper=U),cpu_ns=pre,lp_calls=0)
        else:
            r=solve(q,a,root=root,cap=1 if arm=='milp' else 63,use_milp=arm!='tree_no_milp');r['cpu_ns']+=pre;arms[arm]=r
        replay(q,arms[arm])
    assert max(r['objective'] for r in arms.values())<=min(r['upper'] for r in arms.values())
    return dict(task=q,initial=a,root=root,prepare_ns=pre,arms=arms)


def summary(rows):
    return {a:dict(roots=len(rows),closed=sum(r['arms'][a]['gap']==0 for r in rows),mean_gap=statistics.mean(r['arms'][a]['gap'] for r in rows),mean_cpu_ms=statistics.mean(r['arms'][a]['cpu_ns'] for r in rows)/1e6,nodes=sum(r['arms'][a]['nodes'] for r in rows),max_nodes=max(r['arms'][a]['nodes'] for r in rows),conditional_lp=sum(r['arms'][a]['lp_calls'] for r in rows)) for a in rows[0]['arms']}


def campaign():
    fresh=[];planted=[]
    for r in json.loads(Path('sg28_fresh.json').read_text()):
        z=run(r['task'],r['seed']);z.update(seed=r['seed'],regime=r['regime']);fresh.append(z)
    for g in (2,3,4,5):
        for cost in (1,2,3):
            q=triangle(g);q['lambda']=cost;opt=g*(30-2*cost)
            r=run(q,2800+g*10+cost);r.update(groups=g,cost=cost,optimum=opt)
            for a in r['arms'].values():assert a['objective']<=opt<=a['upper']
            planted.append(r)
            print('planted',g,cost,[(a,v['gap'],v['nodes']) for a,v in r['arms'].items()],flush=True)
    out=dict(fresh=fresh,planted=planted,summary={'fresh':summary(fresh),'planted':summary(planted)})
    Path('sg28_compare.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summary'],indent=2))

if __name__=='__main__':campaign()
