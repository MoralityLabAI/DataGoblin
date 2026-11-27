"""Post-hoc six-entry timing ablation on exposed SG-08 roots."""
import json
import random
import statistics
from pathlib import Path
from unittest.mock import patch

import sg07
import sg08
from datagoblin.training_gym.spectral_cache import CachedDualSession


def native(q):
    def create(task, initial, **kw):
        kw.update(table_cap=6, scalar_cap=16)
        return CachedDualSession(task,initial,**kw)
    with patch.object(sg07,'CachedDualSession',create):
        return sg07.run(q,'lazy12')


def campaign():
    old=json.loads(Path('sg08_results.json').read_text()); rows=[]
    for root in old['rows']:
        arms=['native_lru','lru','fixed_first','learned']; runs={k:[] for k in arms}
        for rep in range(3):
            order=arms.copy(); random.Random(root['seed']+31*rep+500).shuffle(order)
            for k in order:
                z=native(root['task']) if k=='native_lru' else sg08.trace_run(
                    root['task'],k,old['model'].get(k,()),6)
                assert z['receipt']==root['receipt']
                assert z['cache']['component_steps']==root['offline']['6']['lru' if k=='native_lru' else k]['depth_work']*root['task']['d']
                runs[k].append(z['cpu_ns'])
        rows.append(dict(seed=root['seed'],runs=runs))
    base=[statistics.median(r['runs']['native_lru']) for r in rows]
    summary={}
    for k in arms:
        a=[statistics.median(r['runs'][k]) for r in rows]
        summary[k]=dict(cpu_ms=statistics.mean(a)/1e6,wins=sum(x<y for x,y in zip(a,base,strict=True)))
    return dict(evidence='post-hoc same exposed roots',receipt_comparisons=len(rows)*3*len(arms),summary=summary,rows=rows)


if __name__=='__main__':
    z=campaign(); Path('sg08_cap6_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k!='rows'},indent=2))
