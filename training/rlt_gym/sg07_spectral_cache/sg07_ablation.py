"""Post-hoc same-root arithmetic/cache ablation; all policies replay identically."""
import json
import random
import statistics
from pathlib import Path
from unittest.mock import patch

import sg07
from datagoblin.training_gym.spectral_cache import CachedDualSession


CAPS = {'linear_no_cache': (0, 0), 'table_only': (128, 0), 'scalar_only': (0, 128), 'both': (128, 128)}


def run(q, arm):
    if arm == 'original':
        return sg07.run(q, arm)
    tables, scalars = CAPS[arm]

    def construct(task, initial, **kwargs):
        kwargs.update(table_cap=tables, scalar_cap=scalars)
        return CachedDualSession(task, initial, **kwargs)

    with patch.object(sg07, 'CachedDualSession', construct):
        return sg07.run(q, 'cache128')


def campaign():
    original = json.loads(Path('sg07_results.json').read_text())
    rows = []
    for root in original['rows']:
        arms = ['original', *CAPS]
        runs = {arm: [] for arm in arms}
        expected = root['runs']['original'][0]['receipt']
        for rep in range(3):
            order = list(arms)
            random.Random(root['seed'] + 31 * rep + 821).shuffle(order)
            for arm in order:
                result = run(root['task'], arm)
                assert result['receipt'] == expected
                runs[arm].append({k: v for k, v in result.items() if k != 'receipt'})
        rows.append(dict(seed=root['seed'], runs=runs))
    summary = {}
    for arm in ['original', *CAPS]:
        time = [statistics.median(v['cpu_ns'] for v in r['runs'][arm]) for r in rows]
        ref = [statistics.median(v['cpu_ns'] for v in r['runs']['original']) for r in rows]
        summary[arm] = dict(cpu_ms=statistics.mean(time)/1e6,
                            wins=sum(a < b for a,b in zip(time,ref,strict=True)),
                            speedup=statistics.mean(ref)/statistics.mean(time))
    return dict(evidence='post-hoc same exposed roots, separate timing run', roots=len(rows), receipt_comparisons=len(rows)*3*5,
                summary=summary, rows=rows)


if __name__ == '__main__':
    result = campaign()
    Path('sg07_ablation_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))
