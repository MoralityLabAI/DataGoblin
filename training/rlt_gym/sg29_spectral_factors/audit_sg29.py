"""Replay recorded certificates and validate shared-task pairing."""
import json
from pathlib import Path

from sg28_integer import replay as tree_replay
from sg29_factor import replay
from datagoblin.training_gym.spectral_quotient import score


def audit():
    z=json.loads(Path('sg29_results.json').read_text());r=json.loads(Path('sg29_regression.json').read_text())
    counts=dict(tasks=len(z['rows']),runs=0,closed=0,factor_receipts=0,tree_receipts=0,paired_intersections=0,regression=0,regression_closed=0)
    for row in z['rows']:
        q=row['task'];assert len(row['arms'])==5
        for rep in range(z['repeats']):
            receipts=[]
            for arm,rs in row['arms'].items():
                c=rs[rep]['receipt']
                if arm.startswith('factor'):
                    assert c['proof']['initial']==row['initial'] and [c['proof']['eta'],c['proof']['den']]==row['eta']
                    replay(q,c,max_sep=int(arm[-1]));counts['factor_receipts']+=1
                else:
                    tree_replay(q,c,63);counts['tree_receipts']+=1
                if row['optimum'] is not None:assert c['objective']<=row['optimum']<=c['upper']
                counts['runs']+=1;counts['closed']+=c['gap']==0;receipts.append(c)
            assert max(c['objective'] for c in receipts)<=min(c['upper'] for c in receipts)
            counts['paired_intersections']+=1
    inputs=json.loads(Path('sg28_exposed_inputs.json').read_text())
    for row,inp in zip(r['rows'],inputs,strict=True):
        q=row['task'];c=row['receipt'];assert row['label']==inp['label'] and q==inp['task']
        assert c['proof']['initial']==row['initial'] and [c['proof']['eta'],c['proof']['den']]==row['eta']
        if inp['root'] is not None:
            assert row['initial']==inp['initial'] and row['eta']==inp['root']
        if inp['prior_upper'] is not None:assert score(q,row['initial'])['objective']<=c['objective']<=inp['prior_upper']
        replay(q,c,max_sep=2);counts['regression']+=1;counts['regression_closed']+=c['gap']==0
    Path('sg29_audit.json').write_text(json.dumps(counts,indent=2)+'\n');print(counts)


if __name__=='__main__':audit()
