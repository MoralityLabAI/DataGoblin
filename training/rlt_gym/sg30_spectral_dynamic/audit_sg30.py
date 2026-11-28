"""Post-run exact replay and cross-policy comparison for all SG-30 cohorts."""
import json
from pathlib import Path

from sg28_integer import replay as tree_replay
from sg30 import replay


def audit():
    counts=dict(tasks=0,runs=0,closed=0,tree_receipts=0,hybrid_receipts=0,nodes=0,table_rechecks=0,paired=0,memo_pairs=0)
    for kind in ('forced','fresh','bridge','exposed'):
        z=json.loads(Path(f'sg30_{kind}.json').read_text())
        for row in z['rows']:
            counts['tasks']+=1;q=row['task']
            for rr in zip(*row['arms'].values(),strict=True):
                receipts=[]
                for arm,x in zip(row['arms'],rr,strict=True):
                    r=x['receipt'];c=tree_replay(q,r,63) if arm=='tree' else replay(q,r)
                    assert c==x['checked'] and r['gap']>=0
                    if row['opt'] is not None:assert r['objective']<=row['opt']<=r['upper']
                    counts['runs']+=1;counts['closed']+=r['gap']==0;counts['nodes']+=r['nodes'];counts['table_rechecks']+=c.get('unique_tables',0)
                    counts['tree_receipts' if arm=='tree' else 'hybrid_receipts']+=1;receipts.append(r)
                assert max(r['objective'] for r in receipts)<=min(r['upper'] for r in receipts)
                x,y=receipts[-2:];assert all(x[k]==y[k] for k in ('schedule','objective','upper','gap','nodes','lp_calls','work','factor_calls','factor_leaves'))
                counts['paired']+=1;counts['memo_pairs']+=1
    Path('sg30_audit.json').write_text(json.dumps(counts,indent=2)+'\n');print(counts)


if __name__=='__main__':audit()
