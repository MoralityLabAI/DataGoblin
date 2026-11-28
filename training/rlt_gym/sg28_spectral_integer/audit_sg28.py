"""Replay all recorded proofs, independent of the numerical solver."""
import json
from pathlib import Path
from sg28_integer import replay
from sg17_lp import chain_lp
from datagoblin.training_gym.spectral_face import verify_certificate
from datagoblin.training_gym.spectral_quotient import score


def audit():
    inputs=json.loads(Path('sg28_exposed_inputs.json').read_text());rows=json.loads(Path('sg28_exposed.json').read_text())
    count=closed=nodes=0
    for inp,r in zip(inputs,rows,strict=True):
        assert inp['label']==r['label'] and inp['task']==r['task'] and inp['initial']==r['initial']
        q=inp['task'];s=r['receipt'];c=replay(q,s,63)
        if inp['root'] is not None:
            old=verify_certificate(q,inp['initial'],*inp['root'])
            assert old['objective']<=s['objective']<=inp['prior_upper']<=old['integer_upper']
        assert c['gap']==0
        count+=1;closed+=c['gap']==0;nodes+=c['nodes']
    fresh=json.loads(Path('sg28_fresh.json').read_text())
    for r in fresh:
        c=replay(r['task'],r['receipt'],63);count+=1;closed+=c['gap']==0;nodes+=c['nodes']
    comp=json.loads(Path('sg28_compare.json').read_text());comp_count=0
    for group,rr in ((k,comp[k]) for k in ('fresh','planted')):
        for r in rr:
            out=[replay(r['task'],v,63) for v in r['arms'].values()]
            assert max(v['objective'] for v in out)<=min(v['upper'] for v in out)
            if group=='planted':
                assert all(v['objective']<=r['optimum']<=v['upper'] for v in out)
                assert r['arms']['tree']['gap']==r['arms']['tree_no_milp']['gap']==0
            comp_count+=len(out);count+=len(out);closed+=sum(v['gap']==0 for v in out);nodes+=sum(v['nodes'] for v in out)
    # Exact proof from the first exposed SG-26 run (reconstructed from its logged
    # certificate, no reliance on a floating solver's status or old CPU time).
    inp=next(x for x in inputs if x['label']=='SG26-0');q=inp['task']
    s=dict(schedule=[84205608,2751733784,604266552,67412024,621068328,553705512,621068328,67428392,553967656,620806184,84205608,84172840],objective=42,upper=42,gap=0,nodes=3,depth=1,tree=dict(kind='split',j=13,cut=0,children=[dict(kind='bound',eta=[3,0,-2,3,2,-3,1,-6,-5,-3,-6,-18],den=3,upper=41),dict(kind='bound',eta=[4,2,2,4,0,-2,-3,-3,-3,-2,-4,-12],den=2,upper=42)]))
    assert replay(q,s,3)['gap']==0
    Path('sg28_anchor_regression.json').write_text(json.dumps(dict(task=q,receipt=s),indent=2)+'\n')
    count+=1;closed+=1;nodes+=3
    repeat=json.loads(Path('sg28_exposed_repeat.json').read_text())
    repeat_count=0;repeat_closed=0
    for r in repeat:
        c=replay(r['task'],r['receipt'],63);repeat_count+=1;repeat_closed+=c['gap']==0
    return dict(final_receipts=count,closed=closed,proof_nodes=nodes,paired_comparisons=comp_count,repeat_receipts=repeat_count,repeat_closed=repeat_closed)

if __name__=='__main__':
    z=audit();Path('sg28_audit.json').write_text(json.dumps(z,indent=2)+'\n');print(z)
