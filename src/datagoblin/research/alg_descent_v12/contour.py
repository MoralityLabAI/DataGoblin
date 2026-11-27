"""Retrospective strict-contour diagnostic and malformed-proof regression."""
import json
from statistics import mean
from anytime import roots,hmax,check,HERE


def main():
    data=json.loads((HERE/'receipt.json').read_text());out=[];edges=cases=neg=0
    for split in ('iid','ood'):
        for g in roots(split):
            rows=[x for x in data['runs'] if x['root']==g.seed]
            opt=rows[0]['optimum'];d={g.initial:0};cache={}
            def lower(s):
                if s not in cache:cache[s]=hmax(g,s)
                return cache[s]
            for s in range(1<<g.n):
                if s not in d or s&g.goalbit:continue
                for t,c,_ in g.ready(s):
                    assert lower(s)<=c+lower(t);edges+=1
                    d[t]=min(d.get(t,float('inf')),d[s]+c)
            forced={s for s,c in d.items() if not s&g.goalbit and c+lower(s)<opt}
            for row in rows:
                seen={(x['state'],x['g']) for x in row['trace']}
                assert all((s,d[s]) in seen for s in forced);cases+=1
            a=next(x for x in rows if x['arm']=='hmax')
            above=sum(x['g']+lower(x['state'])>opt for x in a['trace'])
            out.append(dict(split=split,root=g.seed,strict=len(forced),astar=len(a['trace']),extra=len(a['trace'])-len(forced),above=above))
            proof=rows[0]['snapshots'][-1]['proof'];assert check(g,proof)==opt
            for bad in ([-1],[len(g.rules)],[True],[],proof[:1]+proof):
                try:check(g,bad)
                except AssertionError:neg+=1
                else:raise AssertionError('malformed proof accepted')
    ood=[x for x in out if x['split']=='ood']
    result=dict(consistency_edges=edges,forced_state_trace_checks=cases,malformed_rejected=neg,ood_mean_strict=mean(x['strict'] for x in ood),ood_mean_extra=mean(x['extra'] for x in ood),ood_astar_exact_contour=sum(x['extra']==0 for x in ood),ood_above_optimum=sum(x['above'] for x in ood),roots=out)
    (HERE/'contour.json').write_text(json.dumps(result,sort_keys=True));print({k:v for k,v in result.items() if k!='roots'})
if __name__=='__main__':main()
