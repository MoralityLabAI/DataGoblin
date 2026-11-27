"""Independent set-based proof replay and retrospective direct FF extraction."""
from pathlib import Path
from collections import defaultdict
from statistics import mean
import json, hashlib, random, time
from proof_planning import roots, CAP
HERE=Path(__file__).resolve().parent

def registry(g):
    return [(r.id,r.head,list(r.premises),r.cost) for r in g.rules]

def check(graph,proof,closed,expected):
    initial=set(graph['initial']);known=set(initial);rules=graph['rules'];total=0
    if not closed:
        assert not proof and expected is None
        return 0
    for i in proof:
        assert type(i) is int and 0<=i<len(rules)
        rid,head,premises,cost=rules[i]
        assert rid==i and head not in known and set(premises)<=known
        known.add(head);total+=cost
    assert graph['goal'] in known and total==expected
    return len(proof)

def direct_ff(graph):
    """One chosen additive-relaxation derivation per head, shared by exact fact ID."""
    initial=set(graph['initial']);rules=graph['rules'];group=defaultdict(list)
    for r in rules:group[r[1]].append(r)
    start=time.perf_counter_ns();cost={i:0 for i in initial};choice={}
    for h in range(3,graph['n']):
        c,i=min((r[3]+sum(cost[p] for p in r[2]),r[0]) for r in group[h]);cost[h]=c;choice[h]=i
    selected=set();seen=set(initial)
    def visit(h):
        if h in seen:return
        i=choice[h];r=rules[i]
        for p in r[2]:visit(p)
        selected.add(i);seen.add(h)
    visit(graph['goal']);proof=sorted(selected,key=lambda i:rules[i][1]);paid=sum(rules[i][3] for i in proof)
    return dict(proof=proof,cost=paid,elapsed_ns=time.perf_counter_ns()-start,rule_scans=len(rules))

def main():
    receipt=json.loads((HERE/'receipt.json').read_text());graphs=[];byroot={};direct=[];steps=closed=0;negative=0
    for split in ('iid','ood'):
        for g in roots(split):
            graph=dict(split=split,root=g.seed,n=g.n,initial=[0,1,2],goal=g.goal,rules=registry(g),registry=g.registry)
            assert hashlib.sha256(json.dumps(graph['rules'],separators=(',',':')).encode()).hexdigest()==g.registry
            graphs.append(graph);byroot[g.seed]=graph
            ref=next(r for r in receipt['reference'] if r['root']==g.seed)
            q=direct_ff(graph);assert q['cost']==g.public(g.initial)[2];check(graph,q['proof'],True,q['cost'])
            direct.append(dict(split=split,root=g.seed,optimum=ref['optimum'],**q))
    for row in receipt['runs']:
        graph=byroot[row['root']];assert row['registry']==graph['registry'];steps+=check(graph,row['proof'],row['closed'],row['cost']);closed+=row['closed']
        assert row['expansions']<=CAP and row['probes']==row['expansions']*len(graph['rules'])
        assert not row['closed'] or row['cost']>=row['optimum']
    assert steps==receipt['proof_steps_checked']
    # Explicit malformed rule-ID, duplicate head, and missing premise rejection fixtures.
    graph=graphs[0]
    valid=next(q for q in direct if q['root']==graph['root']);proof=valid['proof']
    for bad in ([len(graph['rules'])], [proof[0],proof[0]], [proof[-1]]):
        try:check(graph,bad,True,valid['cost'])
        except AssertionError:negative+=1
        else:raise AssertionError('malformed certificate accepted')
    (HERE/'graphs.json').write_text(json.dumps(graphs,sort_keys=True))
    ood=[q for q in direct if q['split']=='ood']
    summary=dict(rows=len(receipt['runs']),closed=closed,proof_steps=steps,negative_fixtures=negative,graphs=len(graphs),direct_ff=direct,direct_ff_ood_mean_ratio=mean(q['cost']/q['optimum'] for q in ood),direct_ff_ood_optimal=sum(q['cost']==q['optimum'] for q in ood),direct_ff_ood_mean_ms=mean(q['elapsed_ns'] for q in ood)/1e6)
    (HERE/'audit.json').write_text(json.dumps(summary,sort_keys=True));print({k:v for k,v in summary.items() if k!='direct_ff'})
if __name__=='__main__':main()
