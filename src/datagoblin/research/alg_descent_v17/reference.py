"""Export fresh public graphs, execute and independently audit the pinned V15 reference."""
import json, random, statistics, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'prior'))
from proof_planning import Graph, Oracle
from experiment import search,stable,REPS
from audit import audit_graph

ARMS=('hmax','lazy_index')
def roots():
    for split,base,count,ns,fs in [('iid',1210000,12,(10,11,12,13),('uniform','clustered')),('ood',1220000,24,(16,17,18),('shared','uniform','clustered'))]:
        for i in range(count):
            yield split,Graph(base+i,ns[i%len(ns)],fs[(i+i//len(ns))%len(fs)] if split=='ood' else fs[(i//len(ns)+i)%2])

def main():
    graphs=list(roots()); rows=[]; audits=[]
    with (HERE/'bank.txt').open('w') as f:
        print(len(graphs),file=f)
        for _,g in graphs:
            print(g.seed,g.n,len(g.rules),file=f)
            for r in g.rules:print(r.head,r.mask,r.cost,file=f)
    for split,g in graphs:
        first={};times={a:[] for a in ARMS}
        for rep in range(REPS):
            order=list(ARMS);random.Random(g.seed+rep*1009).shuffle(order)
            for arm in order:
                r=search(g,arm);times[arm].append(r['snapshots'][-1]['elapsed_ns'])
                if arm in first:assert stable(r)==stable(first[arm])
                else:first[arm]=r
        oracle=Oracle(g);opt=oracle.value(g.initial);oracle.audit()
        batch=[]
        for a,q in first.items():
            assert all(s['L']<=opt<=s['U'] for s in q['snapshots'])
            assert all(v['lower']<=oracle.values[s] for s,v in q['bounds'].items())
            batch.append(dict(split=split,root=g.seed,n=g.n,family=g.family,arm=a,optimum=opt,times_ns=times[a],median_ns=statistics.median(times[a]),**q))
        # JSON conversion preserves the receipt format expected by V15's audit.
        batch=json.loads(json.dumps(batch));audits.append(audit_graph(g,batch));rows.extend(batch)
        print(g.seed,'audited',flush=True)
    (HERE/'reference.json').write_text(json.dumps(dict(runs=rows,audits=audits),sort_keys=True))
    print({k:sum(q[k] for q in audits) for k in audits[0] if k not in ('root','optimum')})
if __name__=='__main__':main()
