"""Independent log-space oracle and split-safe DCF-16 curriculum export."""
import argparse, hashlib, json, itertools, collections, math
from pathlib import Path
import numpy as np


def oracle(scopes, factors, query):
    # Separate implementation: log semiring, min-degree on current primal graph,
    # explicit indexing of each factor into assignments rather than broadcasting.
    live=[(tuple(s), np.log(np.asarray(f,dtype=np.float64)).reshape(-1)) for s,f in zip(scopes,factors)]
    variables=set(itertools.chain.from_iterable(scopes))
    while variables-{query}:
        v=min(variables-{query},key=lambda v:(len(set(itertools.chain.from_iterable(s for s,_ in live if v in s))),-v))
        bucket=[(s,f) for s,f in live if v in s];live=[(s,f) for s,f in live if v not in s]
        u=tuple(sorted(set(itertools.chain.from_iterable(s for s,_ in bucket)))); other=tuple(x for x in u if x!=v)
        if len(u)>18: raise ValueError('audit scope cap')
        x=np.arange(1<<len(other)); score=np.zeros((2,len(x)))
        for val in (0,1):
            for s,f in bucket:
                ix=np.zeros(len(x),dtype=np.int64)
                for a in s: ix=2*ix+(val if a==v else (x>>(len(other)-1-other.index(a)))&1)
                score[val]+=f[ix]
        live.append((other,np.logaddexp(score[0],score[1])));variables.remove(v)
    scores=np.zeros(2)
    for s,f in live: scores+=f
    return float(np.exp(scores[1]-np.logaddexp(*scores)))


def make_fresh():
    for n,family,seed in itertools.product((8,16,32),('ladder','clique_chain','grid'),(1901,1907)):
        rng=np.random.default_rng(seed+n+10000*('ladder','clique_chain','grid').index(family)); edges=set()
        def add(a,b):
            if a!=b:edges.add(tuple(sorted((a,b))))
        if family=='ladder':
            for i in range(n//2):
                add(i,i+n//2)
                if i+1<n//2:add(i,i+1);add(i+n//2,i+n//2+1)
        elif family=='clique_chain':
            for i in range(0,n,4):
                for a,b in itertools.combinations(range(i,min(i+4,n)),2):add(a,b)
                if i:add(i-1,i)
        else:
            for i in range(n):
                if i%4!=3 and i+1<n:add(i,i+1)
                if i+4<n:add(i,i+4)
        scopes=[(i,) for i in range(n)]+sorted(edges)
        for _ in range(8):
            yield dict(split='family_holdout',group=f'{n}-{family}-{seed}',query=int(rng.integers(n)),scopes=scopes,factors=[rng.uniform(.1,1,(2,)*len(s)).tolist() for s in scopes])


def main():
    a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(l) for l in args.source.read_text().splitlines()]; rows+=list(make_fresh()); counts=collections.Counter();max_error=0.; ids=set(); audit=collections.Counter();topologies=collections.defaultdict(set)
    for r in rows:
        p=oracle(r['scopes'],r['factors'],r['query']); action=1 if p>=.8 else 0 if p<=.2 else None
        if 'reference_p' in r:
            err=abs(p-r['reference_p']);max_error=max(max_error,err);assert err<1e-11 and action==r['receipt']['action'];audit['source_agreement']+=1
        split={'development':'train','test':'seed_holdout'}.get(r['split'],r['split']);inp=dict(scopes=r['scopes'],factors=r['factors'],query=r['query'],reward={'correct':1,'wrong':-4,'abstain':0});uid=hashlib.sha256(json.dumps(inp,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert uid not in ids;ids.add(uid)
        topology=hashlib.sha256(json.dumps(r['scopes'],separators=(',',':')).encode()).hexdigest();topologies[split].add(topology)
        target=dict(posterior=p,action=action,expected_utility=max(0,5*p-4,1-5*p),verification='independent_logspace_elimination_fp64')
        frame=dict(schema='dcf16-v1',id=uid,split=split,group=r['group'],input=inp,target=target)
        with (args.out/(split+'.jsonl')).open('a') as f:f.write(json.dumps(frame,separators=(',',':'))+'\n')
        counts[(split,str(action))]+=1
    assert not topologies['train']&topologies['family_holdout']
    report=dict(rows=len(rows),source_agreement=audit['source_agreement'],max_source_error=max_error,counts={s:{a:n for (ss,a),n in counts.items() if ss==s} for s in topologies},topology_overlap_train_seed=len(topologies['train']&topologies['seed_holdout']),topology_overlap_train_family=0)
    (args.out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
