"""Independent cost-partition, set-based proof, priority and frontier audit."""
import hashlib, json, math, random
from functools import lru_cache
from statistics import mean
from experiment import roots, BUDGETS, HERE


def audit_graph(g, rows):
    table = [(r.id, r.head, tuple(r.premises), r.cost) for r in g.rules]
    initial = frozenset(range(3)); goal = g.goal
    groups = {h: [(i, ps, c) for i, q, ps, c in table if q==h]
              for h in range(3, g.n)}
    ancestors = {goal}
    for h in range(goal, 2, -1):
        if h in ancestors:
            for _, ps, _ in groups[h]: ancestors.update(ps)
    digest = hashlib.sha256(json.dumps([g.n, sum(2**x for x in initial), goal, table],
                                       separators=(',',':')).encode()).hexdigest()
    def decode(mask): return frozenset(i for i in range(g.n) if mask//2**i%2)
    def encode(s): return sum(2**i for i in s)

    def proof_cost(path, want=None):
        known = set(initial); paid = 0
        for i in path:
            assert type(i) is int and 0 <= i < len(table)
            rid, h, ps, c = table[i]
            assert rid == i and h not in known and set(ps) <= known
            known.add(h); paid += c
        assert frozenset(known)==want if want is not None else goal in known
        return paid

    @lru_cache(None)
    def transitions(s):
        out = {}
        for i, h, ps, c in table:
            if h in s or not set(ps)<=s: continue
            t = s|{h}
            if t not in out or (c,i)<out[t]: out[t] = (c,i)
        return tuple((t,c,i) for t,(c,i) in out.items())

    @lru_cache(None)
    def optimum(s):
        return 0 if goal in s else min(c+optimum(t) for t,c,_ in transitions(s))

    @lru_cache(None)
    def completion(s):
        @lru_cache(None)
        def additive(h):
            if h in s: return 0,None
            return min((c+sum(additive(p)[0] for p in ps),i) for i,ps,c in groups[h])
        ids = set()
        def collect(h):
            _, i = additive(h)
            if i is None or i in ids: return
            ids.add(i)
            for p in table[i][2]: collect(p)
        collect(goal)
        return sum(table[i][3] for i in ids),tuple(sorted(ids,key=lambda i:table[i][1]))

    def reachable(s, forbidden_ids):
        known = set(s)
        # The public graph is topologically ordered by head.
        for i,h,ps,_ in table:
            if i not in forbidden_ids and set(ps)<=known: known.add(h)
        return goal in known

    @lru_cache(None)
    def mandatory(s):
        return {h for h in ancestors-s if h>=3 and
                not reachable(s,{i for i,_,_ in groups[h]})}

    def distances(s, residual):
        @lru_cache(None)
        def dist(h):
            if h in s: return 0
            return min(residual[i]+max((dist(p) for p in ps), default=0)
                       for i,ps,_ in groups[h])
        return {h:dist(h) for h in range(g.n)}

    opt=optimum(initial)
    count=dict(bound_certificates=0, cuts=0, mandatory_checks=0, expansions=0,
               fields=0, snapshots=0, proof_steps=0, priority_checks=0)
    for row in rows:
        assert row['scope']==digest and row['optimum']==opt
        arm=row['arm']; certified_bounds={}; seen=set()
        for raw, cert in row['bounds'].items():
            s=decode(int(raw)); residual=[c for _,_,_,c in table]
            mand=mandatory(s) if arm in ('mandatory','combined') else set()
            if arm in ('mandatory','combined'): count['mandatory_checks']+=1
            assert encode(mand)==cert['mandatory']
            charge=best=0
            for j in range(len(cert['cuts'])+1):
                d=distances(s,residual); h=d[goal]
                m=sum(min(residual[i] for i,_,_ in groups[p]) for p in mand)
                best=max(best, charge+max(h,m))
                if j==len(cert['cuts']):
                    assert arm in ('hmax','mandatory') or h==0
                    break
                assert arm in ('cut','combined') and h>0
                delta, ids=cert['cuts'][j]; zone={p for p in ancestors if d[p]>=h}
                expected={i for i,q,ps,_ in table if q in zone and not set(ps)&zone}
                assert ids and len(set(ids))==len(ids) and set(ids)==expected
                assert delta>0 and delta==min(residual[i] for i in ids)
                assert not reachable(s,set(ids))
                for i in ids:
                    residual[i]-=delta
                    assert residual[i]>=0
                charge+=delta; count['cuts']+=1
            assert best==cert['lower'] and 0<=best<=optimum(s)
            certified_bounds[s]=best;count['bound_certificates']+=1

        def low(s):
            seen.add(s);return certified_bounds[s]
        U,inc=completion(initial);assert proof_cost(inc)==U
        best={initial:0};opened={};serial=0;at=0
        def admit(s,gc,path):
            nonlocal serial
            b=low(s)
            if gc+b<U:
                serial+=1;opened[s]=(gc,path,b,serial)
        def improve(c,path):
            nonlocal U,inc
            if c<U:
                assert proof_cost(path)==c
                U,inc=c,path
                for t,(gc,_,b,_) in list(opened.items()):
                    if gc+b>=U:del opened[t]
        admit(initial,0,())
        for snap,cap in zip(row['snapshots'],BUDGETS):
            assert snap['budget']==cap
            while at<snap['expansions']:
                e=row['trace'][at];s=decode(e['state'])
                chosen=min(opened,key=lambda t:(opened[t][0]+opened[t][2],opened[t][2],opened[t][0],opened[t][3]))
                assert s==chosen;count['priority_checks']+=1
                gc,path,b,_=opened.pop(s)
                assert e['g']==gc and tuple(e['path'])==path and proof_cost(path,s)==gc and gc+b<U
                f,p=completion(s);improve(gc+f,path+p)
                for t,c,i in transitions(s):
                    new=gc+c
                    if new>=best.get(t,math.inf):continue
                    best[t]=new;opened.pop(t,None)
                    if goal in t:improve(new,path+(i,))
                    else:admit(t,new,path+(i,))
                at+=1;count['expansions']+=1
            L=min(U,min((gc+b for gc,_,b,_ in opened.values()),default=U))
            assert L<=opt<=U
            want=dict(L=L,U=U,proof=list(inc),open=len(opened),certified=L==U,
                      expansions=at,probes=at*len(table),bound_calls=len(seen),
                      cut_rounds=sum(len(row['bounds'][str(encode(s))]['cuts']) for s in seen))
            for k,v in want.items():assert snap[k]==v,(row['root'],arm,k,snap[k],v)
            assert proof_cost(snap['proof'])==U and at<=cap and (at==cap or not opened)
            count['fields']+=len(want);count['snapshots']+=1;count['proof_steps']+=len(inc)
        assert at==len(row['trace']) and seen==set(certified_bounds)
    return dict(root=g.seed,optimum=opt,oracle_states=optimum.cache_info().currsize,**count)


def main():
    data=json.loads((HERE/'receipt.json').read_text());rows=data['runs'];audits=[]
    for split in ('iid','ood'):
        for g in roots(split):audits.append(audit_graph(g,[r for r in rows if r['root']==g.seed]))
        print(split,len(audits),flush=True)
    base={r['root']:r for r in rows if r['split']=='ood' and r['arm']=='hmax'};pairs=[]
    for arm in ('mandatory','cut','combined'):
        a=[r for r in rows if r['split']=='ood' and r['arm']==arm]
        for name,fn in [('expansions',lambda r:r['snapshots'][-1]['expansions']),
                        ('gap16',lambda r:(r['snapshots'][2]['U']-r['snapshots'][2]['L'])/r['optimum'])]:
            d=[fn(r)-fn(base[r['root']]) for r in a];rng=random.Random(13013)
            boot=sorted(mean(rng.choices(d,k=len(d))) for _ in range(5000))
            pairs.append(dict(arm=arm,metric=name,mean=mean(d),wins=sum(v<0 for v in d),
                              ties=sum(v==0 for v in d),losses=sum(v>0 for v in d),interval=[boot[125],boot[4874]]))
    out=dict(roots=len(audits),**{k:sum(r[k] for r in audits) for k in audits[0] if k not in ('root','optimum')},
             pairs=pairs,root_audits=audits,
             scope='Independent set-based exact optimum, cost partition cuts, mandatory facts, priorities, frontier, proof and snapshot replay.')
    (HERE/'audit.json').write_text(json.dumps(out,sort_keys=True));print({k:v for k,v in out.items() if k!='root_audits'})

if __name__=='__main__':main()
