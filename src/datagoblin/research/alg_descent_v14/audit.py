"""Independent V14 cut-prefix, priority, proof and frontier audit."""
import hashlib, json, math, random
from functools import lru_cache
from statistics import mean
from experiment import roots, BUDGETS, HERE, REF_CAP, ARMS, REPS


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
    count=dict(bounds=0,cuts=0,prefixes=0,events=0,expansions=0,refinements=0,snapshots=0,fields=0,proof_steps=0)
    for row in rows:
        assert row['scope']==digest and row['optimum']==opt
        assert len(row['times_ns'])==REPS and sorted(row['times_ns'])[REPS//2]==row['median_ns']
        arm=row['arm'];certs={};used={};opened={};best={initial:0};serial=0;at=0
        c=dict(expansions=0,probes=0,coarse_calls=0,full_bounds=0,cut_rounds=0,refinements=0,bound_scans=0,duplicates=0,reopened=0,pruned=0,peak_open=0,peak_best=1,peak_bound_cache=0,verify_calls=0,verify_steps=0)
        for raw,cert in row['bounds'].items():
            s=decode(int(raw));residual=[x[3] for x in table];charge=0;d=distances(s,residual);b=d[goal]
            assert b==cert['coarse'] and b<=optimum(s)
            prefix=[(b,b==0,sum(len(groups[h]) for h in groups if h not in s))]
            for delta,ids in cert['cuts']:
                h=d[goal];assert h>0
                zone={p for p in ancestors if d[p]>=h};assert goal in zone and not zone&s
                expect={i for i,q,ps,_ in table if q in zone and not set(ps)&zone}
                assert ids and len(ids)==len(set(ids)) and set(ids)==expect
                assert delta>0 and delta==min(residual[i] for i in ids) and not reachable(s,set(ids))
                for i in ids:residual[i]-=delta;assert residual[i]>=0
                charge+=delta;d=distances(s,residual);b=max(b,charge+d[goal]);assert b<=optimum(s)
                prefix.append((b,d[goal]==0,len(table)+len(ids)+prefix[0][2]));count['cuts']+=1
            assert b==cert['lower'] and (d[goal]==0)==cert['done']
            assert arm!='eager' or cert['done'];assert arm!='hmax' or not cert['cuts']
            certs[s]=prefix;count['bounds']+=1;count['prefixes']+=len(prefix)
        def low(s):return certs[s][used[s]][0]
        def step(s,n):
            for _ in range(n):
                assert not certs[s][used[s]][1]
                used[s]+=1;b,done,scans=certs[s][used[s]];c['bound_scans']+=scans;c['cut_rounds']+=1;c['full_bounds']+=done
        def push(s,gc,path,k):
            opened[s]=(gc,path,k);c['peak_open']=max(c['peak_open'],len(opened))
        def admit(s,gc,path):
            nonlocal serial
            if s not in used:
                used[s]=0;c['coarse_calls']+=1;c['bound_scans']+=certs[s][0][2];c['peak_bound_cache']=max(c['peak_bound_cache'],len(used));c['full_bounds']+=int(arm!='hmax' and certs[s][0][1])
            if arm=='eager':step(s,len(certs[s])-1-used[s]);assert certs[s][used[s]][1]
            if gc+low(s)>=U:c['pruned']+=1;return
            serial+=1;push(s,gc,path,serial)
        def verify(path,cost):
            assert proof_cost(path)==cost;c['verify_calls']+=1;c['verify_steps']+=len(path)
        U,inc=completion(initial);verify(inc,U)
        def improve(cost,path):
            nonlocal U,inc
            if cost<U:
                verify(path,cost);U,inc=cost,path
                for s,(gc,_,_) in list(opened.items()):
                    if gc+low(s)>=U:del opened[s];c['pruned']+=1
        admit(initial,0,())
        for snap,cap in zip(row['snapshots'],BUDGETS):
            assert snap['budget']==cap
            while at<snap['events']:
                assert c['expansions']<cap and opened
                e=row['trace'][at];s=decode(e['state']);chosen=min(opened,key=lambda t:(opened[t][0]+low(t),low(t),opened[t][0],opened[t][2]));assert chosen==s
                gc,path,k=opened.pop(s);assert e['g']==gc and gc+low(s)<U
                need=arm.startswith('lazy') and not certs[s][used[s]][1]
                if need:
                    assert e['event']=='refine' and e['old']==low(s) and c['refinements']<REF_CAP
                    n=len(certs[s])-1-used[s] if arm=='lazy_full' else 1
                    assert e['rounds']==n and n>0;step(s,n);assert e['new']==low(s)
                    if arm=='lazy_full':assert certs[s][used[s]][1]
                    c['refinements']+=1
                    if gc+low(s)>=U:c['pruned']+=1
                    else:push(s,gc,path,k)
                else:
                    assert e['event']=='expand' and tuple(e['path'])==path and proof_cost(path,s)==gc
                    c['expansions']+=1;c['probes']+=len(table);f,p=completion(s);improve(gc+f,path+p)
                    for t,cost,i in transitions(s):
                        new=gc+cost;old=best.get(t,math.inf)
                        if new>=old:c['duplicates']+=1;continue
                        if old<math.inf and t not in opened:c['reopened']+=1
                        best[t]=new;opened.pop(t,None);c['peak_best']=max(c['peak_best'],len(best))
                        if goal in t:improve(new,path+(i,))
                        else:admit(t,new,path+(i,))
                at+=1
            L=min(U,min((gc+low(s) for s,(gc,_,_) in opened.items()),default=U));assert L<=opt<=U
            want=dict(L=L,U=U,proof=list(inc),open=len(opened),certified=L==U,events=at,**c)
            for k,v in want.items():assert snap[k]==v,(row['root'],arm,k,snap[k],v)
            assert proof_cost(snap['proof'])==U and c['expansions']<=cap
            assert c['expansions']==cap or not opened or c['refinements']==REF_CAP
            count['fields']+=len(want);count['snapshots']+=1;count['proof_steps']+=len(inc)
        assert at==len(row['trace']) and used.keys()==certs.keys() and all(used[s]==len(certs[s])-1 for s in used)
        count['events']+=at;count['expansions']+=c['expansions'];count['refinements']+=c['refinements']
    return dict(root=g.seed,optimum=opt,oracle_states=optimum.cache_info().currsize,**count)


def main():
    data=json.loads((HERE/'receipt.json').read_text());rows=data['runs'];audits=[]
    for split in ('iid','ood'):
        for g in roots(split):audits.append(audit_graph(g,[r for r in rows if r['root']==g.seed]))
        print(split,len(audits),flush=True)
    base={r['root']:r for r in rows if r['split']=='ood' and r['arm']=='eager'};pairs=[]
    for arm in ('lazy_full','lazy_step'):
        for name,fn in [('scans',lambda r:r['snapshots'][-1]['bound_scans']),('expansions',lambda r:r['snapshots'][-1]['expansions']),('median_ns',lambda r:r['median_ns'])]:
            d=[fn(r)-fn(base[r['root']]) for r in rows if r['split']=='ood' and r['arm']==arm];rng=random.Random(14014);boot=sorted(mean(rng.choices(d,k=len(d))) for _ in range(5000))
            pairs.append(dict(arm=arm,metric=name,mean=mean(d),wins=sum(v<0 for v in d),ties=sum(v==0 for v in d),losses=sum(v>0 for v in d),interval=[boot[125],boot[4874]]))
    out=dict(roots=len(audits),**{k:sum(r[k] for r in audits) for k in audits[0] if k not in ('root','optimum')},pairs=pairs,root_audits=audits,scope='Independent set-based Bellman, cut-prefix, event priority, immutable proof, counters and frontier replay; timing not independently measured.')
    (HERE/'audit.json').write_text(json.dumps(out,sort_keys=True));print({k:v for k,v in out.items() if k!='root_audits'})
if __name__=='__main__':main()
