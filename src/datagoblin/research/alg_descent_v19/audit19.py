"""Independent set-based replay of native V19 traces and matched-work snapshots."""
from pathlib import Path
from collections import Counter, defaultdict
from functools import lru_cache
from statistics import mean, median
import hashlib, heapq, json, math, random, sys
HERE = Path(__file__).resolve().parent
ARMS=('full','static','live_transition','live_all')
ECAPS=(0,4,16,32,64,128)
WCAPS=(256,1024,4096)

def load_bank(path):
    vals=iter(map(int,path.read_text().split())); roots={}
    for _ in range(next(vals)):
        seed,n,nr=(next(vals) for _ in range(3)); rules=[]
        for i in range(nr):
            h,m,c=(next(vals) for _ in range(3)); p=frozenset(j for j in range(n) if m>>j&1)
            assert 3<=h<n and p and max(p)<h and 1<=c<=7
            rules.append((i,h,p,c))
        assert seed not in roots and set(h for _,h,_,_ in rules)==set(range(3,n))
        roots[seed]=(n,rules)
    assert next(vals,None) is None
    return roots

def parse(path):
    rows={}; layout=rss=None; names=[]
    for line in path.read_text().splitlines():
        tag,*a=line.split()
        if tag=='COUNTERS': names=a; continue
        v=list(map(int,a))
        if tag=='LAYOUT': layout=v
        elif tag=='RSS': rss=v[0]
        elif tag=='R':
            key=tuple(v); assert key not in rows; row=dict(events=[],S=[],W=[]); rows[key]=row
        elif tag in ('B','C','E'): row['events'].append((tag,v))
        elif tag in ('S','W'):
            d=dict(zip(('cap','L','U','open')+tuple(names),v)); size=4+len(names); assert v[size]==len(v[size+1:]);d['proof']=v[size+1:];row[tag].append(d)
        elif tag=='T':row['times']=v
        elif tag=='M':row['mem']=dict(zip(('median_ns','peak_requested','live_requested','total_requested','allocations','trace_hash','reps'),v))
        else:raise ValueError(tag)
    return rows,layout,rss,names

def work(d):return sum(d[x] for x in ('probes','distance_scans','ff_scans','index_scans'))

def audit_graph(n,rules,rows,names):
    init=frozenset(range(3));goal=n-1; groups={h:[r for r in rules if r[1]==h] for h in range(3,n)}
    ancestors={goal};desc={h:{h} for h in range(n)}
    for h in range(goal,2,-1):
        if h in ancestors:
            for _,_,p,_ in groups[h]:ancestors.update(p)
    for h in range(n-1,-1,-1):
        for _,q,p,_ in rules:
            if h in p:desc[h].update(desc[q])
    enc=lambda s:sum(1<<i for i in s)
    dec=lambda s:frozenset(i for i in range(n) if s>>i&1)
    costs=tuple(r[3] for r in rules)
    @lru_cache(None)
    def transitions(s):
        out={}
        for i,h,p,c in rules:
            if h not in s and p<=s:
                t=s|{h};out[t]=min(out.get(t,(math.inf,math.inf)),(c,i))
        return sorted(((t,c,i) for t,(c,i) in out.items()),key=lambda v:rules[v[2]][1])
    @lru_cache(None)
    def optimum(s):return 0 if goal in s else min(c+optimum(t) for t,c,_ in transitions(s))
    @lru_cache(None)
    def completion(s):
        @lru_cache(None)
        def additive(h):
            if h in s:return 0,None
            return min((c+sum(additive(q)[0] for q in p),i) for i,_,p,c in groups[h])
        ids=set()
        def collect(h):
            _,i=additive(h)
            if i is None or i in ids:return
            ids.add(i)
            for q in rules[i][2]:collect(q)
        collect(goal);return sum(costs[i] for i in ids),tuple(sorted(ids,key=lambda i:rules[i][1]))
    def proof_cost(path,want=None):
        known=set(init);paid=0
        for i in path:
            assert 0<=i<len(rules);rid,h,p,c=rules[i];assert rid==i and h not in known and p<=known
            known.add(h);paid+=c
        assert (frozenset(known)==want if want is not None else goal in known)
        return paid
    def distances(s,a):
        @lru_cache(None)
        def d(h):return 0 if h in s else min(a[i]+max((d(q) for q in p),default=0) for i,_,p,_ in groups[h])
        return [d(h) for h in range(n)]
    def reachable(s,cut):
        known=set(s)
        for i,h,p,_ in rules:
            if i not in cut and p<=known:known.add(h)
        return goal in known
    @lru_cache(None)
    def live_cone(s):
        found=set()
        def rec(h):
            if h in s or h in found:return
            found.add(h)
            for _,_,p,_ in groups[h]:
                for q in p:rec(q)
        rec(goal)
        return frozenset(found)
    @lru_cache(None)
    def restricted(s,mode):
        if goal in s:return 0
        cone=(ancestors-s) if mode==1 else live_cone(s)
        return min(c+restricted(t,mode) for t,c,i in transitions(s) if rules[i][1] in cone)
    opt=optimum(init);cnt=Counter(oracle_states=optimum.cache_info().currsize)
    # Walk every full-reachable state, including goal-terminal states.
    stack=[init];visited=set()
    while stack:
        state=stack.pop()
        if state in visited:continue
        visited.add(state)
        val=optimum(state)
        assert restricted(state,1)==val==restricted(state,2)
        cnt['filtered_optima']+=2
        if goal not in state:
            for t,_,_ in transitions(state):
                assert live_cone(t)<=live_cone(state)
                cnt['cone_monotonicity']+=1;stack.append(t)
    assert len(visited)==cnt['oracle_states']
    for arm,row in rows.items():
        c=Counter({k:0 for k in names});bc={};ffcache=set();best={init:0};opened={};queue=[];serial=0;cursor=0;work_snaps={}
        cached_cones={}
        c['index_scans']=len(rules) if arm>=2 else 0
        def support(s,expand=False):
            if arm==1:return ancestors-s
            if arm==3:
                if s not in cached_cones:
                    cached_cones[s]=live_cone(s);c['support_heads']+=len(cached_cones[s])
                return cached_cones[s]
            if arm==2 and expand:
                ans=live_cone(s);c['support_heads']+=len(ans);return ans
            return set(groups)-s
        def ff(s):
            if s not in ffcache:
                ffcache.add(s);c['ff_calls']+=1
                c['ff_scans']+=sum(len(groups[h]) for h in support(s))
            return completion(s)
        U,inc=ff(init)
        def verify(path,v):
            assert proof_cost(path)==v;c['verify_calls']+=1;c['verify_steps']+=len(path)
        verify(inc,U)
        def event(tag,values):
            nonlocal cursor
            assert cursor<len(row['events']) and row['events'][cursor]==(tag,values),(arm,cursor,tag,values,row['events'][cursor:cursor+1]);cursor+=1
        def low(s):return bc[s]['b']
        def push(s,gc,path,k):
            opened[s]=(gc,path,k);heapq.heappush(queue,(gc+low(s),low(s),gc,k,enc(s),s));c['peak_open']=max(c['peak_open'],len(opened));c['peak_queue']=max(c['peak_queue'],len(queue))
        def admit(s,gc,path):
            nonlocal serial
            if s not in bc:
                d=distances(s,costs);b=d[goal];assert b<=optimum(s)
                bc[s]=dict(a=list(costs),d=d,b=b,charge=0,rounds=0)
                event('B',[enc(s),b]);c['coarse_calls']+=1;c['peak_bound_cache']=len(bc)
                c['distance_scans']+=sum(len(groups[h]) for h in support(s));cnt['coarse']+=1
            if gc+low(s)>=U:c['pruned']+=1;return
            serial+=1;push(s,gc,path,serial)
        def improve(v,path):
            nonlocal U,inc
            if v>=U:return
            verify(path,v);U,inc=v,path
            for t,(gc,_,_) in list(opened.items()):
                if gc+low(t)>=U:del opened[t];c['pruned']+=1
        def snapshot(cap):
            L=min(U,min((gc+low(t) for t,(gc,_,_) in opened.items()),default=U));assert L<=opt<=U
            assert proof_cost(inc)==U
            return dict(cap=cap,L=L,U=U,open=len(opened),proof=list(inc),**c)
        def record_work():
            for cap in WCAPS:
                if work(c)<=cap:work_snaps[cap]=snapshot(cap)
        admit(init,0,());c['peak_best']=1;record_work()
        for cap,saved in zip(ECAPS,row['S']):
            while c['expansions']<cap and opened:
                t=min(opened,key=lambda s:(opened[s][0]+low(s),low(s),opened[s][0],opened[s][2]));gc,path,k=opened[t]
                while queue:
                    _,b,_,serial0,_,s=queue[0]
                    if s in opened and opened[s][2]==serial0 and low(s)==b:break
                    heapq.heappop(queue)
                assert queue and queue[0][-1]==t
                assert U-gc-low(t)>0
                heapq.heappop(queue);del opened[t]
                assert proof_cost(path,t)==gc
                event('E',[enc(t),gc,len(path),*path]);c['expansions']+=1;cnt['expansions']+=1
                f,p=ff(t);improve(gc+f,path+p)
                eligible=support(t,True);c['probes']+=sum(len(groups[h]) for h in eligible)
                for new,cost,i in transitions(t):
                    if rules[i][1] not in eligible:continue
                    v=gc+cost;old=best.get(new,math.inf)
                    if v>=old:c['duplicates']+=1;continue
                    if old<math.inf and new not in opened:c['reopened']+=1
                    best[new]=v;opened.pop(new,None);c['peak_best']=max(c['peak_best'],len(best))
                    if goal in new:improve(v,path+(i,))
                    else:admit(new,v,path+(i,))
                record_work()
            expected=snapshot(cap);assert expected==saved, (arm,cap,[(k,v,saved.get(k)) for k,v in expected.items() if saved.get(k)!=v]);cnt['fields']+=len(expected);cnt['expansion_snapshots']+=1
        assert cursor==len(row['events']);assert len(row['W'])==len(WCAPS) and len(row['S'])==len(ECAPS)
        for cap,saved in zip(WCAPS,row['W']):
            assert work_snaps[cap]==saved,(arm,cap);cnt['fields']+=len(saved);cnt['work_snapshots']+=1
        cnt['cases']+=1;cnt['events']+=cursor
        assert row['mem']['reps']==len(row['times']) and row['mem']['median_ns']==median(row['times'])
        assert row['mem']['live_requested']<=row['mem']['peak_requested']<=row['mem']['total_requested']
    return opt,dict(cnt)

def main():
    bank=load_bank(HERE/'bank19.txt');rows,layout,rss,names=parse(HERE/'native19.txt');assert len(rows)==36*4
    stats=Counter();opts={}
    for seed,(n,rules) in bank.items():
        opt,count=audit_graph(n,rules,{arm:rows[seed,arm] for arm in range(4)},names);opts[seed]=opt;stats.update(count)
    if (HERE/'sanitized19.txt').exists():
        san,sl,_,sn=parse(HERE/'sanitized19.txt');assert sl==layout and sn==names and rows.keys()==san.keys()
        for k,r in rows.items():
            q=san[k];assert all(r[t]==q[t] for t in ('events','S','W'));assert all(r['mem'][t]==q['mem'][t] for t in ('peak_requested','live_requested','total_requested','allocations','trace_hash'))
    seeds=sorted(s for s in bank if s>=1420000);summary={};pairs={}
    for arm,name in enumerate(ARMS):
        rs=[rows[s,arm] for s in seeds];final=[r['S'][-1] for r in rs]
        summary[name]=dict(certified=sum(r['L']==r['U'] for r in final),expansions=mean(r['expansions'] for r in final),refinements=mean(r['refinements'] for r in final),support_heads=mean(r['support_heads'] for r in final),probes=mean(r['probes'] for r in final),distance_scans=mean(r['distance_scans'] for r in final),ff_scans=mean(r['ff_scans'] for r in final),index_scans=mean(r['index_scans'] for r in final),inspections=mean(work(r) for r in final),time_ns=mean(r['mem']['median_ns'] for r in rs),peak_bytes=mean(r['mem']['peak_requested'] for r in rs),max_peak_bytes=max(r['mem']['peak_requested'] for r in rs),allocations=mean(r['mem']['allocations'] for r in rs),work_caps={cap:dict(gap=mean((r['W'][j]['U']-r['W'][j]['L'])/opts[s] for s,r in zip(seeds,rs)),used=mean(work(r['W'][j]) for r in rs),certified=sum(r['W'][j]['L']==r['W'][j]['U'] for r in rs)) for j,cap in enumerate(WCAPS)})
        if arm==0:continue
        for metric,func in [('time_ns',lambda r,s:r['mem']['median_ns']),('inspections',lambda r,s:work(r['S'][-1])),('peak_bytes',lambda r,s:r['mem']['peak_requested']),('expansions',lambda r,s:r['S'][-1]['expansions']),('gap1024',lambda r,s:(r['W'][1]['U']-r['W'][1]['L'])/opts[s])]:
            ds=[func(rows[s,arm],s)-func(rows[s,0],s) for s in seeds];rng=random.Random(19019);boot=sorted(mean(rng.choices(ds,k=len(ds))) for _ in range(5000));pairs[name+':'+metric]=dict(mean=mean(ds),wins=sum(d<0 for d in ds),ties=sum(d==0 for d in ds),losses=sum(d>0 for d in ds),bootstrap95=[boot[125],boot[4874]])
    out=dict(stats=dict(stats),optima=opts,summary=summary,pairs=pairs,layout=layout,rss_kib=rss,sanitized=(HERE/'sanitized19.txt').exists(),bank_sha256=hashlib.sha256((HERE/'bank19.txt').read_bytes()).hexdigest())
    (HERE/'audit19.json').write_text(json.dumps(out,indent=2,sort_keys=True));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
