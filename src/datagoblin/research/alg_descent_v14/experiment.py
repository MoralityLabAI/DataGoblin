"""V14: lazy residual-cost cut refinement for certified Horn planning."""
import heapq,json,math,random,sys,time,statistics as st
from pathlib import Path
HERE=Path(__file__).resolve().parent
if not (HERE/'proof_planning.py').exists():
    for v in (11,12,13):sys.path.insert(0,str(HERE.parent/f'alg_descent_v{v}'))
from proof_planning import Graph,Oracle
from anytime import scope,check,completion,BUDGETS
from bounds import Bounder
ARMS=('hmax','eager','lazy_full','lazy_step'); REPS=7; REF_CAP=4096

def roots(split):
    base,count,ns,fs=(910000,12,(10,11,12,13),('uniform','clustered')) if split=='iid' else (920000,24,(16,17,18),('shared','uniform','clustered'))
    return [Graph(base+i,ns[i%len(ns)],fs[(i+i//len(ns))%len(fs)] if split=='ood' else fs[(i//len(ns)+i)%2]) for i in range(count)]

def search(g,arm):
    start=time.perf_counter_ns(); bounder=Bounder(g,'hmax')
    bc={};fc={};best={g.initial:0};opened={};heap=[];trace=[];snaps=[];serial=0
    c=dict(expansions=0,probes=0,coarse_calls=0,full_bounds=0,cut_rounds=0,refinements=0,bound_scans=0,bound_ns=0,ff_calls=0,ff_ns=0,verify_calls=0,verify_steps=0,duplicates=0,reopened=0,pruned=0,peak_open=0,peak_best=1,peak_queue=0,peak_bound_cache=0)
    def coarse(s):
        if s not in bc:
            t=time.perf_counter_ns();a=list(bounder.costs);d=bounder.distances(s,a);h=d[g.goal]
            bc[s]=dict(a=a,d=d,charge=0,b=h,done=h==0,cuts=[],coarse=h)
            c['coarse_calls']+=1;c['full_bounds']+=int(arm!='hmax' and h==0);c['peak_bound_cache']=max(c['peak_bound_cache'],len(bc));c['bound_ns']+=time.perf_counter_ns()-t
        return bc[s]
    def refine(s,full):
        q=bc[s];t=time.perf_counter_ns()
        while not q['done']:
            a,d=q['a'],q['d'];h=d[g.goal]
            zone=sum(1<<j for j in range(3,g.n) if d[j]>=h and g.relevant>>j&1)
            assert zone&g.goalbit and not zone&s
            bounder.scans+=len(bounder.rules)
            ids=[i for i,b,ps,_ in bounder.rules if b&zone and not ps&zone]
            assert ids and all(a[i]>0 for i in ids)
            delta=min(a[i] for i in ids);bounder.scans+=len(ids)
            for i in ids:a[i]-=delta
            q['charge']+=delta;q['cuts'].append((delta,ids));c['cut_rounds']+=1
            d=bounder.distances(s,a);q['d']=d;q['b']=max(q['b'],q['charge']+d[g.goal]);q['done']=d[g.goal]==0
            assert len(q['cuts'])<=len(g.rules)
            if q['done']:c['full_bounds']+=1
            if not full:break
        c['bound_ns']+=time.perf_counter_ns()-t
    def ff(s):
        if s not in fc:
            t=time.perf_counter_ns();fc[s]=completion(g,s);c['ff_ns']+=time.perf_counter_ns()-t;c['ff_calls']+=1
        return fc[s]
    def verify(path,cost):
        c['verify_calls']+=1;c['verify_steps']+=len(path);assert check(g,path)==cost
    U,inc=ff(g.initial);verify(inc,U)
    def push(s,gc,path,k):
        b=bc[s]['b'];opened[s]=(gc,path,k);heapq.heappush(heap,(gc+b,b,gc,k,s));c['peak_open']=max(c['peak_open'],len(opened));c['peak_queue']=max(c['peak_queue'],len(heap))
    def admit(s,gc,path):
        nonlocal serial
        q=coarse(s)
        if arm=='eager' and not q['done']:refine(s,True)
        if gc+q['b']>=U:c['pruned']+=1;return
        serial+=1;push(s,gc,path,serial)
    def improve(cost,path):
        nonlocal U,inc
        if cost<U:
            verify(path,cost);U,inc=cost,path
            for s,(gc,_,_) in list(opened.items()):
                if gc+bc[s]['b']>=U:del opened[s];c['pruned']+=1
    admit(g.initial,0,())
    for cap in BUDGETS:
        while c['expansions']<cap and opened:
            while heap:
                f,b,gc,k,s=heap[0]
                if s in opened and opened[s][2]==k and b==bc[s]['b']:break
                heapq.heappop(heap)
            else:raise AssertionError('lost frontier')
            gc,path,k=opened[s];assert gc+bc[s]['b']<U
            q=bc[s];need=arm.startswith('lazy') and not q['done']
            if need and c['refinements']>=REF_CAP:break
            heapq.heappop(heap);del opened[s]
            if need:
                old=q['b'];before=len(q['cuts']);refine(s,arm=='lazy_full');c['refinements']+=1
                trace.append(dict(event='refine',state=s,g=gc,old=old,new=q['b'],rounds=len(q['cuts'])-before))
                if gc+q['b']>=U:c['pruned']+=1
                else:push(s,gc,path,k)
                continue
            trace.append(dict(event='expand',state=s,g=gc,path=path));c['expansions']+=1
            f,p=ff(s);improve(gc+f,path+p);c['probes']+=len(g.rules)
            for child,cost,i in g.ready(s):
                new=gc+cost;old=best.get(child,math.inf)
                if new>=old:c['duplicates']+=1;continue
                if old<math.inf and child not in opened:c['reopened']+=1
                best[child]=new;opened.pop(child,None);c['peak_best']=max(c['peak_best'],len(best))
                if child&g.goalbit:improve(new,path+(i,))
                else:admit(child,new,path+(i,))
        L=min(U,min((gc+bc[s]['b'] for s,(gc,_,_) in opened.items()),default=U))
        c['bound_scans']=bounder.scans
        snaps.append(dict(budget=cap,L=L,U=U,proof=inc,open=len(opened),certified=L==U,events=len(trace),elapsed_ns=time.perf_counter_ns()-start,**c))
    certs={s:dict(coarse=q['coarse'],lower=q['b'],done=q['done'],cuts=q['cuts']) for s,q in bc.items()}
    return dict(scope=scope(g),trace=trace,snapshots=snaps,bounds=certs)

def stable(row):
    return dict(row,snapshots=[{k:v for k,v in s.items() if not k.endswith('_ns')} for s in row['snapshots']])

def main():
    rows=[];refs=[]
    for split in ('iid','ood'):
        for g in roots(split):
            first={};timings={a:[] for a in ARMS};bt={a:[] for a in ARMS}
            for rep in range(REPS):
                arms=list(ARMS);random.Random(g.seed+rep*1009).shuffle(arms)
                for a in arms:
                    q=search(g,a);timings[a].append(q['snapshots'][-1]['elapsed_ns']);bt[a].append(q['snapshots'][-1]['bound_ns'])
                    if a not in first:first[a]=q
                    else:assert stable(q)==stable(first[a])
            oracle=Oracle(g);opt=oracle.value(g.initial);refs.append(dict(split=split,root=g.seed,optimum=opt,states=oracle.audit()))
            for a,q in first.items():
                assert all(s['L']<=opt<=s['U'] for s in q['snapshots'])
                assert all(v['lower']<=oracle.values[s] for s,v in q['bounds'].items())
                rows.append(dict(split=split,root=g.seed,n=g.n,family=g.family,arm=a,optimum=opt,times_ns=timings[a],median_ns=st.median(timings[a]),median_bound_ns=st.median(bt[a]),**q))
        print(split,len(rows),flush=True)
    (HERE/'receipt.json').write_text(json.dumps(dict(protocol='a361d4108565640040bf9528981ab8969b8ad82d',repetitions=REPS,runs=rows,reference=refs),sort_keys=True))
if __name__=='__main__':main()
