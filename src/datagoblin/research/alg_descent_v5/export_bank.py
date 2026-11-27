"""Export fresh shared public candidate banks, and independently replay slot policies."""
import csv
import json
import struct
import sys
import time
from collections import OrderedDict
from pathlib import Path

# In repository this script lives next to alg_descent_v4.py; local v5/ is nested.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alg_descent_v4 import Root, training_data, scores, eligible

CAPS = (0, 64, 128, 256, 512, 1024)

def exact(parent, polys, p):
    if not (2 <= len(polys) <= 8 and len(parent) > 2):
        return False, 0
    if any(len(q) < 2 or len(q) >= len(parent) or q[-1] != 1 or
           any(not 0 <= a < p for a in q) for q in polys):
        return False, 0
    if sum(len(q)-1 for q in polys) != len(parent)-1:
        return False, 0
    cur, paid = [1], 0
    for q in polys:
        paid += len(cur)*len(q)
        cur = [sum(cur[i]*q[j-i] for i in range(len(cur)) if 0 <= j-i < len(q)) % p
               for j in range(len(cur)+len(q)-1)]
    return tuple(cur) == parent, paid


def bank(root, prior):
    spans = [(l,l+d) for d in range(1, root.n+1) for l in range(root.n-d+1)]
    ids = {s:i for i,s in enumerate(spans)}
    nodes = []
    for s in spans:
        parent = root.poly(s)
        actions = []
        if len(parent) > 2:
            a = root.bank(s)
            sc = scores(root,a,prior,'risk',None,[])
            for i in sorted((i for i,x in enumerate(a) if eligible(x)), key=lambda i:(sc[i],i)):
                children, polys, cost, _ = a[i]
                actions.append(tuple((ids[t],p) for t,p in zip(children,polys)))
                # No validity test used to rank or export evaluation candidates.
        nodes.append((parent,actions))
    return nodes


def replay(n,p,nodes,slots,policy):
    cache=OrderedDict(); pending=[]
    work=calls=rejects=hits=peak=pending_peak=0
    def visit(i):
        nonlocal work,calls,rejects,hits,peak,pending_peak
        key,actions=nodes[i]
        if len(key)==2:
            return key[-1]==1 and all(0<=x<p for x in key)
        if key in cache and work<3*n*n:
            cache.move_to_end(key); work+=1; hits+=1; return True
        start=work
        for action in actions:
            polys=[q for _,q in action]
            cost=0; length=1
            for q in polys:
                cost+=length*len(q); length+=len(q)-1
            if work+cost>3*n*n or calls>=8*n:
                return False
            ok,paid=exact(key,polys,p); assert cost==paid
            calls+=1; work+=paid
            if not ok:
                rejects+=1; continue
            assert all(q==nodes[j][0] for j,q in action)
            if slots:
                pending.extend(q for _,q in reversed(action) if len(q)>2)
                pending_peak=max(pending_peak,len(pending))
            for j,q in action:
                if slots and len(q)>2:
                    assert pending.pop()==q
                if not visit(j): return False
            if slots and key not in cache:
                saved=work-start-1
                admit=True
                if len(cache)==slots:
                    value=lambda q:pending.count(q)*cache[q]
                    victim=min(cache,key=value) if policy=='pending' else next(iter(cache))
                    if policy=='pending' and pending.count(key)*saved<value(victim): admit=False
                    if admit: del cache[victim]
                if admit: cache[key]=saved; peak=max(peak,len(cache))
            return True
        return False
    closed=visit(len(nodes)-1)
    assert not closed or not pending
    return dict(closed=int(closed),work=work,calls=calls,rejects=rejects,hits=hits,peak=peak,pending_peak=pending_peak)


def main():
    directory=Path(sys.argv[1] if len(sys.argv)>1 else '.')
    directory.mkdir(exist_ok=True,parents=True)
    slot_bytes=int(sys.argv[2]) if len(sys.argv)>2 else 36
    t=time.perf_counter_ns(); prior,_=training_data(); prior_ns=time.perf_counter_ns()-t
    roots=[]; export_ns=0; bank_ns=0; ref_ns=0; references=[]
    with (directory/'bank.bin').open('wb') as f:
        f.write(b'AD5B'+struct.pack('<I',56))
        for split,base,degrees,count in [('iid',150000,(8,12,16,20),24),('ood',160000,(24,32,40,48),32)]:
            for i in range(count):
                t=time.perf_counter_ns(); r=Root(base+i,degrees[i%4]); nodes=bank(r,prior); bank_ns+=time.perf_counter_ns()-t
                t=time.perf_counter_ns(); f.write(struct.pack('<IBBH',r.seed,r.n,r.p,len(nodes)))
                for key,actions in nodes:
                    f.write(bytes([len(key),*key,len(actions)]))
                    for a in actions:
                        f.write(bytes([len(a)]))
                        for j,q in a:f.write(struct.pack('<H',j)+bytes([len(q),*q]))
                export_ns+=time.perf_counter_ns()-t
                t=time.perf_counter_ns()
                for cap in CAPS:
                    for policy in ('lru','pending') if cap else ('lru',):
                        references.append(dict(seed=r.seed,n=r.n,cap=cap,policy=policy,**replay(r.n,r.p,nodes,cap//slot_bytes,policy)))
                ref_ns+=time.perf_counter_ns()-t
                roots.append(dict(seed=r.seed,n=r.n,p=r.p,split=split,nodes=len(nodes)))
    (directory/'reference.json').write_text(json.dumps(dict(roots=roots,refs=references,slot_bytes=slot_bytes,prior_ns=prior_ns,bank_ns=bank_ns,export_ns=export_ns,reference_ns=ref_ns),sort_keys=True))

if __name__=='__main__': main()
