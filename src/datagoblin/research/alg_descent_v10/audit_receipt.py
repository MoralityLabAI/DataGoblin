"""Independent arithmetic/closure replay of saved V10 traces (not model choice replay)."""
import json
from pathlib import Path
from collections import OrderedDict, Counter
import numpy as np
from experiment import roots
HERE = Path(__file__).resolve().parent

def replay(root,row):
    trace=iter(row['trace']); event=next(trace,None); m=Counter(); cache=OrderedDict(); used=0
    def bind(a):
        expected=tuple(root.poly(t) for t in a[0]); m['bind_calls']+=1
        m['bind_coeff']+=sum(min(len(c),len(e)) for c,e in zip(a[1],expected))
        return a[1]==expected
    def visit(s):
        nonlocal event,used
        p=root.poly(s)
        if len(p)==2:return True
        key=p if row['identity']=='poly' else s
        if row['cap'] and m['work']<3*root.n**2 and key in cache:
            assert event==[list(s),'hit'];event=next(trace,None);cache.move_to_end(key);m['hits']+=1;m['work']+=1;return True
        bank=root.bank(s);allowed=set(range(len(bank)))
        if row['arm']=='binding':
            allowed={j for j,a in enumerate(bank) if bind(a)};m['prefilter_rejects']+=len(bank)-len(allowed)
        while allowed:
            if event is None or tuple(event[0])!=s:
                assert not row['closed'];return False
            _,j,accepted=event;assert j in allowed;allowed.remove(j);a=bank[j]
            c=np.array([1],dtype=np.int64);cost=0
            for child in a[1]:
                cost+=len(c)*len(child);c=np.convolve(c,np.array(child,dtype=np.int64))%root.p
            prod=tuple(c.tolist())==p;bound=bind(a);assert accepted==(prod and bound)
            m['calls']+=1;m['work']+=cost;assert m['calls']<=8*root.n and m['work']<=3*root.n**2
            event=next(trace,None)
            if not accepted:m['product_invalid']+=not prod;m['unbound']+=prod and not bound;continue
            done=all(visit(t) for t in a[0]);size=(len(p) if row['identity']=='poly' else 4)+16
            if done and size<=row['cap'] and key not in cache:
                while used+size>row['cap']:
                    _,n=cache.popitem(last=False);used-=n
                cache[key]=size;used+=size;m['peak']=max(m['peak'],used);m['admitted']+=1
            return done
        return False
    closed=visit((0,root.n));assert event is None and closed==row['closed']
    keys=('work','calls','product_invalid','unbound','hits','bind_calls','bind_coeff','prefilter_rejects','peak','admitted')
    for k in keys:assert m[k]==row[k],(row['root'],row['arm'],k,m[k],row[k])
    return len(keys)+1

if __name__=='__main__':
    receipt=json.loads((HERE/'receipt.json').read_text());n=fields=0
    for split in ('iid','ood'):
        for root in roots(split):
            for row in receipt['runs']:
                if row['root']==root.seed:
                    fields+=replay(root,row);n+=1
        print(split,n,flush=True)
    out={'routes':n,'fields':fields,'scope':'arithmetic, binding, traversal, cache and counters; no independent inference or action ranking replay'}
    (HERE/'audit_receipt.json').write_text(json.dumps(out,indent=2));print(out)
