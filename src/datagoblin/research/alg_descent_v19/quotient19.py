"""Retrospective exact live-cone quotient check on the spent V19 cohort."""
import json
from functools import lru_cache
from statistics import mean
from pathlib import Path
from audit19 import load_bank
HERE=Path(__file__).resolve().parent

def analyze(n,rules):
    initial=frozenset(range(3));goal=n-1
    groups={h:[] for h in range(3,n)}
    for i,h,p,c in rules:groups[h].append((i,p,c))
    @lru_cache(None)
    def cone(s):
        seen=set()
        def walk(h):
            if h in s or h in seen:return
            seen.add(h)
            for _,p,_ in groups[h]:
                for q in p:walk(q)
        walk(goal)
        return frozenset(seen)
    @lru_cache(None)
    def actions(s):
        return tuple((i,h,c) for i,h,p,c in rules if h not in s and p<=s)
    @lru_cache(None)
    def value(s):
        if goal in s:return 0
        return min(c+value(s|{h}) for _,h,c in actions(s))
    value(initial);stack=[initial];seen=set();quotient={};strict=0;edges=0
    while stack:
        s=stack.pop()
        if s in seen:continue
        seen.add(s);a=cone(s)
        signature=tuple((i,h,c,tuple(sorted(cone(s|{h})))) for i,h,c in actions(s) if h in a)
        v=value(s);record=(v,signature)
        if a in quotient:assert quotient[a]==record
        else:quotient[a]=record
        for _,h,c in actions(s):
            if goal not in s:stack.append(s|{h})
        for i,h,c,new in signature:
            assert set(new)<a;edges+=1
        strict+=int(len(a)<len(set(range(3,n))-s))
    return dict(states=len(seen),classes=len(quotient),live_strict=strict,edges=edges)

if __name__=='__main__':
    roots=load_bank(HERE/'bank19.txt');rows=[dict(root=s,**analyze(n,r)) for s,(n,r) in roots.items()]
    total={k:sum(r[k] for r in rows) for k in ('states','classes','live_strict','edges')}
    ood=[r for r in rows if r['root']>=1420000]
    result=dict(total=total,ood_states=sum(r['states'] for r in ood),ood_classes=sum(r['classes'] for r in ood),ood_mean_compression=mean(1-r['classes']/r['states'] for r in ood),roots=rows)
    (HERE/'quotient19.json').write_text(json.dumps(result,sort_keys=True,indent=2));print({k:v for k,v in result.items() if k!='roots'})
