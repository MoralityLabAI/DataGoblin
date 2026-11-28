"""SG-28: exact last-use disjunction certificates; numerical proposals only."""
from datagoblin.training_gym.spectral_quotient import score, validate
from datagoblin.training_gym.spectral_routing import TaskError


def domains(q):
    return tuple((0, q['T']) for _ in range(q['d']))


def split(q, box, j, cut):
    if type(j) is not int or not 0 <= j < q['d'] or type(cut) is not int:
        raise TaskError('invalid split')
    lo, hi = box[j]
    if not lo <= cut < hi:
        raise TaskError('nonpartitioning split')
    a, b = list(box), list(box)
    a[j], b[j] = (lo, cut), (cut + 1, hi)
    return tuple(a), tuple(b)


def conditioned(q, box, eta, den):
    if type(den) is not int or not 1 <= den <= 1_000_000 or not isinstance(eta,list) or len(eta)!=q['T'] or any(type(v) is not int or abs(v)>10**15 for v in eta):
        raise TaskError('invalid rational multipliers')
    total = sum(a*b for a,b in zip(eta,q['k'],strict=True))
    for j,(lo,hi) in enumerate(box):
        prefix=0; best=0 if lo==0 else None
        for i in range(hi):
            z=den*q['u'][i][j]-eta[i]
            value=prefix+z-den*q['lambda']*q['bits'][j]*(q['x'][i+1]-q['x'][0])
            if i+1>=lo:best=value if best is None else max(best,value)
            prefix+=max(0,z)
        assert best is not None
        total+=best
    return total//den


def infeasible(q, box):
    # Each visit must have enough eligible labels. Every label with a positive
    # lower endpoint must receive a read within its allowed last-use window.
    for i,k in enumerate(q['k'],1):
        if sum(hi>=i for _,hi in box)<k:return ['row',i]
    for a in range(1,q['T']+1):
        for b in range(a,q['T']+1):
            if sum(lo>=a and hi<=b for lo,hi in box)>sum(q['k'][a-1:b]):return ['hall',a,b]
    return None


def check_witness(q, box, w):
    if not isinstance(w,list):return False
    if len(w)==2 and w[0]=='row' and type(w[1]) is int and 1<=w[1]<=q['T']:
        return sum(hi>=w[1] for _,hi in box)<q['k'][w[1]-1]
    if len(w)==3 and w[0]=='hall' and all(type(x) is int for x in w[1:]) and 1<=w[1]<=w[2]<=q['T']:
        a,b=w[1:];return sum(lo>=a and hi<=b for lo,hi in box)>sum(q['k'][a-1:b])
    return False


def fixed_schedule(q, box):
    if any(lo!=hi for lo,hi in box):return None
    schedule=[]
    for i,k in enumerate(q['k'],1):
        required=[j for j,(lo,_) in enumerate(box) if lo==i]
        optional=sorted((j for j,(lo,_) in enumerate(box) if lo>i),key=lambda j:(-q['u'][i-1][j],j))
        n=k-len(required)
        if not 0<=n<=len(optional):return None
        schedule.append(sum(1<<j for j in required+optional[:n]))
    return schedule


def replay(task, receipt, cap=127):
    """Independent arithmetic, validates full partition, ignores solver metadata."""
    if type(cap) is not int or not 1<=cap<=127:raise TaskError('invalid proof cap')
    q=validate(task); L=score(q,receipt['schedule'])['objective']; tree=receipt['tree']
    count=0; maxdepth=0
    def walk(node,box,depth):
        nonlocal count,maxdepth
        count+=1;maxdepth=max(maxdepth,depth)
        if count>cap or not isinstance(node,dict):raise TaskError('proof cap or malformed node')
        kind=node.get('kind')
        if kind=='split' and set(node)=={'kind','j','cut','children'} and isinstance(node['children'],list) and len(node['children'])==2:
            children=split(q,box,node['j'],node['cut'])
            a,b=[walk(n,p,depth+1) for n,p in zip(node['children'],children,strict=True)]
            return b if a is None else a if b is None else max(a,b)
        if kind=='infeasible' and set(node)=={'kind','witness'} and check_witness(q,box,node['witness']):return None
        if kind=='bound' and set(node)=={'kind','eta','den','upper'}:
            # Explicit independent last-use formula (no prefix recurrence).
            eta,den=node['eta'],node['den']
            if type(den) is not int or not 1<=den<=1_000_000 or not isinstance(eta,list) or len(eta)!=q['T'] or any(type(v) is not int or abs(v)>10**15 for v in eta):raise TaskError('invalid rational multipliers')
            value=sum(v*k for v,k in zip(eta,q['k'],strict=True))
            for j,(lo,hi) in enumerate(box):
                z=[den*r[j]-e for r,e in zip(q['u'],eta,strict=True)]
                opts=[0] if lo==0 else []
                opts.extend(z[i-1]+sum(max(0,v) for v in z[:i-1])-den*q['lambda']*q['bits'][j]*(q['x'][i]-q['x'][0]) for i in range(max(1,lo),hi+1))
                value+=max(opts)
            u=value//den
            if type(node['upper']) is not int or u!=node['upper']:raise TaskError('bound mismatch')
            return u
        raise TaskError('invalid proof node')
    upper=walk(tree,domains(q),0)
    if upper is None or upper<L:raise TaskError('inconsistent proof')
    result=dict(objective=L,upper=upper,gap=upper-L,nodes=count,depth=maxdepth)
    if any(receipt.get(k)!=v for k,v in result.items()):raise TaskError('receipt mismatch')
    return result
