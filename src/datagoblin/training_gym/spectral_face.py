"""SG-15: exact dual-face recovery; integer arithmetic verifies every certificate."""
from fractions import Fraction
from math import lcm

from datagoblin.training_gym.spectral_quotient import indices, score, validate
from datagoblin.training_gym.spectral_routing import TaskError


def rational_vector(values, max_den=1024):
    """Convert numerical suggestions to a common exact rational denominator."""
    f = [Fraction(float(x)).limit_denominator(max_den) for x in values]
    den = lcm(*(x.denominator for x in f))
    if den > 1_000_000:
        # A common grid controls the integer size even if reconstruction fails.
        den = max_den
        return [round(x * den) for x in f], den
    return [int(x * den) for x in f], den


def face(q, eta, den=1):
    """Return scaled bound and optimal mask boxes (required, possible) by label."""
    if type(den) is not int or not 1 <= den <= 1_000_000:
        raise TaskError('invalid denominator')
    if not isinstance(eta, list) or len(eta) != q['T'] or any(type(v) is not int or abs(v) > 10**15 for v in eta):
        raise TaskError('invalid multiplier numerators')
    total = sum(v*k for v,k in zip(eta,q['k'],strict=True))
    boxes = []
    for j,b in enumerate(q['bits']):
        prefix = required = possible = 0
        best, options = 0, [(0,0)]
        for i in range(q['T']):
            bit = 1 << i
            z = den*q['u'][i][j]-eta[i]
            v = prefix+z-den*q['lambda']*b*(q['x'][i+1]-q['x'][0])
            option = (required|bit,possible|bit)
            if v > best:
                best,options = v,[option]
            elif v == best:
                options.append(option)
            if z > 0:
                prefix += z
                required |= bit
            if z >= 0:
                possible |= bit
        total += best
        boxes.append(tuple(options))
    return total,tuple(boxes)


def exact_bound(q, eta, den):
    """Independent last-use sum, no box construction or prefix recurrence."""
    total = sum(eta[i]*q['k'][i] for i in range(q['T']))
    for j,b in enumerate(q['bits']):
        z = [den*row[j]-v for row,v in zip(q['u'],eta,strict=True)]
        total += max([0,*[z[i]+sum(max(0,w) for w in z[:i])
                         -den*q['lambda']*b*(q['x'][i+1]-q['x'][0]) for i in range(q['T'])]])
    return total


def recover_face(task, eta, den=1, *, node_cap=4096):
    """Exact search over optimal last-use boxes, with row-capacity propagation.

    A capped run may be unknown. No infeasibility claim is made on exhaustion.
    """
    q = validate(task)
    if type(node_cap) is not int or not 1 <= node_cap <= 100_000:
        raise TaskError('invalid face node cap')
    bound, boxes = face(q,eta,den)
    T,d,k = q['T'],q['d'],q['k']
    full = (1 << T)-1
    nodes = reductions = 0
    stack = [boxes]
    while stack and nodes < node_cap:
        domains = list(stack.pop())
        nodes += 1
        # Aggregate per-label rowwise mandatory/possible coverage, then remove
        # boxes that cannot possibly participate in any feasible completion.
        while True:
            lower,upper = [],[]
            for domain in domains:
                a,b = full,0
                for req,poss in domain:
                    a &= req
                    b |= poss
                lower.append(a);upper.append(b)
            lo = [sum((a>>i)&1 for a in lower) for i in range(T)]
            hi = [sum((a>>i)&1 for a in upper) for i in range(T)]
            if any(not a <= v <= b for a,v,b in zip(lo,k,hi,strict=True)):
                break
            changed = False
            for j,domain in enumerate(domains):
                if len(domain)==1:
                    continue
                must = sum(1<<i for i in range(T) if hi[i]-((upper[j]>>i)&1)<k[i])
                forbid = sum(1<<i for i in range(T) if lo[i]-((lower[j]>>i)&1)>=k[i])
                new = tuple((a,b) for a,b in domain if a&forbid==0 and must&~b==0)
                if len(new)!=len(domain):
                    reductions += len(domain)-len(new)
                    domains[j]=new;changed=True
                if not new:
                    break
            if any(not domain for domain in domains):
                break
            if changed:
                continue
            # Test a deterministic choice of one box per label. Once fixed,
            # optional row slots are independent, so this test is exact.
            selected = [v[0] for v in domains]
            schedule=[]
            for i,ki in enumerate(k):
                bit=1<<i
                mandatory=[j for j,(a,b) in enumerate(selected) if a&bit]
                optional=[j for j,(a,b) in enumerate(selected) if b&bit and not a&bit]
                need=ki-len(mandatory)
                if not 0<=need<=len(optional):
                    break
                schedule.append(sum(1<<j for j in mandatory+optional[:need]))
            if len(schedule)==T:
                value=score(q,schedule)['objective']
                if value*den!=bound or exact_bound(q,eta,den)!=bound:
                    raise AssertionError('dual-face certificate mismatch')
                return dict(status='exact',schedule=schedule,objective=value,bound_num=bound,den=den,
                            nodes=nodes,reductions=reductions)
            choices=[j for j,v in enumerate(domains) if len(v)>1]
            if choices:
                j=min(choices,key=lambda j:(len(domains[j]),-(upper[j]^lower[j]).bit_count(),j))
                for box in reversed(domains[j]):
                    child=domains.copy();child[j]=(box,);stack.append(tuple(child))
            break
    return dict(status='unknown' if stack else 'infeasible_face',schedule=None,bound_num=bound,den=den,
                nodes=nodes,reductions=reductions)


def verify_certificate(task,schedule,eta,den=1):
    """Check exact rational dual upper bound against a feasible integer primal."""
    q=validate(task)
    bound,_=face(q,eta,den)  # validates representation
    upper=exact_bound(q,eta,den)
    if upper!=bound:
        raise AssertionError('independent dual mismatch')
    value=score(q,schedule)['objective']
    if value*den>upper:
        raise AssertionError('invalid upper bound')
    return dict(objective=value,bound_num=upper,den=den,integer_upper=upper//den,
                gap=upper//den-value,certified=value==upper//den,exact_face=value*den==upper)
