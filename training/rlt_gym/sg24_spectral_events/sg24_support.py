"""Exact supporting-line crossing; no explicit envelope materialization."""
from fractions import Fraction

from sg24_events import envelope_cross, lines
from datagoblin.training_gym.spectral_routing import TaskError


def support(terms, s):
    """Affine active regret line at s, exact rational comparison, rightmost tie."""
    n,d=s.numerator,s.denominator
    intercept=slope=0
    for z,dz,p,dp,chosen_a,chosen_b in terms:
        a=b=best_a=best_b=best_value=0
        for i,(x,y) in enumerate(zip(z,dz,strict=True)):
            c,v=a+x-p[i+1],b+y-dp[i+1]
            val=c*d+v*n
            if val>best_value or (val==best_value and v>best_b):
                best_a,best_b,best_value=c,v,val
            w=x*d+y*n
            if w>0 or (w==0 and y>0):
                a+=x;b+=y
        intercept+=best_a-chosen_a;slope+=best_b-chosen_b
    return intercept,slope


def cross_support(terms, horizon, threshold, cap=128):
    """Starting from a failing horizon, move to each active line's crossing.

    The active line is a global lower bound on the convex gap. Its crossing
    cannot lie before the true first crossing. Strict moves cannot repeat an
    active line, and the exact envelope fallback caps the number of probes.
    """
    if horizon<=0 or threshold<0 or type(cap) is not int or cap<1:
        raise TaskError('invalid support search')
    s=Fraction(horizon);a,b=support(terms,s);probes=1
    g=a+b*s
    if g<threshold or (threshold==0 and g==0):
        return None,probes,False,(a,b)
    while g>threshold:
        if probes>=cap:
            c,_,_=envelope_cross([{(a-ca,b-cb) for a,b in lines(z,dz,p,dp,horizon)}
                                  for z,dz,p,dp,ca,cb in terms],horizon,threshold)
            return c,probes,True,None
        if b<=0:raise TaskError('invalid supporting slope')
        nxt=Fraction(threshold-a,b)
        if not 0<=nxt<s:raise TaskError('invalid crossing anchor')
        s=nxt;a,b=support(terms,s);probes+=1;g=a+b*s
        assert g>=threshold
    return s,probes,False,(a,b)


def region(ray,state,start,line):
    """Same maximal integer region as SG-22, with exact support-line search."""
    if state.q!=ray.at(start) or not state.certificate()['certified']:
        raise TaskError('invalid anchor')
    eta,den=line.at(start)
    if len(eta)!=state.q.t or any(a*den!=b*state.den for a,b in zip(state.eta,eta,strict=True)):
        raise TaskError('line misses anchor')
    line.at(ray.horizon)
    base,rem=divmod(state.bound*den,state.den)
    if rem:raise TaskError('nonintegral anchor bound')
    H=ray.horizon-start;q=state.q;far=ray.at(ray.horizon)
    cross=None;probes=0;fallback=False;end=start;last_bound=base;objective=state.objective
    if H:
        terms=[];primal_slope=0
        for j,mask in enumerate(state.visits):
            du=[(v-u)//H for u,v in zip(q.columns[j],far.columns[j],strict=True)]
            p=[q.lam*q.bits[j]*(x-q.x[0]) for x in q.x]
            dp=[(far.lam*far.bits[j]*(x-far.x[0])-v)//H for v,x in zip(p,far.x,strict=True)]
            z=[den*u-e for u,e in zip(q.columns[j],eta,strict=True)]
            dz=[den*u-v for u,v in zip(du,line.slope,strict=True)]
            ids=[i for i in range(q.t) if mask&(1<<i)]
            last=mask.bit_length()
            ca=sum(z[i] for i in ids)-den*p[last];cb=sum(dz[i] for i in ids)-den*dp[last]
            primal_slope+=sum(du[i] for i in ids)-dp[last]
            terms.append((z,dz,[den*v for v in p],[den*v for v in dp],ca,cb))
        cross,probes,fallback,active=cross_support(terms,H,den)
        offset=H if cross is None else (cross.numerator-1)//cross.denominator
        end=start+offset;objective=state.objective+offset*primal_slope
        if offset:
            if cross is not None or active is None:
                active=support(terms,Fraction(offset));probes+=1
            a,b=active;last_bound=den*objective+a+b*offset
        assert den*objective<=last_bound<den*(objective+1)
    return dict(start=start,end=end,origin=line.origin,eta=list(line.eta),slope=list(line.slope),den=den,
                schedule=list(state.schedule),bounds=[base,last_bound],objectives=[state.objective,objective],
                probes=0,crossing=None if cross is None else [cross.numerator,cross.denominator],
                support_evals=probes,fallback=fallback)
