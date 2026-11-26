"""Typed lambda normalization microbenchmark. Original terms, no Lean import."""
import json

# Terms: atom c, variable de Bruijn index, lambda, application.
C = ('c',)
ATOM = ('o',)


def infer(t, ctx=()):
    match t:
        case ('c',):
            return ATOM
        case ('v', i):
            return ctx[i]
        case ('l', ty, body):
            return ('arr', ty, infer(body, (ty, *ctx)))
        case ('a', f, x):
            ft, xt = infer(f,ctx), infer(x,ctx)
            assert ft[0] == 'arr' and ft[1] == xt
            return ft[2]
        case _:
            raise ValueError(t)


def shift(t, delta, cutoff=0):
    match t:
        case ('c',):
            return t
        case ('v', i):
            return ('v', i+delta if i>=cutoff else i)
        case ('l', ty, body):
            return ('l', ty, shift(body,delta,cutoff+1))
        case ('a', f, x):
            return ('a', shift(f,delta,cutoff), shift(x,delta,cutoff))


def subst(t, j, s, depth=0):
    match t:
        case ('c',):
            return t
        case ('v', i):
            return shift(s,depth) if i==j+depth else t
        case ('l', ty, body):
            return ('l',ty,subst(body,j,s,depth+1))
        case ('a', f, x):
            return ('a',subst(f,j,s,depth),subst(x,j,s,depth))


def beta(body, x):
    return shift(subst(body,0,shift(x,1)),-1)


def reduce(t, eager):
    if t[0]=='a':
        f,x=t[1:]
        if not eager and f[0]=='l':
            return beta(f[2],x)
        nxt=reduce(f,eager)
        if nxt is not None:
            return ('a',nxt,x)
        if eager:
            nxt=reduce(x,eager)
            if nxt is not None:
                return ('a',f,nxt)
            if f[0]=='l':
                return beta(f[2],x)
        else:
            nxt=reduce(x,eager)
            if nxt is not None:
                return ('a',f,nxt)
    if t[0]=='l':
        nxt=reduce(t[2],eager)
        if nxt is not None:
            return ('l',t[1],nxt)
    return None


def nf(t,eager):
    steps=0
    while (nxt:=reduce(t,eager)) is not None:
        t=nxt
        steps+=1
    return t,steps


def nest(n):
    t=C
    identity=('l',ATOM,('v',0))
    for _ in range(n):
        t=('a',identity,t)
    return t


def main():
    rows=[]
    for n in [1,4,16,64,128]:
        for family,t in [('discarded_argument',('a',('l',ATOM,C),nest(n))),('required_argument',nest(n))]:
            assert infer(t)==ATOM
            a,ac=nf(t,True)
            b,bc=nf(t,False)
            assert a==b==C
            rows.append({'family':family,'n':n,'eager_beta':ac,'normal_order_beta':bc})
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
