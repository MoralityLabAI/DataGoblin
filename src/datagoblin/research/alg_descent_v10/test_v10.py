import pytest
from experiment import BoundRoot, ClosureCache, CertificateReceipt, audit, route, VERSION, exact
from certificate_contract import verify_bound
from alg_descent import product


def test_binding_swap_and_compensating_edit():
    a,b=(0,1,0,1),(0,0,1,1);parent,_=product((a,b),2)
    assert verify_bound(parent,(a,b),(a,b),2).accepted
    bad=verify_bound(parent,(b,a),(a,b),2)
    assert bad.product_valid and not bad.child_bound and not bad.accepted
    # Equal-degree redistribution of linear factors in characteristic 3.
    f,g,h,j=(0,1),(1,1),(2,1),(0,1)
    u,_=product((f,g),3);v,_=product((h,j),3)
    x,_=product((f,h),3);y,_=product((g,j),3);p,_=product((u,v),3)
    rec=verify_bound(p,(x,y),(u,v),3)
    assert rec.product_valid and not rec.child_bound and not rec.accepted


def test_malformed_rejected():
    rec=verify_bound((0,1,1),((0,1),(2,1)),((0,1),(1,1)),2)
    assert not rec.accepted


def test_cache_contract_scope_and_failure():
    c=ClosureCache(128,(42,2,VERSION)); good=CertificateReceipt(True,True,6)
    assert not c.admit((0,1,1),good,False,19)
    assert not c.admit((0,1,1),CertificateReceipt(True,False,6),True,19)
    assert not c.admit((0,1,1),CertificateReceipt(False,True,6),True,19)
    assert not c.hit((0,1,1))
    assert c.admit((0,1,1),good,True,19) and c.hit((0,1,1))
    c.scope=(42,2,'another-version');assert not c.hit((0,1,1))
    c.scope=(42,3,VERSION);assert not c.hit((0,1,1))
    c.scope=(43,2,VERSION);assert not c.hit((0,1,1))
    c.scope=(42,2,VERSION);assert c.hit((0,1,1))
    for i in range(20):c.admit((i,1,1),good,True,19)
    assert c.peak<=128 and c.used<=128


def test_joint_teacher_and_twins():
    roots=[BoundRoot(630000+i,12,'second') for i in range(4)]
    rows=[audit(r) for r in roots]
    assert sum(q['twins'] for q in rows)>0
    assert all(not q.get('bound_invalid',0) for q in rows)
    for r in roots:
        q=route(r,{},'oracle');assert q['closed'] and q['work']==r.oracle((0,r.n))[0]
        for k in ('poly','interval'):
            z=route(r,{},'binding',cap=128,identity=k)
            assert z['closed'] and not z['product_invalid'] and not z['unbound']
