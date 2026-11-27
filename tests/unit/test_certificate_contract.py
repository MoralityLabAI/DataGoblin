"""Minimal regression for product-valid, identity-invalid descent proposals."""
try:
    from datagoblin.research.alg_descent import product
    from datagoblin.research.certificate_contract import verify_bound
except ModuleNotFoundError:
    from alg_descent import product
    from certificate_contract import verify_bound


def test_product_valid_child_swap_rejected():
    p = 2
    a = (0, 1, 0, 1)
    b = (0, 0, 1, 1)
    parent, _ = product((a, b), p)
    assert a != b
    for z in range(p):
        assert sum(c * z**i for i, c in enumerate(a)) % p == sum(
            c * z**i for i, c in enumerate(b)
        ) % p
    good = verify_bound(parent, (a, b), (a, b), p)
    bad = verify_bound(parent, (b, a), (a, b), p)
    assert good.accepted
    assert bad.product_valid and not bad.child_bound and not bad.accepted
    assert good.work == bad.work == 20


def test_bound_but_invalid_product_rejected():
    a = (0, 1)
    b = (1, 1)
    receipt = verify_bound((1, 0, 1), (a, b), (a, b), 2)
    assert receipt.child_bound and not receipt.product_valid
    assert not receipt.accepted and receipt.work == 6


def test_missing_child_binding_rejected():
    a = (0, 1)
    b = (1, 1)
    parent, _ = product((a, b), 2)
    receipt = verify_bound(parent, (a, b), (a,), 2)
    assert receipt.product_valid and not receipt.accepted
