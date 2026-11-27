"""Exact indexed-child acceptance contract for the synthetic descent gym."""
from typing import NamedTuple

try:
    from .alg_descent import verify
except ImportError:
    from alg_descent import verify


class CertificateReceipt(NamedTuple):
    product_valid: bool
    child_bound: bool
    work: int

    @property
    def accepted(self) -> bool:
        return self.product_valid and self.child_bound


def verify_bound(parent, children, expected_children, p):
    """Charge the product check; require every proposed child to bind its index.

    expected_children must come from the trusted root-local state registry,
    not the proposer. This wrapper preserves the existing coefficient meter;
    binding comparisons have a separate, unmetered cost in this proxy.
    """
    product_valid, work = verify(parent, children, p)
    child_bound = tuple(children) == tuple(expected_children)
    return CertificateReceipt(product_valid, child_bound, work)
