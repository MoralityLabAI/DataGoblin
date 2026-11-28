"""SG-16: staged acquisition of exact routing certificates."""
from __future__ import annotations

from datagoblin.training_gym.spectral_block_replay import verify
from datagoblin.training_gym.spectral_blocks import BlockSearch
from datagoblin.training_gym.spectral_face import recover_face, verify_certificate
from datagoblin.training_gym.spectral_primal import dual_round, row_sweep
from datagoblin.training_gym.spectral_quotient import score, validate
from datagoblin.training_gym.spectral_routing import TaskError


def face_candidate(q, initial, eta, den=1, *, node_cap=512):
    """Rounding and bounded face recovery; trust only exact certificates."""
    initial_cert = verify_certificate(q, initial, eta, den)
    if not initial_cert['gap']:
        return dict(kind='dual', schedule=list(initial), eta=list(eta), den=den,
                    certificate=initial_cert, face_status='already_certified', face_nodes=0)
    scaled = {**q, 'u': [[u * den for u in row] for row in q['u']],
              'lambda': q['lambda'] * den}
    rounded = dual_round(scaled, eta)
    rounded, _ = row_sweep(q, rounded)
    candidate = max((initial, rounded), key=lambda a: score(q, a)['objective'])
    cert = verify_certificate(q, candidate, eta, den)
    status, nodes = 'already_certified', 0
    if cert['gap']:
        z = recover_face(q, eta, den, node_cap=node_cap)
        status, nodes = z['status'], z['nodes']
        if z['schedule'] is not None:
            candidate = z['schedule']
        cert = verify_certificate(q, candidate, eta, den)
    return dict(kind='dual', schedule=list(candidate), eta=list(eta), den=den,
                certificate=cert, face_status=status, face_nodes=nodes)


class Portfolio:
    """Intersection of independent certificates for one immutable task."""

    def __init__(self, task, initial):
        self.q = validate(task)
        self.schedule = list(initial)
        self.lower = score(self.q, self.schedule)['objective']
        self.upper = None
        self.records = []

    def _accept(self, record, schedule, upper):
        value = score(self.q, schedule)['objective']
        lower = max(self.lower, value)
        upper = upper if self.upper is None else min(self.upper, upper)
        if lower > upper:
            raise TaskError('inconsistent certificate intersection')
        if value > self.lower:
            self.schedule, self.lower = list(schedule), value
        self.upper = upper
        self.records.append(record)

    def dual(self, eta, den=1, *, recover=True, node_cap=512):
        if recover:
            record = face_candidate(self.q, self.schedule, eta, den, node_cap=node_cap)
        else:
            cert = verify_certificate(self.q, self.schedule, eta, den)
            record = dict(kind='dual', schedule=list(self.schedule), eta=list(eta), den=den,
                          certificate=cert, face_status='skipped', face_nodes=0)
        self._accept(record, record['schedule'], record['certificate']['integer_upper'])

    def block(self, extra, cap, *, deadline=None):
        """Run the unchanged block engine, stopping at a verified portfolio bound."""
        initial = list(self.schedule)
        s = BlockSearch(self.q, initial, event_cap=cap, extra=extra)
        while len(s.events) < cap and (i := s.best()) is not None:
            if (self.upper is not None and s.incumbent >= self.upper) or (deadline and deadline()):
                break
            b = s.live[i]
            free = ((1 << self.q['d']) - 1) & ~(b.suffix | b.include | b.exclude)
            s.advance(i, s.relax.choose(b) if b.t > 1 and free else None)
        r = s.receipt()
        # BlockSearch is a trusted solver; the separate replay below validates
        # its entire transcript outside the timed search interval.
        record = dict(kind='block', initial=initial, extra=[list(e) for e in extra], cap=cap, receipt=r)
        self._accept(record, r['schedule'], r['upper'])

    @property
    def gap(self):
        return None if self.upper is None else self.upper - self.lower

    def receipt(self):
        return dict(schedule=list(self.schedule), objective=self.lower, upper=self.upper,
                    gap=self.gap, records=list(self.records))


def replay(task, initial, records):
    """Recompute every dual bound and independently replay each block transcript."""
    p = Portfolio(task, initial)
    for record in records:
        if record['kind'] == 'dual':
            cert = verify_certificate(p.q, record['schedule'], record['eta'], record['den'])
            if cert != record['certificate']:
                raise TaskError('incorrect dual receipt')
            p._accept(record, record['schedule'], cert['integer_upper'])
        elif record['kind'] == 'block':
            if record['initial'] != p.schedule:
                raise TaskError('incorrect block incumbent')
            r = record['receipt']
            checked = verify(p.q, record['initial'], record['extra'], r['events'], event_cap=record['cap'])
            if checked != {k: v for k, v in r.items() if k != 'peak_stored_blocks'}:
                raise TaskError('incorrect block receipt')
            p._accept(record, r['schedule'], r['upper'])
        else:
            raise TaskError('unknown acquisition record')
    return p.receipt()
