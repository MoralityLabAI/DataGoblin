"""Bounded arithmetic caches; original verifier transcripts still replay."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

from datagoblin.training_gym.spectral_dual import DualSearchSession, _checked
from datagoblin.training_gym.spectral_routing import TaskError
from datagoblin.training_gym.spectral_search import Node


class CachedDualSession(DualSearchSession):
    """Evict only recomputable arithmetic, never proof obligations or proposals."""

    def __init__(self, task: dict[str, Any], initial: list[int], *, transition_cap: int = 4096,
                 proposal_cap: int = 4096, table_cap: int = 128, scalar_cap: int = 128):
        if any(type(v) is not int or not 0 <= v <= 100_000 for v in (table_cap, scalar_cap)):
            raise TaskError("invalid arithmetic cache cap")
        self.table_cap, self.scalar_cap = table_cap, scalar_cap
        self.tables: OrderedDict[tuple[int, tuple[int, ...]], tuple[int, tuple[int, ...]]] = OrderedDict()
        self.scalars: OrderedDict[tuple[int, int], int] = OrderedDict()
        self.overlay: dict[tuple[int, int], int] = {}
        self.counts = dict(table_hits=0, table_misses=0, table_evictions=0, scalar_hits=0,
                           scalar_misses=0, scalar_evictions=0, component_steps=0,
                           mask_additions=0, peak_tables=0, peak_scalars=0, peak_overlay=0)
        super().__init__(task, initial, transition_cap=transition_cap, proposal_cap=proposal_cap)

    def _table(self, t: int, eta: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
        key = (t, eta)
        if key in self.tables:
            self.counts['table_hits'] += 1
            self.tables.move_to_end(key)
            return self.tables[key]
        self.counts['table_misses'] += 1
        self.counts['component_steps'] += t * self.d
        base = sum(eta[i] * self.k[i] for i in range(t))
        delta = []
        for j, bits in enumerate(self.bits):
            prefix = best = 0
            for i in range(t):
                z = self.u[i][j] - eta[i]
                best = max(best, prefix + z - self.lam * bits * (self.x[i + 1] - self.x[0]))
                prefix += max(0, z)
            base += best
            delta.append(prefix - best)
        value = (base, tuple(delta))
        if self.table_cap:
            if len(self.tables) == self.table_cap:
                self.tables.popitem(last=False)
                self.counts['table_evictions'] += 1
            self.tables[key] = value
            self.counts['peak_tables'] = max(self.counts['peak_tables'], len(self.tables))
        return value

    def _value(self, t: int, suffix: int, eta: tuple[int, ...]) -> int:
        value, delta = self._table(t, eta)
        while suffix:
            bit = suffix & -suffix
            value += delta[bit.bit_length() - 1]
            self.counts['mask_additions'] += 1
            suffix ^= bit
        return value

    def remaining_bound(self, t: int, suffix: int) -> int:
        if type(t) is not int or type(suffix) is not int or not 0 <= t <= self.t or not 0 <= suffix < 1 << self.d:
            raise TaskError("invalid bound state")
        key = (t, suffix)
        if key in self.overlay:
            return self.overlay[key]
        if key in self.scalars:
            self.counts['scalar_hits'] += 1
            self.scalars.move_to_end(key)
            return self.scalars[key]
        self.counts['scalar_misses'] += 1
        value = min([self.weak[t], *[self._value(t, suffix, tuple(e[:t])) for e in self.etas]])
        if self.scalar_cap:
            if len(self.scalars) == self.scalar_cap:
                self.scalars.popitem(last=False)
                self.counts['scalar_evictions'] += 1
            self.scalars[key] = value
            self.counts['peak_scalars'] = max(self.counts['peak_scalars'], len(self.scalars))
        return value

    def candidate_bound(self, t: int, suffix: int, eta: list[int]) -> int:
        """Evaluate a candidate exactly without changing the search certificate."""
        _checked(self.task, t, suffix, eta)
        return self._value(t, suffix, tuple(eta))

    def propose(self, t: int, suffix: int, eta: list[int]) -> None:
        if type(t) is not int or type(suffix) is not int or (t, suffix) not in self.live:
            raise TaskError("state is not live")
        if self.proposals >= self.proposal_cap:
            raise TaskError("proposal cap exceeded")
        value = self.candidate_bound(t, suffix, eta)
        key = (t, suffix)
        n = self.live[key]
        bound = min(self.remaining_bound(t, suffix), value)
        self.overlay[key] = bound
        self.scalars.pop(key, None)
        self.counts['peak_overlay'] = max(self.counts['peak_overlay'], len(self.overlay))
        self.live[key] = Node(n.value, n.tail, n.value + bound)
        self.proposals += 1
        self.events.append({'kind': 'dual', 'state': [t, suffix], 'eta': list(eta)})
        self._prune()
