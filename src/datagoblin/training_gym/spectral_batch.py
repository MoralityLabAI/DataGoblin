"""Exact quotient expansion and optional NumPy split-bank queries."""
from __future__ import annotations

import numpy as np

from datagoblin.training_gym.spectral_bank import BankSession
from datagoblin.training_gym.spectral_routing import TaskError, score
from datagoblin.training_gym.spectral_search import Node


class BatchedBankSession(BankSession):
    """Preserve complete-action accounting and earliest maximizing action ties."""

    def __init__(self, task, initial, *, mode='hybrid', **kwargs):
        if mode not in ('dedup', 'batch', 'hybrid'):
            raise TaskError('unknown expansion mode')
        self.mode = mode
        self._action_arrays = {}
        self._data_view = None
        self._row_arrays = None
        super().__init__(task, initial, split=True, **kwargs)
        self.counts.update(expansions=0, raw_actions=0, unique_unions=0,
                           dominated=0, bound_masks=0, vector_actions=0,
                           vector_batches=0, vector_masks=0)

    def _quotient(self, t, suffix):
        actions = self.actions[t - 1]
        if self.mode == 'batch' or (self.mode == 'hybrid' and len(actions) >= 64):
            if t not in self._action_arrays:
                a = np.asarray(actions, dtype=np.int64)
                self._action_arrays[t] = (a[:, 0].copy(), a[:, 1].copy())
            masks, utilities = self._action_arrays[t]
            unions = (masks | suffix).tolist()
            triples = zip(unions, masks.tolist(), utilities.tolist(), strict=True)
            self.counts['vector_actions'] += len(actions)
        else:
            triples = ((suffix | m, m, u) for m, u in actions)
        best = {}
        for union, mask, utility in triples:
            old = best.get(union)
            if old is None or utility > old[1]:
                best[union] = (mask, utility)
        return best

    def _bounds(self, t, masks):
        if self.mode == 'dedup':
            return [self.remaining_bound(t, m) for m in masks]
        # Snapshot cache hits before inserting misses. Eviction order may differ
        # from scalar replay, but every returned integer bound is identical.
        out, misses = {}, []
        for m in masks:
            key = (t, m)
            if key in self.overlay:
                out[m] = self.overlay[key]
            elif key in self.scalars:
                self.counts['scalar_hits'] += 1
                self.scalars.move_to_end(key)
                out[m] = self.scalars[key]
            else:
                misses.append(m)
        if misses and (self.mode == 'batch' or len(misses) >= 16):
            self._ensure_bank()
            if self._data_view is None:
                self._data_view = np.frombuffer(self.bank.data, dtype=np.int64)
                self._row_arrays = [np.asarray(row, dtype=np.int64)[:, None] for row in self.bank.rows]
            a = np.asarray(misses, dtype=np.int64)
            off = self._row_arrays[t]
            v = self._data_view[off + (a & self.bank.low_mask)]
            v += self._data_view[off + self.bank.low_n + (a >> self.bank.cut)]
            values = np.minimum(v.min(axis=0), self.weak[t]).tolist()
            self.counts['vector_batches'] += 1
            self.counts['vector_masks'] += len(misses)
            self.counts['bank_queries'] += len(misses) * len(self.bank.rows[t])
            self.counts['scalar_misses'] += len(misses)
            for m, value in zip(misses, values, strict=True):
                out[m] = value
                if self.scalar_cap:
                    if len(self.scalars) == self.scalar_cap:
                        self.scalars.popitem(last=False)
                        self.counts['scalar_evictions'] += 1
                    self.scalars[t, m] = value
                    self.counts['peak_scalars'] = max(self.counts['peak_scalars'], len(self.scalars))
        else:
            for m in misses:
                out[m] = self.remaining_bound(t, m)
        return [out[m] for m in masks]

    def advance(self, t, suffix):
        if type(t) is not int or type(suffix) is not int or (t, suffix) not in self.live:
            raise TaskError('state is not live')
        count = len(self.actions[t - 1])
        if self.transitions + count > self.cap:
            raise TaskError('transition cap would be exceeded')
        node = self.live.pop((t, suffix))
        self.trace.append([t, suffix])
        self.transitions += count
        self.counts['expansions'] += 1
        self.counts['raw_actions'] += count
        price = self.lam * (self.x[t] - self.x[0])
        if t == 1:
            # All terminal children are complete schedules; the original
            # strict-improvement loop keeps the earliest maximum in action order.
            mask, value = max(((m, node.value + u - price * self.weights[m & ~suffix])
                               for m, u in self.actions[0]), key=lambda pair: pair[1])
            if value > self.incumbent:
                tail = [mask, *node.tail]
                if score(self.task, tail)['objective'] != value:
                    raise AssertionError('incumbent accounting failed')
                self.incumbent, self.schedule = value, tail
        else:
            groups = self._quotient(t, suffix)
            self.counts['unique_unions'] += len(groups)
            pending = []
            for union, (mask, utility) in groups.items():
                value = node.value + utility - price * self.weights[union & ~suffix]
                old = self.seen.get((t - 1, union))
                if old is not None and value <= old:
                    self.counts['dominated'] += 1
                    continue
                pending.append((union, mask, value))
            self.counts['bound_masks'] += len(pending)
            bounds = self._bounds(t - 1, [p[0] for p in pending])
            for (union, mask, value), bound in zip(pending, bounds, strict=True):
                upper = value + bound
                if upper <= self.incumbent:
                    continue
                key = (t - 1, union)
                self.seen[key] = value
                self.live[key] = Node(value, (mask, *node.tail), upper)
        self._prune()
        self.events.append({'kind': 'expand', 'state': [t, suffix]})
