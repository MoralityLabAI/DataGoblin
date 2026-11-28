"""Exact prefix-shared fixed multiplier bank for spectral routing."""
from __future__ import annotations

from array import array

from datagoblin.training_gym.spectral_cache import CachedDualSession
from datagoblin.training_gym.spectral_routing import TaskError


def subset_sums(values):
    out = [0]
    for v in values:
        out += [s + v for s in out]
    return out


class PrefixBank:
    """Task-local fixed vectors; no student data enters the bank."""

    def __init__(self, task, etas, split=False):
        self.d, self.t = task['d'], task['T']
        self.cut = self.d // 2
        self.low_mask = (1 << self.cut) - 1
        self.low_n, self.high_n = 1 << self.cut, 1 << (self.d - self.cut)
        self.stride = self.low_n + self.high_n if split else self.d + 1
        self.split = split
        self.data = array('q')
        self.offsets = {}
        self.rows = [[] for _ in range(self.t + 1)]
        self.steps = 0
        for eta in etas:
            a, b, constant = [0] * self.d, [0] * self.d, 0
            for t in range(self.t + 1):
                if t:
                    i = t - 1
                    constant += eta[i] * task['k'][i]
                    for j, bits in enumerate(task['bits']):
                        z = task['u'][i][j] - eta[i]
                        price = task['lambda'] * bits * (task['x'][t] - task['x'][0])
                        b[j] = max(b[j], a[j] + z - price)
                        a[j] += max(0, z)
                    self.steps += self.d
                key = (t, tuple(eta[:t]))
                if key in self.offsets:
                    continue
                base = constant + sum(b)
                delta = [v - w for v, w in zip(a, b, strict=True)]
                off = len(self.data)
                self.offsets[key] = off
                self.rows[t].append(off)
                if split:
                    self.data.extend(base + v for v in subset_sums(delta[:self.cut]))
                    self.data.extend(subset_sums(delta[self.cut:]))
                else:
                    self.data.extend([base, *delta])

    def value(self, off, mask):
        data = self.data
        if self.split:
            return data[off + (mask & self.low_mask)] + data[off + self.low_n + (mask >> self.cut)]
        value = data[off]
        while mask:
            bit = mask & -mask
            value += data[off + bit.bit_length()]
            mask ^= bit
        return value

    def fixed_bound(self, t, mask, weak):
        return min(weak, *(self.value(off, mask) for off in self.rows[t]))

    def batch(self, t, masks, weak):
        if type(t) is not int or not 0 <= t <= self.t:
            raise TaskError('invalid bank depth')
        if any(type(m) is not int or not 0 <= m < 1 << self.d for m in masks):
            raise TaskError('invalid bank mask')
        return [self.fixed_bound(t, m, weak) for m in masks]


class BankSession(CachedDualSession):
    """Fixed bank plus bounded cache of arbitrary proposed multiplier vectors."""

    def __init__(self, task, initial, *, split=False, **kwargs):
        self.bank = None
        self.split = split
        super().__init__(task, initial, **kwargs)

    def _ensure_bank(self):
        if self.bank is None:
            self.bank = PrefixBank(self.task, self.etas, self.split)
            self.counts['bank_steps'] = self.bank.steps
            self.counts['bank_bytes'] = self.bank.data.itemsize * len(self.bank.data)
            self.counts['bank_queries'] = 0

    def _value(self, t, suffix, eta):
        self._ensure_bank()
        off = self.bank.offsets.get((t, eta))
        if off is not None:
            self.counts['bank_queries'] += 1
            return self.bank.value(off, suffix)
        return super()._value(t, suffix, eta)

    def remaining_bound(self, t, suffix):
        if type(t) is not int or type(suffix) is not int or not 0 <= t <= self.t or not 0 <= suffix < 1 << self.d:
            raise TaskError('invalid bound state')
        key = (t, suffix)
        if key in self.overlay:
            return self.overlay[key]
        if key in self.scalars:
            self.counts['scalar_hits'] += 1
            self.scalars.move_to_end(key)
            return self.scalars[key]
        self.counts['scalar_misses'] += 1
        self._ensure_bank()
        self.counts['bank_queries'] += len(self.bank.rows[t])
        value = self.bank.fixed_bound(t, suffix, self.weak[t])
        if self.scalar_cap:
            if len(self.scalars) == self.scalar_cap:
                self.scalars.popitem(last=False)
                self.counts['scalar_evictions'] += 1
            self.scalars[key] = value
            self.counts['peak_scalars'] = max(self.counts['peak_scalars'], len(self.scalars))
        return value
