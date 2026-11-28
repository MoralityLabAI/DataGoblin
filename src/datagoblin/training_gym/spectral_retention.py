"""SG-08: bounded retention of recomputable spectral tables."""
from collections import OrderedDict
from dataclasses import dataclass

from datagoblin.training_gym.spectral_cache import CachedDualSession
from datagoblin.training_gym.spectral_routing import TaskError


@dataclass
class Entry:
    value: object
    last: int
    count: int = 1
    gap: int = 0


def features(key, entry, incoming, clock, cap, fixed):
    age = clock - entry.last
    return (1, age.bit_length(), entry.count.bit_length(), entry.gap.bit_length(),
            key[0], int(key in fixed), int(key[0] == incoming[0]), cap,
            int(age <= 6), int(0 < entry.gap <= 6), int(not any(key[1])))


class Retention:
    """Mandatory insertion, resident-only history, deterministic ties."""

    def __init__(self, cap, kind, fixed, beta=(), observer=None):
        if type(cap) is not int or not 0 <= cap <= 100000:
            raise TaskError('invalid capacity')
        if kind not in ('lru', 'lfu', 'weighted', 'fixed_first', 'learned', 'shuffled'):
            raise TaskError('invalid policy')
        if kind in ('learned', 'shuffled') and (len(beta) != 11 or any(type(v) is not int for v in beta)):
            raise TaskError('invalid coefficients')
        self.cap, self.kind, self.fixed, self.beta = cap, kind, fixed, tuple(beta)
        self.entries = OrderedDict()
        self.clock = self.hits = self.misses = self.work = self.evictions = self.peak = 0
        self.observer = observer

    def access(self, key, factory):
        self.clock += 1
        if key in self.entries:
            e = self.entries[key]
            e.gap, e.last, e.count = self.clock-e.last, self.clock, e.count+1
            self.entries.move_to_end(key)
            self.hits += 1
            return e.value
        self.misses += 1
        self.work += key[0]
        value = factory()
        if not self.cap:
            return value
        if len(self.entries) == self.cap:
            if self.observer:
                self.observer(self, key)
            def rank(k):
                e = self.entries[k]
                if self.kind == 'lfu':
                    return e.count, e.last
                if self.kind == 'weighted':
                    return k[0]*e.count/(self.clock-e.last), e.last
                if self.kind == 'fixed_first':
                    return k in self.fixed, e.last
                return sum(a*b for a,b in zip(self.beta, features(
                    k,e,key,self.clock,self.cap,self.fixed),strict=True)), e.last
            victim = next(iter(self.entries)) if self.kind == 'lru' else min(self.entries,key=rank)
            del self.entries[victim]
            self.evictions += 1
        self.entries[key] = Entry(value,self.clock)
        self.peak = max(self.peak,len(self.entries))
        return value


def fixed_keys(task):
    cutoff = [sorted(row,reverse=True)[k-1] if k else max(row)+1
              for row,k in zip(task['u'],task['k'],strict=True)]
    eta = [[0]*task['T'], *[[x+s for x in cutoff] for s in (-6,-3,0,3,6)]]
    return frozenset((t,tuple(e[:t])) for t in range(task['T']+1) for e in eta)


class RetentionSession(CachedDualSession):
    """Replace only table cache; scalar cache and proof state are unchanged."""

    def __init__(self,task,initial,*,table_cap=12,scalar_cap=16,kind='lru',beta=(),record=False,**kwargs):
        self.retention = Retention(table_cap,kind,fixed_keys(task),beta)
        self.requests = [] if record else None
        super().__init__(task,initial,table_cap=0,scalar_cap=scalar_cap,**kwargs)

    def _table(self,t,eta):
        key = (t,eta)
        if self.requests is not None:
            self.requests.append(key)
        def build():
            return super(RetentionSession,self)._table(t,eta)
        old = self.retention.hits
        value = self.retention.access(key,build)
        self.counts['table_hits'] += self.retention.hits-old
        self.counts['peak_tables'] = self.retention.peak
        self.counts['table_evictions'] = self.retention.evictions
        return value
