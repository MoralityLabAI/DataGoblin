"""Admissible mandatory-fact and residual goal-threshold cut bounds."""
MODES = ('hmax', 'mandatory', 'cut', 'combined')


class Bounder:
    def __init__(self, graph, mode):
        assert mode in MODES
        self.g = graph
        self.mode = mode
        self.groups = tuple(tuple((r.id, r.premises) for r in group)
                            for group in graph.groups)
        self.costs = tuple(r.cost for r in graph.rules)
        self.rules = graph.compact
        self.scans = 0

    def distances(self, s, residual):
        d = [0] * self.g.n
        for head in range(3, self.g.n):
            if s >> head & 1:
                continue
            rules = self.groups[head]
            self.scans += len(rules)
            d[head] = min(residual[i] + max((d[p] for p in ps), default=0)
                          for i, ps in rules)
        return d

    def mandatory(self, s):
        required = [0] * self.g.n
        for head in range(3, self.g.n):
            if s >> head & 1:
                continue
            common = (1 << self.g.n) - 1
            for _, ps in self.groups[head]:
                self.scans += 1
                union = 0
                for p in ps:
                    union |= required[p]
                common &= union
            required[head] = common | (1 << head)
        return required[self.g.goal]

    def mandatory_cost(self, mandatory, residual):
        paid = 0
        while mandatory:
            bit = mandatory & -mandatory
            h = bit.bit_length() - 1
            group = self.groups[h]
            self.scans += len(group)
            paid += min(residual[i] for i, _ in group)
            mandatory ^= bit
        return paid

    def evaluate(self, s):
        residual = list(self.costs)
        mand = self.mandatory(s) if self.mode in ('mandatory', 'combined') else 0
        charge = best = 0
        cuts = []
        while True:
            d = self.distances(s, residual)
            h = d[self.g.goal]
            m = self.mandatory_cost(mand, residual) if mand else 0
            best = max(best, charge + max(h, m))
            if self.mode in ('hmax', 'mandatory') or h == 0:
                break
            zone = sum(1 << j for j in range(3, self.g.n)
                       if d[j] >= h and self.g.relevant >> j & 1)
            assert zone & self.g.goalbit and not zone & s
            self.scans += len(self.rules)
            cut = [i for i, b, ps, _ in self.rules if b & zone and not ps & zone]
            assert cut and all(residual[i] > 0 for i in cut)
            delta = min(residual[i] for i in cut)
            self.scans += len(cut)
            for i in cut:
                residual[i] -= delta
            charge += delta
            cuts.append((delta, cut))
            assert len(cuts) <= len(self.rules)
        return dict(lower=best, mandatory=mand, cuts=cuts)
