import itertools
import unittest
from dataclasses import replace

from datagoblin.training_gym.frontier import Bound, Receipt, dominates, frontier, optimistic_index


class FrontierTests(unittest.TestCase):
    def make(self, key, gain, cost):
        return Receipt(key, ('task', 'dev', 'init', 'cpu', 'exposure', 'acc'), 'measured',
                       (Bound(gain, gain),), (Bound(cost, cost), Bound(1, 1), Bound(0, 0), Bound(0, 0)), True)

    def test_grid(self):
        points = list(itertools.product(range(3), repeat=2))
        for i, a in enumerate(points):
            for j, b in enumerate(points):
                expected = a[0] >= b[0] and a[1] <= b[1] and a != b
                self.assertEqual(dominates(self.make(str(i), *a), self.make(str(j), *b)), expected)
        for mask in range(1 << len(points)):
            rows = [self.make(str(i), *p) for i, p in enumerate(points) if mask & (1 << i)]
            expected = tuple(sorted(r.key for r in rows if not any(
                s.benefits[0].low >= r.benefits[0].low and s.costs[0].low <= r.costs[0].low
                and (s.benefits != r.benefits or s.costs != r.costs) for s in rows)))
            self.assertEqual(frontier(rows, (2, 1, 0, 0)), expected)

    def test_guards(self):
        a = self.make('a', 1, 1)
        b = self.make('b', 2, 1)
        self.assertEqual(frontier([a, replace(b, competent=False)], (2, 1, 0, 0)), ('a',))
        self.assertEqual(frontier([a, b], (0, 1, 0, 0)), ())
        self.assertFalse(dominates(replace(a, benefits=(Bound(0, 3),)), b))
        for bad in (replace(b, status='estimated'), replace(b, context=('other', *b.context[1:]))):
            with self.assertRaises(ValueError):
                frontier([a, bad], (2, 1, 0, 0))
        with self.assertRaises(ValueError):
            frontier([a, a], (2, 1, 0, 0))
        with self.assertRaises(ValueError):
            Bound(float('nan'), 1)
        self.assertEqual(optimistic_index(Bound(1, 3), Bound(2, 4), 1), .75)


if __name__ == '__main__':
    unittest.main()
