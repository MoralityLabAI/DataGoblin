import unittest
from dataclasses import replace

from datagoblin.training_gym.steering_feedback import Action, Audit, Cost, select_window


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.cap = Cost(100, 100, 0, 100, 10, 100, 0)
        cost = Cost(50, 50, 0, 50, 5, 50, 0)
        self.a = Audit(Action('a', 'core.1@visit2', 'donor', .5), 'ckpt', 'cal', 20, 'feedback', .1, .5, .02, cost)
        self.b = replace(self.a, action=Action('b', 'core.2@visit3', 'donor', 1.), specificity=.7,
                         ideal=1., gain_lower=.03)

    def choose(self, rows, **kwargs):
        args = dict(checkpoint='ckpt', feedback_cohort='cal', step=20, window=10,
                    remaining=self.cap, competent=True)
        args.update(kwargs)
        return select_window(rows, **args)

    def test_gain_vs_deficit(self):
        self.assertEqual(self.choose([self.a, self.b]).action.key, 'b')
        self.assertEqual(self.choose([self.a, self.b], policy='deficit').action.key, 'a')
        self.assertEqual(self.choose([self.a]).stop_step, 30)

    def test_budget_and_unknown(self):
        for cost in (None, replace(self.cap, ram_bytes=101), replace(self.cap, wall_ns=0)):
            self.assertIsNone(self.choose([replace(self.a, window_cost_upper=cost)]).action)
        self.assertIsNone(self.choose([replace(self.a, gain_lower=None)]).action)
        self.assertIsNone(self.choose([replace(self.a, gain_lower=0)]).action)
        self.assertIsNone(self.choose([self.a], competent=False).action)

    def test_binding(self):
        for row in (replace(self.a, checkpoint='old'), replace(self.a, scope='evaluation'),
                    replace(self.a, feedback_cohort='other'), replace(self.a, step=19)):
            with self.assertRaises(ValueError):
                self.choose([row])
        with self.assertRaises(ValueError):
            self.choose([self.a, self.a])

    def test_tie(self):
        b = replace(self.b, specificity=.2+1e-12, ideal=1.)
        for rows in ([self.a, b], [b, self.a]):
            self.assertEqual(self.choose(rows, policy='deficit').action.key, 'a')

    def test_invalid(self):
        with self.assertRaises(ValueError):
            replace(self.cap, visits=-1)
        with self.assertRaises(ValueError):
            replace(self.a, specificity=float('nan'))
        with self.assertRaises(ValueError):
            replace(self.a, ideal=0)
        with self.assertRaises(ValueError):
            self.choose([self.a], window=0)


if __name__ == '__main__':
    unittest.main()
