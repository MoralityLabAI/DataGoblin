"""DG-METTA-12 CPU wiring tests with no runtime dependencies."""
import sys
import unittest
from pathlib import Path

# Import the isolated reference without importing metta_research_bench.__init__,
# which loads the heavy pinned Verifiers runtime.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                       "metta_research_bench" / "courier_transfer"))

from reference import (
    CourierState,
    CourierWorld,
    FiniteOracle,
    hashed_singleton,
    initial,
    legal_actions,
    serialize_state,
    success,
    transition,
)


class CourierReferenceTests(unittest.TestCase):
    def test_two_equal_routes_produce_two_optimal_actions(self):
        world = CourierWorld(
            start="A", edges=(("A", "B", ""), ("A", "C", ""),
                              ("B", "D", ""), ("C", "D", "")),
            offices=(), parcels=(("p", "A", "D", ""),),
        )
        oracle = FiniteOracle(world)
        state = transition(world, initial(world), "pickup:p")
        self.assertEqual(oracle.action_set(state, 3), ("move:B", "move:C"))
        self.assertTrue(oracle.can_finish(state, 3))
        for action in oracle.action_set(state, 3):
            nxt = transition(world, state, action)
            self.assertEqual(oracle.value(nxt, 2)[0], 1)

    def test_permit_and_pickup_tie_on_optimal_path(self):
        world = CourierWorld("A", (("A", "B", "gate"),), (("gate", "A"),),
                             (("p", "A", "B", "gate"),))
        oracle = FiniteOracle(world)
        self.assertEqual(oracle.action_set(initial(world), 4),
                         ("permit:gate", "pickup:p"))
        with self.assertRaises(ValueError):
            transition(world, initial(world), "move:B")

    def test_shared_source_aliases_are_not_independent_proofs(self):
        world = CourierWorld(
            "A", (("A", "B", "gate"),), (("gate", "A"),),
            (("p", "A", "B", ""),),
            witnesses=(("one", "A", "source-1"),
                       ("alias", "A", "source-1"),
                       ("two", "A", "source-2")),
            permit_sources=(("gate", ("source-1", "source-2")),),
        )
        state = transition(world, initial(world), "inspect:one")
        self.assertNotIn("inspect:alias", legal_actions(world, state))
        self.assertNotIn("permit:gate", legal_actions(world, state))
        state = transition(world, state, "inspect:two")
        self.assertIn("permit:gate", legal_actions(world, state))

    def test_stop_requires_verified_delivery(self):
        world = CourierWorld("A", (), (), (("p", "A", "A", ""),))
        state = initial(world)
        self.assertFalse(success(world, state, stopped=True))
        state = transition(world, state, "pickup:p")
        state = transition(world, state, "deliver:p")
        self.assertFalse(success(world, state, stopped=False))
        self.assertTrue(success(world, state, stopped=True))
        self.assertEqual(FiniteOracle(world).action_set(state, 3), ("STOP",))

    def test_oracle_horizon_and_hash_control(self):
        world = CourierWorld("A", (("A", "B", ""),), (),
                             (("p", "B", "B", ""),))
        state = initial(world)
        oracle = FiniteOracle(world)
        self.assertEqual(oracle.action_set(state, 0), ("STOP",))
        self.assertFalse(oracle.can_finish(state, 2))
        self.assertTrue(oracle.can_finish(state, 3))
        choices = ("move:B", "permit:x")
        a = hashed_singleton("test", state, choices, 1101)
        self.assertEqual(a, hashed_singleton("test", state, tuple(reversed(choices)), 1101))
        shuffled = CourierState("A", frozenset(("p2", "p1")))
        self.assertEqual(serialize_state(shuffled),
                         serialize_state(CourierState("A", frozenset(("p1", "p2")))))

    def test_pytorch_set_loss_if_installed(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch optional in stdlib smoke")
        from reference import set_nll
        logits = torch.zeros(4, requires_grad=True)
        self.assertAlmostEqual(float(set_nll(logits, (0, 2)).detach()),
                               0.69314718, places=5)
        set_nll(logits, (0, 2)).backward()
        self.assertIsNotNone(logits.grad)
        self.assertAlmostEqual(float(set_nll(logits, (0, 1, 2, 3)).detach()),
                               0.0, places=6)


if __name__ == "__main__":
    unittest.main()
