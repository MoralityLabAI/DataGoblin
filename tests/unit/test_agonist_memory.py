import copy
import unittest

from Agonist.memory_curriculum import (
    MENU,
    NAMES,
    RESOURCES,
    PermitEpisode,
    actor_action,
    author_ticket,
    checks,
    replay,
    validate_ticket,
)


class PermitMemoryTest(unittest.TestCase):
    def test_query_requires_all_observations_then_memory_commit(self):
        engine = PermitEpisode(author_ticket(3, 2), 5)
        with self.assertRaises(ValueError):
            engine.reveal_query(0)
        with self.assertRaises(ValueError):
            engine.commit_memory(0)
        bits = [engine.next_observation()["observation"]["bit"] for _ in range(3)]
        self.assertEqual(bits, [1, 0, 1])
        with self.assertRaises(ValueError):
            engine.reveal_query(0)
        with self.assertRaises(ValueError):
            engine.commit_memory(2)
        engine.commit_memory(1)
        packet = engine.reveal_query(0)
        self.assertEqual(packet["retained_code"], 1)
        self.assertNotIn("observation", packet)
        with self.assertRaises(ValueError):
            engine.next_observation()
        with self.assertRaises(ValueError):
            engine.commit_memory(0)
        self.assertEqual(engine.score(1)["success"], 1)
        with self.assertRaises(ValueError):
            engine.score(1)

    def test_constant_names_menu_and_resources_across_every_world(self):
        ticket = author_ticket(4, 2)
        for word in range(16):
            for query in range(4):
                engine = PermitEpisode(ticket, word)
                for phase in range(4):
                    packet = engine.next_observation()
                    self.assertEqual(packet["names"], NAMES)
                    self.assertEqual(packet["resources"], RESOURCES)
                    self.assertEqual(packet["menu"], [])
                    self.assertEqual(packet["phase"], phase)
                    self.assertEqual(set(packet), {
                        "schema", "stage", "phase", "names", "resources", "menu", "observation",
                    })
                    self.assertEqual(set(packet["observation"]), {"kind", "bit"})
                engine.commit_memory(0)
                packet = engine.reveal_query(query)
                self.assertEqual(packet["names"], NAMES)
                self.assertEqual(packet["resources"], RESOURCES)
                self.assertEqual(packet["menu"], MENU)
                self.assertEqual(set(packet), {
                    "schema", "stage", "phase", "names", "resources", "menu", "retained_code", "query_index", "rule",
                })

    def test_observation_prefix_has_no_query_information(self):
        ticket = author_ticket(3, 2)
        for word in range(8):
            prefixes = []
            for query in range(3):
                engine = PermitEpisode(ticket, word)
                prefix = [engine.next_observation() for _ in range(3)]
                engine.commit_memory(0)
                engine.reveal_query(query)
                prefixes.append(prefix)
            self.assertEqual(prefixes[0], prefixes[1])
            self.assertEqual(prefixes[1], prefixes[2])

    def test_labels_stay_in_private_receipt(self):
        engine = PermitEpisode(author_ticket(2, 1), 2)
        packets = [engine.next_observation() for _ in range(2)]
        engine.commit_memory(0)
        packets.append(engine.reveal_query(0))
        for packet in packets:
            self.assertNotIn("expected_action", packet)
            self.assertNotIn("private_word", packet)
            self.assertNotIn("root_id", packet)
        receipt = engine.score(1)
        self.assertEqual(receipt["expected_action"], 1)
        self.assertEqual(receipt["private_word"], 2)

    def test_private_fields_cannot_enter_author_ticket(self):
        ticket = author_ticket(3, 2)
        for field in ("expected_action", "private_word", "training_ready"):
            changed = copy.deepcopy(ticket)
            changed[field] = 1
            with self.assertRaises(ValueError):
                validate_ticket(changed)

    def test_compiled_actor_ignores_private_extras(self):
        controller = {"outputs": [[0, 1], [1, 0]]}
        packet = {"stage": "query", "retained_code": 0, "query_index": 1}
        altered = dict(packet, private_word=0, expected_action=0, secret_name="authorize")
        self.assertEqual(actor_action(controller, packet), actor_action(controller, altered))

    def test_shared_core_replays_through_independent_story_mechanics(self):
        self.assertEqual(len(checks()), 12)
        from Agonist.mic18_core import compile_exact, make_parity, make_register

        self.assertEqual(replay(make_parity(4))["reward"], "1")
        self.assertEqual(replay(make_register(4))["reward"], "1")
        self.assertLess(replay(compile_exact(4, 2)["controller"])["successes"], 64)


if __name__ == "__main__":
    unittest.main()
