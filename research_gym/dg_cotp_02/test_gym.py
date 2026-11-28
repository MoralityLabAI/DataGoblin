"""Offline unit tests for pilot protocol, fixture oracles, and budget logic."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from fractions import Fraction

from gym import (make_fixture, oracle, prepare, mock, parse_initial, read,
                 observe, initial_prompt, continuation_prompt, write, digest)
from fable_batch import reserved_micro, make_requests, submit


class FixtureTests(unittest.TestCase):
    def test_risk(self):
        f = {"kind":"risk","weights":[1,1],"transcript":[0,1],
             "events":[[0],[1]],"question":"risk"}
        self.assertEqual(oracle(f)["value"], "1")
        f["transcript"] = [0,0]
        self.assertEqual(oracle(f)["value"], "1/2")

    def test_adaptive(self):
        f = {"kind":"adaptive","weights":[1,2,3],"queries":[{"members":[2],"cost":1}],
             "budget":1,"question":"identification"}
        self.assertEqual(oracle(f)["value"], "5/6")
        f["budget"] = 0
        self.assertEqual(oracle(f)["value"], "1/2")

    def test_joint(self):
        f = {"kind":"joint","marginals":["1/2"]*3,"pairs_ab_ac_bc":["1/4"]*3,
             "question":"maximum triple intersection"}
        self.assertEqual(oracle(f)["value"], "1/4")

    def test_cover(self):
        f = {"kind":"cover","hypotheses":3,"question":"cost","probes":[
             {"outcomes":[0,1,2],"cost":1,"routed":False},
             {"outcomes":[0,1,2],"cost":5,"routed":True},
             {"outcomes":[0,1,0],"cost":2,"routed":True},
             {"outcomes":[0,0,1],"cost":2,"routed":True}]}
        self.assertEqual(oracle(f)["value"], "4")

    def test_generated(self):
        for kind in ("risk","adaptive","joint","cover"):
            for seed in range(32):
                f = make_fixture(kind,seed)
                self.assertEqual(f,make_fixture(kind,seed))
                value = Fraction(oracle(f)["value"])
                self.assertGreaterEqual(value,0)
                if kind != "cover":
                    self.assertLessEqual(value,1)


class ProtocolTests(unittest.TestCase):
    def test_offline(self):
        with tempfile.TemporaryDirectory() as d:
            out = prepare(Path(d))
            mock(out)
            mock(out)
            manifest = read(out / "manifest.json")
            self.assertEqual(len(manifest["roots"]),12)
            self.assertEqual(read(out / "mock" / "receipt.json")["api_calls"],0)
            for rid in manifest["roots"]:
                packet = read(out / "public" / (rid+".json"))
                self.assertNotIn('"oracle":',initial_prompt(packet))
                obs = read(out / "mock" / rid / "observation.json")
                self.assertEqual(obs["packet_sha256"],digest(packet))
                self.assertFalse(obs["prediction_correct"])
            with self.assertRaises(ValueError):
                make_requests(out,1)

    def test_schema(self):
        with self.assertRaises(ValueError):
            parse_initial('{"action":"shell"}')

    def test_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"x.json"
            write(p,{"x":1})
            with self.assertRaises(ValueError):
                write(p,{"x":2})

    def test_budget(self):
        self.assertEqual(reserved_micro(1000,2000),49096)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                submit(SimpleNamespace(),Path(d),1,10_000_000,False)
            with self.assertRaises(ValueError):
                submit(SimpleNamespace(),Path(d),1,10_000_001,True)


if __name__ == "__main__":
    unittest.main()
