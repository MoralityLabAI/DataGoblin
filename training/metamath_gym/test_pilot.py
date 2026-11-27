import unittest

from pilot import apply, metrics, similarity


def task():
    return {"id":"p","split":"train","candidate_signatures":{
        "pair":{"floating":[["setvar","x"],["setvar","y"]],
                "essential":[],"distinct":[["x","y"]],"statement":["pair","x","y"]}},
        "prefix_stack":[["setvar","u"],["setvar","v"]],
        "variables":["u","v"],"distinct":[["u","v"]]}

class TestPilot(unittest.TestCase):
    def test_dv(self):
        x=task()
        self.assertEqual(apply(x,"pair"),["pair","u","v"])
        x["distinct"]=[]
        self.assertIsNone(apply(x,"pair"))
        x["prefix_stack"][1]=["setvar","u"]
        self.assertIsNone(apply(x,"pair"))
    def test_wrong_type(self):
        x=task()
        x["prefix_stack"][1]=["wff","v"]
        self.assertIsNone(apply(x,"pair"))
    def test_score(self):
        rows=[{"id":"p","split":s} for s in ("train","dev","public_challenge")]
        y={"p":{"accepted_actions":["alt"]}}
        r=metrics(rows,y,{"p":["bad","alt"]})
        self.assertEqual(r["dev"]["hits"],{"1":0,"3":1,"5":1})
        self.assertEqual(r["dev"]["hypothetical_full_replays"],{"1":1,"3":2,"5":2})
        self.assertEqual(similarity(["x"],["x"]),1)

if __name__=="__main__":
    unittest.main()
