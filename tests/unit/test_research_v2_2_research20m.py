"""Independent cache ledger and budget allocation regression."""
from datagoblin.research_v2_2 import research20m as m


def test_quota_and_memory_replay():
    assert sum(m.QUOTAS.values()) == m.TARGET == 15_778_719
    for n in (12,10001,m.TARGET):
        assert sum(m.quota_plan(n).values())==n
    for i in range(32):
        for profile in ("offline_cache","probe_alias"):
            d = m.world(None,"research_memory",profile,199,i)
            after,change = m.revised("research_memory",d,__import__("random").Random(i))
            rows = m.memory_rows(d,after,change)
            assert len(rows)==4
            if profile=="offline_cache":
                assert m.cache_exact(d)["cost"]<=m.cache_delayed(d)["cost"]
            else:
                assert m.probe_exact(d)["missed"]>=0


def test_parity_revisions():
    from datagoblin.research_v2_2.exact_research import ExactResearch
    import random
    for i in range(12):
        d = m.world(None,"adaptive_evidence","parity_synergy",199,i)
        after,_ = m.revised("adaptive_evidence",d,random.Random(i))
        a,b = ExactResearch().solve(d),ExactResearch().solve(after)
        assert a["training_admission"] is False and b["training_admission"] is False
        assert len(d["tests"])<=4
