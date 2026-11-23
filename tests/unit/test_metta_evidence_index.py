import random
from dataclasses import replace

from MeTTa.evidence_index import ProofIndex
from MeTTa.evidence_index_pilot import PILOT_FAMILIES, stress_episode
from MeTTa.evidence_proofs import Record, Rule, Snapshot
from MeTTa.evidence_revision import summary


def test_retiring_a_short_proof_recovers_the_previously_dominated_route():
    index = ProofIndex()
    rule = Rule("Long", ("A", "B"), "Review")
    direct = Record("Direct", "Review", "SourceA")
    premises = (Record("First", "A", "SourceA"), Record("Second", "B", "SourceB"))
    before, _ = index.evaluate(Snapshot((direct, *premises), (rule,), 1))
    after, work = index.evaluate(Snapshot(premises, (rule,), 1))
    assert before["minimal_sources"] == [["SourceA"]]
    assert after["minimal_sources"] == [["SourceA", "SourceB"]]
    assert work["computed_cells"] == 1 and work["cache_hits"] == 2


def test_cached_empty_cell_is_invalidated_on_evidence_arrival():
    index = ProofIndex()
    rules = (Rule("Goal", ("A", "B"), "Review"),)
    first = Record("First", "A", "SourceA")
    assert not index.evaluate(Snapshot((first,), rules, 1))[0]["proofs"]
    pair = (first, Record("Second", "B", "SourceB"))
    result, work = index.evaluate(Snapshot(pair, rules, 1))
    assert result["minimal_sources"] == [["SourceA", "SourceB"]]
    assert work["invalidated_cells"] == 2
    assert index.evaluate(Snapshot(pair, rules, 1))[1] == {"invalidated_cells": 0, "cache_hits": 1}


def test_index_matches_independent_oracle_through_topology_and_visibility_changes():
    for family in PILOT_FAMILIES:
        for seed in range(8):
            index = ProofIndex()
            for snapshot in stress_episode(family, seed):
                result, _ = index.evaluate(snapshot)
                assert result == summary(snapshot)


def test_wide_rule_update_reuses_unchanged_joins():
    initial, changed, *_ = stress_episode("wide_alternatives", 4)
    index = ProofIndex()
    index.evaluate(initial)
    result, work = index.evaluate(changed)
    assert result == summary(changed)
    assert work["rule_joins"] == 1
    assert work["join_cache_hits"] == len(initial.rules) - 1



def test_index_handles_reordered_and_rebound_facts_in_bounded_cycles():
    rng = random.Random(314)
    claims = ("A", "B", "C", "Review")
    for _ in range(30):
        index = ProofIndex()
        records = [Record(f"R{i}", rng.choice(claims), f"S{rng.randrange(3)}") for i in range(5)]
        rules = [Rule(f"R{i}", tuple(rng.choices(claims, k=rng.randrange(1, 3))), rng.choice(claims))
                 for i in range(4)]
        for _ in range(6):
            snapshot = Snapshot(tuple(records), tuple(rules), rng.randrange(4))
            assert index.evaluate(snapshot)[0] == summary(snapshot)
            moved, changed = rng.randrange(len(records)), rng.randrange(len(rules))
            records[moved] = replace(records[moved], claim=rng.choice(claims), root=f"S{rng.randrange(3)}",
                                     visible=rng.choice((True, False)), current=rng.choice((True, False)))
            rules[changed] = replace(rules[changed], premises=(rng.choice(claims),), conclusion=rng.choice(claims))
            rng.shuffle(records)
            rng.shuffle(rules)
