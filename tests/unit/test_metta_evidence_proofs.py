from MeTTa.evidence_proofs import FAMILIES, Record, Rule, Snapshot, proof_oracle, public_query, source_pair


def test_generated_pairs_flip_support_at_matched_visible_counts():
    for family in FAMILIES:
        for seed in range(16):
            positive, negative = source_pair(family, seed)
            assert proof_oracle(positive)
            assert not proof_oracle(negative)
            assert len(positive.records) == len(negative.records)
            assert sum(r.current for r in positive.records) == sum(r.current for r in negative.records)
            assert [len(r.premises) for r in positive.rules] == [len(r.premises) for r in negative.rules]


def test_hidden_record_and_stale_witness_do_not_enter_a_proof():
    snapshot = Snapshot(
        (Record("Hidden", "A", "PrivateRoot", visible=False), Record("Old", "A", "Archive", current=False)),
        (Rule("Direct", ("A",), "Review"),),
        depth=1,
    )
    assert "PrivateRoot" not in public_query(snapshot)
    assert not proof_oracle(snapshot)
