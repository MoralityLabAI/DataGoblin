"""Exact finite-distribution and independent optimizer checks for the decoder tournament."""

import numpy as np
import pytest
import torch
from decoder_lab import (
    CONDITIONS,
    WORLDS,
    Reader,
    cells,
    fit_table,
    labels,
    lookup,
    make_reader,
    metric_vectors,
    observations,
    train_ensemble,
    world_labels,
)
from receipt_gym import controls


def toy():
    query = np.zeros((2, 12))
    query[:, 0] = query[:, 5] = 1
    query[0, 6] = query[1, 7] = 1
    query[:, 10] = 1
    side = WORLDS[:, [0, 1]]
    return np.zeros((8, 12)), query, world_labels(query), side, np.array([0.125, 0.875])


def test_conditional_shuffle_preserves_marginals_and_breaks_target_dependence():
    raw = toy()
    code = WORLDS[:, 2]
    masses, positive = {}, {}
    for condition in CONDITIONS:
        _, masses[condition], positive[condition] = cells(raw, code, condition)
        assert masses[condition].sum() == pytest.approx(1)
    assert np.array_equal(masses["valid"], masses["shuffle"])
    assert np.array_equal(masses["side"], masses["constant"])
    oracle, _ = controls(raw)
    risk = {}
    for condition in CONDITIONS:
        x = metric_vectors(masses[condition], positive[condition], np.full(len(masses[condition]), 0.5))
        risk[condition] = x["bayes"]
        assert x["brier"] == pytest.approx(0.25)
    assert risk["side"] == pytest.approx(oracle[0])
    assert risk["constant"] == pytest.approx(risk["side"])
    assert risk["shuffle"] == pytest.approx(risk["side"])
    assert risk["valid"] == pytest.approx(oracle[240 // 2])
    assert risk["valid"] < risk["shuffle"]
    with pytest.raises(ValueError, match="binary"):
        cells(raw, code + 0.01, "valid")


def test_structured_side_inference_matches_oracle_with_no_actual_world_as_input():
    raw = toy()
    x, mass, yes = cells(raw, WORLDS[:, 2], "side")
    assert (x[:, -1] == 0).all()
    model = Reader("structured")
    probs = model(torch.from_numpy(x)).detach().numpy()
    assert metric_vectors(mass, yes, probs)["regret"] == pytest.approx(0)
    for condition in CONDITIONS:
        x = observations(raw[1], condition)
        assert x.shape == (8, 14)
    table = fit_table([cells(raw, WORLDS[:, 2], "valid")])
    assert all(0 <= lookup(table, row) <= 1 for row in x)


def test_inverting_unseen_frame_bit_eliminates_its_pooled_oracle_value():
    from decoder_pooled import pooled_oracle

    raw = toy()
    code = WORLDS[:, 2]
    valid = [cells(raw, bits, "valid") for bits in (code, 1 - code)]
    side = [cells(raw, bits, "side") for bits in (code, 1 - code)]
    assert pooled_oracle(valid)["brier"] == pytest.approx(pooled_oracle(side)["brier"])
    assert pooled_oracle([valid[0]])["brier"] < pooled_oracle(valid)["brier"]


@pytest.mark.parametrize("architecture", ["C", "D", "MLP"])
def test_valid_positive_and_negative_symbols_reuse_native_receiver_exactly(architecture):
    reader = make_reader(architecture, 137)
    x = torch.from_numpy(observations(toy()[1], "valid"))
    native = reader.actor.decode(x[:, :12], (x[:, 12] + 1) / 2, (x[:, 13] + 1) / 2).sigmoid()
    torch.testing.assert_close(reader(x), native, atol=0, rtol=0)


def test_vectorized_logistic_matches_individual_adam_and_clipping(monkeypatch):
    import decoder_lab

    monkeypatch.setattr(decoder_lab, "STEPS", 3)
    raw, seed = toy(), 137
    keys = labels(seed)
    channels = {key: [cells(raw, WORLDS[:, i % 3], key[1])] for i, key in enumerate(keys)}
    state = train_ensemble("logistic", seed, channels)
    for index in (0, 5, 11):
        expected = make_reader("logistic", seed)
        optimizer = torch.optim.Adam(expected.parameters(), lr=0.003)
        x, mass, yes = [torch.tensor(v, dtype=torch.float32) for v in channels[keys[index]][0]]
        for _ in range(3):
            loss = (mass * (expected(x) - yes / mass.clamp_min(1e-15)).square()).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.0)
            optimizer.step()
        for name, parameter in expected.named_parameters():
            torch.testing.assert_close(state[index][name], parameter)
