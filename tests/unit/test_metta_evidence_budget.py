import pytest

from MeTTa.evidence_budget import Catalog, Episode, bayes_check, check_run, observe
from MeTTa.evidence_budget_pilot import features, train_rows


def test_budget_controls_receipts_and_only_public_state_reaches_policy():
    catalog = Catalog(((0, 1), (2, 3)), (50, 60, 70, 80, 50, 50, 50))
    episode = Episode(catalog, (False, True, True, True, False, False, False))
    observations = []

    def choose(public):
        observations.append(public)
        return public.legal[0]

    result = check_run(episode, choose, budget=3)
    assert result["accepted_route"] == (2, 3)
    assert result["checks"] == 3
    assert observations[1] == observe(catalog, (-1, 0, 0, 0, 0, 0, 0), 2)
    assert not hasattr(observations[0], "current")
    assert not check_run(episode, choose, budget=1)["completed"]
    with pytest.raises(ValueError, match="unavailable"):
        check_run(episode, lambda public: 6)


def test_bayes_and_features_use_priors_to_seek_the_best_feasible_route():
    catalog = Catalog(((0, 1), (2, 3)), (30, 40, 90, 90, 80, 80, 80))
    episode = Episode(catalog, (True,) * 7)
    assert check_run(episode, bayes_check(catalog), budget=2)["accepted_route"] == (2, 3)
    x, legal = features(observe(catalog, (0,) * 7, 2))
    assert x.shape == (7, 12)
    assert legal.tolist() == [True, True, True, True, False, False, False]
    inputs, targets, masks = train_rows([episode])
    assert len(inputs) == len(targets) <= 4
    assert all(mask[target] for mask, target in zip(masks, targets, strict=True))


def test_bayes_work_tie_break_checks_the_riskiest_conjunct_first():
    catalog = Catalog(((0, 1),), (90, 30, 80, 80, 80, 80, 80))
    episode = Episode(catalog, (True, False, True, True, True, True, True))
    early = check_run(episode, bayes_check(catalog, minimize_work=True), budget=2)
    original = check_run(episode, bayes_check(catalog), budget=2)
    assert early["checks"] == 1
    assert original["checks"] == 2
    assert not early["completed"] and not original["completed"]
