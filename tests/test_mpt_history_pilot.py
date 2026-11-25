from datagoblin.training_gym.mpt_history_pilot import (
    ARMS,
    STEPS,
    fit,
    generate,
    label,
    lean_source,
    loss,
    rollout,
    verify,
)


def test_generator_and_verifier():
    a = list(generate(101, 2))
    b = list(generate(102, 2))
    assert len(a) == 8 and len({w["root"] for w in [*a, *b]}) == 16
    for w in [*a, *b]:
        for t in w["types"]:
            assert verify(t, label(t)) and not verify(t, 1 - label(t))


def test_matched_budget_and_oracle():
    for w in generate(101, 2):
        d = len(w["types"])
        budget = d + d // 2
        for arm in ARMS:
            row = rollout(w, (0.0, 1.0, 1.0), arm, budget)
            assert row["checks"] <= budget and row["checks"] == len(row["attempts"])
            assert all(a["passed"] == verify(w["types"][a["obligation"]], a["candidate"]) for a in row["attempts"])
            if arm == "typed_oracle":
                assert row["solved"] and row["checks"] == d


def test_training_and_lean_sentinels():
    train = list(generate(901721, 16))
    dev = list(generate(901722, 4))
    c = fit(train)
    assert tuple(c) == STEPS and loss(dev, c[64]) < loss(dev, c[0])
    text = lean_source(dev)
    n = sum(len(w["types"]) for w in dev)
    assert text.count("fail_if_success exact") == n and text.count("#print axioms") == n
