"""Independent MMS-06 adversarial checks, including non-power-of-two capacities."""
from __future__ import annotations

import numpy as np
import pytest

from datagoblin.training_gym import fast_gradient_bank as mms

MULT = 6364136223846793005


def signed_oracle(key: int, cap: int) -> int:
    word = (int(key) * MULT) % (1 << 64)
    signed = word - (1 << 64) if word >= (1 << 63) else word
    return signed % cap


@pytest.mark.parametrize("cap", [1, 2, 3, 5, 7, 16, 17, 127, 128, 257, 4093, 29127, 65535, 65536])
def test_slot_against_integer_oracle(cap: int) -> None:
    bank = mms.FastGradientBank((cap * 18) / (1 << 20), (1,))
    rng = np.random.default_rng(603)
    keys = np.concatenate((np.arange(512, dtype=np.int64), rng.integers(0, 2**63-1, 1024), [2**63-1]))
    assert bank.capacity == cap
    _, slots = bank._slots(keys)
    assert np.array_equal(slots, np.array([signed_oracle(k, cap) for k in keys]))


def test_non_power_of_two_collision_witness() -> None:
    cap = 29127
    mib = cap * 18 / (1 << 20)
    signed = mms.FastGradientBank(mib, (1,))
    unsigned = mms.FastGradientBank(mib, (1,), hash_scheme=mms.UNSIGNED_HASH)
    keys = np.array([0, 23269], dtype=np.int64)
    for bank in (signed, unsigned):
        bank.write(keys[:1], np.array([[1]], dtype=np.float32), 1)
        bank.write(keys[1:], np.array([[2]], dtype=np.float32), 2)
    assert signed.collisions == 1
    assert unsigned.collisions == 0
    assert signed.read(keys, 3, 10)[1].tolist() == [False, True]
    assert unsigned.read(keys, 3, 10)[1].tolist() == [True, True]

    historic = unsigned.state_dict()
    historic.pop("hash_scheme")
    with pytest.raises(ValueError, match="hash scheme"):
        signed.load_state_dict(historic)
    unsigned.load_state_dict(historic)
    assert unsigned.read(keys, 4, 10)[1].tolist() == [True, True]
    assert signed.receipt()["hash_scheme"] == mms.SIGNED_HASH


def test_invalid_resume_is_atomic() -> None:
    bank = mms.FastGradientBank(0.01, (2,))
    bank.write(np.array([7]), np.ones((1, 2), dtype=np.float32), 1)
    state = bank.state_dict()
    bad = dict(state, tags=state["tags"][:-1])
    with pytest.raises(ValueError, match="shape or dtype"):
        bank.load_state_dict(bad)
    assert np.array_equal(bank.tags, state["tags"])
    bad = dict(state, tags=state["tags"].copy(), age=state["age"].copy())
    index = np.flatnonzero(bad["tags"] >= 0)[0]
    bad["tags"][index] = -2
    with pytest.raises(ValueError, match="tag/age"):
        bank.load_state_dict(bad)
    assert np.array_equal(bank.tags, state["tags"])
    bank.load_state_dict(state)
    assert bank.read(np.array([7]), 2, 10)[1].tolist() == [True]


def test_live_ledger_counts_open_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(mms.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(mms.time, "process_time", lambda: clock[0])
    ledger = mms.CostLedger()
    with ledger.scope("train"):
        clock[0] = 1.0
        assert ledger.total_cpu_seconds == 1.0
        with ledger.scope("lookup"):
            clock[0] = 2.0
            assert ledger.total_cpu_seconds == 2.0
            assert ledger.receipt()["cpu_seconds"]["lookup"] == 1.0
        clock[0] = 3.0
        assert ledger.total_cpu_seconds == 3.0
    assert ledger.receipt()["total_cpu_seconds"] == 3.0
    assert ledger.receipt()["cpu_seconds"]["train"] == 2.0
