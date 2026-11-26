"""Regression tests for vectorized FastGradientBank and non-overlapping CostLedger."""

from __future__ import annotations

import time

import numpy as np
import pytest

from datagoblin.training_gym.fast_gradient_bank import (
    CostLedger,
    FastGradientBank,
    bf16_bits_to_f32,
    f32_to_bf16_bits,
)


def test_bf16_roundtrip_and_budget_cap() -> None:
    vals = np.array([-3.5, -1.0, 0.0, 0.25, 1.5, 12.75], dtype=np.float32)
    rt = bf16_bits_to_f32(f32_to_bf16_bits(vals))
    assert np.allclose(rt, vals, atol=1e-2)

    bank = FastGradientBank(mib=0.05, shape=(64,))
    assert bank.allocated_bytes <= bank.budget_bytes
    assert bank.capacity == bank.budget_bytes // (2 * 64 + 16)

    with pytest.raises(ValueError):
        FastGradientBank(mib=0.0, shape=(64,))


def test_duplicate_keys_last_write_wins_collisions_expiration_and_resume() -> None:
    # Tiny bank holding ~2 slots so collisions are guaranteed
    row_bytes = 2 * 4 + 16  # 24 bytes/slot
    bank = FastGradientBank(mib=(2 * row_bytes + 4) / (1 << 20), shape=(4,))
    assert bank.capacity == 2

    # Duplicate key 7 in same batch: last write (v2) must win!
    keys = np.array([7, 7], dtype=np.int64)
    v1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    v2 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    bank.write(keys, np.stack([v1, v2]), step=10)
    assert bank.writes == 1
    assert bank.collisions == 0

    # Same-step read must miss (same_step_skips += 1)
    _, hit_same = bank.read(np.array([7]), step=10, max_age=5)
    assert not bool(hit_same[0])
    assert bank.same_step_skips == 1

    # Next-step read hits v2
    out, hit_ok = bank.read(np.array([7]), step=12, max_age=5)
    assert bool(hit_ok[0])
    assert np.allclose(out[0], v2, atol=1e-2)

    # Expired read misses (stale_misses += 1)
    _, hit_stale = bank.read(np.array([7]), step=20, max_age=5)
    assert not bool(hit_stale[0])
    assert bank.stale_misses == 1

    # State dict resume preserves exact state
    sd = bank.state_dict()
    bank2 = FastGradientBank(mib=(2 * row_bytes + 4) / (1 << 20), shape=(4,))
    bank2.load_state_dict(sd)
    out2, hit2 = bank2.read(np.array([7]), step=13, max_age=5)
    assert bool(hit2[0])
    assert np.allclose(out2[0], v2, atol=1e-2)

    # Force collision by writing keys 0, 1, 2 into capacity-2 bank
    bank.write(np.array([0, 1, 2], dtype=np.int64), np.ones((3, 4), dtype=np.float32), step=14)
    assert bank.collisions >= 1


def test_cost_ledger_non_overlapping_nested_scopes() -> None:
    ledger = CostLedger()
    t0 = time.perf_counter()
    with ledger.scope("train"):
        _ = sum(i * i for i in range(50_000))
        with ledger.scope("lookup"):
            _ = sum(i * i for i in range(50_000))
        with ledger.scope("scheduler"):
            _ = sum(i * i for i in range(25_000))
    elapsed = time.perf_counter() - t0
    rcpt = ledger.receipt()
    # Non-overlapping guarantee: sum of all wall buckets must not exceed outer wall + tiny slack
    assert rcpt["total_wall_seconds"] <= elapsed + 0.02
    assert rcpt["calls"]["train"] == 1
    assert rcpt["calls"]["lookup"] == 1
    assert rcpt["calls"]["scheduler"] == 1
