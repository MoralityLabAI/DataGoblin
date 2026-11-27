"""Vectorized BF16 direct-mapped GradientBank and non-overlapping CPU/wall CostLedger.

Eliminates per-page Python loops from the baseline GradientBank while preserving
exact collision accounting, max-age expiration, duplicate-key last-write-wins
semantics, hard byte budget caps, and state_dict/load_state_dict resume parity.

Pure NumPy + stdlib so base CI runs without PyTorch.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

SIGNED_HASH = "torch_signed_v1"
UNSIGNED_HASH = "numpy_unsigned_v0"
HASH_SCHEMES = (SIGNED_HASH, UNSIGNED_HASH)

LEDGER_BUCKETS = (
    "train",
    "map",
    "lookup",
    "refresh",
    "feedback",
    "scheduler",
    "verifier",
    "other",
)


def f32_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> BF16 stored as uint16."""
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    return ((u32 + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """Exact BF16 uint16 -> float32."""
    u32 = np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


class CostLedger:
    """Stack-aware non-overlapping process-CPU and wall-time ledger.

    Nested ``with ledger.scope(bucket):`` calls suspend the enclosing bucket so
    ``sum(cpu_seconds.values())`` equals exact elapsed attributed CPU with zero
    double counting.
    """

    def __init__(self) -> None:
        self.cpu: dict[str, float] = dict.fromkeys(LEDGER_BUCKETS, 0.0)
        self.wall: dict[str, float] = dict.fromkeys(LEDGER_BUCKETS, 0.0)
        self.calls: dict[str, int] = dict.fromkeys(LEDGER_BUCKETS, 0)
        self._stack: list[tuple[str, float, float]] = []

    @contextmanager
    def scope(self, bucket: str) -> Iterator[None]:
        if bucket not in self.cpu:
            raise ValueError(f"unknown cost ledger bucket: {bucket!r}")
        now_w, now_c = time.perf_counter(), time.process_time()
        # Flush enclosing scope up to this boundary so inner time is never double-counted
        if self._stack:
            p_bucket, p_w, p_c = self._stack[-1]
            self.wall[p_bucket] += max(0.0, now_w - p_w)
            self.cpu[p_bucket] += max(0.0, now_c - p_c)
        self._stack.append((bucket, now_w, now_c))
        self.calls[bucket] += 1
        try:
            yield
        finally:
            end_w, end_c = time.perf_counter(), time.process_time()
            cur_bucket, cur_w, cur_c = self._stack.pop()
            self.wall[cur_bucket] += max(0.0, end_w - cur_w)
            self.cpu[cur_bucket] += max(0.0, end_c - cur_c)
            if self._stack:
                p_bucket, _, _ = self._stack[-1]
                self._stack[-1] = (p_bucket, end_w, end_c)

    def _snapshot(self) -> tuple[dict[str, float], dict[str, float]]:
        """Include accrued work in the currently active scope, without mutating the ledger."""
        cpu, wall = dict(self.cpu), dict(self.wall)
        if self._stack:
            bucket, start_w, start_c = self._stack[-1]
            wall[bucket] += max(0.0, time.perf_counter() - start_w)
            cpu[bucket] += max(0.0, time.process_time() - start_c)
        return cpu, wall

    @property
    def total_cpu_seconds(self) -> float:
        cpu, _ = self._snapshot()
        return float(sum(cpu.values()))

    @property
    def total_wall_seconds(self) -> float:
        _, wall = self._snapshot()
        return float(sum(wall.values()))

    def receipt(self) -> dict[str, Any]:
        cpu, wall = self._snapshot()
        return {
            "cpu_seconds": {k: round(v, 6) for k, v in cpu.items()},
            "wall_seconds": {k: round(v, 6) for k, v in wall.items()},
            "calls": dict(self.calls),
            "total_cpu_seconds": round(sum(cpu.values()), 6),
            "total_wall_seconds": round(sum(wall.values()), 6),
        }


class FastGradientBank:
    """Vectorized flat direct-mapped BF16 gradient bank under a hard byte cap.

    Stores ``data[capacity, *shape]`` (uint16 BF16), ``tags[capacity]`` (int64),
    and ``age[capacity]`` (int64). All read/write operations are single vectorized
    NumPy passes with zero per-page Python loops.
    """

    def __init__(
        self, mib: float, shape: tuple[int, ...], *, hash_scheme: str = SIGNED_HASH
    ) -> None:
        if hash_scheme not in HASH_SCHEMES:
            raise ValueError("unknown MMS hash scheme")
        self.hash_scheme = hash_scheme
        if not math.isfinite(mib) or mib <= 0:
            raise ValueError("positive finite MMS budget required")
        self.shape = tuple(int(d) for d in shape)
        if not self.shape or any(d <= 0 for d in self.shape):
            raise ValueError("invalid bank row shape")
        self.budget_bytes = int(mib * (1 << 20))
        row_bytes = 2 * math.prod(self.shape) + 16
        self.capacity = self.budget_bytes // row_bytes
        if self.capacity < 1:
            raise ValueError("MMS budget cannot hold one row")

        self.data = np.zeros((self.capacity, *self.shape), dtype=np.uint16)
        self.tags = np.full(self.capacity, -1, dtype=np.int64)
        self.age = np.full(self.capacity, -1, dtype=np.int64)

        self.hits = 0
        self.queries = 0
        self.writes = 0
        self.collisions = 0
        self.stale_misses = 0
        self.same_step_skips = 0

    @property
    def owned_array_bytes(self) -> int:
        return int(self.data.nbytes + self.tags.nbytes + self.age.nbytes)

    @property
    def allocated_bytes(self) -> int:
        return self.owned_array_bytes

    @property
    def occupied_rows(self) -> int:
        return int((self.tags >= 0).sum())

    def _slots(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = np.asarray(keys, dtype=np.int64).reshape(-1)
        if k.size and bool((k < 0).any()):
            raise ValueError("negative MMS key")
        # Legacy Torch multiplies in signed int64, wrapping modulo 2**64,
        # then applies a nonnegative remainder. Unsigned v0 is retained for
        # explicit historical replay; it has a different collision relation.
        with np.errstate(over="ignore"):
            if self.hash_scheme == SIGNED_HASH:
                prod = k * np.int64(6364136223846793005)
                slots = np.remainder(prod, self.capacity)
            else:
                prod = k.astype(np.uint64) * np.uint64(6364136223846793005)
                slots = (prod % np.uint64(self.capacity)).astype(np.int64)
        return k, slots

    def read(self, keys: np.ndarray, step: int, max_age: int) -> tuple[np.ndarray, np.ndarray]:
        k, slots = self._slots(keys)
        n = k.shape[0]
        self.queries += n
        result = np.zeros((n, *self.shape), dtype=np.float32)
        if n == 0:
            return result, np.zeros(0, dtype=bool)
        stored_tags = self.tags[slots]
        stored_age = self.age[slots]
        tag_match = stored_tags == k
        age_diff = step - stored_age
        valid = tag_match & (stored_age >= 0) & (stored_age < step) & (age_diff <= max_age)
        self.same_step_skips += int((tag_match & (stored_age >= step)).sum())
        self.stale_misses += int((tag_match & (stored_age >= 0) & (age_diff > max_age)).sum())
        if bool(valid.any()):
            result[valid] = bf16_bits_to_f32(self.data[slots[valid]])
            self.hits += int(valid.sum())
        return result, valid

    def write(self, keys: np.ndarray, values: np.ndarray, step: int) -> None:
        k, slots = self._slots(keys)
        vals = np.asarray(values, dtype=np.float32)
        if tuple(vals.shape) != (k.shape[0], *self.shape):
            raise ValueError(f"MMS row shape mismatch: expected {(k.shape[0], *self.shape)}, got {vals.shape}")
        n = k.shape[0]
        if n == 0:
            return
        # Deduplicate slots preserving last-write-wins semantics within the batch
        rev_slots = slots[::-1]
        uniq_slots, rev_first = np.unique(rev_slots, return_index=True)
        last_ix = (n - 1) - rev_first

        k_last = k[last_ix]
        existing = self.tags[uniq_slots]
        self.collisions += int(((existing >= 0) & (existing != k_last)).sum())

        self.data[uniq_slots] = f32_to_bf16_bits(vals[last_ix])
        self.tags[uniq_slots] = k_last
        self.age[uniq_slots] = int(step)
        self.writes += int(uniq_slots.shape[0])

    def state_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "hash_scheme": self.hash_scheme,
            "capacity": self.capacity,
            "budget_bytes": self.budget_bytes,
            "data": self.data.copy(),
            "tags": self.tags.copy(),
            "age": self.age.copy(),
            "hits": self.hits,
            "queries": self.queries,
            "writes": self.writes,
            "collisions": self.collisions,
            "stale_misses": self.stale_misses,
            "same_step_skips": self.same_step_skips,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            tuple(state["shape"]) != self.shape
            or int(state["capacity"]) != self.capacity
            or int(state["budget_bytes"]) != self.budget_bytes
        ):
            raise ValueError("MMS checkpoint budget/shape mismatch")
        # Historical unversioned FastGradientBank states used unsigned hashing.
        # Do not silently resume them under a different collision policy.
        if state.get("hash_scheme", UNSIGNED_HASH) != self.hash_scheme:
            raise ValueError("MMS checkpoint hash scheme mismatch")
        arrays = (
            (state["data"], np.dtype(np.uint16), (self.capacity, *self.shape)),
            (state["tags"], np.dtype(np.int64), (self.capacity,)),
            (state["age"], np.dtype(np.int64), (self.capacity,)),
        )
        for array, dtype, shape in arrays:
            if not isinstance(array, np.ndarray) or array.dtype != dtype or array.shape != shape:
                raise ValueError("invalid MMS checkpoint array shape or dtype")
        data, tags, age = (a for a, _, _ in arrays)
        occupied = tags >= 0
        if bool((tags < -1).any()) or bool((age[~occupied] != -1).any()) or bool((age[occupied] < 0).any()):
            raise ValueError("invalid MMS checkpoint tag/age")
        _, slots = self._slots(tags[occupied])
        if not np.array_equal(slots, np.flatnonzero(occupied)):
            raise ValueError("MMS checkpoint tag does not match its hash slot")
        fields = ("hits", "queries", "writes", "collisions", "stale_misses", "same_step_skips")
        counters = {field: state[field] for field in fields}
        if any(type(v) is not int or v < 0 for v in counters.values()):
            raise ValueError("invalid MMS checkpoint counter")
        if data.nbytes + tags.nbytes + age.nbytes > self.budget_bytes:
            raise ValueError("MMS checkpoint exceeds payload budget")
        # Validate before mutating live state; temporary copy memory is additional.
        self.data, self.tags, self.age = data.copy(), tags.copy(), age.copy()
        for field, value in counters.items():
            setattr(self, field, value)

    def receipt(self) -> dict[str, Any]:
        return {
            "budget_bytes": self.budget_bytes,
            "hash_scheme": self.hash_scheme,
            "allocated_bytes": self.allocated_bytes,
            "owned_array_bytes": self.owned_array_bytes,
            "capacity": self.capacity,
            "occupied_rows": self.occupied_rows,
            "hits": self.hits,
            "queries": self.queries,
            "writes": self.writes,
            "collisions": self.collisions,
            "stale_misses": self.stale_misses,
            "same_step_skips": self.same_step_skips,
        }
