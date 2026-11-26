"""DG-Gym v0.2: deterministic planning primitives, not a hardware profiler."""
from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass

@dataclass(frozen=True)
class Storage:
    key: str
    pool: str
    size: int
    start: int
    end: int


def peak_bytes(storages: Iterable[Storage]) -> dict[str, int]:
    """Half-open lifetimes; aliases must use identical storage records."""
    seen = {}
    events = {}
    for s in storages:
        if (
            not s.key or not s.pool or type(s.size) is not int or s.size < 0
            or type(s.start) is not int or type(s.end) is not int or not 0 <= s.start < s.end
        ):
            raise ValueError("invalid storage record")
        if s.key in seen:
            if seen[s.key] != s:
                raise ValueError("conflicting storage alias")
            continue
        seen[s.key] = s
        e = events.setdefault(s.pool, {})
        e[s.start] = e.get(s.start, 0) + s.size
        e[s.end] = e.get(s.end, 0) - s.size
    out = {}
    for pool, e in events.items():
        current = maximum = 0
        for t in sorted(e):
            current += e[t]
            maximum = max(current, maximum)
        out[pool] = maximum
    return out

@dataclass(frozen=True)
class Job:
    key: str
    ram: int
    vram: int
    cores: int
    ticks: int
    dependencies: tuple[str, ...] = ()


def schedule(jobs: Iterable[Job], caps: tuple[int, int, int]) -> tuple[dict[str, tuple[int, int]], int]:
    """Deterministic nonpreemptive list scheduler using conservative job reservations.

    RAM/VRAM values are incremental exclusive reservations; shared resident pools
    must be reserved outside this scheduler. Time is integer estimated ticks.
    """
    pending = {}
    if len(caps) != 3 or any(type(x) is not int or x < 0 for x in caps):
        raise ValueError("invalid caps")
    for j in jobs:
        if not j.key or j.key in pending or type(j.ticks) is not int or j.ticks <= 0:
            raise ValueError("invalid job identity or duration")
        demands = (j.ram, j.vram, j.cores)
        if any(type(x) is not int or x < 0 or x > cap for x, cap in zip(demands, caps, strict=True)):
            raise ValueError("job exceeds resource cap")
        pending[j.key] = j
    if any(set(j.dependencies) - pending.keys() for j in pending.values()):
        raise ValueError("unknown dependency")
    active = {}
    completed = set()
    intervals = {}
    t = 0
    while pending or active:
        for key, (end, _j) in list(active.items()):
            if end == t:
                completed.add(key)
                del active[key]
        used = [sum(getattr(j, attr) for _, j in active.values()) for attr in ("ram", "vram", "cores")]
        for key in sorted(pending):
            j = pending[key]
            demands = (j.ram, j.vram, j.cores)
            if set(j.dependencies) <= completed and all(u + x <= cap for u, x, cap in zip(used, demands, caps, strict=True)):
                active[key] = (t + j.ticks, j)
                intervals[key] = (t, t + j.ticks)
                used = [u + x for u, x in zip(used, demands, strict=True)]
                del pending[key]
        if active:
            t = min(end for end, _ in active.values())
        elif pending:
            raise ValueError("cyclic dependencies")
    return intervals, t


def paired_outcomes(a: Iterable[int], b: Iterable[int]) -> dict:
    """Independent paired units must be supplied by the caller; no CI is inferred."""
    a, b = tuple(a), tuple(b)
    if not a or len(a) != len(b) or any(type(x) is not int or x not in (0, 1) for x in a + b):
        raise ValueError("invalid paired binary outcomes")
    n00 = n01 = n10 = n11 = 0
    for x, y in zip(a, b, strict=True):
        n00 += (x, y) == (0, 0)
        n01 += (x, y) == (0, 1)
        n10 += (x, y) == (1, 0)
        n11 += (x, y) == (1, 1)
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "gain": (n01 - n10) / len(a),
            "recovery": n01 / (n00+n01) if n00+n01 else None,
            "corruption": n10 / (n10+n11) if n10+n11 else None}

@dataclass(frozen=True)
class WorkerWindow:
    pid: int
    start_identity: str
    owned_leaf: bool
    disposable: bool
    rss_bytes: int
    cpu_core_seconds: float
    io_bytes: int
    gpu_active: bool | None
    lease_expired: bool
    heartbeat_stale: bool


def reclaim_candidates(
    windows: Iterable[WorkerWindow], *, pressure: bool, min_rss: int, max_cpu_seconds: float
) -> list[tuple[int, str]]:
    """Advisory only. The caller must revalidate process identity before any kill.

    CPU-idle GPU/I/O workers and unknown GPU activity are protected. RSS is a
    ranking proxy, not a guaranteed number of reclaimable bytes.
    """
    if not pressure:
        return []
    eligible = [w for w in windows if w.pid > 0 and w.start_identity and w.owned_leaf and w.disposable
                and w.rss_bytes >= min_rss and 0 <= w.cpu_core_seconds <= max_cpu_seconds
                and w.io_bytes == 0 and w.gpu_active is False and w.lease_expired and w.heartbeat_stale]
    return [(w.pid, w.start_identity) for w in sorted(eligible, key=lambda w: (-w.rss_bytes, w.pid, w.start_identity))]
