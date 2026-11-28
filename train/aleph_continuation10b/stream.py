"""Deterministic multi-lane packed stream and detached replay for Aleph continuation stages.

Builds directly on the frozen ``aleph.packed-stream.v1`` (``training/aleph_rlt_placement/packing.py``)
format without mutating the 10M pilot runner or verifier. Enforces:
  - exact frozen SmolLM2 tokenizer digest (``9ca9acdd...``), context 2048, vocab 49152, pad 0;
  - immutable per-lane manifest, SQLite registry, and shard SHA-256 checks;
  - deterministic deficit-first round-robin across stage lanes (tie-break by lane name);
  - zero wraparound on exhaustion and exact boundary remainders
    (e.g. 90,000,000 = 343 * 262,144 + 84,608 = 344 optimizer steps).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from collections import Counter
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(
    (parent for parent in (HERE, *HERE.parents) if (parent / "training" / "aleph_next" / "candidate01").is_dir()),
    HERE.parent.parent,
)
TRAINING_ROOT = REPO_ROOT / "training"
ALEPH_NEXT = TRAINING_ROOT / "aleph_next"
for p in (str(TRAINING_ROOT / "aleph_rlt_placement"), str(ALEPH_NEXT / "candidate01")):
    if p not in sys.path:
        sys.path.insert(0, p)

from packing import PackedStream, fingerprint  # noqa: E402
from production import TOKENIZER_SHA256, digest, read  # noqa: E402

STAGE_INDEX_SCHEMA = "aleph.continuation-stage-index.v1"
STAGE_REPLAY_SCHEMA = "aleph.continuation-stage-replay.v1"
DEFAULT_GROUP_POSITIONS = 262_144


def stage_group_boundaries(
    stage_new_positions: int,
    group_positions: int = DEFAULT_GROUP_POSITIONS,
) -> list[int]:
    """Return cumulative stage-local non-padding position boundaries for every update group."""
    if stage_new_positions <= 0 or group_positions <= 0:
        raise ValueError("stage_new_positions and group_positions must be positive")
    stops: list[int] = []
    pos = 0
    while pos < stage_new_positions:
        pos = min(stage_new_positions, pos + group_positions)
        stops.append(pos)
    return stops


def verify_lane_manifest(
    manifest_path: Path,
    expected_manifest_sha256: str,
    *,
    split: str = "train",
    expected_tokenizer_sha256: str = TOKENIZER_SHA256,
    context: int = 2048,
    vocab_size: int = 49152,
    pad_id: int = 0,
) -> dict[str, Any]:
    """Validate an immutable ``aleph.packed-stream.v1`` lane manifest, registry, and shards."""
    path = Path(manifest_path).resolve()
    actual_sha = digest(path)
    if actual_sha != expected_manifest_sha256:
        raise ValueError(f"lane manifest digest mismatch for {path.name}: {actual_sha} != {expected_manifest_sha256}")
    manifest = read(path)
    if manifest.get("schema") != "aleph.packed-stream.v1":
        raise ValueError(f"unexpected lane manifest schema: {manifest.get('schema')}")
    if (
        manifest["tokenizer"]["sha256"],
        manifest["context"],
        manifest["vocab_size"],
        manifest["pad_id"],
    ) != (expected_tokenizer_sha256, context, vocab_size, pad_id):
        raise ValueError("unexpected tokenizer/context/vocab/pad configuration in lane manifest")
    for spec in (manifest["registry"], manifest["tokenizer"], *manifest["shards"]):
        member = (path.parent / spec["path"]).resolve()
        if not member.is_relative_to(path.parent) or member.is_symlink():
            raise ValueError(f"unsafe lane artifact path: {spec['path']}")
        if digest(member) != spec["sha256"]:
            raise ValueError(f"lane artifact SHA-256 mismatch: {spec['path']}")
    shard_splits = {s["split"] for s in manifest["shards"]}
    if shard_splits != {split}:
        raise ValueError(f"lane shards must strictly belong to split={split!r}, got {sorted(shard_splits)}")

    uri = (path.parent / manifest["registry"]["path"]).resolve().as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        rows = db.execute(
            "SELECT roots.root, families.family, families.split FROM roots "
            "JOIN families ON roots.family = families.family"
        ).fetchall()
        max_pres = db.execute("SELECT COALESCE(MAX(presentation), 0) FROM views").fetchone()[0]
    if any(sp != split for _, _, sp in rows):
        raise ValueError("non-matching split found inside lane registry")
    if int(max_pres) > 4:
        raise ValueError(f"static-root presentation cap violated: max={max_pres} > 4")
    return {
        "manifest": manifest,
        "roots": {r for r, _, _ in rows},
        "families": {f for _, f, _ in rows},
        "max_presentation": int(max_pres),
    }


def _pack_digest(lane: str, pack: dict[str, Any]) -> str:
    payload = {
        "lane": lane,
        "input_ids": pack["input_ids"],
        "position_ids": pack["position_ids"],
        "segment_ids": pack["segment_ids"],
        "labels_at_query": pack["labels_at_query"],
        "nonpadding": pack["nonpadding"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


class StageMixedStream:
    """Generalized deficit-first round-robin stream across arbitrary continuation stage lanes."""

    def __init__(
        self,
        base: Path,
        index_path: Path,
        expected_index_sha256: str,
        stack: ExitStack,
        cursor: dict[str, Any] | None = None,
        *,
        verify_all_members: bool = False,
    ) -> None:
        self.base = Path(base).resolve()
        self.index_path = Path(index_path).resolve()
        actual_sha = digest(self.index_path)
        if actual_sha != expected_index_sha256:
            raise ValueError(f"stage index digest mismatch: {actual_sha} != {expected_index_sha256}")
        self.index_sha256 = expected_index_sha256
        self.index = read(self.index_path)
        if self.index.get("schema") != STAGE_INDEX_SCHEMA:
            raise ValueError(f"unexpected stage index schema: {self.index.get('schema')}")
        self.split = str(self.index.get("split", "train"))
        self.stage_start = int(self.index["stage_start_positions"])
        self.stage_stop = int(self.index["stage_stop_positions"])
        self.stage_new_positions = int(self.index["stage_new_positions"])
        self.group_positions = int(self.index.get("update_group_positions", DEFAULT_GROUP_POSITIONS))
        self.context = int(self.index.get("context", 2048))
        self.vocab_size = int(self.index.get("vocab_size", 49152))
        self.tokenizer_sha256 = str(self.index["tokenizer_sha256"])

        if self.stage_stop - self.stage_start != self.stage_new_positions:
            raise ValueError("stage_stop - stage_start != stage_new_positions")
        self.quotas: dict[str, int] = {k: int(v) for k, v in self.index["lane_quotas"].items()}
        if sum(self.quotas.values()) != self.stage_new_positions:
            raise ValueError("sum(lane_quotas) != stage_new_positions")
        self.lanes_order = sorted(self.quotas.keys())

        if cursor is not None:
            payload = cursor.get("payload")
            if not isinstance(payload, dict) or fingerprint(payload) != cursor.get("sha256"):
                raise ValueError("stage cursor checksum mismatch")
            if payload.get("index_sha256") != expected_index_sha256:
                raise ValueError("stage cursor index_sha256 mismatch (stale cursor)")
            if payload.get("split") != self.split:
                raise ValueError("stage cursor split mismatch")
            if set(payload.get("totals", {}).keys()) != set(self.lanes_order):
                raise ValueError("stage cursor lane set mismatch")
            self.totals = {k: int(v) for k, v in payload["totals"].items()}
            self.target_totals = {k: int(v) for k, v in payload["target_totals"].items()}
            self.groups_completed = int(payload["groups_completed"])
            lane_cursors = payload["lane_cursors"]
        else:
            self.totals = {k: 0 for k in self.lanes_order}
            self.target_totals = {k: 0 for k in self.lanes_order}
            self.groups_completed = 0
            lane_cursors = {k: None for k in self.lanes_order}

        self.streams: dict[str, PackedStream] = {}
        seen_roots: set[str] = set()
        for lane in self.lanes_order:
            meta = self.index["lanes"][lane]
            mpath = (self.base / meta["manifest_path"]).resolve()
            if not mpath.is_relative_to(self.base):
                raise ValueError(f"lane manifest escapes base dir: {meta['manifest_path']}")
            if verify_all_members:
                checked = verify_lane_manifest(
                    mpath,
                    meta["manifest_sha256"],
                    split=self.split,
                    expected_tokenizer_sha256=self.tokenizer_sha256,
                    context=self.context,
                    vocab_size=self.vocab_size,
                )
                overlap = seen_roots & checked["roots"]
                if overlap:
                    raise ValueError(f"cross-lane duplicate root detected in {lane}: {len(overlap)}")
                seen_roots |= checked["roots"]
            self.streams[lane] = stack.enter_context(
                PackedStream(mpath, meta["manifest_sha256"], self.split, lane_cursors[lane])
            )

    @property
    def processed_stage_positions(self) -> int:
        return sum(self.totals.values())

    @property
    def absolute_positions(self) -> int:
        return self.stage_start + self.processed_stage_positions

    def cursor(self) -> dict[str, Any]:
        payload = {
            "schema": "aleph.continuation-stage-cursor.v1",
            "index_sha256": self.index_sha256,
            "split": self.split,
            "groups_completed": self.groups_completed,
            "totals": dict(self.totals),
            "target_totals": dict(self.target_totals),
            "lane_cursors": {lane: self.streams[lane].checkpoint() for lane in self.lanes_order},
        }
        return {"payload": payload, "sha256": fingerprint(payload)}

    def next_pack(self, group_remaining: int) -> tuple[str, dict[str, Any]]:
        """Choose the most under-target lane using exact integer arithmetic (tie-break by lane name)."""
        if group_remaining <= 0:
            raise ValueError("group_remaining must be positive")
        done = self.processed_stage_positions
        next_pos = min(done + self.context, done + group_remaining, self.stage_new_positions)
        remaining = [lane for lane in self.lanes_order if self.totals[lane] < self.quotas[lane]]
        if not remaining:
            raise RuntimeError("stage corpus exhausted before requested group completion (no wraparound permitted)")
        chosen_lane = min(
            remaining,
            key=lambda lane: (self.totals[lane] * self.stage_new_positions - self.quotas[lane] * next_pos, lane),
        )
        rem_lane = self.quotas[chosen_lane] - self.totals[chosen_lane]
        cap = min(next_pos - done, rem_lane, self.context)
        target_for_stream = self.totals[chosen_lane] + cap
        pack = self.streams[chosen_lane].next_pack(target_for_stream)
        if pack is None:
            raise RuntimeError(
                f"lane {chosen_lane!r} exhausted at {self.totals[chosen_lane]} / {self.quotas[chosen_lane]} "
                "(no wraparound permitted)"
            )
        pos = int(pack["nonpadding"])
        if pos != cap:
            raise RuntimeError(
                f"lane {chosen_lane!r} exhausted mid-pack: got {pos} positions, expected {cap} "
                "(no wraparound permitted)"
            )
        labels = sum(1 for v in pack["labels_at_query"] if v != -100)
        self.totals[chosen_lane] += pos
        self.target_totals[chosen_lane] += labels
        return chosen_lane, pack


def collect_stage_group(
    stream: StageMixedStream,
    expected_stop_stage: int,
    *,
    expected_digest: str | None = None,
    expected_labels: int | None = None,
) -> tuple[list[tuple[str, dict[str, Any], int]], int, dict[str, Any]]:
    """Collect one exact update group up to ``expected_stop_stage`` stage-local positions."""
    start_pos = stream.processed_stage_positions
    if expected_stop_stage <= start_pos:
        raise ValueError(f"group stop {expected_stop_stage} <= current stage position {start_pos}")
    packs: list[tuple[str, dict[str, Any], int]] = []
    h = hashlib.sha256()
    lane_positions: Counter[str] = Counter()
    lane_labels: Counter[str] = Counter()
    total_padding = 0
    total_labels = 0
    while stream.processed_stage_positions < expected_stop_stage:
        rem = expected_stop_stage - stream.processed_stage_positions
        lane, pack = stream.next_pack(rem)
        k = sum(1 for v in pack["labels_at_query"] if v != -100)
        packs.append((lane, pack, k))
        h.update(_pack_digest(lane, pack).encode("ascii"))
        lane_positions[lane] += int(pack["nonpadding"])
        lane_labels[lane] += k
        total_padding += stream.context - int(pack["nonpadding"])
        total_labels += k
    stream.groups_completed += 1
    group_sha = h.hexdigest()
    positions = expected_stop_stage - start_pos
    if expected_digest is not None and group_sha != expected_digest:
        raise ValueError(f"group digest mismatch at stop={expected_stop_stage}: {group_sha} != {expected_digest}")
    if expected_labels is not None and total_labels != expected_labels:
        raise ValueError(
            f"group label count mismatch at stop={expected_stop_stage}: {total_labels} != {expected_labels}"
        )
    if total_labels <= 0:
        raise ValueError(f"empty supervised label group at stop={expected_stop_stage}")
    meta = {
        "stage_group": stream.groups_completed,
        "stage_stop": expected_stop_stage,
        "absolute_stop": stream.stage_start + expected_stop_stage,
        "positions": positions,
        "labels": total_labels,
        "packs": len(packs),
        "padding": total_padding,
        "lane_positions": dict(sorted(lane_positions.items())),
        "lane_labels": dict(sorted(lane_labels.items())),
        "sha256": group_sha,
    }
    return packs, total_labels, meta


def replay_stage_release(
    base: Path,
    index_sha256: str,
    *,
    restart_after_group: int | None = None,
) -> dict[str, Any]:
    """Deterministically replay every group in a stage release and verify mid-stage restart equality."""
    base_dir = Path(base).resolve()
    index_path = base_dir / "index.json"
    idx = read(index_path)
    boundaries = stage_group_boundaries(
        int(idx["stage_new_positions"]),
        int(idx.get("update_group_positions", DEFAULT_GROUP_POSITIONS)),
    )
    if restart_after_group is None:
        restart_after_group = max(1, len(boundaries) // 2) if len(boundaries) > 1 else 0

    steps: list[dict[str, Any]] = []
    saved_cursor = None
    with ExitStack() as stack:
        stream = StageMixedStream(base_dir, index_path, index_sha256, stack, verify_all_members=True)
        for g_idx, stop in enumerate(boundaries, start=1):
            _, _, meta = collect_stage_group(stream, stop)
            steps.append(meta)
            if g_idx == restart_after_group:
                saved_cursor = stream.cursor()
        terminal_cursor = stream.cursor()
        lane_totals = dict(stream.totals)
        lane_targets = dict(stream.target_totals)

    # Verify exact mid-stage restart parity on the suffix
    restart_verified = True
    if saved_cursor is not None and 0 < restart_after_group < len(boundaries):
        with ExitStack() as stack2:
            stream2 = StageMixedStream(base_dir, index_path, index_sha256, stack2, cursor=saved_cursor)
            for g_idx in range(restart_after_group + 1, len(boundaries) + 1):
                stop = boundaries[g_idx - 1]
                expected = steps[g_idx - 1]
                _, _, meta2 = collect_stage_group(
                    stream2,
                    stop,
                    expected_digest=expected["sha256"],
                    expected_labels=expected["labels"],
                )
                if meta2 != expected:
                    raise ValueError(f"mid-stage replay mismatch at group {g_idx}")
            if stream2.cursor()["sha256"] != terminal_cursor["sha256"]:
                raise ValueError("terminal cursor SHA mismatch after mid-stage restart")

    return {
        "schema": STAGE_REPLAY_SCHEMA,
        "index_sha256": index_sha256,
        "stage_start_positions": int(idx["stage_start_positions"]),
        "stage_stop_positions": int(idx["stage_stop_positions"]),
        "stage_new_positions": int(idx["stage_new_positions"]),
        "update_group_positions": int(idx.get("update_group_positions", DEFAULT_GROUP_POSITIONS)),
        "update_groups": len(steps),
        "full_groups": sum(
            1 for s in steps if s["positions"] == int(idx.get("update_group_positions", DEFAULT_GROUP_POSITIONS))
        ),
        "final_group_positions": steps[-1]["positions"],
        "total_labels": sum(s["labels"] for s in steps),
        "total_packs": sum(s["packs"] for s in steps),
        "total_padding": sum(s["padding"] for s in steps),
        "lane_positions": lane_totals,
        "lane_labels": lane_targets,
        "restart_after_group": restart_after_group,
        "restart_parity_verified": restart_verified,
        "saved_cursor_sha256": saved_cursor["sha256"] if saved_cursor else None,
        "terminal_cursor_sha256": terminal_cursor["sha256"],
        "steps": steps,
    }
