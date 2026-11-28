"""Stream admitted, already-tokenized JSONL into segment-causal next-token packs.

A separately frozen SQLite index owns split assignments and admitted record
digests. A cursor belongs to exactly one manifest, split and packing context.
See ARCHITECTURE.md for the producer and optimizer checkpoint contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from itertools import pairwise
from pathlib import Path

IGNORE = -100
INDEX_SCHEMA = """
CREATE TABLE families(family TEXT PRIMARY KEY, split TEXT NOT NULL);
CREATE TABLE roots(root TEXT PRIMARY KEY, family TEXT NOT NULL REFERENCES families);
CREATE TABLE views(
    root TEXT NOT NULL REFERENCES roots, presentation INTEGER NOT NULL CHECK(presentation BETWEEN 1 AND 4),
    shard_sha256 TEXT NOT NULL, ordinal INTEGER NOT NULL, record_sha256 TEXT NOT NULL,
    receipt_sha256 TEXT, PRIMARY KEY(root,presentation), UNIQUE(shard_sha256,ordinal)
);
"""


def file_digest(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        digest.update(block)
    return digest.hexdigest()


def fingerprint(data):
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def read_manifest(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("manifest digest mismatch")
    data = json.loads(raw)
    if data["schema"] != "aleph.packed-stream.v1":
        raise ValueError("unknown packed-stream schema")
    for key in ("context", "vocab_size"):
        if type(data[key]) is not int or data[key] < 2:
            raise ValueError("invalid context or vocabulary")
    if type(data["pad_id"]) is not int or not 0 <= data["pad_id"] < data["vocab_size"]:
        raise ValueError("invalid padding ID")
    return data


class PackedStream:
    def __init__(self, manifest_path, expected_sha256, split, cursor=None):
        path = Path(manifest_path).resolve()
        self.base = path.parent
        self.manifest = read_manifest(path, expected_sha256)
        self.manifest_sha = expected_sha256
        self.split = split
        shards = self.manifest["shards"]
        if len({s["sha256"] for s in shards}) != len(shards):
            raise ValueError("duplicate shard digest")
        self.shards = [s for s in shards if s["split"] == split]
        if not self.shards:
            raise ValueError("unavailable split")
        self.state = {
            "shard": 0,
            "byte": 0,
            "ordinal": 0,
            "within": 0,
            "processed": 0,
            "padding": 0,
            "targets": 0,
            "packs": 0,
            "started_views": 0,
            "canonical_view_positions": 0,
            "positions_by_lane": {},
            "targets_by_lane": {},
        }
        if cursor is not None:
            payload = cursor["payload"]
            if (
                fingerprint(payload) != cursor["sha256"]
                or payload["manifest_sha256"] != expected_sha256
                or payload["split"] != split
                or payload["state"].keys() != self.state.keys()
            ):
                raise ValueError("cursor checksum or stream identity mismatch")
            self.state = deepcopy(payload["state"])
            if not 0 <= self.state["shard"] <= len(self.shards):
                raise ValueError("cursor outside shard list")
            if any(type(v) is not int or v < 0 for v in self.state.values() if not isinstance(v, dict)):
                raise ValueError("invalid stream counters")
            if (
                self.state["targets"] > self.state["processed"]
                or sum(self.state["positions_by_lane"].values()) != self.state["processed"]
                or sum(self.state["targets_by_lane"].values()) != self.state["targets"]
            ):
                raise ValueError("inconsistent cursor totals")
        self.resources = ExitStack()
        try:
            for field in ("tokenizer", "registry"):
                with self.member(self.manifest[field]).open("rb") as member:
                    if file_digest(member) != self.manifest[field]["sha256"]:
                        raise ValueError(f"{field} digest mismatch")
            uri = self.member(self.manifest["registry"]).as_uri() + "?mode=ro&immutable=1"
            self.registry = sqlite3.connect(uri, uri=True)
            self.resources.callback(self.registry.close)
            self.handle = None
            self.open_shard = None
            self.cached = None
        except BaseException:
            self.resources.close()
            raise

    def member(self, entry):
        name = Path(entry["path"])
        out = (self.base / name).resolve()
        if name.is_absolute() or not out.is_relative_to(self.base):
            raise ValueError("artifact escapes corpus directory")
        return out

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.close()
        self.resources.close()

    def checkpoint(self):
        payload = {"manifest_sha256": self.manifest_sha, "split": self.split, "state": deepcopy(self.state)}
        return {"payload": payload, "sha256": fingerprint(payload)}

    def _row(self, state):
        index, offset = state["shard"], state["byte"]
        key = (index, offset, state["ordinal"])
        if self.cached is not None and self.cached[0] == key:
            return self.cached[1:]
        spec = self.shards[index]
        if self.open_shard != index:
            if self.handle is not None:
                self.handle.close()
            self.handle = self.member(spec).open("rb")
            if file_digest(self.handle) != spec["sha256"]:
                self.handle.close()
                self.open_shard = None
                raise ValueError("shard digest mismatch")
            self.open_shard = index
        self.handle.seek(offset)
        raw = self.handle.readline()
        if not raw:
            if state["ordinal"] != spec["records"] or state["within"] != 0:
                raise ValueError("unexpected end of shard")
            return None, offset
        if state["ordinal"] >= spec["records"]:
            raise ValueError("shard exceeds declared record count")
        row = json.loads(raw)
        bound = self.registry.execute(
            """
            SELECT v.root,r.family,f.split,v.presentation,v.record_sha256,v.receipt_sha256
            FROM views v JOIN roots r ON r.root=v.root JOIN families f ON f.family=r.family
            WHERE v.shard_sha256=? AND v.ordinal=?
        """,
            (spec["sha256"], state["ordinal"]),
        ).fetchone()
        fields = (
            row["root"],
            row["family"],
            self.split,
            row["presentation"],
            hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest(),
            row.get("receipt_sha256"),
        )
        if (
            bound != fields
            or row["split"] != self.split
            or type(row["presentation"]) is not int
            or not 1 <= row["presentation"] <= 4
        ):
            raise ValueError("record differs from custody index")
        ids, mask = row["token_ids"], row["target_mask"]
        if (
            not ids
            or len(ids) != len(mask)
            or mask[0]
            or any(type(v) is not bool for v in mask)
            or any(type(i) is not int or not 0 <= i < self.manifest["vocab_size"] for i in ids)
        ):
            raise ValueError("bad IDs or next-token target mask")
        if row["objective"] == "checked_response":
            start = row["response_start"]
            if type(start) is not int or not 1 <= start <= len(ids) or any(mask[:start]):
                raise ValueError("checked label in prompt")
            receipt = row.get("receipt_sha256")
            if any(mask) and (
                not isinstance(receipt, str) or len(receipt) != 64 or any(x not in "0123456789abcdef" for x in receipt)
            ):
                raise ValueError("supervised checked action lacks receipt")
        elif row["objective"] != "next_token":
            raise ValueError("unsupported token objective")
        if not isinstance(row["lane"], str) or not row["lane"]:
            raise ValueError("missing lane")
        self.cached = (key, row, self.handle.tell())
        return self.cached[1:]

    def next_pack(self, stop_at):
        """Emit at most context tokens, respecting an absolute nonpadding stop.

        Use the same registered stop positions on both arms. A new stop inside
        an existing pack changes its available left context even if IDs match.
        """
        if type(stop_at) is not int or stop_at < self.state["processed"]:
            raise ValueError("invalid absolute token stop")
        state = deepcopy(self.state)
        context = self.manifest["context"]
        cap = min(context, stop_at - state["processed"])
        if cap == 0:
            return None
        ids, segments, positions, labels = [], [], [], []
        source_counts, target_counts = Counter(), Counter()
        segment = 0
        while len(ids) < cap and state["shard"] < len(self.shards):
            row, after = self._row(state)
            if row is None:
                state.update(shard=state["shard"] + 1, byte=0, ordinal=0, within=0)
                continue
            start = state["within"]
            if not 0 <= start < len(row["token_ids"]):
                raise ValueError("cursor outside record")
            end = min(start + cap - len(ids), len(row["token_ids"]))
            part = row["token_ids"][start:end]
            targets = [row["token_ids"][j] if row["target_mask"][j] else IGNORE for j in range(start + 1, end)]
            targets.append(IGNORE)
            ids.extend(part)
            segments.extend([segment] * len(part))
            positions.extend(range(len(part)))
            labels.extend(targets)
            segment += 1
            source_counts[row["lane"]] += len(part)
            target_counts[row["lane"]] += sum(v != IGNORE for v in targets)
            state["started_views"] += int(start == 0)
            if row["presentation"] == 1:
                state["canonical_view_positions"] += len(part)
            if end == len(row["token_ids"]):
                state.update(within=0, byte=after, ordinal=state["ordinal"] + 1)
            else:
                state["within"] = end
        if not ids:
            self.state = state
            return None
        n = len(ids)
        pad = context - n
        ids.extend([self.manifest["pad_id"]] * pad)
        labels.extend([IGNORE] * pad)
        positions.extend([0] * pad)
        segments.extend([-1] * pad)
        state["processed"] += n
        state["padding"] += pad
        state["targets"] += sum(target_counts.values())
        state["packs"] += 1
        for name, additions in (("positions_by_lane", source_counts), ("targets_by_lane", target_counts)):
            lane_counts = state[name]
            for lane, v in additions.items():
                lane_counts[lane] = lane_counts.get(lane, 0) + v
        self.state = state
        return {
            "input_ids": ids,
            "segment_ids": segments,
            "position_ids": positions,
            "labels_at_query": labels,
            "nonpadding": n,
        }


def sdpa_mask(segment_ids, device=None):
    """Boolean allowed-pairs mask for SDPA parity. Padding queries see only themselves."""
    import torch

    segments = torch.as_tensor(segment_ids, device=device)
    if segments.ndim == 1:
        segments = segments.unsqueeze(0)
    if segments.ndim != 2:
        raise ValueError("segment IDs require [T] or [B,T]")
    k = segments.shape[-1]
    ix = torch.arange(k, device=segments.device)
    same = segments[:, :, None] == segments[:, None, :]
    allowed = same & (segments[:, :, None] >= 0) & (ix[:, None] >= ix[None, :])
    allowed |= (segments[:, :, None] < 0) & (ix[:, None] == ix[None, :])
    return allowed.unsqueeze(1)


def reference_logits(model, pack, arm="direct", r=2, s=1):
    """Exact segment isolation through Aleph's native forward, for backend parity."""
    import torch
    from torch.nn import functional as F

    device = model.embed.weight.device
    ids = torch.tensor(pack["input_ids"], dtype=torch.long, device=device)
    n = pack["nonpadding"]
    bounds = [0]
    bounds.extend(i for i in range(1, n) if pack["segment_ids"][i] != pack["segment_ids"][i - 1])
    bounds.append(n)
    pieces = [model(ids[None, a:b], arm, r, s) for a, b in pairwise(bounds)]
    return F.pad(torch.cat(pieces, dim=1), (0, 0, 0, len(ids) - n))


def token_loss_sum(logits, labels_at_query):
    """Return the differentiable numerator and valid count, including an all-masked pack."""
    import torch
    from torch.nn import functional as F

    labels = torch.as_tensor(labels_at_query, device=logits.device).reshape(-1)
    numerator = F.cross_entropy(logits.flatten(end_dim=-2), labels, ignore_index=IGNORE, reduction="sum")
    return numerator, (labels != IGNORE).sum()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--sha256", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--stop-at", required=True, type=int)
    p.add_argument("--cursor", type=Path)
    args = p.parse_args()
    previous = json.loads(args.cursor.read_text()) if args.cursor else None
    with PackedStream(args.manifest, args.sha256, args.split, previous) as stream:
        while stream.next_pack(args.stop_at) is not None:
            pass
        if stream.state["processed"] != args.stop_at:
            p.error("insufficient admitted positions for requested stop")
        print(json.dumps(stream.checkpoint(), indent=2))


if __name__ == "__main__":
    main()
