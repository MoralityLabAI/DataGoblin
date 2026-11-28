"""Build and replay the first 10M-position corpus against the frozen direct-A recipe."""

import argparse
import hashlib
import json
import shutil
import sqlite3
import struct
import sys
import tempfile
import time
from array import array
from collections import Counter
from contextlib import ExitStack, closing
from pathlib import Path

from production import TOKENIZER_SHA256, digest, read
from readiness import PIN, audit, milestone_supply
from snapshot import INDEX_SCHEMA, RECIPE, MixedStream, dump, membership, scan

ORDER = b"aleph-direct-A-10m/root-order/137\0"


def integers(values):
    """Unambiguous little-endian signed 32-bit token/mask representation."""
    raw = array("i", values)
    if raw.itemsize != 4:
        raise ValueError("expected four-byte integers")
    if sys.byteorder == "big":
        raw.byteswap()
    return raw.tobytes()


def require_train(row, lane, mapping, denied):
    if row["split"] != "train" or row["presentation"] != 1 or mapping.get(row["lane"]) != lane:
        raise ValueError("split, view or production lane mismatch")
    if row["family"] in denied or set(row["family"].split("/")) & denied or row["root"] in denied:
        raise ValueError("excluded family or root")
    mask = row["target_mask"]
    if row["objective"] == "checked_response" and lane != "general":
        start = row["response_start"]
        if not row.get("receipt_sha256") or not 0 < start < len(mask) or not any(mask[start:]) or any(mask[:start]):
            raise ValueError("unverified or empty response")
    elif row["objective"] == "next_token" and lane == "general":
        if mask != [False] + [True] * (len(mask) - 1):
            raise ValueError("unexpected document supervision")
    else:
        raise ValueError("unexpected lane objective")


def detached_receipts(folders):
    receipts = {}
    for folder in folders:
        with (folder / "receipts.jsonl").open("rb") as f:
            for line in f:
                receipt = json.loads(line)
                root = receipt["root"]
                if receipt.get("split") != "train" or receipt.get("accepted") is False or root in receipts:
                    raise ValueError("invalid or duplicate detached receipt")
                receipts[root] = hashlib.sha256(line.rstrip(b"\r\n")).hexdigest()
    return receipts


def read_rows(path, lane, mapping, denied, roots, sequences, shards, receipts):
    manifest = read(path)
    members, _ = membership(path, "train")
    refs = []
    positions = 0
    for spec in manifest["shards"]:
        shard = path.parent / spec["path"]
        shard_id = len(shards)
        shards.append(shard)
        with shard.open("rb") as f:
            for _ in range(spec["records"]):
                offset = f.tell()
                raw = f.readline()
                row = json.loads(raw)
                require_train(row, lane, mapping, denied)
                root = row["root"]
                if lane != "general" and row["receipt_sha256"] != receipts.get(root):
                    raise ValueError("missing or different detached verification receipt")
                encoded = hashlib.sha256(integers(row["token_ids"])).digest()
                if root in roots or encoded in sequences or root not in members:
                    raise ValueError("duplicate root, duplicate token sequence or absent registry member")
                roots.add(root)
                sequences.add(encoded)
                refs.append((hashlib.sha256(ORDER + root.encode()).digest(), root, shard_id, offset))
                positions += len(row["token_ids"])
            if f.read(1):
                raise ValueError("additional unregistered rows")
    if len(refs) != len(members):
        raise ValueError("registry contains unread roots")
    # Validate the ORIGINAL registry-to-shard and detached receipt bindings before reordering.
    verified = scan(path, "train", positions)
    if verified["started_views"] != len(refs):
        raise ValueError("unexpected source row count")
    return refs, verified


def merge_lane(folder, lane, paths, mapping, denied, roots, sequences, receipts):
    folder.mkdir()
    refs, paths_to_shards, sources = [], [], []
    for path in paths:
        additions, ledger = read_rows(path, lane, mapping, denied, roots, sequences, paths_to_shards, receipts)
        refs.extend(additions)
        sources.append({"manifest": digest(path), **ledger})
    refs.sort()
    data = folder / "rows.jsonl"
    views, families = [], set()
    with data.open("wb") as target, ExitStack() as stack:
        handles = [stack.enter_context(p.open("rb")) for p in paths_to_shards]
        for _, root, shard_id, offset in refs:
            src = handles[shard_id]
            src.seek(offset)
            line = src.readline().rstrip(b"\r\n")
            row = json.loads(line)
            if row["root"] != root:
                raise ValueError("source changed while building")
            target.write(line + b"\n")
            families.add(row["family"])
            views.append((root, row["family"], hashlib.sha256(line).hexdigest(), row.get("receipt_sha256")))
    shard_sha = digest(data)
    dbpath = folder / "registry.sqlite3"
    with closing(sqlite3.connect(dbpath)) as db:
        db.executescript(INDEX_SCHEMA)
        db.executemany("INSERT INTO families VALUES (?, 'train')", [(name,) for name in sorted(families)])
        db.executemany("INSERT INTO roots VALUES (?, ?)", [(r, f) for r, f, _, _ in views])
        db.executemany(
            "INSERT INTO views VALUES (?,1,?,?,?,?)",
            [(r, shard_sha, i, h, receipt) for i, (r, _, h, receipt) in enumerate(views)],
        )
        db.commit()
    tokenizer = folder / "tokenizer.json"
    shutil.copyfile(paths[0].parent / read(paths[0])["tokenizer"]["path"], tokenizer)
    if digest(tokenizer) != TOKENIZER_SHA256:
        raise ValueError("wrong tokenizer after merge")
    manifest = {
        "schema": "aleph.packed-stream.v1",
        "vocab_size": 49152,
        "context": 2048,
        "pad_id": 0,
        "tokenizer": {"path": tokenizer.name, "sha256": TOKENIZER_SHA256},
        "registry": {"path": dbpath.name, "sha256": digest(dbpath)},
        "shards": [{"path": data.name, "sha256": shard_sha, "records": len(views), "split": "train"}],
    }
    p = folder / "manifest.json"
    dump(p, manifest)
    positions = sum(s["processed"] for s in sources)
    ledger = scan(p, "train", positions)
    return {
        "path": f"{lane}/manifest.json",
        "manifest_sha256": digest(p),
        "positions": positions,
        "train_families": sorted(families),
        "sources": sources,
        "admitted_ledger": ledger,
    }


def source_evidence(folder):
    return {p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def check_admission(folder, lane, evidence):
    if lane == "general":
        record = read(folder / "admission.json")
        if record["manifest_sha256"] != evidence["manifest.json"]:
            raise ValueError("text source audit used a different manifest")
        attribution = read(folder / "manifest.json")["attribution"]
        if (
            record["attribution_sha256"] != evidence[attribution["path"]]
            or record["attribution_sha256"] != attribution["sha256"]
        ):
            raise ValueError("text attribution differs from source audit")
        return
    admission = read(folder / "admissions.json")
    for name, stamp in admission["members"].items():
        if evidence.get(name) != stamp:
            raise ValueError("admission artifact changed: " + name)
    if lane == "custom_research":
        checked = read(folder / "audit.json")
        if (
            not checked["checks_passed"]
            or checked["admissions_sha256"] != evidence["admissions.json"]
            or checked["members"] != admission["members"]
        ):
            raise ValueError("research gym verification differs from admitted records")
    if lane in ("verified", "procedural", "failure_directed"):
        verified = read(folder / "verified-replay.json")
        if (verified["admissions_sha256"], verified["manifest_sha256"]) != (
            evidence["admissions.json"],
            evidence["manifest.json"],
        ):
            raise ValueError("verified supplement binding changed")
        if (
            verified["independently_recomputed_responses"] != admission["rows"]
            or verified["positions"] != admission["positions"]
        ):
            raise ValueError("incomplete response verification")


def freeze(prior, extra, out):
    if out.exists():
        raise FileExistsError(out)
    historical = audit(prior, PIN)
    parent = read(prior / read(PIN)["index_path"])
    recipe = read(RECIPE)
    quotas = milestone_supply(recipe, parent)
    denied = {f for fs in read(prior / "verified/admissions.json")["excluded_families"].values() for f in fs}
    for role, group in parent["private_math"]["roles"].items():
        denied.update(group["families"])
        roots, families = membership(prior / "prose" / role / "manifest.json", role)
        denied.update(roots | families)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp) / "mix"
        stage.mkdir()
        cells, evidence = {}, {}
        roots, seqs = set(), set()
        receipts = detached_receipts(
            [
                prior / "verified",
                prior / "tournament-train",
                *(extra / lane for lane in sorted(parent["lanes"]) if lane != "general"),
            ]
        )
        for lane, cell in sorted(parent["lanes"].items()):
            addon = extra / lane
            evidence[lane] = source_evidence(addon)
            check_admission(addon, lane, evidence[lane])
            sources = [prior / cell["path"], addon / "manifest.json"]
            info = merge_lane(stage / lane, lane, sources, recipe["stream"]["subtasks"], denied, roots, seqs, receipts)
            if info["sources"][0]["processed"] != cell["positions"] or digest(sources[0]) != cell["manifest_sha256"]:
                raise ValueError("parent lane differs from original freeze")
            if info["positions"] < quotas[lane]["required"]:
                raise ValueError(f"insufficient new supply for {lane}")
            cells[lane] = info
        if set(receipts) - roots:
            raise ValueError("unused detached receipt")
        index = {
            "schema": "aleph.pilot-supply.v1",
            "recipe_sha256": digest(RECIPE),
            "parent": historical["snapshot"],
            "order": ORDER.decode().rstrip("\0"),
            "lanes": cells,
            "target_by_lane": parent["target_by_lane"],
            "first_checkpoint": {k: v["required"] for k, v in quotas.items()},
            "new_source_evidence": evidence,
            "total_positions": sum(c["positions"] for c in cells.values()),
            "train_roots": len(roots),
        }
        dump(stage / "index.json", index)
        stage.rename(out)
    return digest(out / "index.json")


def audit_pack(pack, lane, sha):
    n = pack["nonpadding"]
    fields = [pack[name] for name in ("input_ids", "segment_ids", "position_ids", "labels_at_query")]
    ids, seg, pos, labels = fields
    if not 0 < n <= 2048 or any(len(v) != 2048 for v in fields):
        raise ValueError("pack geometry changed")
    if any(ids[n:]) or any(pos[n:]) or any(v != -1 for v in seg[n:]) or any(v != -100 for v in labels[n:]):
        raise ValueError("bad padding")
    for i in range(n):
        new = i == 0 or seg[i] != seg[i - 1]
        if new:
            if pos[i] != 0 or seg[i] != (0 if i == 0 else seg[i - 1] + 1):
                raise ValueError("segment boundary changed")
        elif pos[i] != pos[i - 1] + 1:
            raise ValueError("invalid position")
        end = i + 1 == n or seg[i + 1] != seg[i]
        if (end and labels[i] != -100) or (not end and labels[i] not in (-100, ids[i + 1])):
            raise ValueError("label crosses document or differs from next token")
    sha.update(lane.encode() + b"\0" + struct.pack("<I", n))
    for v in fields:
        sha.update(integers(v))
    return n, sum(v != -100 for v in labels)


class ReferenceLane:
    """Read the published JSONL independently and check every shifted mask at document and pack cuts."""

    def __init__(self, folder, lane, stack):
        manifest = read(folder / "manifest.json")
        files = [stack.enter_context((folder / s["path"]).open("rb")) for s in manifest["shards"]]
        self.rows = (json.loads(line) for f in files for line in f)
        self.lane = lane
        self.row = None
        self.offset = 0
        self.family_positions = Counter()
        self.family_labels = Counter()

    def verify(self, pack):
        filled = 0
        while filled < pack["nonpadding"]:
            if self.row is None:
                self.row = next(self.rows, None)
                self.offset = 0
                if self.row is None:
                    raise ValueError(f"independent {self.lane} input ended early")
            row = self.row
            count = min(len(row["token_ids"]) - self.offset, pack["nonpadding"] - filled)
            end = self.offset + count
            expected_ids = row["token_ids"][self.offset : end]
            expected_labels = [
                row["token_ids"][i] if row["target_mask"][i] else -100 for i in range(self.offset + 1, end)
            ] + [-100]
            sl = slice(filled, filled + count)
            if pack["input_ids"][sl] != expected_ids or pack["labels_at_query"][sl] != expected_labels:
                raise ValueError(f"independent source IDs or supervision differ in {self.lane}")
            if pack["position_ids"][sl] != list(range(count)):
                raise ValueError("independent document boundary differs")
            self.family_positions[row["family"]] += count
            self.family_labels[row["family"]] += sum(v != -100 for v in expected_labels)
            filled += count
            self.offset = end
            if end == len(row["token_ids"]):
                self.row = None


def walk_step(stream, stop, reference=None):
    sha = hashlib.sha256()
    totals = Counter()
    while (item := stream.next_pack(stop)) is not None:
        lane, pack = item
        n, labels = audit_pack(pack, lane, sha)
        if reference is not None:
            reference[lane].verify(pack)
        totals.update(positions=n, labels=labels, padding=2048 - n, packs=1)
    if not totals["labels"]:
        raise ValueError("empty supervision for optimizer update")
    return {"stop": stop, "sha256": sha.hexdigest(), **totals}


def replay(base, expected):
    index = base / "index.json"
    ix, cfg = read(index), read(RECIPE)
    tick, cpu = time.perf_counter(), time.process_time()
    limit = cfg["stream"]["checkpoints"][0]
    size = cfg["stream"]["batch_nonpadding"]
    stops = [min(start + size, limit) for start in range(0, limit, size)]
    cut = len(stops) // 2
    with ExitStack() as stack:
        stream = MixedStream(base, index, expected, stack)
        reference = {
            lane: ReferenceLane((base / cell["path"]).parent, lane, stack) for lane, cell in ix["lanes"].items()
        }
        steps = []
        for i, stop in enumerate(stops, 1):
            steps.append(walk_step(stream, stop, reference))
            if i == cut:
                saved = stream.checkpoint()
        final = stream.checkpoint()
        lanes = {
            k: {
                f: v.state[f]
                for f in ("processed", "targets", "padding", "packs", "started_views", "canonical_view_positions")
            }
            for k, v in stream.sources.items()
        }
    with ExitStack() as stack:
        resumed = MixedStream(base, index, expected, stack, saved)
        for ref in steps[cut:]:
            if walk_step(resumed, ref["stop"]) != ref:
                raise ValueError("resumed batch changed")
        if resumed.checkpoint() != final:
            raise ValueError("restored cursor changed")
    if {k: v["processed"] for k, v in lanes.items()} != ix["first_checkpoint"] or sum(
        v["processed"] for v in lanes.values()
    ) != limit:
        raise ValueError("10M mixture boundary mismatch")
    if any(entry["canonical_view_positions"] != entry["processed"] for entry in lanes.values()):
        raise ValueError("repeated presentation")
    return {
        "schema": "aleph.pilot-stream-replay.v1",
        "snapshot": expected,
        "recipe": digest(RECIPE),
        "lanes": lanes,
        "steps": steps,
        "full_positions": limit,
        "update_groups": len(stops),
        "training_updates_executed": 0,
        "independently_checked_positions": limit,
        "positions_by_family": {lane: dict(ref.family_positions) for lane, ref in reference.items()},
        "targets_by_family": {lane: dict(ref.family_labels) for lane, ref in reference.items()},
        "restored_after_update": cut,
        "restart_position": stops[cut - 1],
        "identical_resumed_positions": limit - stops[cut - 1],
        "saved_cursor": saved,
        "terminal_cursor": final,
        "total_targets": sum(v["targets"] for v in lanes.values()),
        "total_padding": sum(v["padding"] for v in lanes.values()),
        "total_packs": sum(v["packs"] for v in lanes.values()),
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - tick,
        "accelerator_credits": 0,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    build = commands.add_parser("freeze")
    for name in ("prior", "extra", "out"):
        build.add_argument("--" + name, type=Path, required=True)
    check = commands.add_parser("replay")
    check.add_argument("--base", type=Path, required=True)
    check.add_argument("--sha256", required=True)
    check.add_argument("--receipt", type=Path, required=True)
    args = cli.parse_args()
    if args.command == "freeze":
        print(freeze(args.prior, args.extra, args.out))
    else:
        if args.receipt.exists():
            raise FileExistsError("use a fresh receipt path")
        result = replay(args.base, args.sha256)
        dump(args.receipt, result)
        print(json.dumps({k: v for k, v in result.items() if k not in {"steps", "saved_cursor", "terminal_cursor"}}))


if __name__ == "__main__":
    main()
