"""Freeze actual admitted data and replay the fixed five-lane mixture without a training backend."""

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack, closing
from pathlib import Path

from production import HERE, TOKENIZER_SHA256, digest, read
from reserve import verify as verify_reserve

sys.path.insert(0, str(HERE.parents[1] / "aleph_rlt_placement"))
from packing import INDEX_SCHEMA, PackedStream, fingerprint

RECIPE = HERE / "RECIPE.json"
ROLES = ("train", "confirmation", "adjudication", "final")


def dump(path, data):
    path.write_bytes(json.dumps(data, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")


def membership(path, split):
    manifest = read(path)
    for spec in (manifest["registry"], manifest["tokenizer"], *manifest["shards"]):
        member = (path.parent / spec["path"]).resolve()
        if not member.is_relative_to(path.parent.resolve()) or digest(member) != spec["sha256"]:
            raise ValueError("changed or escaped data artifact")
    if (manifest["tokenizer"]["sha256"], manifest["context"], manifest["vocab_size"], manifest["pad_id"]) != (
        TOKENIZER_SHA256,
        2048,
        49152,
        0,
    ):
        raise ValueError("unexpected packing configuration")
    if {s["split"] for s in manifest["shards"]} != {split}:
        raise ValueError("held and training payloads must live in separate stores")
    uri = (path.parent / manifest["registry"]["path"]).resolve().as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        members = db.execute(
            "SELECT roots.root,families.family,families.split FROM roots JOIN families ON roots.family=families.family"
        ).fetchall()
    if any(s != split for _, _, s in members):
        raise ValueError("role mismatch in registry")
    return {r for r, _, _ in members}, {f for _, f, _ in members}


def scan(path, role, demand):
    sha = digest(path)
    with PackedStream(path, sha, role) as stream:
        while stream.next_pack(demand + 1) is not None:
            pass
        info = stream.state
        if info["processed"] != demand:
            raise ValueError("emitted positions differ from source ledger")
        return {
            k: info[k]
            for k in (
                "processed",
                "targets",
                "padding",
                "packs",
                "positions_by_lane",
                "targets_by_lane",
                "started_views",
            )
        }


def checked_lanes(sources, stage, cfg):
    by_lane = defaultdict(list)
    ledgers = {}
    for verified in sources:
        path = verified / "manifest.json"
        membership(path, "train")
        manifest = read(path)
        if len(manifest["shards"]) != 1:
            raise ValueError("expected one verified admissions shard")
        source = path.parent / manifest["shards"][0]["path"]
        positions = Counter()
        with source.open("rb") as f:
            for raw in f:
                row = json.loads(raw)
                if (
                    row["split"] != "train"
                    or row["presentation"] != 1
                    or row["objective"] != "checked_response"
                    or not row["receipt_sha256"]
                ):
                    raise ValueError("unadmitted checked view")
                lane = cfg["stream"]["subtasks"][row["lane"]]
                if lane == "general":
                    raise ValueError("checked row mislabeled as text")
                positions[row["lane"]] += len(row["token_ids"])
                by_lane[lane].append((row, raw))
        # Pass through the original admission registry before building lane-specific cursors.
        original = scan(path, "train", sum(positions.values()))
        if original["positions_by_lane"] != dict(positions):
            raise ValueError("source ledger mismatch")
        ledgers[verified.name] = original
    out = {}
    for lane, rows in sorted(by_lane.items()):
        folder = stage / lane
        folder.mkdir()
        data = folder / "checked.jsonl"
        with data.open("wb") as f:
            for _, line in rows:
                f.write(line)
        dbpath = folder / "index.sqlite3"
        sha = digest(data)
        db = sqlite3.connect(dbpath)
        try:
            db.executescript(INDEX_SCHEMA)
            db.executemany(
                "INSERT INTO families VALUES (?,'train')", [(v,) for v in sorted({r["family"] for r, _ in rows})]
            )
            db.executemany("INSERT INTO roots VALUES (?,?)", [(r["root"], r["family"]) for r, _ in rows])
            db.executemany(
                "INSERT INTO views VALUES (?,1,?,?,?,?)",
                [
                    (r["root"], sha, i, hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest(), r["receipt_sha256"])
                    for i, (r, raw) in enumerate(rows)
                ],
            )
            db.commit()
        finally:
            db.close()
        shutil.copyfile(verified / manifest["tokenizer"]["path"], folder / "tokenizer.json")
        result = {
            **{k: manifest[k] for k in ("schema", "vocab_size", "context", "pad_id")},
            "tokenizer": {"path": "tokenizer.json", "sha256": TOKENIZER_SHA256},
            "registry": {"path": dbpath.name, "sha256": digest(dbpath)},
            "shards": [{"path": data.name, "sha256": sha, "records": len(rows), "split": "train"}],
        }
        dump(folder / "manifest.json", result)
        out[lane] = sum(len(r["token_ids"]) for r, _ in rows)
    return out, ledgers


def next_allocation(totals, goals, stop):
    done = sum(totals.values())
    budget = sum(goals.values())
    if type(stop) is not int or not done <= stop <= budget:
        raise ValueError("invalid absolute global stop")
    if done == stop:
        return None
    next_pos = min(done + 2048, stop)
    remaining = (k for k, v in totals.items() if v < goals[k])
    lane = min(remaining, key=lambda k: (totals[k] * budget - goals[k] * next_pos, k))
    n = min(next_pos - done, goals[lane] - totals[lane])
    return lane, n


class MixedStream:
    """Select a lane before each pack and retain every constituent cursor."""

    def __init__(self, base, manifest, expected, stack, cursor=None):
        if digest(manifest) != expected:
            raise ValueError("wrong production snapshot")
        self.sha = expected
        self.sources = {}
        self.totals = {}
        self.index = read(manifest)
        if self.index["recipe_sha256"] != digest(RECIPE):
            raise ValueError("recipe differs from snapshot")
        self.goals = self.index["target_by_lane"]
        if cursor:
            if cursor["sha256"] != fingerprint(cursor["payload"]) or cursor["payload"]["snapshot"] != expected:
                raise ValueError("invalid mixture cursor")
            if set(cursor["payload"]["lanes"]) != set(self.goals):
                raise ValueError("incomplete mixture checkpoint")
        for lane, info in self.index["lanes"].items():
            path = (base / info["path"]).resolve()
            if not path.is_relative_to(base.resolve()):
                raise ValueError("manifest escapes snapshot root")
            other = cursor["payload"]["lanes"][lane] if cursor else None
            self.sources[lane] = stack.enter_context(PackedStream(path, info["manifest_sha256"], "train", other))
            self.totals[lane] = self.sources[lane].state["processed"]
        if set(self.goals) != set(self.sources) or self.goals != read(RECIPE)["stream"]["lane_positions"]:
            raise ValueError("incomplete production lane set")

    def checkpoint(self):
        p = {"snapshot": self.sha, "lanes": {k: s.checkpoint() for k, s in self.sources.items()}}
        return {"payload": p, "sha256": fingerprint(p)}

    def next_pack(self, stop):
        allocation = next_allocation(self.totals, self.goals, stop)
        if allocation is None:
            return None
        lane, n = allocation
        source = self.sources[lane]
        if self.totals[lane] + n > self.index["lanes"][lane]["positions"]:
            raise ValueError(f"insufficient admitted positions in {lane}, no recycling")
        p = source.next_pack(self.totals[lane] + n)
        if not p or p["nonpadding"] != n:
            raise ValueError("supply declaration exceeds corpus")
        self.totals[lane] += n
        return lane, p


def freeze(base, supplement=False):
    cfg = read(RECIPE)
    prose = read(base / "prose/audit.json")
    reserve = verify_reserve(base / "private-evaluations")
    seen_roots = set()
    seen_families = set()
    roles = {}
    for role in ROLES:
        path = base / "prose" / role / "manifest.json"
        info = prose["distributions"][role]
        if (
            digest(path) != info["manifest_sha256"]
            or digest(path.parent / "attribution.jsonl") != info["attribution_sha256"]
        ):
            raise ValueError("attribution or text manifest differs")
        roots, families = membership(path, role)
        if roots & seen_roots or families & seen_families:
            raise ValueError("text crossed custody roles")
        roles[role] = {
            "manifest_sha256": digest(path),
            "records": len(roots),
            "lineage_groups": len(families),
            **scan(path, role, info["tokens"]),
        }
        seen_roots.update(roots)
        seen_families.update(families)
    verified = base / "verified"
    sources = [verified]
    if supplement:
        sources.append(base / "tournament-train")
    for source in sources:
        roots, families = membership(source / "manifest.json", "train")
        if seen_roots & roots or seen_families & families:
            raise ValueError("verified source identity collision")
        seen_roots.update(roots)
        seen_families.update(families)
    out = base / ("snapshot-final" if supplement else "snapshot")
    if out.exists():
        raise FileExistsError("existing snapshot; use check")
    with tempfile.TemporaryDirectory(dir=base) as tmp:
        stage = Path(tmp) / "snapshot"
        stage.mkdir()
        positions, from_source = checked_lanes(sources, stage, cfg)
        positions["general"] = roles["train"]["processed"]
        goals = cfg["stream"]["lane_positions"]
        if set(positions) != set(goals) or sum(goals.values()) != cfg["stream"]["target"]:
            raise ValueError("incomplete lane plan")
        cells = {}
        for lane, n in sorted(positions.items()):
            rel = f"{out.name}/{lane}/manifest.json" if lane != "general" else "prose/train/manifest.json"
            p = stage / lane / "manifest.json" if lane != "general" else base / rel
            cells[lane] = {
                "path": rel,
                "manifest_sha256": digest(p),
                "positions": n,
                "deficit": max(0, goals[lane] - n),
            }
        index = {
            "schema": "aleph.actual-supply.v1",
            "recipe_sha256": digest(RECIPE),
            "lanes": cells,
            "target_by_lane": goals,
            "total_train_positions": sum(positions.values()),
            "total_deficit": sum(c["deficit"] for c in cells.values()),
            "complete_10b": all(n["deficit"] == 0 for n in cells.values()),
            "verification_source_manifest": digest(verified / "manifest.json"),
            "verified_pack_ledger": from_source[verified.name],
            "text_roles": roles,
            "private_math": reserve,
            "paid_authorization": False,
        }
        if supplement:
            index["supplement_manifest"] = digest(sources[1] / "manifest.json")
            index["supplement_pack_ledger"] = from_source[sources[1].name]
        dump(stage / "index.json", index)
        stage.rename(out)
    return index


def check(base, expected, supplement=False):
    p = base / ("snapshot-final" if supplement else "snapshot") / "index.json"
    if digest(p) != expected:
        raise ValueError("index digest differs from expected")
    ix = read(p)
    # No model, no assessment answers. Recheck immutable custody and the realized first four packs.
    if verify_reserve(base / "private-evaluations") != ix["private_math"]:
        raise ValueError("private assessment commitment changed")
    for role, e in ix["text_roles"].items():
        path = base / "prose" / role / "manifest.json"
        if digest(path) != e["manifest_sha256"]:
            raise ValueError("text manifest changed")
        membership(path, role)
        member = read(path)["attribution"]
        if digest(path.parent / member["path"]) != member["sha256"]:
            raise ValueError("text attribution changed")
    if digest(base / "verified/manifest.json") != ix["verification_source_manifest"]:
        raise ValueError("original verifier manifest changed")
    membership(base / "verified/manifest.json", "train")
    if supplement:
        extra = base / "tournament-train/manifest.json"
        if digest(extra) != ix["supplement_manifest"]:
            raise ValueError("train tournament manifest changed")
        membership(extra, "train")
    with ExitStack() as stack:
        mix = MixedStream(base, p, expected, stack)
        for info in ix["lanes"].values():
            membership(base / info["path"], "train")
        for _ in range(2):
            mix.next_pack(8192)
        c = mix.checkpoint()
        bs = [mix.next_pack(8192), mix.next_pack(8192)]
        restored = MixedStream(base, p, expected, stack, c)
        if bs != [restored.next_pack(8192), restored.next_pack(8192)] or mix.checkpoint() != restored.checkpoint():
            raise ValueError("mixture restart changed the emitted packs")
    return dict(
        snapshot=expected,
        mixture_restart_exact=True,
        test_positions=8192,
        ready_for_10b=ix["complete_10b"],
        paid_authorization=False,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("freeze", "check"))
    p.add_argument("base", type=Path)
    p.add_argument("--sha256")
    p.add_argument("--supplement", action="store_true")
    p.add_argument("--require-supply", action="store_true")
    args = p.parse_args()
    if args.mode == "freeze":
        print(json.dumps(freeze(args.base, args.supplement), indent=2))
    else:
        if not args.sha256:
            p.error("check needs the frozen index SHA-256")
        result = check(args.base, args.sha256, args.supplement)
        print(json.dumps(result))
        if args.require_supply and not result["ready_for_10b"]:
            raise SystemExit(2)
