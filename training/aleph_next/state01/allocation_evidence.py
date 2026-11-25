"""Archive public equations and measured costs, and replay every primary allocation decision."""

import argparse
import hashlib
import io
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from allocation_replay import DEFAULT, MEDIAN
from allocation_routes import route
from allocation_tasks import Leaf, equal, pair_key


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def text_json(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()


def pack(root, output):
    payload = {}
    for name in ("train.json", "held.json", "FROZEN.json", "POLICIES.json", "train-only.jsonl"):
        payload[name] = (root / name).read_bytes()
    fr = json.loads(payload["FROZEN.json"])
    for m in fr["fits"]:
        payload[m["checkpoint"]] = (root / m["checkpoint"]).read_bytes()
    records = {}
    for path in sorted((root / "singleton").glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        sha = r["source"]
        source = path.with_suffix(".lean").read_bytes()
        raw = path.with_suffix(".log").read_bytes()
        if checksum(source) != sha or checksum(raw) != r["log"]:
            raise ValueError("singleton contents changed")
        payload[f"lean/{sha}.lean"] = source
        log = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
        # Keep line, column and type diagnostics; remove the machine's absolute prefix.
        scrubbed = re.sub(r"(?m)^.*?([0-9a-f]{64}\.lean:)", r"\1", log)
        payload[f"diagnostics/{sha}.txt"] = scrubbed.encode()
        records[sha] = r
    payload["singleton.json"] = text_json(records)
    payload["manifest.json"] = text_json({k: checksum(v) for k, v in sorted(payload.items())})
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    with ZipFile(tmp, "w", compression=ZIP_DEFLATED, compresslevel=9) as z:
        for k, v in sorted(payload.items()):
            meta = ZipInfo(k, date_time=(1980, 1, 1, 0, 0, 0))
            z.writestr(meta, v, compress_type=ZIP_DEFLATED, compresslevel=9)
    tmp.replace(output)
    return verify(output)


def verify(path):
    with ZipFile(io.BytesIO(path.read_bytes())) as z:

        def obj(k):
            return json.loads(z.read(k))

        manifest = obj("manifest.json")
        if set(z.namelist()) != set(manifest) | {"manifest.json"} or len(z.namelist()) != len(set(z.namelist())):
            raise ValueError("unexpected or duplicated archive member")
        for k, sha in manifest.items():
            if checksum(z.read(k)) != sha:
                raise ValueError("archive member hash mismatch")
        held, fr, p = obj("held.json"), obj("FROZEN.json"), obj("POLICIES.json")
        if (
            fr["train_sha256"] != manifest["train.json"]
            or held["freeze_sha256"] != manifest["FROZEN.json"]
            or p["held_sha256"] != manifest["held.json"]
            or p["fit_sha256"] != manifest["FROZEN.json"]
        ):
            raise ValueError("broken freeze chain")
        if any(manifest[i["checkpoint"]] != i["sha256"] for i in fr["fits"]):
            raise ValueError("fit receipt changed")
        train = obj("train.json")
        singletons = obj("singleton.json")

        def restore(v):
            return tuple(restore(x) for x in v) if isinstance(v, list) else v

        for dataset in (train, held):
            for row in dataset["rows"]:
                leaves = tuple(Leaf(domain, restore(x), restore(y)) for domain, x, y in row["public"])
                if row["key"] != pair_key(leaves) or row["gate"] != equal(leaves):
                    raise ValueError("public equation or independent denotation changed")
                checks = [r["kernel"] for r in [*row["source"], *row["actions"].values()] if r["kernel"]]
                if row["stale"]:
                    checks.extend(row["stale"].values())
                if any(singletons.get(k["source"]) != k for k in checks):
                    raise ValueError("an action references an unbound kernel measurement")
        if {r["key"] for r in train["rows"]} & {r["key"] for r in held["rows"]}:
            raise ValueError("overlapping theorem")
        decisions = 0
        for policy, choices in p["choices"].items():
            for idx, (r, o, inf) in enumerate(
                zip(held["rows"], choices, p["inference_seconds_by_row"][policy], strict=True)
            ):
                needs = policy == "train_symbolic" or policy.startswith(("ldt_", "mlp_"))
                if route(r, o, 4 * MEDIAN, inf, needs and r["gate"]) != p["details"][policy][idx]:
                    raise ValueError("primary policy replay divergence")
                decisions += 1
        if any(checksum(z.read(f"lean/{k}.lean")) != k for k in singletons):
            raise ValueError("unbound Lean source")
        return dict(
            archive_sha256=checksum(path.read_bytes()),
            members=len(manifest),
            primary_decisions=decisions,
            singleton_sources=len(singletons),
            train_lineages=len({r["lineage"] for r in train["rows"]}),
            held_lineages=len({r["lineage"] for r in held["rows"]}),
            promotion=p["passes"],
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("pack", "verify"))
    p.add_argument("--root", type=Path, default=DEFAULT)
    p.add_argument("--evidence", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(pack(a.root, a.evidence) if a.mode == "pack" else verify(a.evidence), sort_keys=True))


if __name__ == "__main__":
    main()
