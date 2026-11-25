"""Join independently fitted seed shards before the single sealed architecture evaluation."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def join(main, *shards):
    directories = (main, *shards)
    if len({p.resolve() for p in directories}) != len(directories):
        raise ValueError("each source directory must be distinct")
    for p in directories:
        if (p / "private.seed").exists() or (p / "scores.json").exists():
            raise ValueError("join fit-only directories before opening a draw")
    manifests = [p / "training.json" for p in directories]
    records = [json.loads(p.read_text(encoding="utf-8")) for p in manifests]
    base = records[0]
    cfg = base["contract"]["recipe"]
    expected = {f"{a}-{s}" for a in cfg["arms"] for s in cfg["seeds"]}
    combined, origins = {}, {}
    for directory, record in zip(directories, records, strict=True):
        spec = record["contract"]
        if {k: v for k, v in spec.items() if k != "recipe"} != {
            k: v for k, v in base["contract"].items() if k != "recipe"
        }:
            raise ValueError("source or numerical runtime mismatch")
        sub = spec["recipe"]
        if {k: v for k, v in cfg.items() if k not in ("arms", "seeds")} != {
            k: v for k, v in sub.items() if k not in ("arms", "seeds")
        }:
            raise ValueError("only independent arm/seed partitioning is allowed")
        own = {f"{a}-{s}" for a in sub["arms"] for s in sub["seeds"]}
        if not own <= expected or not set(record["fits"]) <= own:
            raise ValueError("unregistered shard cells")
        for key, row in record["fits"].items():
            if key != f"{row['arm']}-{row['seed']}" or digest(directory / f"{key}.pt") != row["checkpoint_sha256"]:
                raise ValueError("wrong fit identity or damaged archived state")
            if key in combined and combined[key]["checkpoint_sha256"] != row["checkpoint_sha256"]:
                raise ValueError("duplicate cell fits disagree")
            combined.setdefault(key, row)
            origins.setdefault(key, directory)
    if set(combined) != expected:
        raise ValueError("finish the complementary fits first")
    for seed in cfg["seeds"]:
        fits = [r for r in combined.values() if r["seed"] == seed]
        if any(len({r[k] for r in fits}) != 1 for k in ("backbone_init", "stream_sha256", "positions")):
            raise ValueError("within-seed backbone or token stream mismatch")
    previous = digest(manifests[0])
    pre = main / "training.pre-union.json"
    if not pre.exists():
        shutil.copyfile(manifests[0], pre)
    imported = [k for k, origin in origins.items() if origin != main]
    for key in imported:
        target = main / f"{key}.pt"
        if not target.exists():
            temporary = target.with_suffix(".join-part")
            with (origins[key] / target.name).open("rb") as src, temporary.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if digest(temporary) != combined[key]["checkpoint_sha256"]:
                raise ValueError("partial checkpoint copy mismatch")
            temporary.replace(target)
        if digest(target) != combined[key]["checkpoint_sha256"]:
            raise ValueError("checkpoint copy mismatch")
    base["fits"] = combined
    temp = main / "training.union.tmp"
    temp.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(manifests[0])
    receipt = {
        "main_before": previous,
        "shard_sha256": [digest(p) for p in manifests[1:]],
        "joined_sha256": digest(manifests[0]),
        "joined_keys": sorted(imported),
        "paired_stream_and_backbone_checks": True,
    }
    (main / "shard-custody.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("main", type=Path)
    cli.add_argument("shard", nargs="+", type=Path)
    a = cli.parse_args()
    join(a.main, *a.shard)
