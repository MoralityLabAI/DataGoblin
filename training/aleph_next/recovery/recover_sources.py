"""Recover immutable training-source candidates from the two public research repositories.

This is an input recovery packet. It deliberately does not assert production
admission or reproduce the missing 10M corpus. Held data never enter the output.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

DG = "MoralityLabAI/DataGoblin"
LT = "MoralityLabAI/LoopedTransformers"
DG_EVIDENCE = "48c99a7d9c2d37c82fe127369dfa0124b88bc8ab"
LT_PIN = "b14a51015996f0dee81ef033e2e198d9f84951b2"
EVIDENCE_PATH = "assets/evidence/DataGoblin-1B-Pretrain-Evidence-v1.zip"
EVIDENCE_BLOB = "c8b859ba326bb34fd48ef30ded6588ed0c2d4341"
MIXED_SHA256 = "a0d2dfbe5a93e7630c60e1c182dccabc03d60efea376eb65503b62e812c66383"
NATIVE_TRAIN = (
    "text-train.jsonl",
    "lean-train.public.jsonl",
    "lean-train.receipts.jsonl",
    "revision/train.tasks.jsonl",
    "revision/train.targets.jsonl",
)
SCHEDULE = "experiments/loop_schedule_architecture_discovery_v1/datasets"
RAM = "experiments/lean_ram11f/curriculum"
LRM = "mechinterp/revision_memory_gym/results"
LBF = "research/lite_braille_fresco/results/lbf18/train_teacher_certificates"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def fetch(repo, commit, name, blob, size, out):
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{name}"
    with urllib.request.urlopen(url, timeout=90) as response:
        data = response.read(size + 1)
    if len(data) != size or git_blob(data) != blob:
        raise ValueError(f"Git blob or size changed: {repo}/{name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return {
        "repo": repo,
        "commit": commit,
        "path": name,
        "git_blob": blob,
        "sha256": sha256(out),
        "bytes": size,
    }


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def count_rows(path):
    return sum(1 for _ in rows(path))


def verify_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path.name}")


def check_role(path):
    """Check an explicit row-level split if present; file role is pinned separately."""
    for row in rows(path):
        for field in ("split", "partition"):
            if field in row and isinstance(row[field], str) and row[field].lower() not in ("train", "training"):
                raise ValueError(f"unexpected {field}={row[field]} in {path.name}")


def recover_native(work, out):
    # Use the repository's original hash/manifest verifier and native checked loader.
    placement = Path(__file__).resolve().parents[2] / "aleph_rlt_placement"
    sys.path.insert(0, str(placement))
    try:
        from corpus import load_corpus
        from unpack_evidence import unpack

        archive = work / "historical-evidence.zip"
        source = fetch(DG, DG_EVIDENCE, EVIDENCE_PATH, EVIDENCE_BLOB, 5622851, archive)
        mixed = work / "mixed"
        receipt = unpack(archive, mixed)
        verify_hash(mixed / "dataset.json", MIXED_SHA256)
        train, dev, audit = load_corpus(mixed)
        dest = out / "datagoblin_mixed_train"
        members = {}
        for name in NATIVE_TRAIN:
            original = mixed / name
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            members[name] = {"sha256": sha256(target), "rows": count_rows(target)}
        # The original mixed manifest, and its held rows, stay in temporary work.
        # The native loader's counts include deterministic repair/counterexample
        # views that were generated in memory and are not copied to this packet.
        return {
            "source": source,
            "historical_archive_audit": receipt,
            "mixed_view_digest": audit["view_digest"],
            "native_train_views": {k: len(train[k]) for k in ("text", "lean", "rule")},
            "native_dev_views_checked_but_excluded": {k: len(dev[k]) for k in ("text", "lean", "rule")},
            "members": members,
        }
    finally:
        sys.path.remove(str(placement))


def recover_looped(out):
    sources = []
    root = out / "looped_train_candidates"

    def put(path, blob, size):
        dest = root / path
        sources.append(fetch(LT, LT_PIN, path, blob, size, dest))
        return dest

    sched_manifest = put(
        SCHEDULE + "/manifest.json", "c602d262659b2a36bda9ca67ce9158afdcceafc7", 2763
    )
    sched = put(SCHEDULE + "/train.jsonl", "0dd0cf1b122ddf983d3021d5219d424e963e98bb", 4074626)
    spec = read(sched_manifest)
    verify_hash(sched, spec["files"]["train"]["sha256"])
    expected = sum(c["train"] for c in spec["counts"].values())
    if count_rows(sched) != expected or expected != 7507:
        raise ValueError("loop-schedule training count changed")
    check_role(sched)

    ram_manifest = put(RAM + "/manifest.json", "e83f2b9c50d50ecea15513e9aeaad15a508043f6", 1325)
    scripts = put(RAM + "/certified_complete_scripts.jsonl", "3e63e62190b0c529a8ccadbf476e928b7f47586d", 59647)
    prefixes = put(
        RAM + "/certified_positive_prefix_action_labels.jsonl",
        "bb762212c4e440a19167342e3372889120138462",
        99649,
    )
    ram_spec = read(ram_manifest)
    verify_hash(scripts, ram_spec["script_sha256"])
    verify_hash(prefixes, ram_spec["positive_prefix_sha256"])
    if count_rows(scripts) != 76 or count_rows(prefixes) != 230:
        raise ValueError("RAM11F training count changed")

    lrm_manifest = put(
        LRM + "/lrm_rev02_v1_publication_manifest.json", "298afb09f9c5b36430d9cad3c995ffa08a999245", 7979
    )
    lrm_spec = {item["path"]: item for item in read(lrm_manifest)["files"]}
    lrm_counts = []
    for i in range(4):
        name = f"{LRM}/lrm_rev02_v1/root_{i}/training_episodes.jsonl.gz"
        pin = lrm_spec[name]
        shard = put(name, pin["git_blob_sha"], pin["size"])
        verify_hash(shard, pin["sha256"])
        n = count_rows(shard)
        if n != 2048:
            raise ValueError("LRM train episode count changed")
        lrm_counts.append(n)

    lbf_manifest = put(LBF + "/manifest.json", "c937d482579779b8d587042eaedd4303bd002328", 358)
    lbf = read(lbf_manifest)
    for sub, count, ms, mb, ps, pb in (
        ("references", 64, "d6bb40a6cd08e55f99573a5de704b503ce27c248", 2831,
         "fb21bd75dc16f1e2b9a05f28c9311fce0f3b9165", 224082),
        ("counterfactuals", 24, "4a4b815f82d81c684f9be06d3ecf955cd76ba0f0", 829,
         "88d7a37ae2a5b7f37047c83f081c1a7d1ae58ecf", 58659),
    ):
        manifest = put(f"{LBF}/{sub}/manifest.json", ms, mb)
        proof = put(f"{LBF}/{sub}/Proofs.lean", ps, pb)
        m = read(manifest)
        verify_hash(proof, m["source_sha256"])
        if (
            not m["actual_lean_compiler"] or m["compiled_axiom_free"] != count or len(m["declarations"]) != count
        ):
            raise ValueError("unexpected archived LBF certificate count")
    if (lbf["train_roots"], lbf["reference_certificates"], lbf["counterfactual_certificates"]) != (64, 64, 24):
        raise ValueError("LBF train teacher manifest changed")

    return {
        "sources": sources,
        "loop_schedule_training_records": expected,
        "ram11f_scripts": 76,
        "ram11f_prefix_labels": 230,
        "ram11f_original_theorem_roots": ram_spec["unique_original_theorems"],
        "lrm_training_episodes": sum(lrm_counts),
        "lbf_reference_certificates": 64,
        "lbf_counterfactual_certificates": 24,
        "lbf_train_roots": 64,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    out = args.out.resolve()
    if out.exists():
        p.error("use a fresh output directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as folder:
        work = Path(folder)
        stage = work / "packet"
        stage.mkdir()
        native = recover_native(work, stage)
        looped = recover_looped(stage)
        result = {
            "schema": "datagoblin.aleph.recovered-source-candidates.v1",
            "production_admitted": False,
            "original_10m_recovered": False,
            "accelerator_credits": 0,
            "native": native,
            "looped": looped,
            "next": "Recheck rights, lineage and independent outcomes; tokenize and freeze a new train snapshot.",
            "members": {
                p.relative_to(stage).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
                for p in sorted(stage.rglob("*"))
                if p.is_file()
            },
        }
        (stage / "RECOVERY.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        stage.rename(out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
