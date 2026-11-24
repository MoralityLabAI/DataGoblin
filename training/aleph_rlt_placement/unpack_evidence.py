"""Verify the pinned original evidence archive and materialize only its mixed dataset."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

EVIDENCE_COMMIT = "48c99a7d9c2d37c82fe127369dfa0124b88bc8ab"
BLOB = "c8b859ba326bb34fd48ef30ded6588ed0c2d4341"
MEMBER = "assets/evidence/DataGoblin-1B-Pretrain-Evidence-v1.zip"


def unpack(archive, out):
    raw = archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as outer:
        if MEMBER in outer.namelist():
            raw = outer.read(MEMBER)
    if hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != BLOB:
        raise ValueError("archive differs from the pinned Git blob")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        manifest = json.loads(z.read("bundle_manifest.json"))
        names = z.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest["files"]) | {"bundle_manifest.json"}:
            raise ValueError("archive inventory mismatch")
        for name, expected in manifest["files"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError(f"archive digest changed: {name}")
        dataset = json.loads(z.read("mixed/dataset.json"))
        revision = json.loads(z.read("mixed/revision/manifest.json"))
        members = {"dataset.json", *dataset["files"]}
        for split in revision["splits"]:
            members.update({f"revision/{split}.tasks.jsonl", f"revision/{split}.targets.jsonl"})
        for name in sorted(members):
            target = (out / name).resolve()
            if not target.is_relative_to(out.resolve()):
                raise ValueError("unexpected member path")
            target.parent.mkdir(parents=True, exist_ok=True)
            contents = z.read("mixed/" + name)
            if target.exists() and target.read_bytes() != contents:
                raise ValueError("refusing to overwrite changed dataset")
            target.write_bytes(contents)
    return {
        "git_commit": EVIDENCE_COMMIT,
        "git_blob": BLOB,
        "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "verified_members": len(manifest["files"]),
        "dataset_members": len(members),
        "dataset_schema": dataset["schema"],
        "dataset_sha256": hashlib.sha256((out / "dataset.json").read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    print(json.dumps(unpack(args.archive, args.out), indent=2))
