"""Verify the published replacement archive against its independently pinned release receipt."""

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

VERSION = "aleph-direct-A-10m-replacement-v2"
RECIPE_SHA = "58fd8c2805f04ed918b4547d5a177624f839648c6be98dc749426ba39cd32e87"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def load_release(path, expected):
    need(digest(path) == expected, "external release receipt hash differs")
    r = read(path)
    need(r["schema"] == VERSION and r["recipe_sha256"] == RECIPE_SHA, "wrong release identity")
    need(r["full_positions"] == 10_000_000 and r["update_groups"] == 39, "wrong update budget")
    need(r["fresh_replay_pass"] is True and r["train_only"] is True, "release has not passed replay")
    need(r["production_launch_authorized"] is False, "production is outside this pilot")
    return r


def verify_extracted(base, release, *, full=True):
    need(digest(base / "index.json") == release["index_sha256"], "index digest differs")
    need(digest(base / "REPLAY.json") == release["replay_sha256"], "replay digest differs")
    receipt = read(base / "REPLAY.json")
    need(receipt["recipe"] == RECIPE_SHA and receipt["snapshot"] == release["index_sha256"],
         "recipe or snapshot identity differs")
    steps = receipt["steps"]
    need(len(steps) == 39 and sum(s["positions"] for s in steps) == 10_000_000, "step ledger differs")
    need(sum(s["labels"] for s in steps) == release["supervised_labels"], "supervision ledger differs")
    need([s["stop"] for s in steps] == [min(262144*i, 10_000_000) for i in range(1, 40)],
         "update boundaries differ")
    need(all(s["labels"] > 0 for s in steps), "empty update")
    if full:
        members = read(base / "MEMBERS.json")
        paths = {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()}
        need(paths == set(members) | {"MEMBERS.json"}, "unexpected member inventory")
        for name, spec in members.items():
            p = base / name
            need(p.resolve().is_relative_to(base.resolve()) and not p.is_symlink(), "unsafe member")
            need(p.stat().st_size == spec["bytes"] and digest(p) == spec["sha256"], "member bytes differ")
        from build10m import verify_release

        again = verify_release(base, release["index_sha256"])
        for key in ("steps", "saved_cursor", "terminal_cursor", "total_targets", "full_positions", "lanes"):
            need(again[key] == receipt[key], "independent replay differs: " + key)
    return receipt


def extract(archive, out, release):
    need(archive.stat().st_size == release["archive_bytes"], "archive size differs")
    need(digest(archive) == release["archive_sha256"], "archive digest differs")
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "mix"
        stage.mkdir()
        with zipfile.ZipFile(archive) as z:
            names = set()
            total = 0
            for item in z.infolist():
                path = PurePosixPath(item.filename)
                need(not path.is_absolute() and ".." not in path.parts and "\\" not in item.filename,
                     "unsafe archive path")
                need(not item.is_dir() and item.filename not in names, "duplicate or directory member")
                need((item.external_attr >> 16) & 0o170000 != 0o120000, "symlink member")
                need(item.file_size <= 512_000_000, "member too large")
                total += item.file_size
                need(total < 2_000_000_000, "archive too large")
                names.add(item.filename)
            z.extractall(stage)
        verify_extracted(stage, release)
        stage.rename(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    release = load_release(args.release, args.release_sha256)
    extract(args.archive, args.out.resolve(), release)
    print(json.dumps({
        "verified": True, "archive_sha256": release["archive_sha256"],
        "index_sha256": release["index_sha256"], "replay_sha256": release["replay_sha256"],
        "positions": release["full_positions"], "supervised_labels": release["supervised_labels"],
        "update_groups": release["update_groups"], "base": str(args.out.resolve()),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
