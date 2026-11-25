"""Read-only pinned external Lean source staging; no proof or training admission."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import BinaryIO

PIN = "e1cf80b80c2fe9d22d83f1dc3673410a2ac1bf2e"
TREE = "820e408af56584901ee105f696a9f9aa2a140e3b"
REPO = "1337isnot1337/solved-conjectures"
PREFIXES = ("Loneliness", "Frankl", "Hadamard", "CutPeriod")
IMPORTS = ("CKZUpstream", "LineSizeImported", "MotzkinImported")


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def tree_entries(repo: Path, pin: str) -> list[tuple[str, str, int]]:
    entries = []
    for raw in git(repo, "ls-tree", "-r", "-z", "-l", pin, "--", "lean/").split(b"\0"):
        if not raw:
            continue
        meta, path_bytes = raw.split(b"\t", 1)
        path = path_bytes.decode("utf-8", errors="strict")
        mode, kind, oid, size = meta.decode("ascii").split()
        if not path.endswith(".lean"):
            continue
        if mode not in ("100644", "100755") or kind != "blob":
            raise ValueError(f"non-regular Lean path: {path}")
        entries.append((path, oid, int(size)))
    return sorted(entries)


def read_blob(stream: BinaryIO, oid: str, size: int) -> bytes:
    expected = f"{oid} blob {size}\n".encode()
    if stream.readline() != expected:
        raise ValueError(f"unexpected cat-file header for {oid}")
    body = stream.read(size)
    if len(body) != size or stream.read(1) != b"\n":
        raise ValueError(f"truncated blob: {oid}")
    prefix = f"blob {size}\0".encode()
    if hashlib.sha1(prefix + body, usedforsecurity=False).hexdigest() != oid:
        raise ValueError(f"Git blob mismatch: {oid}")
    return body


def triage(path: str, size: int) -> tuple[str, str]:
    parts = Path(path).parts
    name = Path(path).stem
    if any(part in IMPORTS for part in parts):
        return "imported_rights_review", "imported"
    for prefix in PREFIXES:
        if name.startswith(prefix):
            return "certificate_family_review", prefix
    if size > 100_000:
        return "large_file_review", "large_other"
    # A path is a review hint only, never an independent problem or split ID.
    return "ordinary_source_review", "unmapped"


def reference_hashes(paths: list[Path]) -> set[str]:
    hashes = set()
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                digest = row["sha256"]
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError(f"invalid reference digest in {path}")
                hashes.add(digest)
    return hashes


def stage(
    repo: Path,
    out: Path,
    references: list[Path],
    *,
    pin: str = PIN,
    expected_tree: str = TREE,
    expected_counts: tuple[int, int] | None = (5215, 262579836),
) -> dict:
    actual = git(repo, "rev-parse", f"{pin}^{{tree}}").decode().strip()
    if actual != expected_tree:
        raise ValueError("pinned source tree mismatch")
    entries = tree_entries(repo, pin)
    raw_bytes = sum(size for _, _, size in entries)
    if expected_counts is not None and (len(entries), raw_bytes) != expected_counts:
        raise ValueError("pinned primary-source inventory mismatch")
    external = reference_hashes(references)
    out.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    lanes: Counter[str] = Counter()
    lane_bytes: Counter[str] = Counter()
    duplicates: Counter[str] = Counter()
    review: list[dict] = []
    manifest = out / "candidates.jsonl"
    with subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    ) as proc, manifest.open("w", encoding="utf-8", newline="\n") as stream:
        assert proc.stdin is not None and proc.stdout is not None
        try:
            for path, oid, size in entries:
                proc.stdin.write(f"{oid}\n".encode())
                proc.stdin.flush()
                body = read_blob(proc.stdout, oid, size)
                digest = hashlib.sha256(body).hexdigest()
                lane, family = triage(path, size)
                duplicate = "reference_exact" if digest in external else (
                    "within_source_exact" if digest in seen else None
                )
                seen.add(digest)
                lanes[lane] += 1
                lane_bytes[lane] += size
                if duplicate:
                    duplicates[duplicate] += 1
                row = {
                    "repo": REPO, "revision": pin, "path": path,
                    "git_blob": oid, "sha256": digest, "bytes": size,
                    "triage": lane, "family_hint": family, "duplicate": duplicate,
                    "problem_root": None, "split": None, "disposition": "QUARANTINE",
                    "training_ready": False, "independent_replay": False,
                }
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                if duplicate is None:
                    review.append(row)
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError("git cat-file failed")
        except BaseException:
            proc.kill()
            raise
    # Deterministic review queue: reserve space for ordinary files, and limit
    # known high-volume groups. This is not sampling for a training split.
    groups: Counter[str] = Counter()
    shortlist = []
    for row in sorted(review, key=lambda r: (
        r["triage"] != "ordinary_source_review", r["sha256"], r["path"]
    )):
        group = row["family_hint"]
        if group != "unmapped" and groups[group] >= 4:
            continue
        if group == "unmapped" and groups[group] >= 48:
            continue
        groups[group] += 1
        shortlist.append(row)
        if len(shortlist) == 80:
            break
    with (out / "review_queue.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in shortlist:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "repo": REPO, "revision": pin, "tree": actual,
        "primary_paths": len(entries), "primary_bytes": raw_bytes,
        "distinct_content_sha256": len(seen), "reference_unique_hashes": len(external),
        "reference_manifests_supplied": len(references), "duplicates": dict(duplicates),
        "triage_files": dict(lanes), "triage_bytes": dict(lane_bytes),
        "review_queue_files": len(shortlist), "review_queue_groups": dict(groups),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "independent_replay": False, "training_ready": False,
        "accepted_unique_tokens": None, "accepted_loss_tokens": None,
    }
    (out / "inventory.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(stage(args.source, args.out, args.reference), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
