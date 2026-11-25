"""Build disjoint, attributed math-text snapshots from the pinned LibreTexts shard."""

import argparse
import gzip
import hashlib
import heapq
import json
import re
import shutil
import sqlite3
import sys
import tempfile
import unicodedata
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

from production import TOKENIZER_SHA256, digest, load_tokenizer

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "aleph_rlt_placement")]
from confirm_text import folded, native_bytes, panel_data  # noqa: E402
from packing import INDEX_SCHEMA  # noqa: E402

SOURCE = "71851a2b5acfccf80659729a0eccd07ab6ed737f0edc659a662aa9c1cd2194db"
REV = "204d01e40af72b3b6d3f926e1f6148ee9ed3970a"
REPO = "common-pile/libretexts_filtered"
PANEL = "82761739ea4f078bd1631dba2851be6551109692fbfe5166295429094fe45c51"
ROLES = ("final", "adjudication", "confirmation", "train")
LICENSE = "Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/"


def raw(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(buf):
    return hashlib.sha256(buf).hexdigest()


def words(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())


def family_keys(row):
    m = row["metadata"]
    author = " ".join(words(m["author"]))
    bits = unquote(urlsplit(row["id"]).path).strip("/").split("/")
    title = " ".join(words(bits[2])) if len(bits) >= 3 else ""
    return ("by:" + author, "book:" + m["book_url"], "title:" + title)


def eligibility(row):
    m = row.get("metadata") or {}
    if urlsplit(row.get("id", "")).hostname != "math.libretexts.org":
        return "outside_math"
    if m.get("license") != LICENSE:
        return "license_outside_cc_by_4"
    if (
        m.get("url") != row["id"]
        or not str(m.get("book_url", "")).startswith("https://commons.libretexts.org/book/")
        or any(not isinstance(m.get(k), str) or not m[k].strip() for k in ("author", "title", "provenance"))
        or not family_keys(row)[2][6:]
    ):
        return "missing_provenance"
    if m["author"].strip().lower() in {"unknown", "anonymous", "various", "libretexts", "multiple authors"}:
        return "unspecified_author"
    if re.search(
        r"all rights reserved|used (by|with) permission|non.?commercial|no.?derivatives|except where otherwise noted",
        row["text"],
        re.I,
    ):
        return "possible_third_party_terms"
    if not 512 <= len(row["text"].encode()) <= 65536:
        return "length"
    return None


def assign_groups(rows):
    parents = {}

    def root(x):
        parents.setdefault(x, x)
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    for row in rows:
        a, b, t = family_keys(row)
        for other in (b, t):
            left, right = sorted((root(a), root(other)))
            parents[right] = left
    groups = {root(family_keys(r)[0]) for r in rows}
    ordered = sorted(groups, key=lambda x: sha(("aleph-libretexts-roles-v1/" + x).encode()))
    if len(groups) < 4:
        raise ValueError("need four independent groups for source splits")
    # Reserve one whole author/book component per evaluation per twenty groups.
    role = {g: ROLES[min(i % 20, 3)] for i, g in enumerate(ordered)}
    return {r["id"]: (root(family_keys(r)[0]), role[root(family_keys(r)[0])]) for r in rows}


def duplicate_signature(text):
    tokens = [re.sub(r"\d+", "#", w) for w in words(text)]
    encoded = [t.encode() for t in tokens]
    keys = {zlib.crc32(b" ".join(encoded[i : i + 7])) for i in range(len(encoded) - 6)}
    return sha(b" ".join(encoded)), frozenset(heapq.nsmallest(32, keys))


def historical(pack, panel):
    _, docs = panel_data(panel, PANEL)
    seen = [bytes.fromhex(r["hex"]) for r in docs] + native_bytes(pack)["text-held.jsonl"]
    # Complete 33-byte window and its normalized form. Needles are evaluator-owned.
    # Index by first 8 bytes; scanning is linear in the document length.
    prior = defaultdict(list)
    for b in seen:
        for value in (b, folded(b)):
            if len(value) >= 12:
                prior[value[:8]].append(value)
    return prior


def historical_overlap(body, prior):
    for value in (body, folded(body)):
        for i in range(len(value) - 7):
            endings = prior.get(value[i : i + 8])
            if endings and any(value.startswith(needle, i) for needle in endings):
                return True
    return False


def write_split(base, role, tok_path, rows):
    directory = base / role
    directory.mkdir()
    shard = directory / "text.jsonl"
    attribution = directory / "attribution.jsonl"
    families, entries = set(), []
    tokens = Counter()
    with shard.open("wb") as f, attribution.open("wb") as attrs:
        for row, family, ids, norm in rows:
            root = "libretexts/" + sha(row["id"].encode())
            rec = dict(
                root=root,
                family=family,
                split=role,
                lane="general",
                objective="next_token",
                presentation=1,
                token_ids=ids,
                target_mask=[False] + [True] * (len(ids) - 1),
            )
            b = raw(rec)
            f.write(b + b"\n")
            meta = row["metadata"]
            attrs.write(
                raw(
                    dict(
                        root=root,
                        family=family,
                        split=role,
                        author=meta["author"],
                        title=meta["title"],
                        original_url=row["id"],
                        book_url=meta["book_url"],
                        license=meta["license"],
                        upstream_line=meta["provenance"],
                        source_revision=REV,
                        normalized_text_sha256=norm,
                        text_sha256=sha(row["text"].encode()),
                        transformations="Upstream filtered plaintext tokenized unchanged.",
                        tokens=len(ids),
                    )
                )
                + b"\n"
            )
            families.add(family)
            entries.append((root, family, sha(b)))
            tokens["processed"] += len(ids)
    if not entries:
        raise ValueError("empty role: " + role)
    db = directory / "registry.sqlite3"
    shard_sha = digest(shard)
    conn = sqlite3.connect(db)
    try:
        conn.executescript(INDEX_SCHEMA)
        conn.executemany("INSERT INTO families VALUES (?,?)", [(fam, role) for fam in sorted(families)])
        conn.executemany("INSERT INTO roots VALUES (?,?)", [(r, f) for r, f, _ in entries])
        conn.executemany(
            "INSERT INTO views VALUES (?,1,?,?,?,NULL)", [(r, shard_sha, i, h) for i, (r, _, h) in enumerate(entries)]
        )
        conn.commit()
    finally:
        conn.close()
    shutil.copyfile(tok_path, directory / "tokenizer.json")
    manifest = dict(
        schema="aleph.packed-stream.v1",
        context=2048,
        vocab_size=49152,
        pad_id=0,
        tokenizer=dict(path="tokenizer.json", sha256=TOKENIZER_SHA256),
        registry=dict(path=db.name, sha256=digest(db)),
        attribution=dict(path=attribution.name, sha256=digest(attribution)),
        shards=[dict(path=shard.name, sha256=shard_sha, records=len(entries), split=role)],
    )
    path = directory / "manifest.json"
    path.write_bytes(raw(manifest) + b"\n")
    return dict(
        manifest_sha256=digest(path),
        records=len(entries),
        families=len(families),
        tokens=tokens["processed"],
        attribution_sha256=digest(attribution),
        shard_sha256=shard_sha,
    )


def build(args):
    if digest(args.source) != SOURCE:
        raise ValueError("upstream LFS blob differs")
    tokenizer = load_tokenizer(args.tokenizer)
    deny = historical(args.native, args.history)
    rejected = Counter()
    rows = []
    total = 0
    with gzip.open(args.source, "rt", encoding="utf-8") as f:
        for line in f:
            total += 1
            r = json.loads(line)
            why = eligibility(r)
            if why:
                rejected[why] += 1
            else:
                rows.append(r)
    grouped = assign_groups(rows)
    rank = {v: i for i, v in enumerate(ROLES)}
    ordered = sorted(rows, key=lambda r: (rank[grouped[r["id"]][1]], sha(r["id"].encode())))
    by_role = defaultdict(list)
    by_book = Counter()
    inverted = defaultdict(list)
    seen_ids, seen_normalized = set(), set()
    sketches = []
    for row in ordered:
        family, role = grouped[row["id"]]
        book = row["metadata"]["book_url"]
        if by_book[book] >= 128:
            rejected["book_cap"] += 1
            continue
        norm, sketch = duplicate_signature(row["text"])
        if row["id"] in seen_ids or norm in seen_normalized:
            rejected["exact_or_numeral_template"] += 1
            continue
        hits = Counter(idx for word in sketch for idx in inverted[word])
        if any(n >= 0.75 * max(len(sketch), len(sketches[idx])) for idx, n in hits.items()):
            rejected["near_duplicate"] += 1
            continue
        if historical_overlap(row["text"].encode(), deny):
            rejected["observed_historical_window"] += 1
            continue
        ids = tokenizer.encode(row["text"], add_special_tokens=False).ids
        by_role[role].append((row, "libretexts/" + family, ids, norm))
        seen_ids.add(row["id"])
        seen_normalized.add(norm)
        for word in sketch:
            inverted[word].append(len(sketches))
        sketches.append(sketch)
        by_book[book] += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp) / "snapshot"
        stage.mkdir()
        result = {role: write_split(stage, role, args.tokenizer, by_role[role]) for role in ROLES}
        audit = dict(
            source_repo=REPO,
            source_revision=REV,
            source_sha256=SOURCE,
            tokenizer_sha256=TOKENIZER_SHA256,
            rights=LICENSE,
            source_rows=total,
            rejected=dict(rejected),
            role_order=list(ROLES),
            split_unit="connected author, book ID and normalized book title",
            distributions=result,
            historical_panel=PANEL,
            limitations=[
                "lexical and numeral-template duplicate screens do not prove mathematical inequivalence",
                "section metadata is an attribution source; individual third-party exceptions may remain",
                "one shard restricted to mathematics; no claim of representative general prose",
            ],
        )
        if sum(r["records"] for r in result.values()) + sum(rejected.values()) != total:
            raise ValueError("intake ledger mismatch")
        (stage / "audit.json").write_bytes(raw(audit) + b"\n")
        if args.out.exists():
            raise FileExistsError(args.out)
        stage.rename(args.out)
    return audit


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "tokenizer", "native", "history", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(build(p.parse_args()), indent=2))
