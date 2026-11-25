"""Freeze and score the private, document-blocked fresh-text confirmation panel."""

from __future__ import annotations

import argparse
import codecs
import hashlib
import hmac
import importlib.util
import json
import math
import random
import re
import secrets
import tempfile
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from xml.etree import ElementTree

SOURCE_PINS = Path(__file__).with_name("confirm_text_sources.json")
SCHEMA = "datagoblin.aleph.confirm14.text.v1"
BOOKS, BLOCKS, WINDOW, MATCH = 12, 32, 33, 16
NATIVE = {
    "text-train.jsonl": "258b87e7d01efb130dd69f168e2c391827c83800d813a8291e1c1ecd3f4be7c2",
    "text-held.jsonl": "90b3e12ae3bdf656d78b5109b36ba463dcbc7e4ac86c0edd4363984462240320",
}
NS = {"pg": "http://www.gutenberg.org/2009/pgterms/", "dc": "http://purl.org/dc/terms/"}
START = re.compile(r"^(?:\*{3}\s*START OF (?:THIS|THE) PROJECT GUTENBERG|\*END THE SMALL PRINT!).*$", re.I | re.M)
END = re.compile(
    r"^(?:\*{3}\s*END OF (?:THIS|THE) PROJECT GUTENBERG|END OF (?:THE )?PROJECT GUTENBERG).*$", re.I | re.M
)


def encode(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pins():
    docs = json.loads(SOURCE_PINS.read_bytes())
    if (
        len(docs) != BOOKS
        or len({d["id"] for d in docs}) != BOOKS
        or len({d["author"] for d in docs}) != BOOKS
        or sorted(Counter(d["stratum"] for d in docs).values()) != [4, 4, 4]
    ):
        raise ValueError("Expected twelve independent author-documents in three equal strata")
    return docs


def cached(doc, kind, cache):
    suffix = ".txt" if kind == "text" else ".rdf"
    return Path(cache) / (doc["id"] + suffix)


def read_verified(path, checksum):
    data = path.read_bytes()
    if sha(data) != checksum:
        raise ValueError(f"Hash mismatch for {path.name}")
    return data


def fetch(cache):
    """Explicit network-only entry point; freezing and evaluation run offline."""
    cache.mkdir(parents=True, exist_ok=True)
    for doc in pins():
        for kind in ("text", "catalogue"):
            path = cached(doc, kind, cache)
            ref = doc[kind]
            if path.exists():
                read_verified(path, ref["sha256"])
                continue
            url = f"https://raw.githubusercontent.com/{doc['repository']}/{doc['commit']}/{urllib.parse.quote(ref['path'])}"
            with urllib.request.urlopen(url, timeout=45) as reply:
                data = reply.read()
            if sha(data) != ref["sha256"]:
                raise ValueError(f"Upstream source changed for {doc['id']}")
            with path.open("xb") as stream:
                stream.write(data)


def body_bytes(doc, cache):
    raw = read_verified(cached(doc, "text", cache), doc["text"]["sha256"])
    rdf = read_verified(cached(doc, "catalogue", cache), doc["catalogue"]["sha256"])
    book = ElementTree.fromstring(rdf).find("pg:ebook", NS)
    if (
        book is None
        or book.findtext("dc:rights", namespaces=NS) != "Public domain in the USA."
        or book.findtext("dc:creator/pg:agent/pg:name", namespaces=NS) != doc["author"]
        or book.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about") != "ebooks/" + doc["id"][2:]
    ):
        raise ValueError(f"Catalogue rights, identity or attribution mismatch for {doc['id']}")
    declared = re.search(rb"(?im)^character set encoding:\s*([A-Za-z0-9-]+)", raw[:8192])
    encoding = declared[1].decode("ascii").lower() if declared else "ascii"
    if encoding not in {"ascii", "utf-8", "us-ascii", "iso-8859-1"}:
        raise ValueError(f"Unsupported encoding {encoding}")
    text = raw.removeprefix(codecs.BOM_UTF8).decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
    start = START.search(text)
    end = END.search(text, start.end()) if start else None
    if end is None:
        raise ValueError(f"Missing Gutenberg body markers for {doc['id']}")
    normalized = " ".join(unicodedata.normalize("NFC", text[start.end() : end.start()]).split()).encode()
    return normalized, encoding


def folded(raw):
    text = unicodedata.normalize("NFKC", raw.decode("utf-8", errors="ignore")).casefold()
    return " ".join(re.findall(r"[^\W_]+", text)).encode()


def shingles(raw):
    return {raw[i : i + MATCH] for i in range(len(raw) - MATCH + 1)}


def overlap_keys(raw):
    return shingles(raw) | shingles(folded(raw))


def native_bytes(pack):
    original = {}
    for name, checksum in NATIVE.items():
        content = read_verified(Path(pack) / name, checksum)
        original[name] = [bytes(json.loads(line)["tokens"]) for line in content.splitlines()]
        if any(len(row) != WINDOW for row in original[name]):
            raise ValueError(f"Unexpected native window length in {name}")
    return original


def root_id(doc_id, offset, raw):
    return sha(encode([SCHEMA, doc_id, offset, raw.hex()]))


def draw(cache, pack, key):
    if len(key) != 32:
        raise ValueError("Expected a 256-bit private draw key")
    source = pins()
    native = native_bytes(pack)
    prior = {span for windows in native.values() for raw in windows for span in overlap_keys(raw)}
    selected_keys, rows, docs = set(), [], []
    for doc in source:
        body, encoding = body_bytes(doc, cache)
        lo, hi = len(body) // 5, len(body) * 4 // 5
        buckets = [[] for _ in range(BLOCKS)]
        counts = Counter()
        for offset in range(lo, hi - WINDOW + 1, WINDOW):
            raw = body[offset : offset + WINDOW]
            counts["grid"] += 1
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if sum(ch.isalpha() for ch in text) < 15 or text.count(" ") < 3:
                continue
            counts["prose"] += 1
            if not prior.isdisjoint(overlap_keys(raw)):
                counts["native_overlaps"] += 1
                continue
            block = (offset - lo) * BLOCKS // (hi - lo)
            rank = hmac.digest(key, encode([doc["id"], offset]), "sha256")
            buckets[block].append((rank, offset))
        for block, bucket in enumerate(buckets):
            for _, offset in sorted(bucket):
                raw = body[offset : offset + WINDOW]
                keys = overlap_keys(raw)
                if selected_keys.isdisjoint(keys):
                    selected_keys.update(keys)
                    break
                counts["earlier_selected_overlap_skips"] += 1
            else:
                raise ValueError(f"No distinct eligible window in {doc['id']} block {block}")
            rows.append(
                dict(doc=doc["id"], block=block, offset=offset, root=root_id(doc["id"], offset, raw), hex=raw.hex())
            )
        docs.append(
            dict(
                id=doc["id"],
                stratum=doc["stratum"],
                body_sha256=sha(body),
                body_bytes=len(body),
                encoding=encoding,
                counts=dict(counts),
                eligible_per_block=[len(b) for b in buckets],
            )
        )
    windows = b"".join(encode(row) for row in rows)
    manifest = dict(
        schema=SCHEMA,
        source_pins_sha256=sha(encode(source)),
        key_sha256=sha(key),
        native_sha256=NATIVE,
        native_counts={k: len(v) for k, v in native.items()},
        overlap_bytes=MATCH,
        documents=docs,
        rows=len(rows),
        supervised_bytes=len(rows) * (WINDOW - 1),
        windows_sha256=sha(windows),
    )
    return encode(manifest), windows


def freeze(cache, pack, panel, key=None):
    if panel.exists():
        raise FileExistsError(f"Refusing to replace a frozen panel at {panel}")
    panel.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32) if key is None else key
    manifest, rows = draw(cache, pack, key)
    with tempfile.TemporaryDirectory(dir=panel.parent) as temp:
        staged = Path(temp) / "panel"
        staged.mkdir()
        (staged / "draw.key").write_bytes(key)
        (staged / "windows.jsonl").write_bytes(rows)
        (staged / "manifest.json").write_bytes(manifest)
        staged.rename(panel)
    return sha(manifest)


def audit(cache, pack, panel):
    manifest, rows = draw(cache, pack, (panel / "draw.key").read_bytes())
    if (panel / "manifest.json").read_bytes() != manifest or (panel / "windows.jsonl").read_bytes() != rows:
        raise ValueError("Private draw differs from offline reconstruction")
    return sha(manifest)


def panel_data(panel, expected_manifest_sha256=None):
    panel = Path(panel)
    manifest = (panel / "manifest.json").read_bytes()
    digest = sha(manifest)
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        raise ValueError("Different confirmation panel commitment")
    metadata = json.loads(manifest)
    if (
        metadata["schema"] != SCHEMA
        or metadata["source_pins_sha256"] != sha(encode(pins()))
        or metadata["native_sha256"] != NATIVE
    ):
        raise ValueError("Unexpected sources or selection protocol")
    rows = [
        json.loads(line) for line in read_verified(panel / "windows.jsonl", metadata["windows_sha256"]).splitlines()
    ]
    groups = {d["id"]: set(range(BLOCKS)) for d in pins()}
    if len(rows) != BOOKS * BLOCKS or len({r["root"] for r in rows}) != len(rows):
        raise ValueError("Missing or duplicate confirmation window")
    for row in rows:
        raw = bytes.fromhex(row["hex"])
        if len(raw) != WINDOW or row["root"] != root_id(row["doc"], row["offset"], raw):
            raise ValueError("Invalid confirmation window")
        groups[row["doc"]].remove(row["block"])
    if any(groups.values()):
        raise ValueError("Missing document block")
    return digest, rows


def load_rows(panel, expected_manifest_sha256=None):
    """Return native corpus rows. The model sees only the first 32 bytes of each window."""
    if importlib.util.find_spec("corpus") is None:
        from training.aleph_rlt_placement.corpus import Row
    else:
        from corpus import Row

    _, private = panel_data(panel, expected_manifest_sha256)
    rows = []
    for info in private:
        raw = tuple(bytes.fromhex(info["hex"]))
        rows.append(Row("text", info["root"], raw[:-1], raw[1:], 256, info["doc"], info["block"]))
    return rows


def document_scores(scored_rows, panel, expected_manifest_sha256=None):
    """Reduce study.score output; keep one observation per authored document."""
    digest, private = panel_data(panel, expected_manifest_sha256)
    expected = {(r["root"], r["doc"], r["block"]) for r in private}
    masses = defaultdict(list)
    for row in scored_rows:
        if row["lane"] != "text":
            continue
        key = row["root"], row["group"], row["order"]
        if key not in expected or row["positions"] != WINDOW - 1 or not math.isfinite(row["nll"]) or row["nll"] < 0:
            raise ValueError("Scored row missing, duplicated, nonfinite or from another text panel")
        expected.remove(key)
        masses[row["group"]].append(row["nll"])
    if expected:
        raise ValueError("Scored text panel is incomplete")
    per_doc = {k: math.fsum(v) / (BLOCKS * (WINDOW - 1)) for k, v in sorted(masses.items())}
    return dict(
        panel_sha256=digest,
        nats_per_byte=mean(per_doc.values()),
        per_document=per_doc,
        documents=BOOKS,
        windows=BOOKS * BLOCKS,
        supervised_bytes=BOOKS * BLOCKS * (WINDOW - 1),
    )


def quantile(values, q):
    index = q * (len(values) - 1)
    low = int(index)
    return values[low] + (index - low) * (values[min(low + 1, len(values) - 1)] - values[low])


def compare(a_by_seed, b_by_seed, *, draws=10000, seed=37019):
    """Paired B minus A, averaging the fixed matched training seeds within each author."""
    if not a_by_seed or set(a_by_seed) != set(b_by_seed) or draws < 2:
        raise ValueError("Expected identical nonempty seed coverage and at least two draws")
    docs = pins()
    ids = {d["id"] for d in docs}
    pairs = sorted(a_by_seed, key=str)
    all_scores = [scores[s] for scores in (a_by_seed, b_by_seed) for s in pairs]
    if len({r["panel_sha256"] for r in all_scores}) != 1 or any(
        set(r["per_document"]) != ids or not all(math.isfinite(x) for x in r["per_document"].values())
        for r in all_scores
    ):
        raise ValueError("Different documents, panel commitments or nonfinite losses in pairing")
    delta = {s: {d: b_by_seed[s]["per_document"][d] - a_by_seed[s]["per_document"][d] for d in ids} for s in pairs}
    docmeans = {d: mean(delta[s][d] for s in pairs) for d in sorted(ids)}
    strata = defaultdict(list)
    for doc in docs:
        strata[doc["stratum"]].append(docmeans[doc["id"]])
    rng = random.Random(seed)
    samples = sorted(
        mean(x for block in strata.values() for x in rng.choices(block, k=len(block))) for _ in range(draws)
    )
    return dict(
        panel_sha256=all_scores[0]["panel_sha256"],
        paired_B_minus_A_nats_per_byte=mean(docmeans.values()),
        per_seed={s: mean(delta[s].values()) for s in pairs},
        per_document=docmeans,
        per_stratum={k: mean(v) for k, v in strata.items()},
        documents_favoring_B=sum(x < 0 for x in docmeans.values()),
        paired_document_interval_95=[quantile(samples, 0.025), quantile(samples, 0.975)],
        bootstrap=dict(unit="document", strata=3, fixed_matched_seeds=len(pairs), draws=draws, seed=seed),
    )


def baselines(pack, panel, expected_manifest_sha256=None):
    """Frozen native-train-only byte unigram and interpolated trigram for held evaluation."""
    train = native_bytes(pack)["text-train.jsonl"]
    unigram, bigram, trigram = Counter(), defaultdict(Counter), defaultdict(Counter)
    for raw in train:
        unigram.update(raw[1:])
        for i, token in enumerate(raw[1:], 1):
            bigram[raw[i - 1]][token] += 1
            if i >= 2:
                trigram[raw[i - 2 : i]][token] += 1
    total = sum(unigram.values()) + 256
    _, private = panel_data(panel, expected_manifest_sha256)
    scored = {"unigram": [], "interpolated_trigram": []}
    for info in private:
        raw = bytes.fromhex(info["hex"])
        losses = {k: 0.0 for k in scored}
        for i, token in enumerate(raw[1:], 1):
            p = (unigram[token] + 1) / total
            losses["unigram"] -= math.log(p)
            for counts in (bigram[raw[i - 1]], trigram[raw[i - 2 : i]] if i >= 2 else {}):
                p = (counts.get(token, 0) + 8 * p) / (sum(counts.values()) + 8)
            losses["interpolated_trigram"] -= math.log(p)
        for name, nll in losses.items():
            scored[name].append(
                dict(
                    lane="text",
                    root=info["root"],
                    group=info["doc"],
                    order=info["block"],
                    positions=WINDOW - 1,
                    nll=nll,
                )
            )
    return {k: document_scores(v, panel, expected_manifest_sha256) for k, v in scored.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "freeze", "audit"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--panel", type=Path)
    args = parser.parse_args()
    if args.action == "fetch":
        fetch(args.cache)
        return
    if args.pack is None or args.panel is None:
        parser.error("freeze and audit require --pack and --panel")
    operation = freeze if args.action == "freeze" else audit
    print(operation(args.cache, args.pack, args.panel))


if __name__ == "__main__":
    main()
