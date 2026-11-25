"""Independent raw-source and text-token reconciliation for the train-only general supplement."""

import gzip
import json
from collections import Counter
from statistics import median

import text_intake as old
from packing import PackedStream
from production import digest, load_tokenizer, read


def audit(base, folder):
    source = base / "libretexts-checked.json.gz"
    if digest(source) != old.SOURCE:
        raise ValueError("source archive checksum differs")
    manifest = read(folder / "manifest.json")
    adm = read(folder / "admission.json")
    stamp = digest(folder / "manifest.json")
    if adm["manifest_sha256"] != stamp or len(manifest["shards"]) != 1:
        raise ValueError("unbound text manifest")
    attributes = manifest["attribution"]
    af = folder / attributes["path"]
    if digest(af) != attributes["sha256"]:
        raise ValueError("attribution digest differs")
    rs = manifest["shards"][0]
    shard = folder / rs["path"]
    if digest(shard) != rs["sha256"] or rs["split"] != "train":
        raise ValueError("unbound text stream")
    for item in manifest["licenses"].values():
        if digest(folder / item["path"]) != item["sha256"]:
            raise ValueError("changed rights notice")
    tok = load_tokenizer(base / "tokenizer.json")
    pending = {}
    source_starts = set()
    by_domain = Counter()
    tokens = []
    whole = 0
    with af.open("rb") as attr, shard.open("rb") as packed:
        for a, r in zip(attr, packed, strict=True):
            a = json.loads(a)
            r = json.loads(r)
            url = a["original_url"]
            if url in pending or r["root"] != "libretexts/" + old.sha(url.encode()) or r["root"] != a["root"]:
                raise ValueError("duplicate or mismatched original source")
            if r["family"] != a["family"] or r["split"] != a["split"] or a["split"] != "train":
                raise ValueError("family/split mismatch")
            if r["lane"] != "general" or r["objective"] != "next_token" or r["presentation"] != 1:
                raise ValueError("published row has an unexpected role")
            ids = r["token_ids"]
            if r["target_mask"] != [False] + [True] * (len(ids) - 1):
                raise ValueError("unexpected text supervision mask")
            if a["tokens"] != len(ids) or a["source_revision"] != old.REV or a["license"] != old.LICENSE:
                raise ValueError("token or rights metadata differs")
            pending[url] = (a, ids)
    with gzip.open(source, "rt", encoding="utf8") as stream:
        for line in stream:
            record = json.loads(line)
            url = record["id"]
            if url not in pending:
                continue
            a, ids = pending.pop(url)
            meta = record["metadata"]
            if (
                any(a[k] != meta[k] for k in ("license", "title", "author", "book_url"))
                or a["upstream_line"] != meta["provenance"]
            ):
                raise ValueError("row does not retain upstream attribution")
            raw = record["text"].encode()
            lo, hi = a["source_start_byte"], a["source_end_byte"]
            if (
                not 0 <= lo < hi <= len(raw)
                or a["source_text_bytes"] != len(raw)
                or a["source_text_sha256"] != old.sha(raw)
            ):
                raise ValueError("original byte range differs")
            span = raw[lo:hi]
            if old.sha(span) != a["text_sha256"] or tok.encode(span.decode(), add_special_tokens=False).ids != ids:
                raise ValueError("next-token document is not the recorded contiguous source passage")
            source_starts.add((url, lo))
            by_domain[url.split("//", 1)[1].split(".")[0]] += len(ids)
            tokens.append(len(ids))
            whole += int(lo == 0 and hi == len(raw))
    if pending or len(tokens) != rs["records"] or len(source_starts) != len(tokens):
        raise ValueError("source inventory incomplete")
    target = sum(tokens) + 1
    with PackedStream(folder / "manifest.json", stamp, "train") as ps:
        while ps.next_pack(target) is not None:
            pass
        state = ps.state
    if (
        sum(tokens) != adm["distinct_positions"]
        or dict(by_domain) != adm["subject_positions"]
        or state["targets"] != adm["packing"]["targets"]
    ):
        raise ValueError("independent source counts differ from admission")
    return {
        "schema": "datagoblin.milestone.general-source-audit.v1",
        "manifest_sha256": stamp,
        "admission_sha256": digest(folder / "admission.json"),
        "source_sha256": old.SOURCE,
        "checked_document_sources": len(tokens),
        "whole_sections": whole,
        "contiguous_excerpts": len(tokens) - whole,
        "native_processed": state["processed"],
        "native_targets": state["targets"],
        "native_padding": state["padding"],
        "median_tokens_per_source": median(tokens),
        "all_original_byte_spans_and_token_arrays_match": True,
    }
