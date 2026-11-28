"""Deterministic CPU corpus intake, screening, packing, and 344-group replay for Aleph Stage-100M.

Builds an immutable 90,000,000-new-position train-only release (10M -> 100M) plus a
strictly disjoint fixed development panel using the frozen SmolLM2 tokenizer
(SHA-256 ``9ca9acdd...``, ``tokenizers==0.22.2``) and ``aleph.packed-stream.v1``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(
    (parent for parent in (HERE, *HERE.parents) if (parent / "training" / "aleph_next" / "candidate01").is_dir()),
    HERE.parent.parent,
)
TRAINING_ROOT = REPO_ROOT / "training"
ALEPH_NEXT = TRAINING_ROOT / "aleph_next"
for p in (str(HERE), str(TRAINING_ROOT / "aleph_rlt_placement"), str(ALEPH_NEXT / "candidate01")):
    if p not in sys.path:
        sys.path.insert(0, p)

from packing import INDEX_SCHEMA  # noqa: E402
from production import TOKENIZER_SHA256, digest, load_tokenizer  # noqa: E402
from stream import STAGE_INDEX_SCHEMA, replay_stage_release  # noqa: E402

ADMISSION_RECEIPT_SCHEMA = "aleph.stage100m.admission-receipt.v1"

# Pinned Stage-1 (90M) source specifications (exact HF commits + LFS SHA-256 OIDs)
PINNED_FILES: dict[str, list[dict[str, Any]]] = {
    "educational_reference": [
        {
            "source_id": "libretexts_filtered",
            "repo": "common-pile/libretexts_filtered",
            "revision": "204d01e40af72b3b6d3f926e1f6148ee9ed3970a",
            "path": "libretexts-0000.json.gz",
            "sha256": "71851a2b5acfccf80659729a0eccd07ab6ed737f0edc659a662aa9c1cd2194db",
            "bytes": 115037937,
            "license": "per-document CC-BY/CC-BY-SA/CC0/PD (non-math only; math.libretexts.org excluded)",
            "format": "jsonl_gz",
            "adapter": "libretexts",
        },
        {
            "source_id": "fineweb_edu_10bt",
            "repo": "HuggingFaceFW/fineweb-edu",
            "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
            "path": "sample/10BT/000_00000.parquet",
            "sha256": "b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871",
            "bytes": 2152819114,
            "license": "odc-by",
            "format": "parquet",
            "adapter": "fineweb_edu",
        },
    ],
    "scientific_papers": [
        {
            "source_id": "pes2o_v2",
            "repo": "allenai/peS2o",
            "revision": "636a503e44a3ca1b58e01fb61eab0825cd574de0",
            "path": "data/v2/train-00000-of-00020.json.gz",
            "sha256": "3d611791c7d89d7915a4089e5f70e6dfc66b0c3e85ace1e5e624e2b96f0e2a29",
            "bytes": 1575272287,
            "license": "odc-by",
            "format": "stream_jsonl_gz",
            "adapter": "pes2o_stream",
        },
    ],
    "code_tests": [
        {
            "source_id": "opc_annealing_algo_1",
            "repo": "OpenCoder-LLM/opc-annealing-corpus",
            "revision": "078e4bc1d39725cccc18514d2e3278a3fe191110",
            "path": "algorithmic_corpus/1.jsonl",
            "sha256": "290c2b20775ea9faa08d6d6512b035bbd3f9d9343e6b3cb3373794e457286e51",
            "bytes": 68917100,
            "license": "odc-by",
            "format": "jsonl",
            "adapter": "opc_algo",
        },
        {
            "source_id": "opc_annealing_algo_10",
            "repo": "OpenCoder-LLM/opc-annealing-corpus",
            "revision": "078e4bc1d39725cccc18514d2e3278a3fe191110",
            "path": "algorithmic_corpus/10.jsonl",
            "sha256": "51443acd019054fc10fcc62569a9f5b1674265f7dee40148c99eec401cb9c6e2",
            "bytes": 72437021,
            "license": "odc-by",
            "format": "jsonl",
            "adapter": "opc_algo",
        },
        {
            "source_id": "opc_annealing_algo_11",
            "repo": "OpenCoder-LLM/opc-annealing-corpus",
            "revision": "078e4bc1d39725cccc18514d2e3278a3fe191110",
            "path": "algorithmic_corpus/11.jsonl",
            "sha256": "f2f6c176b0ba50364f7444a10b9bb32b0dd3c5b3d91bb8800de69a6a1f9a74ad",
            "bytes": 72807474,
            "license": "odc-by",
            "format": "jsonl",
            "adapter": "opc_algo",
        },
        {
            "source_id": "opc_annealing_algo_12",
            "repo": "OpenCoder-LLM/opc-annealing-corpus",
            "revision": "078e4bc1d39725cccc18514d2e3278a3fe191110",
            "path": "algorithmic_corpus/12.jsonl",
            "sha256": "2947e6d18406581004a5678d353d839d6162550ad6a576154d1da58d69fd49b1",
            "bytes": 73319686,
            "license": "odc-by",
            "format": "jsonl",
            "adapter": "opc_algo",
        },
        {
            "source_id": "opc_annealing_algo_13",
            "repo": "OpenCoder-LLM/opc-annealing-corpus",
            "revision": "078e4bc1d39725cccc18514d2e3278a3fe191110",
            "path": "algorithmic_corpus/13.jsonl",
            "sha256": "99bb62b36ab1b85991bcee2e869235b762f0a79028458af0f76d6064a2d88ef2",
            "bytes": 72599763,
            "license": "odc-by",
            "format": "jsonl",
            "adapter": "opc_algo",
        },
    ],
    "informal_math": [
        {
            "source_id": "finemath_4plus_00000",
            "repo": "HuggingFaceTB/finemath",
            "revision": "e92b25a616738fe95dc186b64dfb19f9c8525594",
            "path": "finemath-4plus/train-00000-of-00064.parquet",
            "sha256": "5d0b26114b4cf309c82a4ba8f6f45857480b27812e23a46b61b45e8bbed61fe5",
            "bytes": 286267316,
            "license": "odc-by",
            "format": "parquet",
            "adapter": "finemath",
        },
        {
            "source_id": "finemath_4plus_00001",
            "repo": "HuggingFaceTB/finemath",
            "revision": "e92b25a616738fe95dc186b64dfb19f9c8525594",
            "path": "finemath-4plus/train-00001-of-00064.parquet",
            "sha256": "4f38efdb251638c1a71be4ed380119bdd9966b1d6f200f6ff637e4768cc3930c",
            "bytes": 288474354,
            "license": "odc-by",
            "format": "parquet",
            "adapter": "finemath",
        },
    ],
}

LANE_QUOTAS: dict[str, int] = {
    "educational_reference": 40_000_000,
    "scientific_papers": 10_000_000,
    "code_tests": 20_000_000,
    "informal_math": 20_000_000,
}

# Screening regexes (fail-closed per document)
RE_SECRET = re.compile(
    r"(?:-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|hf_[A-Za-z0-9]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|sk-(?:proj-)?[A-Za-z0-9_-]{24,}"
    r"|AIza[0-9A-Za-z_-]{35}"
    r"|xox[baprs]-[0-9A-Za-z-]{10,})"
)
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_SSN = re.compile(r"(?i)(?:ssn|social\s+security).{0,24}\b\d{3}-\d{2}-\d{4}\b")
RE_CONTAM = re.compile(
    r"(?i)\b(?:DataGoblin|Research\s+Verifiers\s+v2|Jinn/Beast|MTR-02[A-I]|GoblinBench"
    r"|MATH-500|AIME\s*202[456]|GPQA\s+Diamond|HumanEval\+|MBPP\+)\b"
)
RE_ALLOWED_LIBRETEXTS_LICENSE = re.compile(
    r"(?i)(?:creativecommons\.org/licenses/by(?:-sa)?/|creative\s+commons\s*-\s*attribution"
    r"|public\s+domain|cc0|gnu\s+free\s+documentation|gnu\s+fdl|ck-12|openstax)"
)
RE_WS = re.compile(r"\s+")


@dataclass
class NormalizedDoc:
    root: str
    family: str
    text: str
    attribution: dict[str, Any]


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def sha24(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def norm_domain(url: str) -> str:
    try:
        host = urllib.parse.urlparse(url).netloc.lower().split(":")[0]
    except Exception:
        host = ""
    if host.startswith("www."):
        host = host[4:]
    return host or "unknown"


def split_for_family(family: str) -> str:
    bucket = int(hashlib.sha256(f"aleph-stage100m-split-v1:{family}".encode()).hexdigest()[:8], 16) % 100
    return "dev" if bucket < 5 else "train"


def screen_text(text: str, *, min_chars: int = 200, max_chars: int = 250_000) -> str | None:
    if not text or len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    if "\x00" in text:
        return "null_bytes"
    if RE_SECRET.search(text):
        return "secret_pattern"
    if RE_SSN.search(text):
        return "pii_ssn"
    if RE_EMAIL.search(text):
        return "pii_email"
    if RE_CONTAM.search(text):
        return "benchmark_contamination"
    return None


def dedup_keys(text: str) -> tuple[str, str]:
    norm = RE_WS.sub(" ", text.lower()).strip()
    exact = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    # Near-dup anchor: prefix + middle + suffix normalized slices (catches boilerplate/mirror variants)
    n = len(norm)
    mid = n // 2
    anchor = norm[:192] + "|" + norm[max(0, mid - 96) : mid + 96] + "|" + norm[max(0, n - 192) :]
    near = hashlib.sha256(anchor.encode("utf-8")).hexdigest()[:24]
    return exact, near


def ensure_pinned_download(spec: dict[str, Any], download_dir: Path, *, workers: int = 8) -> Path:
    """Download a pinned HF LFS file using parallel HTTP Range requests with strict SHA-256 verification."""
    safe_name = f"{spec['source_id']}__{Path(spec['path']).name}"
    dst = download_dir / safe_name
    if dst.is_file() and dst.stat().st_size == spec["bytes"]:
        got = digest(dst)
        if got == spec["sha256"]:
            print(f"[cache-hit] {safe_name} verified sha256={got[:16]}...", flush=True)
            return dst
        print(f"[cache-bad] {safe_name} sha256 mismatch; re-downloading", flush=True)
        dst.unlink()

    url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['path']}"
    tmp = dst.with_name(dst.name + ".part")
    total_size = int(spec["bytes"])
    print(
        f"[download] {safe_name} ({total_size / 1e6:.1f} MB, {workers}x range) "
        f"from {spec['repo']}@{spec['revision'][:10]}",
        flush=True,
    )
    t0 = time.monotonic()

    chunk_size = 8 * 1024 * 1024
    ranges = [(s, min(total_size - 1, s + chunk_size - 1)) for s in range(0, total_size, chunk_size)]

    def fetch_part(rng: tuple[int, int]) -> tuple[int, bytes]:
        s, e = rng
        for attempt in range(4):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"Range": f"bytes={s}-{e}", "User-Agent": "aleph-stage100m-intake/1.0"},
                )
                with urllib.request.urlopen(req, timeout=90) as resp:
                    data = resp.read()
                if len(data) != (e - s + 1):
                    raise OSError(f"short range read {len(data)} != {e - s + 1}")
                return s, data
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("unreachable")

    # Stream chunks in bounded windows so RAM stays <= ~128 MB while maintaining 8x concurrency
    h = hashlib.sha256()
    written = 0
    with tmp.open("wb") as out, ThreadPoolExecutor(max_workers=workers) as pool:
        window = workers * 2
        for base_i in range(0, len(ranges), window):
            batch = ranges[base_i : base_i + window]
            results = list(pool.map(fetch_part, batch))
            results.sort(key=lambda item: item[0])
            for _, data in results:
                out.write(data)
                h.update(data)
                written += len(data)
    got = h.hexdigest()
    if written != total_size or got != spec["sha256"]:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"download verification failed for {safe_name}: bytes={written}/{total_size} sha={got}/{spec['sha256']}"
        )
    os.replace(tmp, dst)
    dt = max(0.01, time.monotonic() - t0)
    print(f"[verified] {safe_name} in {dt:.1f}s ({written / dt / 1e6:.1f} MB/s) sha256={got[:16]}...", flush=True)
    return dst


class _HashingPrefixStream(io.RawIOBase):
    """Wraps an HTTP response stream, verifying HF LFS ETag and hashing exact consumed bytes."""

    def __init__(self, url: str, expected_lfs_sha256: str) -> None:
        self.url = url
        self.expected_lfs_sha256 = expected_lfs_sha256
        # Verify pinned upstream LFS SHA-256 via HEAD X-Linked-Etag before streaming
        head_req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "aleph-stage100m-intake/1.0"})

        # Avoid following redirect so we see HuggingFace's X-Linked-Etag header directly
        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        opener = urllib.request.build_opener(_NoRedirect)
        try:
            opener.open(head_req, timeout=30)
            linked_etag = ""
        except urllib.error.HTTPError as exc:
            linked_etag = str(exc.headers.get("X-Linked-Etag") or exc.headers.get("ETag") or "").strip().strip('"')
        if linked_etag and linked_etag != expected_lfs_sha256:
            raise ValueError(f"upstream HF X-Linked-Etag mismatch: {linked_etag} != {expected_lfs_sha256}")
        self.linked_etag = linked_etag or expected_lfs_sha256
        req = urllib.request.Request(url, headers={"User-Agent": "aleph-stage100m-intake/1.0"})
        self.resp = urllib.request.urlopen(req, timeout=90)
        self.hasher = hashlib.sha256()
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, b: bytearray) -> int:
        data = self.resp.read(len(b))
        n = len(data)
        if n:
            b[:n] = data
            self.hasher.update(data)
            self.bytes_read += n
        return n

    def close(self) -> None:
        try:
            self.resp.close()
        finally:
            super().close()


def iter_pes2o_stream(spec: dict[str, Any], rej: Counter[str], stream_meta: dict[str, Any]) -> Iterator[NormalizedDoc]:
    """Stream gzipped peS2o v2 JSONL directly from pinned HF LFS revision, hashing exact prefix bytes."""
    url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['path']}"
    print(
        f"[stream] {spec['source_id']} from {spec['repo']}@{spec['revision'][:10]} "
        f"(verifying LFS OID {spec['sha256'][:16]}...)",
        flush=True,
    )
    raw_stream = _HashingPrefixStream(url, spec["sha256"])
    stream_meta["verified_upstream_lfs_sha256"] = raw_stream.linked_etag
    t0 = time.monotonic()
    try:
        with gzip.GzipFile(fileobj=io.BufferedReader(raw_stream, buffer_size=1024 * 1024), mode="rb") as gz:
            text_stream = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
            for line_no, raw in enumerate(text_stream, start=1):
                try:
                    row = json.loads(raw)
                except Exception:
                    rej["parse_error"] += 1
                    continue
                doc_id = str(row.get("id") or "").strip()
                src = str(row.get("source") or "").strip()
                text = str(row.get("text") or "").strip()
                if not doc_id or not src:
                    rej["missing_provenance"] += 1
                    continue
                if len(text) < 400:
                    rej["too_short"] += 1
                    continue
                sample = text[:1000]
                ascii_ratio = sum(1 for ch in sample if ord(ch) < 128) / max(1, len(sample))
                if ascii_ratio < 0.85:
                    rej["non_english_or_garbled"] += 1
                    continue
                first_line = RE_WS.sub(" ", text.split("\n", 1)[0].lower()).strip()[:120]
                base_id = re.sub(r"v\d+$", "", doc_id)
                fam_key = f"{src}|{first_line}" if len(first_line) >= 24 else f"{src}|{base_id}"
                yield NormalizedDoc(
                    root=f"pes2o:{sha24(f'{src}:{doc_id}:{line_no}')}",
                    family=f"pes2o_paper:{sha16(fam_key)}",
                    text=text,
                    attribution={
                        "source_id": spec["source_id"],
                        "repo": spec["repo"],
                        "revision": spec["revision"],
                        "file": spec["path"],
                        "id": doc_id,
                        "source": src,
                        "created": str(row.get("created") or ""),
                        "license": spec["license"],
                    },
                )
    finally:
        raw_stream.close()
        stream_meta["streamed_compressed_bytes"] = raw_stream.bytes_read
        stream_meta["streamed_prefix_sha256"] = raw_stream.hasher.hexdigest()
        stream_meta["stream_seconds"] = round(time.monotonic() - t0, 2)
        print(
            f"[stream-done] {spec['source_id']} read {raw_stream.bytes_read / 1e6:.2f} MB in "
            f"{stream_meta['stream_seconds']}s prefix_sha256={stream_meta['streamed_prefix_sha256'][:16]}...",
            flush=True,
        )


def iter_source_docs(spec: dict[str, Any], local_path: Path, rej: Counter[str]) -> Iterator[NormalizedDoc]:
    adapter = spec["adapter"]
    if adapter == "libretexts":
        with gzip.open(local_path, "rt", encoding="utf-8", errors="replace") as f:
            for line_no, raw in enumerate(f, start=1):
                try:
                    row = json.loads(raw)
                except Exception:
                    rej["parse_error"] += 1
                    continue
                meta = row.get("metadata") or {}
                url = str(meta.get("url") or row.get("id") or "")
                book_url = str(meta.get("book_url") or "")
                lic = str(meta.get("license") or "")
                # Strict exclusion of math.libretexts.org (10M pilot & reserved math eval custody)
                if "math.libretexts.org" in url.lower() or "/book/math-" in book_url.lower():
                    rej["pilot_libretexts_math_excluded"] += 1
                    continue
                if not lic or not RE_ALLOWED_LIBRETEXTS_LICENSE.search(lic):
                    rej["license_rejected"] += 1
                    continue
                text = str(row.get("text") or "").strip()
                author = str(meta.get("author") or "").strip()
                fam_key = book_url or f"{norm_domain(url)}|{author.lower()}"
                yield NormalizedDoc(
                    root=f"libretexts:{sha24(url or f'{line_no}:{text[:120]}')}",
                    family=f"libretexts_book:{sha16(fam_key)}",
                    text=text,
                    attribution={
                        "source_id": spec["source_id"],
                        "repo": spec["repo"],
                        "revision": spec["revision"],
                        "file": spec["path"],
                        "url": url,
                        "book_url": book_url,
                        "title": str(meta.get("title") or ""),
                        "author": author,
                        "license": lic,
                    },
                )
        return

    if adapter == "fineweb_edu":
        pf = pq.ParquetFile(local_path)
        cols = ["text", "id", "dump", "url", "language", "int_score"]
        for batch in pf.iter_batches(batch_size=2048, columns=cols):
            data = batch.to_pydict()
            n = len(data["text"])
            for i in range(n):
                if data["language"][i] != "en":
                    rej["non_english"] += 1
                    continue
                score = data["int_score"][i]
                if score is None or int(score) < 4:
                    rej["low_edu_score"] += 1
                    continue
                url = str(data["url"][i] or "")
                dom = norm_domain(url)
                if "libretexts.org" in dom or "arxiv.org" in dom or "semanticscholar.org" in dom:
                    rej["cross_lane_or_pilot_domain"] += 1
                    continue
                text = str(data["text"][i] or "").strip()
                doc_id = str(data["id"][i] or url)
                yield NormalizedDoc(
                    root=f"fineweb_edu:{sha24(doc_id)}",
                    family=f"fineweb_domain:{dom}",
                    text=text,
                    attribution={
                        "source_id": spec["source_id"],
                        "repo": spec["repo"],
                        "revision": spec["revision"],
                        "file": spec["path"],
                        "id": doc_id,
                        "url": url,
                        "dump": str(data["dump"][i] or ""),
                        "int_score": int(score),
                        "license": spec["license"],
                    },
                )
        return

    if adapter == "pes2o":
        with gzip.open(local_path, "rt", encoding="utf-8", errors="replace") as f:
            for line_no, raw in enumerate(f, start=1):
                try:
                    row = json.loads(raw)
                except Exception:
                    rej["parse_error"] += 1
                    continue
                doc_id = str(row.get("id") or "").strip()
                src = str(row.get("source") or "").strip()
                text = str(row.get("text") or "").strip()
                if not doc_id or not src:
                    rej["missing_provenance"] += 1
                    continue
                if len(text) < 400:
                    rej["too_short"] += 1
                    continue
                # Quick ASCII/English heuristic for scientific papers
                sample = text[:1000]
                ascii_ratio = sum(1 for ch in sample if ord(ch) < 128) / max(1, len(sample))
                if ascii_ratio < 0.85:
                    rej["non_english_or_garbled"] += 1
                    continue
                first_line = RE_WS.sub(" ", text.split("\n", 1)[0].lower()).strip()[:120]
                base_id = re.sub(r"v\d+$", "", doc_id)
                fam_key = f"{src}|{first_line}" if len(first_line) >= 24 else f"{src}|{base_id}"
                yield NormalizedDoc(
                    root=f"pes2o:{sha24(f'{src}:{doc_id}:{line_no}')}",
                    family=f"pes2o_paper:{sha16(fam_key)}",
                    text=text,
                    attribution={
                        "source_id": spec["source_id"],
                        "repo": spec["repo"],
                        "revision": spec["revision"],
                        "file": spec["path"],
                        "id": doc_id,
                        "source": src,
                        "created": str(row.get("created") or ""),
                        "license": spec["license"],
                    },
                )
        return

    if adapter == "opc_algo":
        with local_path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, raw in enumerate(f, start=1):
                try:
                    row = json.loads(raw)
                except Exception:
                    rej["parse_error"] += 1
                    continue
                text = str(row.get("text") or "").strip()
                lang = str(row.get("lang") or "unknown").lower()
                rtype = str(row.get("type") or "unknown").lower()
                if not text:
                    rej["too_short"] += 1
                    continue
                # Group by first substantive prompt/header line to keep problem variants in one family
                header = ""
                for ln in text.splitlines():
                    s = RE_WS.sub(" ", ln.strip().lower())
                    if len(s) >= 20:
                        header = s[:140]
                        break
                fam_seed = f"{lang}|{header}" if header else f"{lang}|{text[:140].lower()}"
                src_id = spec["source_id"]
                yield NormalizedDoc(
                    root=f"opc_algo:{sha24(f'{src_id}:{line_no}:{text[:160]}')}",
                    family=f"opc_algo_fam:{sha16(fam_seed)}",
                    text=text,
                    attribution={
                        "source_id": spec["source_id"],
                        "repo": spec["repo"],
                        "revision": spec["revision"],
                        "file": spec["path"],
                        "ordinal": line_no,
                        "lang": lang,
                        "type": rtype,
                        "license": spec["license"],
                    },
                )
        return

    if adapter == "finemath":
        pf = pq.ParquetFile(local_path)
        cols = ["url", "text", "int_score", "language", "crawl"]
        for batch in pf.iter_batches(batch_size=2048, columns=cols):
            data = batch.to_pydict()
            n = len(data["text"])
            for i in range(n):
                if data["language"][i] != "en":
                    rej["non_english"] += 1
                    continue
                score = data["int_score"][i]
                if score is None or int(score) < 4:
                    rej["low_math_score"] += 1
                    continue
                url = str(data["url"][i] or "")
                dom = norm_domain(url)
                if "libretexts.org" in dom:
                    rej["pilot_libretexts_excluded"] += 1
                    continue
                text = str(data["text"][i] or "").strip()
                yield NormalizedDoc(
                    root=f"finemath:{sha24(url or text[:160])}",
                    family=f"finemath_domain:{dom}",
                    text=text,
                    attribution={
                        "source_id": spec["source_id"],
                        "repo": spec["repo"],
                        "revision": spec["revision"],
                        "file": spec["path"],
                        "url": url,
                        "crawl": str(data["crawl"][i] or ""),
                        "int_score": int(score),
                        "license": spec["license"],
                    },
                )
        return

    raise ValueError(f"unknown adapter: {adapter}")


class LaneStoreWriter:
    """Streams admitted tokenized records into <=64MB JSONL shards + SQLite custody registry."""

    def __init__(
        self,
        lane_dir: Path,
        lane_name: str,
        split: str,
        tokenizer_src: Path,
        *,
        max_shard_bytes: int = 48 * 1024 * 1024,
    ) -> None:
        self.lane_dir = lane_dir
        self.lane_dir.mkdir(parents=True, exist_ok=True)
        self.lane_name = lane_name
        self.split = split
        self.max_shard_bytes = max_shard_bytes

        self.tok_dst = self.lane_dir / "tokenizer.json"
        shutil.copyfile(tokenizer_src, self.tok_dst)
        self.tok_sha = digest(self.tok_dst)

        self.attr_path = self.lane_dir / "attribution.jsonl"
        self.attr_fp = self.attr_path.open("wb")

        self.db_path = self.lane_dir / "registry.sqlite3"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA synchronous = OFF")
        self.conn.execute("PRAGMA journal_mode = MEMORY")
        self.conn.executescript(INDEX_SCHEMA)

        self.shard_idx = 0
        self.shard_fp = None
        self.shard_hasher = None
        self.shard_path: Path | None = None
        self.shard_bytes = 0
        self.shard_records = 0
        self.shard_views_buffer: list[tuple[str, int, str, int, str, None]] = []
        self.shards_meta: list[dict[str, Any]] = []

        self.families_seen: set[str] = set()
        self.total_records = 0
        self.total_positions = 0
        self.total_labels = 0
        self._open_new_shard()

    def _open_new_shard(self) -> None:
        self.shard_path = self.lane_dir / f"shard_{self.shard_idx:04d}.jsonl"
        self.shard_fp = self.shard_path.open("wb")
        self.shard_hasher = hashlib.sha256()
        self.shard_bytes = 0
        self.shard_records = 0
        self.shard_views_buffer = []

    def _flush_shard(self) -> None:
        if self.shard_fp is None or self.shard_path is None or self.shard_hasher is None:
            return
        self.shard_fp.close()
        self.shard_fp = None
        if self.shard_records == 0:
            self.shard_path.unlink(missing_ok=True)
            return
        shard_sha = self.shard_hasher.hexdigest()
        self.conn.executemany(
            "INSERT INTO views VALUES (?, ?, ?, ?, ?, ?)",
            [(r, p, shard_sha, o, rsha, rcpt) for (r, p, _, o, rsha, rcpt) in self.shard_views_buffer],
        )
        self.shards_meta.append(
            {
                "path": self.shard_path.name,
                "sha256": shard_sha,
                "records": self.shard_records,
                "split": self.split,
            }
        )
        self.shard_idx += 1

    def add(self, doc: NormalizedDoc, token_ids: list[int]) -> None:
        if self.shard_bytes >= self.max_shard_bytes and self.shard_records > 0:
            self._flush_shard()
            self._open_new_shard()
        row = {
            "root": doc.root,
            "family": doc.family,
            "split": self.split,
            "presentation": 1,
            "lane": self.lane_name,
            "objective": "next_token",
            "token_ids": token_ids,
            "target_mask": [False] + [True] * (len(token_ids) - 1),
            "receipt_sha256": None,
        }
        raw = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        line = raw + b"\n"
        assert self.shard_fp is not None and self.shard_hasher is not None
        self.shard_fp.write(line)
        self.shard_hasher.update(line)
        if doc.family not in self.families_seen:
            self.conn.execute("INSERT INTO families VALUES (?, ?)", (doc.family, self.split))
            self.families_seen.add(doc.family)
        self.conn.execute("INSERT INTO roots VALUES (?, ?)", (doc.root, doc.family))
        self.shard_views_buffer.append((doc.root, 1, "", self.shard_records, hashlib.sha256(raw).hexdigest(), None))
        self.shard_records += 1
        self.shard_bytes += len(line)
        self.total_records += 1
        self.total_positions += len(token_ids)
        self.total_labels += len(token_ids) - 1

        attr_row = {
            "root": doc.root,
            "family": doc.family,
            "split": self.split,
            "lane": self.lane_name,
            "positions": len(token_ids),
            **doc.attribution,
        }
        self.attr_fp.write(
            (json.dumps(attr_row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        )

    def finalize(self) -> tuple[Path, str, dict[str, Any]]:
        self._flush_shard()
        self.attr_fp.close()
        self.conn.commit()
        self.conn.close()
        db_sha = digest(self.db_path)
        attr_sha = digest(self.attr_path)
        manifest = {
            "schema": "aleph.packed-stream.v1",
            "context": 2048,
            "vocab_size": 49152,
            "pad_id": 0,
            "tokenizer": {"path": self.tok_dst.name, "sha256": self.tok_sha},
            "registry": {"path": self.db_path.name, "sha256": db_sha},
            "attribution": {"path": self.attr_path.name, "sha256": attr_sha},
            "shards": self.shards_meta,
        }
        mpath = self.lane_dir / "manifest.json"
        raw = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
        mpath.write_bytes(raw)
        msha = hashlib.sha256(raw).hexdigest()
        stats = {
            "manifest_path": mpath.name,
            "manifest_sha256": msha,
            "registry_sha256": db_sha,
            "attribution_sha256": attr_sha,
            "records": self.total_records,
            "families": len(self.families_seen),
            "admitted_positions": self.total_positions,
            "admitted_labels": self.total_labels,
            "shards": len(self.shards_meta),
        }
        return mpath, msha, stats


def build_members_ledger(root_dir: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(root_dir.rglob("*")):
        if path.is_file() and path.name != "MEMBERS.json":
            rel = path.relative_to(root_dir).as_posix()
            sz = path.stat().st_size
            total_bytes += sz
            entries.append({"path": rel, "bytes": sz, "sha256": digest(path)})
    return {
        "schema": "aleph.stage100m.members-ledger.v1",
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir", type=Path, required=True, help="Local scratch directory (e.g. /path/to/stage100m_work)"
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="Persistent download cache directory")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Path to frozen SmolLM2 tokenizer.json")
    parser.add_argument("--dev-cap-per-lane", type=int, default=250_000, help="Held-out dev panel tokens per lane")
    parser.add_argument("--scale", type=float, default=1.0, help="Quota scale (<1.0 for smoke test only)")
    parser.add_argument(
        "--receipt-out", type=Path, default=None, help="Optional path to copy sanitized release receipt JSON"
    )
    args = parser.parse_args()

    t_start = time.monotonic()
    tok = load_tokenizer(args.tokenizer)
    print(
        f"[tokenizer] verified frozen SmolLM2 sha256={TOKENIZER_SHA256[:16]}... vocab={tok.get_vocab_size()}",
        flush=True,
    )

    work_dir = args.work_dir.resolve()
    dl_dir = args.cache_dir.resolve() if args.cache_dir else (work_dir / "downloads")
    dl_dir.mkdir(parents=True, exist_ok=True)
    release_dir = work_dir / "stage_100m"
    dev_dir = work_dir / "dev_panel"
    if release_dir.exists() or dev_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing {release_dir} or {dev_dir}; use a clean work-dir")
    release_dir.mkdir(parents=True)
    dev_dir.mkdir(parents=True)

    quotas = {lane: max(2048, round(q * args.scale)) for lane, q in LANE_QUOTAS.items()}
    dev_cap = max(2048, round(args.dev_cap_per_lane * args.scale))
    # Global cross-source deduplication sets (train + dev never share exact/near docs or roots)
    seen_exact: set[str] = set()
    seen_near: set[str] = set()
    seen_roots: set[str] = set()

    lane_receipts: dict[str, Any] = {}
    index_lanes: dict[str, Any] = {}
    dev_index_lanes: dict[str, Any] = {}

    for lane in ("educational_reference", "scientific_papers", "code_tests", "informal_math"):
        quota = quotas[lane]
        print(f"\n=== Lane: {lane} (target train={quota:,} positions, dev_cap={dev_cap:,}) ===", flush=True)
        train_writer = LaneStoreWriter(release_dir / lane, lane, "train", args.tokenizer)
        dev_writer = LaneStoreWriter(dev_dir / lane, lane, "dev", args.tokenizer)

        source_reports: list[dict[str, Any]] = []
        for spec in PINNED_FILES[lane]:
            if train_writer.total_positions >= quota and dev_writer.total_positions >= dev_cap:
                source_reports.append(
                    {
                        "source_id": spec["source_id"],
                        "repo": spec["repo"],
                        "revision": spec["revision"],
                        "path": spec["path"],
                        "sha256": spec["sha256"],
                        "license": spec["license"],
                        "status": "not_needed_quota_met",
                    }
                )
                continue

            stream_meta: dict[str, Any] = {}
            rej: Counter[str] = Counter()
            if spec["adapter"] == "pes2o_stream":
                doc_iter = iter_pes2o_stream(spec, rej, stream_meta)
            else:
                local_path = ensure_pinned_download(spec, dl_dir)
                doc_iter = iter_source_docs(spec, local_path, rej)
            scanned = 0
            admitted_train_docs = 0
            admitted_train_pos = 0
            admitted_dev_docs = 0
            admitted_dev_pos = 0

            # Batch encode for fast Rust tokenizer throughput
            batch_docs: list[tuple[NormalizedDoc, str]] = []

            def _flush_batch(
                _batch: list[tuple[NormalizedDoc, str]] = batch_docs,
                _rej: Counter[str] = rej,
                _train_w: LaneStoreWriter = train_writer,
                _dev_w: LaneStoreWriter = dev_writer,
                _quota: int = quota,
            ) -> None:
                nonlocal admitted_train_docs, admitted_train_pos, admitted_dev_docs, admitted_dev_pos
                if not _batch:
                    return
                encs = tok.encode_batch([d.text for d, _ in _batch], add_special_tokens=False)
                for (doc, sp), enc in zip(_batch, encs, strict=True):
                    ids = enc.ids
                    if len(ids) < 32:
                        _rej["too_few_tokens"] += 1
                        continue
                    if sp == "train":
                        if _train_w.total_positions >= _quota:
                            continue
                        _train_w.add(doc, ids)
                        admitted_train_docs += 1
                        admitted_train_pos += len(ids)
                    else:
                        if _dev_w.total_positions >= dev_cap:
                            continue
                        _dev_w.add(doc, ids)
                        admitted_dev_docs += 1
                        admitted_dev_pos += len(ids)
                _batch.clear()

            try:
                for doc in doc_iter:
                    scanned += 1
                    reason = screen_text(doc.text)
                    if reason is not None:
                        rej[reason] += 1
                        continue
                    if doc.root in seen_roots:
                        rej["duplicate_root"] += 1
                        continue
                    ex_key, near_key = dedup_keys(doc.text)
                    if ex_key in seen_exact:
                        rej["exact_duplicate"] += 1
                        continue
                    if near_key in seen_near:
                        rej["near_duplicate"] += 1
                        continue
                    sp = split_for_family(doc.family)
                    if sp == "train" and train_writer.total_positions >= quota:
                        if dev_writer.total_positions >= dev_cap:
                            break
                        rej["train_quota_already_met"] += 1
                        continue
                    if sp == "dev" and dev_writer.total_positions >= dev_cap:
                        if train_writer.total_positions >= quota:
                            break
                        rej["dev_cap_already_met"] += 1
                        continue
                    seen_roots.add(doc.root)
                    seen_exact.add(ex_key)
                    seen_near.add(near_key)
                    batch_docs.append((doc, sp))
                    if len(batch_docs) >= 256:
                        _flush_batch()
                        if train_writer.total_positions >= quota and dev_writer.total_positions >= dev_cap:
                            break
                _flush_batch()
            finally:
                if hasattr(doc_iter, "close"):
                    doc_iter.close()

            source_reports.append(
                {
                    "source_id": spec["source_id"],
                    "repo": spec["repo"],
                    "revision": spec["revision"],
                    "path": spec["path"],
                    "sha256": spec["sha256"],
                    "bytes": spec["bytes"],
                    "license": spec["license"],
                    "status": "ingested",
                    **({"streaming": stream_meta} if stream_meta else {}),
                    "rows_scanned": scanned,
                    "admitted_train_docs": admitted_train_docs,
                    "admitted_train_positions": admitted_train_pos,
                    "admitted_dev_docs": admitted_dev_docs,
                    "admitted_dev_positions": admitted_dev_pos,
                    "rejections": dict(sorted(rej.items())),
                }
            )
            print(
                f"  [{spec['source_id']}] scanned={scanned:,} train=+{admitted_train_pos:,} "
                f"(cum {train_writer.total_positions:,}/{quota:,}) dev=+{admitted_dev_pos:,} "
                f"rej={sum(rej.values()):,}",
                flush=True,
            )

        _, train_msha, train_stats = train_writer.finalize()
        _, dev_msha, dev_stats = dev_writer.finalize()

        shortfall = max(0, quota - train_stats["admitted_positions"])
        if shortfall > 0:
            raise RuntimeError(
                f"SHORTFALL in lane {lane}: admitted={train_stats['admitted_positions']:,} < quota={quota:,} "
                "(refusing to fabricate quota or wrap around)"
            )

        index_lanes[lane] = {
            "manifest_path": f"{lane}/manifest.json",
            "manifest_sha256": train_msha,
            "available_positions": train_stats["admitted_positions"],
            "available_labels": train_stats["admitted_labels"],
            "records": train_stats["records"],
            "families": train_stats["families"],
        }
        dev_index_lanes[lane] = {
            "manifest_path": f"{lane}/manifest.json",
            "manifest_sha256": dev_msha,
            "available_positions": dev_stats["admitted_positions"],
            "available_labels": dev_stats["admitted_labels"],
            "records": dev_stats["records"],
            "families": dev_stats["families"],
        }
        lane_receipts[lane] = {
            "quota_positions": quota,
            "train": train_stats,
            "dev": dev_stats,
            "sources": source_reports,
        }

    total_new = sum(quotas.values())
    stage_start = 10_000_000
    stage_stop = stage_start + total_new
    index_obj = {
        "schema": STAGE_INDEX_SCHEMA,
        "split": "train",
        "stage_start_positions": stage_start,
        "stage_stop_positions": stage_stop,
        "stage_new_positions": total_new,
        "update_group_positions": 262_144,
        "context": 2048,
        "vocab_size": 49152,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "lane_quotas": quotas,
        "lanes": index_lanes,
    }
    index_raw = json.dumps(index_obj, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    (release_dir / "index.json").write_bytes(index_raw)
    index_sha = hashlib.sha256(index_raw).hexdigest()

    # Detached full CPU replay + mid-stage restart verification
    print(f"\n=== Running full detached CPU replay on {total_new:,} new positions ===", flush=True)
    t_rep = time.monotonic()
    replay = replay_stage_release(release_dir, index_sha)
    replay_raw = json.dumps(replay, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    (release_dir / "REPLAY.json").write_bytes(replay_raw)
    replay_sha = hashlib.sha256(replay_raw).hexdigest()
    print(
        f"[replay-ok] groups={replay['update_groups']} full={replay['full_groups']} "
        f"final={replay['final_group_positions']} labels={replay['total_labels']:,} "
        f"padding={replay['total_padding']:,} restart_verified={replay['restart_parity_verified']} "
        f"wall={time.monotonic() - t_rep:.1f}s",
        flush=True,
    )

    # Write dev panel index separately (strictly outside train release_dir)
    dev_index = {
        "schema": "aleph.stage100m.dev-panel-index.v1",
        "split": "dev",
        "context": 2048,
        "vocab_size": 49152,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "lanes": dev_index_lanes,
    }
    dev_index_raw = json.dumps(dev_index, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    (dev_dir / "dev_index.json").write_bytes(dev_index_raw)
    dev_index_sha = hashlib.sha256(dev_index_raw).hexdigest()

    dev_members = build_members_ledger(dev_dir)
    (dev_dir / "MEMBERS.json").write_bytes(
        json.dumps(dev_members, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    )

    admission_receipt = {
        "schema": ADMISSION_RECEIPT_SCHEMA,
        "stage_start_positions": stage_start,
        "stage_stop_positions": stage_stop,
        "stage_new_positions": total_new,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tokenizer_library_version": "0.22.2",
        "stage_index_sha256": index_sha,
        "stage_replay_sha256": replay_sha,
        "dev_panel_index_sha256": dev_index_sha,
        "max_presentations_per_root": 1,
        "wraparound_permitted": False,
        "pilot_exclusion": [
            "Excluded 100% of math.libretexts.org and /book/math-* from common-pile/libretexts_filtered",
            "Excluded libretexts.org domains from fineweb-edu and finemath-4plus",
            "Zero pilot records recycled",
        ],
        "deferred_or_rejected_stage1_candidates": [
            {
                "repo": "HuggingFaceTB/stack-edu",
                "status": "deferred_metadata_only",
                "reason": "Ships Software Heritage blob identifiers without executable source bytes",
            },
            {
                "repo": "common-pile/comma_v0.1_training_dataset",
                "status": "not_used_lacks_per_doc_provenance",
                "reason": (
                    "Compressed training chunks expose only raw 'text' without per-document provenance; "
                    "allenai/peS2o v2 provides explicit per-document id/source/created provenance"
                ),
            },
        ],
        "replay_summary": {k: v for k, v in replay.items() if k != "steps"},
        "lanes": lane_receipts,
        "wall_seconds": round(time.monotonic() - t_start, 2),
    }
    receipt_raw = json.dumps(admission_receipt, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    (release_dir / "ADMISSION_RECEIPT.json").write_bytes(receipt_raw)

    members = build_members_ledger(release_dir)
    members_raw = json.dumps(members, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    (release_dir / "MEMBERS.json").write_bytes(members_raw)
    admission_receipt["members_ledger_sha256"] = hashlib.sha256(members_raw).hexdigest()
    admission_receipt["release_file_count"] = members["file_count"]
    admission_receipt["release_total_bytes"] = members["total_bytes"]
    admission_receipt["dev_panel_total_bytes"] = dev_members["total_bytes"]

    if args.receipt_out is not None:
        args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_out.write_bytes(
            json.dumps(admission_receipt, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
        )
    print(
        json.dumps(
            {
                "status": "OK",
                "index_sha256": index_sha,
                "replay_sha256": replay_sha,
                "members_ledger_sha256": admission_receipt["members_ledger_sha256"],
                "release_total_bytes": members["total_bytes"],
                "dev_panel_total_bytes": dev_members["total_bytes"],
                "wall_seconds": admission_receipt["wall_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
