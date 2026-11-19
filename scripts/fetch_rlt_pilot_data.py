"""Fetch bounded, pinned dataset-viewer slices for the pilot; save no credentials."""

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

SOURCES = {
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", (0, 100, 200)),
    "openwebtext": ("Skylion007/openwebtext", "plain_text", "train", (0, 100, 200)),
    "lambada": ("EleutherAI/lambada_openai", "en", "test", (0,)),
    "hellaswag": ("Rowan/hellaswag", "default", "validation", (0,)),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test", (0,)),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test", (0,)),
}


def fetch(url, limit):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DataGoblin-RLT-small-pilot/1"})
            with urllib.request.urlopen(req, timeout=22) as response:
                raw = response.read(limit + 1)
            if len(raw) > limit:
                raise ValueError(f"response exceeds {limit} bytes: {url}")
            return json.loads(raw), hashlib.sha256(raw).hexdigest()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2)


def retrieve(root, name, spec):
    dataset, config, split, offsets = spec
    revision, _ = fetch(f"https://huggingface.co/api/datasets/{dataset}", 6_000_000)
    splits_url = "https://datasets-server.huggingface.co/splits?" + urllib.parse.urlencode({"dataset": dataset})
    splits, _ = fetch(splits_url, 2_000_000)
    available = [r["config"] for r in splits["splits"] if r["split"] == split]
    if config not in available and len(available) == 1:
        config = available[0]
    if config not in available:
        raise ValueError(f"{name}: expected config {config!r} not found in {available}")
    rows, pulls = [], []
    for offset in offsets:
        url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode({
            "dataset": dataset, "config": config, "split": split, "offset": offset, "length": 100,
        })
        result, sha = fetch(url, 16_000_000)
        returned = result["rows"]
        assert [r["row_idx"] for r in returned] == list(range(offset, offset + 100))
        rows.extend(returned)
        pulls.append({"url": url, "response_sha256": sha})
    if name in ("fineweb_edu", "openwebtext"):
        # Keep only a bounded excerpt per document. Store the identity of its full text.
        for row in rows:
            raw = row["row"]["text"]
            row["full_text_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
            row["row"]["text"] = raw[:8192]
    path = root / f"{name}.json"
    partial = path.with_suffix(".part")
    partial.write_text(json.dumps({
        "dataset": dataset, "config": config, "split": split, "observed_repo_revision": revision["sha"],
        "pulls": pulls, "rows": rows,
    }), encoding="utf-8")
    partial.replace(path)
    return name, len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root / "sources"
    root.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(retrieve, root, name, spec)
            for name, spec in SOURCES.items()
            if not (root / f"{name}.json").exists()
        ]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
