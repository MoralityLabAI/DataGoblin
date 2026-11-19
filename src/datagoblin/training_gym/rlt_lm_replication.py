"""Held-out, multi-seed replication of the frozen LM feedback comparison."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from itertools import product
from pathlib import Path

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import canonical_json, dump_structured, load_structured

FEEDBACK = ("direct", "rlt", "matched_damping")
CORPORA = ("fineweb_edu", "openwebtext")
DEPTHS = (1, 4, 8, 16)
NEW_SEEDS = (149, 191)


def select_windows(source, index, seen_documents, forbidden_windows, *, count=128, width=96):
    selected = []
    rejected = Counter()
    for row in source["rows"]:
        document = row["full_text_sha256"]
        if document in seen_documents:
            rejected["document_hash"] += 1
            continue
        text = row["row"]["text"].encode()
        n = len(text) - width + 1
        if n <= 0:
            rejected["short"] += 1
            continue
        if any(text[i : i + width] in forbidden_windows for i in range(n)):
            rejected["overlapping_window"] += 1
            continue
        key = sha256_json([index, source["observed_repo_revision"], row["row_idx"], document])
        offset = int(key[:16], 16) % n
        window = text[offset : offset + width]
        selected.append({"id": key, "ordinal": row["row_idx"], "bytes": list(window), "byte_offset": offset})
        seen_documents.add(document)
        forbidden_windows.add(window)
        if len(selected) == count:
            return selected, dict(rejected)
    raise ValueError(f"{index} has fewer than {count} eligible held-out documents")


def freeze(root: Path, new_sources: Path, out: Path):
    original = root / "frozen"
    manifest_path = original / "manifest.json"
    fixture = load_structured(manifest_path)
    text_path, known_path = original / "text_train.json", original / "known.json"
    for path, key in ((text_path, "train_text_sha256"), (known_path, "known_sha256")):
        if sha256_file(path) != fixture[key]:
            raise ValueError("the original text fixture changed")
    train = json.loads(text_path.read_text(encoding="utf-8"))
    seen = {bytes(row[1:]) for row in train}
    known = load_structured(known_path)
    seen.update(bytes(row["bytes"]) for corpus in CORPORA for row in known[corpus])
    seen_documents = set()
    for corpus in CORPORA:
        path = root / "sources" / f"{corpus}.json"
        if sha256_file(path) != fixture["benchmark_sources"][corpus]["snapshot_sha256"]:
            raise ValueError("original text snapshot changed")
        seen_documents.update(row["full_text_sha256"] for row in load_structured(path)["rows"])

    sources, items = {}, {}
    for corpus in CORPORA:
        path = new_sources / f"{corpus}.json"
        source = load_structured(path)
        prior = fixture["benchmark_sources"][corpus]
        if any(source[k] != prior[k] for k in ("dataset", "config", "split")):
            raise ValueError("held-out corpus identity changed")
        indices = [r["row_idx"] for r in source["rows"]]
        if min(indices) < 1000 or indices != sorted(set(indices)):
            raise ValueError("held-out source rows must be ordered, distinct, and outside the original slices")
        rows, rejected = select_windows(source, corpus, seen_documents, seen)
        items[corpus] = rows
        sources[corpus] = {
            **{k: v for k, v in source.items() if k != "rows"},
            "snapshot_sha256": sha256_file(path), "windows": len(rows),
            "rejection_counts_before_cutoff": rejected,
        }
    out.mkdir(parents=True, exist_ok=False)
    windows = out / "windows.json"
    dump_structured(windows, items)
    dump_structured(out / "manifest.json", {
        "schema": "datagoblin.rlt-lm-replication-holdout.v1", "source_fixture_sha256": sha256_file(manifest_path),
        "windows_sha256": sha256_file(windows), "sources": sources,
    })


def completed_resources(path):
    resources = load_structured(path)
    if resources["segments"][-1]["status"] != "complete":
        raise ValueError("run must complete before scoring")
    if any(s["peak_commit_bytes"] > s["hard_commit_cap_bytes"] for s in resources["segments"]):
        raise ValueError("memory cap exceeded")
    return resources


def frozen_fits(root, repeat):
    from .rlt_lm_audit import model_receipts

    _, _, original, _ = model_receipts(root)
    replication = load_structured(repeat / "fit-protocol.json")
    completed_resources(repeat / "fit-resources.json")
    if replication["fixture_sha256"] != sha256_file(root / "frozen" / "manifest.json"):
        raise ValueError("replication used a different training fixture")
    if replication["seeds"] != list(NEW_SEEDS) or replication["feedback"] != list(FEEDBACK):
        raise ValueError("replication fit protocol changed")
    if replication["source_sha256"] != sha256_file(Path(__file__).with_name("rlt_pilot.py")):
        raise ValueError("replication fit source changed")
    original_seed = original["direct"]["seed"]
    budget_keys = ("steps", "train_depth", "unique_parameters", "text_target_bytes",
                   "rule_targets", "shared_block_token_visits_including_padding")
    budget = tuple(original["direct"][key] for key in budget_keys)
    fits = {}
    for seed in (original_seed, *NEW_SEEDS):
        fits[seed] = {}
        initializations = set()
        for mode in FEEDBACK:
            path = root / "run" / mode if seed == original_seed else repeat / "fits" / str(seed) / mode
            fit = load_structured(path / "train.json")
            if (fit["seed"] != seed or fit["feedback"] != mode or tuple(fit[k] for k in budget_keys) != budget
                    or fit["checkpoint_sha256"] != sha256_file(path / "weights.npz")
                    or fit["train_trace_sha256"] != sha256_file(path / "train.jsonl")):
                raise ValueError("checkpoint or fit budget differs from the recorded receipt")
            initializations.add(fit["init_sha256"])
            fits[seed][mode] = (path / "weights.npz", fit)
        if len(initializations) != 1:
            raise ValueError("initialization differs across arms of a seed")
    return fits


def evaluate(root, repeat, holdout, out, guard):
    import numpy as np

    from .rlt_lm_audit import text_rows
    from .rlt_pilot import pending_rows

    manifest_path = holdout / "manifest.json"
    manifest = load_structured(manifest_path)
    windows = holdout / "windows.json"
    if sha256_file(windows) != manifest["windows_sha256"]:
        raise ValueError("held-out windows changed")
    if manifest["source_fixture_sha256"] != sha256_file(root / "frozen" / "manifest.json"):
        raise ValueError("held-out split was checked against a different training fixture")
    fits = frozen_fits(root, repeat)
    items = load_structured(windows)
    protocol = {
        "schema": "datagoblin.rlt-lm-replication.v1", "depths": DEPTHS, "feedback": FEEDBACK,
        "holdout_sha256": sha256_file(manifest_path),
        "source_sha256": {path.name: sha256_file(path) for path in (
            Path(__file__), Path(__file__).with_name("rlt_lm_audit.py"), Path(__file__).with_name("rlt_pilot.py"),
        )},
        "fits": {str(seed): {k: fit for k, (_, fit) in arms.items()} for seed, arms in fits.items()},
    }
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "protocol.json"
    if frozen.exists() and canonical_json(load_structured(frozen)) != canonical_json(protocol):
        raise ValueError("replication evaluation protocol changed")
    if not frozen.exists():
        dump_structured(frozen, protocol)
    for seed, arms in fits.items():
        for trained, (weights, _) in arms.items():
            with np.load(weights, allow_pickle=False) as data:
                model = dict(data)
            for depth, mode in cells():
                path = out / f"{seed}--{trained}--{mode}--{depth}.jsonl"
                if path.exists():
                    continue
                with pending_rows(path) as handle:
                    for corpus in CORPORA:
                        for row in text_rows(model, mode, depth, items[corpus], (depth, depth), guard):
                            handle.write(canonical_json({"corpus": corpus, **row}) + "\n")
            print(canonical_json({"seed": seed, "trained": trained, "status": "evaluated"}), flush=True)


def cells():
    yield 1, "direct"
    yield from product(DEPTHS[1:], FEEDBACK)


def score(run, holdout, out):
    resources = {
        stage: completed_resources(run.parent / f"{stage}-resources.json") for stage in ("fit", "eval")
    }
    protocol = load_structured(run / "protocol.json")
    manifest_path = holdout / "manifest.json"
    manifest = load_structured(manifest_path)
    window_path = holdout / "windows.json"
    if (sha256_file(manifest_path) != protocol["holdout_sha256"]
            or sha256_file(window_path) != manifest["windows_sha256"]):
        raise ValueError("held-out fixture changed")
    items = load_structured(window_path)
    expected = {(corpus, item["id"]): len(item["bytes"]) for corpus, rows in items.items() for item in rows}
    summaries, cache = [], {}
    for seed, trained in product(protocol["fits"], FEEDBACK):
        for depth, runtime in cells():
            path = run / f"{seed}--{trained}--{runtime}--{depth}.jsonl"
            counts, losses = Counter(), {corpus: [] for corpus in CORPORA}
            seen = set()
            for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
                key = row["corpus"], row["id"]
                if key in seen or expected.get(key) != row["bytes"]:
                    raise ValueError("duplicate, missing or invalid evaluation case")
                seen.add(key)
                if not math.isfinite(row["nll_nats"]) or row["nll_nats"] < 0:
                    raise ValueError("invalid byte loss")
                counts[key[0]] += row["bytes"]
                losses[key[0]].append(row["nll_nats"])
            if seen != expected.keys():
                raise ValueError("incomplete prediction file")
            metrics = {k: math.fsum(v) / (counts[k] * math.log(2)) for k, v in losses.items()}
            cache[seed, trained, runtime, depth] = metrics
            summaries.append({
                "seed": int(seed), "trained": trained, "runtime": runtime, "depth": depth, "bpb": metrics,
                "predictions_sha256": sha256_file(path), "file": path.name,
            })
    effects = []
    for seed, corpus, depth in product(protocol["fits"], CORPORA, DEPTHS):
        baseline = cache[seed, "direct", "direct", depth][corpus]
        fitted = cache[seed, "rlt", "direct", depth][corpus]
        runtime = "direct" if depth == 1 else "rlt"
        joint = cache[seed, "rlt", runtime, depth][corpus]
        effect = {
            "seed": int(seed), "corpus": corpus, "depth": depth, "direct_baseline_bpb": baseline,
            "fit_only_bpb": fitted - baseline, "matched_rlt_bpb": joint - baseline,
        }
        if depth > 1:
            switched = cache[seed, "direct", "rlt", depth][corpus]
            damped = cache[seed, "rlt", "matched_damping", depth][corpus]
            effect.update(
                runtime_only_bpb=switched - baseline, rlt_direct_swap_cost_bpb=fitted - joint,
                interaction_bpb=joint - fitted - switched + baseline, runtime_damping_on_rlt_bpb=damped - joint,
            )
        effects.append(effect)
    dump_structured(out, {
        "schema": "datagoblin.rlt-lm-replication-report.v1", "protocol": protocol,
        "holdout": manifest, "cells": summaries, "effects": effects, "resources": resources,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    build = commands.add_parser("freeze")
    for name in ("root", "sources", "out"):
        build.add_argument(f"--{name}", type=Path, required=True)
    scoring = commands.add_parser("score")
    for name in ("run", "holdout", "out"):
        scoring.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.root, args.sources, args.out)
    else:
        score(args.run, args.holdout, args.out)


if __name__ == "__main__":
    main()
