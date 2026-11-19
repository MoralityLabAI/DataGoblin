"""Produce a compact, paired evidence bundle from a finished local RLT pilot."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections.abc import Sequence
from pathlib import Path

from datagoblin.hashing import sha256_file
from datagoblin.serialization import canonical_json, dump_structured, load_structured

from .rlt_pilot_data import PUBLIC, RULE_SPLITS, TEXT, jsonl
from .rlt_revision import score


def by_id(rows):
    mapped = {r["id"]: r for r in rows}
    if len(mapped) != len(rows):
        raise ValueError("duplicate pilot rows")
    return mapped


def compare(items, baseline, arm, field):
    wins = sum(bool(v[arm][field]) and not v[baseline][field] for v in items)
    losses = sum(bool(v[baseline][field]) and not v[arm][field] for v in items)
    return {"wins": wins, "losses": losses, "ties": len(items) - wins - losses, "total": len(items)}


def aggregate(root: Path, dest: Path, *, completed_depths_only: bool = False) -> dict:
    run = root / "run"
    resource = load_structured(run / "run.json")
    if resource["status"] != "complete" and not completed_depths_only:
        raise ValueError("incomplete run, use --completed-depths-only to export fully matched depths")
    if resource["peak_job_commit_bytes"] > resource["hard_job_commit_limit_bytes"]:
        raise ValueError("completed run exceeded the recorded job limit")
    protocol = load_structured(run / "protocol.json")
    manifest_path = root / "frozen" / "manifest.json"
    if sha256_file(manifest_path) != protocol["frozen_manifest_sha256"]:
        raise ValueError("frozen manifest changed")
    manifest = load_structured(manifest_path)
    known_path = root / "frozen" / "known.json"
    if sha256_file(known_path) != manifest["known_sha256"]:
        raise ValueError("known dataset changed")
    known_inputs = load_structured(known_path)
    rule_manifest = root / "rules" / "manifest.json"
    if sha256_file(rule_manifest) != manifest["rules_manifest_sha256"]:
        raise ValueError("rule manifest changed")
    modes = protocol["feedback_modes"]
    depths = [
        depth for depth in protocol["depths"]
        if all(
            (run / mode / f"known-k{depth}.jsonl").exists()
            and all((run / mode / f"rule-{split}-k{depth}.jsonl").exists() for split in RULE_SPLITS)
            for mode in modes
        )
    ]
    if not depths or (not completed_depths_only and depths != protocol["depths"]):
        raise ValueError("insufficient complete matched evaluation files")
    training = {mode: load_structured(run / mode / "train.json") for mode in modes}
    for mode, receipt in training.items():
        if (
            sha256_file(run / mode / "weights.npz") != receipt["checkpoint_sha256"]
            or sha256_file(run / mode / "train.jsonl") != receipt["train_trace_sha256"]
        ):
            raise ValueError(f"training artifact changed for {mode}")
    for key in ("seed", "steps", "init_sha256", "unique_parameters", "text_target_bytes", "rule_targets",
                "shared_block_token_visits_including_padding"):
        if len({row[key] for row in training.values()}) != 1:
            raise ValueError(f"unmatched training field: {key}")
    dest.mkdir(parents=True, exist_ok=False)
    summary: dict = {
        "protocol": protocol, "dataset_manifest": manifest, "rule_manifest": load_structured(rule_manifest),
        "training": training, "resources": resource, "depths": {},
        "scope": "One-seed reduced-scale pilot with 64 items per public benchmark. Task effects need replication.",
        "matched_depths": depths,
        "missing_depths": [depth for depth in protocol["depths"] if depth not in depths],
    }
    aborted_path = root / "aborted-before-resumable" / "run.json"
    if aborted_path.exists():
        summary["excluded_attempt"] = load_structured(aborted_path)
    items_path = dest / "paired-items.jsonl"
    with items_path.open("x", encoding="utf-8", newline="\n") as handle:
        for k in depths:
            panel: dict = {}
            known = {mode: by_id(jsonl(run / mode / f"known-k{k}.jsonl")) for mode in modes}
            if any(rows.keys() != known["direct"].keys() for rows in known.values()):
                raise ValueError("known item coverage differs across arms")
            for name in PUBLIC:
                group = []
                expected = {item["id"] for item in known_inputs[name]}
                observed = {key for key, row in known["direct"].items() if row["benchmark"] == name}
                if observed != expected:
                    raise ValueError(f"wrong frozen item coverage for {name}")
                for key, row in known["direct"].items():
                    if row["benchmark"] != name:
                        continue
                    pair = {mode: known[mode][key] for mode in modes}
                    if any(r["benchmark"] != name for r in pair.values()):
                        raise ValueError("benchmark differs across arms")
                    group.append(pair)
                    handle.write(canonical_json({"depth": k, "id": key, "benchmark": name, "arms": pair}) + "\n")
                if len(group) != manifest["known_items"][name]:
                    raise ValueError("wrong known benchmark count")
                if name in TEXT:
                    panel[name] = {
                        mode: {
                            "bits_per_byte": sum(p[mode]["nll_nats"] for p in group)
                            / (math.log(2) * sum(p[mode]["bytes"] for p in group)),
                            "n": len(group),
                        } for mode in modes
                    }
                else:
                    primary = "correct_normalized" if name == "hellaswag" else "correct"
                    panel[name] = {
                        "counts": {mode: [sum(p[mode][primary] for p in group), len(group)] for mode in modes},
                        "primary": primary,
                        "paired_vs_direct": {
                            mode: compare(group, "direct", mode, primary) for mode in modes if mode != "direct"
                        },
                    }
            for split in RULE_SPLITS:
                paths = {mode: run / mode / f"rule-{split}-k{k}.jsonl" for mode in modes}
                panel[split] = {mode: score(root / "rules", split, path) for mode, path in paths.items()}
                predictions = {mode: by_id(jsonl(path)) for mode, path in paths.items()}
                paired_rule = []
                for target in jsonl(root / "rules" / f"{split}.targets.jsonl"):
                    pair = {}
                    for mode in modes:
                        prediction = predictions[mode][target["id"]]
                        before = prediction["before"] == target["before"]
                        after = prediction["after"] == target["after"]
                        pair[mode] = {
                            "before": prediction["before"], "after": prediction["after"],
                            "before_correct": before, "after_correct": after, "episode_correct": before and after,
                        }
                    paired_rule.append(pair)
                    handle.write(canonical_json({
                        "depth": k, "id": target["id"], "lineage": target["lineage"],
                        "benchmark": f"rule_{split}", "changed": target["before"] != target["after"],
                        "arms": pair,
                    }) + "\n")
                panel[split]["paired_episode_vs_direct"] = {
                    mode: compare(paired_rule, "direct", mode, "episode_correct") for mode in modes if mode != "direct"
                }
            summary["depths"][str(k)] = panel
    diagnostics = {}
    for mode in modes:
        path = run / mode / "diagnostics.json"
        if path.exists():
            diagnostics[mode] = json.loads(path.read_text(encoding="utf-8"))
        elif not completed_depths_only:
            raise ValueError(f"missing dynamics measurements for {mode}")
    dump_structured(dest / "diagnostics.json", diagnostics)
    traces = {"resources.jsonl": run / "resources.jsonl"} | {
        f"train-{mode}.jsonl": run / mode / "train.jsonl" for mode in modes
    }
    summary["trace_sha256"] = {}
    for name, path in traces.items():
        shutil.copyfile(path, dest / name)
        summary["trace_sha256"][name] = sha256_file(dest / name)
    summary["paired_items_sha256"] = sha256_file(items_path)
    summary["diagnostics_sha256"] = sha256_file(dest / "diagnostics.json")
    dump_structured(dest / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--completed-depths-only", action="store_true")
    a = p.parse_args(argv)
    result = aggregate(a.root, a.out, completed_depths_only=a.completed_depths_only)
    print(canonical_json({"depths": list(result["depths"]), "paired_items_sha256": result["paired_items_sha256"]}))


if __name__ == "__main__":
    main()
