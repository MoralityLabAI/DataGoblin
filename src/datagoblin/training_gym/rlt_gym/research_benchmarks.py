"""Read evaluation-only benchmark releases and format the separate synthetic training data."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .hypergame_taste import _digest, _file_sha256, _json
from .research_campaign import FORMATS, _write_json, records

EXTERNAL = {
    "tastebench": {"license": "CC-BY-4.0", "revision": "v1.0", "label_kind": "paired_trajectory_outcome"},
    "rtaste": {"license": "CC-BY-4.0", "label_kind": "independent_rerun_and_model_audit"},
}


def tastebench_rows(source):
    """Accept a user-authorized research JSONL export. Emit both published orders."""
    seen = set()
    for row in records(source):
        if row["domain"] != "research":
            raise ValueError("the research adapter accepts only the research configuration")
        uid = row["id"]
        answer = row["answer"]
        choices = row["choices"]
        if (uid in seen or len(choices) != 2 or answer not in ("A", "B")
                or row["answer_index"] != "AB".index(answer)):
            raise ValueError("invalid or repeated Taste-Bench item")
        seen.add(uid)
        for order, options in enumerate((choices, list(reversed(choices)))):
            yield {
                "root": row.get("task_id") or uid, "item": uid, "order": order, "split": "external_test",
                "input": {"task": row["query"], "visible_history": row["prefix_text"], "choices": options},
                "target": {"answer": answer if order == 0 else "BA"["AB".index(answer)]},
            }


def _optional_boolean(text):
    if text not in ("", "True", "False"):
        raise ValueError("expected a released rTaste boolean")
    return None if text == "" else text == "True"


def _optional_number(text):
    if text == "":
        return None
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("expected a finite rTaste number")
    return value


def rtaste_rows(source):
    """Group verified outcomes by registered condition, preserving unknown results."""
    cohorts = defaultdict(list)
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            cohorts[row["seed"], row["condition"]].append(row)
    for (seed, condition), rows in sorted(cohorts.items()):
        rows.sort(key=lambda row: int(row["mve"]))
        if [int(row["mve"]) for row in rows] != [1, 2, 3, 4, 5]:
            raise ValueError("an rTaste cohort must contain each registered MVE exactly once")
        yield {
            "root": seed, "item": condition, "split": "external_test",
            "input": {"proposals": [{k: row[k] for k in (
                "mve", "title", "relation", "threshold", "event_form", "expected_minutes", "needs_gpu",
            )} for row in rows]},
            "target": {"outcomes": [{
                "mve": row["mve"], "event": _optional_boolean(row["event"]),
                "sound": _optional_boolean(row["sound"]), "rank": _optional_number(row["mean_value_rank"]),
                "value": _optional_number(row["research_value"]),
            } for row in rows]},
        }


def import_evaluation(name, source, output, source_revision, expected_sha256):
    actual = _file_sha256(source)
    if actual != expected_sha256 or not source_revision:
        raise ValueError("external source digest or revision differs")
    details = EXTERNAL[name]
    if details.get("revision", source_revision) != source_revision:
        raise ValueError("unexpected external benchmark revision")
    read_rows = {"tastebench": tastebench_rows, "rtaste": rtaste_rows}[name]
    output.mkdir(parents=True)
    paths = ("eval_inputs.jsonl", "eval_labels.jsonl")
    count = 0
    with (output / paths[0]).open("xb") as inputs, (output / paths[1]).open("xb") as labels:
        for row in read_rows(source):
            uid = _digest([name, row["root"], row["item"], row.get("order")])
            identity = {"id": uid, "root": row["root"], "split": row["split"], "source": name}
            inputs.write((_json({**identity, "input": row["input"]}) + "\n").encode())
            labels.write((_json({**identity, "target": row["target"]}) + "\n").encode())
            count += 1
    if not count:
        raise ValueError("empty evaluation import")
    _write_json(output / "evaluation_manifest.json", {
        **details, "source": name, "revision": source_revision, "source_sha256": actual,
        "training_eligible": False, "rows": count, "artifacts": {n: _file_sha256(output / n) for n in paths},
    })


def format_training(archive, output):
    """Render the six capped synthetic exports as chat records with data lineage."""
    manifest = json.loads((archive / "manifest.json").read_bytes())
    ledger = archive / "training_ledger.jsonl"
    if _file_sha256(ledger) != manifest["artifacts"][ledger.name]["sha256"]:
        raise ValueError("training ledger digest differs")
    index = {}
    by_root = defaultdict(int)
    for row in records(ledger):
        item = row["root"], row["presentation"]
        by_root[item[0]] += 1
        if item in index or by_root[item[0]] > 4 or row["format"] not in FORMATS:
            raise ValueError("training presentation custody differs")
        index[item] = row["format"]
    output.mkdir(parents=True)
    for name in FORMATS:
        original = archive / f"{name}.jsonl"
        if _file_sha256(original) != manifest["artifacts"][original.name]["sha256"]:
            raise ValueError("training stream digest differs")
        with (output / original.name).open("xb") as stream:
            for row in records(original):
                identity = row["root"], row["presentation"]
                if (row["split"] != "train" or row["evidence_class"] != "synthetic_exact_reference"
                        or index.pop(identity, None) != name):
                    raise ValueError("unexpected training record")
                chat = [
                    {"role": "system", "content": f"Return a JSON record for the {name} research task."},
                    {"role": "user", "content": _json(row["input"])},
                    {"role": "assistant", "content": _json(row["target"])},
                ]
                lineage = {k: row[k] for k in ("root", "presentation", "split", "evidence_class")}
                stream.write((_json({"messages": chat, "task_kind": name, **lineage}) + "\n").encode())
    if index:
        raise ValueError("training records missing from exports")
    _write_json(output / "manifest.json", {
        "schema": "datagoblin.research_chat.v1", "source_manifest_sha256": _file_sha256(archive / "manifest.json"),
        "training_roots": len(by_root), "max_train_presentations_per_root": max(by_root.values(), default=0),
        "streams": {f"{name}.jsonl": _file_sha256(output / f"{name}.jsonl") for name in FORMATS},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    external = commands.add_parser("external")
    external.add_argument("--name", choices=EXTERNAL, required=True)
    external.add_argument("--source", type=Path, required=True)
    external.add_argument("--revision", required=True)
    external.add_argument("--sha256", required=True)
    external.add_argument("--out", type=Path, required=True)
    synthetic = commands.add_parser("format-training")
    synthetic.add_argument("--archive", type=Path, required=True)
    synthetic.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "external":
        import_evaluation(args.name, args.source, args.out, args.revision, args.sha256)
    else:
        format_training(args.archive, args.out)


if __name__ == "__main__":
    main()
