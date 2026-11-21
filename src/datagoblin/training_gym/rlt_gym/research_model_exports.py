"""Convert replayed model choices into train-only demonstrations and repair records."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .hypergame_taste import _file_sha256, _json
from .research_campaign import _write_json, best, case, records
from .research_controller import REGRET_LIMIT, verified
from .research_teacher import messages, replay
from .research_world import Game


def export(archive, output, key):
    receipt = replay(archive, key)
    output.mkdir(parents=True)
    counts = Counter()
    path = output / "model_training.jsonl"
    with path.open("xb") as stream:
        for row in records(archive / "model_rollouts.jsonl"):
            task, sealed = case("train", row["K"], row["ordinal"], 0, key)
            game = Game(task["public"])
            state = game.initial
            for visit, step in enumerate(row["events"]):
                verdict = step["verdict"]
                action = verdict["action"]
                result = messages(task, state, game)
                values = game.values(state)
                if verdict["status"] == "valid":
                    justified = verified(values, action)
                    preferred = action if justified else best(values)
                    if not justified:
                        result[-1]["content"] = _json({
                            "decision_context": json.loads(result[-1]["content"]),
                            "candidate_to_revise": action,
                            "feedback": "The verifier rejected this action under the expected-utility criterion.",
                        })
                    result.append({"role": "assistant", "content": _json({"action": preferred})})
                    kind = "verified_model_demonstration" if justified else "synthetic_repair_of_model_choice"
                    counts[kind] += 1
                    stream.write((_json({
                        "root": row["root"], "presentation": visit, "split": "train", "messages": result,
                        "source_model": row["requested_model"], "evidence_class": kind,
                        "post_choice_expected_regret": verdict["regret"],
                    }) + "\n").encode())
                state, _ = game.outcome(state, action, sealed)
            game._solve.cache_clear()
    _write_json(output / "manifest.json", {
        "schema": "datagoblin.research_model_training.v1",
        "source_sha256": _file_sha256(archive / "model_rollouts.jsonl"),
        "stream_sha256": _file_sha256(path), "rows": dict(counts),
        "max_train_presentations_per_root": 4, "verifier_regret_tolerance": REGRET_LIMIT,
        "archive_replay": receipt,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--private-seed-file", type=Path, required=True)
    args = parser.parse_args()
    key = hashlib.sha256(args.private_seed_file.read_bytes() + b"/model").digest()
    export(args.archive, args.out, key)


if __name__ == "__main__":
    main()
