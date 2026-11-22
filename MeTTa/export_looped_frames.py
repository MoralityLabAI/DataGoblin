"""Alternate *view* of admitted-candidate SFT rows for LoopedTransformers.

The export matches LoopedTransformers/docs/sft_export.md's instruction/input/
output/metadata layout. It is never a second set of training presentations:
train the canonical DataGoblin JSONL OR this view, and keep shared root IDs.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .collect import sha256_bytes
from .looped_sft import SCHEMA, audit_bundle, canon


def export_frames(bundle: Path, destination: Path) -> dict[str, Any]:
    # A full replay keeps the target generator and content hashes fixed.
    audit = audit_bundle(bundle, full_replay=True)
    if not audit["valid"]:
        raise ValueError("source corpus was not audited")
    if destination.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("adapter output must be distinct from its canonical source")
    roots: Counter[str] = Counter()
    count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (bundle / "train_candidates.jsonl").open(encoding="utf-8") as source, destination.open(
        "w", encoding="utf-8"
    ) as out:
        for line in source:
            row = json.loads(line)
            if row["schema_version"] != SCHEMA or row["training_ready"] is not False:
                raise ValueError("unsupported or prematurely released row")
            roots[row["root"]] += 1
            if roots[row["root"]] > 4:
                raise ValueError("source root presentation cap exceeded")
            input_ = row["messages"][0]["content"]
            output = row["messages"][1]["content"]
            view = {
                "instruction": (
                    "Given the MeTTa curation graph and current evidence selection, "
                    "emit the next bounded edit or STOP."
                ),
                "input": input_,
                "output": output,
                "metadata": {
                    "id": f'{row["root"]}:{row["presentation"]}',
                    "root": row["root"],
                    "presentation": row["presentation"],
                    "family": row["family"],
                    "source": "DataGoblin.MeTTa.looped_sft",
                    "supervision_scope": "synthetic_bounded_python_oracle",
                    "soundness_type": "SIMULATED_NOT_ENVIRONMENT_CERTIFIED",
                    "candidate_release": "DISCOVERY_ONLY",
                    "training_ready": False,
                    "canonical_edits_remaining": row["recurrent_supervision"][
                        "canonical_edits_remaining_before"
                    ],
                    "stop_target": row["recurrent_supervision"]["stop_target"],
                    "metta_query_supervision": row["metta_query_supervision"],
                    "verifier_method": row["verifier_receipt"]["method"],
                },
            }
            out.write(canon(view) + "\n")
            count += 1
    receipt = {
        "schema_version": "metta-looped-frames-view-v1",
        "canonical_source_sha256": sha256_bytes((bundle / "train_candidates.jsonl").read_bytes()),
        "frames_sha256": sha256_bytes(destination.read_bytes()),
        "rows": count,
        "roots": len(roots),
        "max_presentations_per_root": max(roots.values(), default=0),
        "alternative_view_not_extra_presentations": True,
        "training_ready": False,
        "native_metta_verified": False,
    }
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_frames(args.bundle, args.out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
