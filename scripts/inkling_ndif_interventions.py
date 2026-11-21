"""Run NDIF pass-aware ablation controls and save bounded, unverified receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="thinkingmachines/TML-d24-varloop-research")
    parser.add_argument("--revision", required=True, help="Pinned checkpoint revision (prefer an immutable commit SHA)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--task-digest", help="Existing frozen RLT task digest; otherwise prompt SHA-256")
    parser.add_argument("--donor-prompt")
    parser.add_argument("--donor-task-digest")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--pass-index", type=int, default=1, help="Zero-based architectural pass to modify")
    parser.add_argument("--site", default="model.language_model.layers.17")
    parser.add_argument("--features", type=int, nargs="+", default=[2027])
    parser.add_argument("--dose", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-token-id", type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    key = os.environ.get("NDIF_API_KEY")
    if not key:
        raise SystemExit("Set NDIF_API_KEY in the environment; it will not be written to receipts")
    from nnsight import CONFIG, TransformersModel, is_model_running
    from datagoblin.training_gym.rlt_gym.nnsight_probe import ProbePlan, run_control_suite

    plan = ProbePlan(
        model_id=args.model, revision=args.revision,
        task_digest=args.task_digest or _digest(args.prompt),
        site_path=args.site, expected_passes=args.passes,
        intervention_pass=args.pass_index, feature_ids=tuple(args.features),
        dose=args.dose, seed=args.seed, target_token_id=args.target_token_id,
    )
    if args.donor_task_digest and not args.donor_prompt:
        parser.error("--donor-task-digest requires --donor-prompt")
    if not is_model_running(args.model, task="text-generation", revision=args.revision):
        raise SystemExit(f"NDIF has no confirmed running deployment for {args.model} at {args.revision}")
    CONFIG.set_default_api_key(key)
    model = TransformersModel(args.model, task="text-generation", revision=args.revision)
    result = run_control_suite(
        model, args.prompt, plan, remote=True,
        donor_prompt=args.donor_prompt,
        donor_task_digest=(args.donor_task_digest or _digest(args.donor_prompt)) if args.donor_prompt else None,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as file:
        file.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.out), "sha256": result["sha256"],
                      "arms": list(result["arms"]), "verifier_status": result["verifier_status"]}))


if __name__ == "__main__":
    main()
