"""One bounded NDIF activation smoke test. Requires NDIF_API_KEY in the environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from importlib.metadata import version

GPT2 = "openai-community/gpt2"
INKLING = "thinkingmachines/TML-d24-varloop-research"
PROMPT = "The Eiffel Tower is in the city of"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(GPT2, INKLING), default=GPT2)
    parser.add_argument("--expected-passes", type=int, default=3, help="Architectural passes for Inkling (default: 3)")
    parser.add_argument("--revision", default="main", help="Checkpoint revision; pin a SHA for research receipts")
    args = parser.parse_args()

    from nnsight import CONFIG, TransformersModel, is_model_running

    key = os.environ.get("NDIF_API_KEY")
    if not key:
        raise SystemExit("Set NDIF_API_KEY in the environment; credentials are never printed or stored")
    if not is_model_running(args.model, task="text-generation", revision=args.revision):
        raise SystemExit(f"NDIF has no confirmed running deployment for {args.model} at {args.revision}")
    CONFIG.set_default_api_key(key)
    model = TransformersModel(args.model, task="text-generation", revision=args.revision)
    receipt = {
        "model": args.model, "revision": args.revision, "provider": "NDIF",
        "nnsight_version": version("nnsight"), "transformers_version": version("transformers"),
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
    }
    started = time.monotonic()
    if args.model == GPT2:
        with model.trace(PROMPT, remote=True):
            activation = model.transformer.h[3].output[0, -1].cpu().save()
            token = model.output.logits[0, -1].argmax().cpu().save()
        receipt["activation_shape"] = list(activation.shape)
        if receipt["activation_shape"] != [768]:
            raise RuntimeError("unexpected GPT-2 block activation shape")
        receipt["token"] = model.tokenizer.decode([token.item()])
    else:
        from datagoblin.training_gym.rlt_gym.nnsight_probe import ProbePlan, trace_once

        plan = ProbePlan(
            model_id=args.model, revision=args.revision,
            task_digest=receipt["prompt_sha256"], expected_passes=args.expected_passes,
        )
        observed = trace_once(model, PROMPT, plan, remote=True)
        receipt["probe"] = observed
        receipt["token"] = model.tokenizer.decode([observed["readout"]["top_token_id"]])
    receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
