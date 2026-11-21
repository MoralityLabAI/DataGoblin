"""Check remote activation access on a single public prompt through NDIF."""

from __future__ import annotations

import argparse
import json
import os
import time
from getpass import getpass
from importlib.metadata import version

GPT2 = "openai-community/gpt2"
INKLING = "thinkingmachines/TML-d24-varloop-research"
PROMPT = "The Eiffel Tower is in the city of"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(GPT2, INKLING), default=GPT2)
    args = parser.parse_args()

    from nnsight import CONFIG, TransformersModel, is_model_running

    if not is_model_running(args.model, task="text-generation"):
        raise SystemExit(f"NDIF reports no running text-generation deployment for {args.model}")
    CONFIG.API.APIKEY = os.environ.get("NDIF_API_KEY") or getpass("NDIF API key: ")
    model = TransformersModel(args.model, task="text-generation")
    receipt = {
        "model": args.model, "provider": "NDIF", "nnsight_version": version("nnsight"),
        "transformers_version": version("transformers"), "prompt": PROMPT,
    }
    started = time.monotonic()
    if args.model == GPT2:
        with model.trace(PROMPT, remote=True):
            activation = model.transformer.h[3].output[0, -1].cpu().save()
            token = model.output.logits[0, -1].argmax().cpu().save()
        receipt["activation_shape"] = list(activation.shape)
        if receipt["activation_shape"] != [768]:
            raise RuntimeError("unexpected GPT-2 block activation shape")
    else:
        layer = model.model.language_model.layers[17]
        with model.trace(PROMPT, remote=True) as tracer:
            norms = list().save()
            for _ in tracer.iter[:4]:
                norms.append(layer.output[0, -1].float().norm().item())
            token = model.output.logits[0, -1].argmax().cpu().save()
        receipt.update(layer=17, component="active_block", observed_passes=len(norms),
                       pass_observation_cap=4, pass_boundary_l2=norms)
    receipt.update(token=model.tokenizer.decode([token.item()]), elapsed_seconds=round(time.monotonic() - started, 3))
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
