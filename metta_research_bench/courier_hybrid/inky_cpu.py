"""Run the preregistered Inky prompts against explicitly supplied offline weights."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from .experiment import file_sha, json_out
from .inky import Reply
from .native import canonical, digest

REGISTRATION = Path(__file__).with_suffix(".json")


def generate(requests: Path, model_path: Path, out: Path) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    config = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if (
        torch.__version__ != config["torch"]
        or transformers.__version__ != config["transformers"]
        or file_sha(model_path / config["weights_file"]) != config["weights_sha256"]
    ):
        raise ValueError("Local inference runtime or weights differ from the auxiliary registration")
    entries = [json.loads(line) for line in requests.read_text(encoding="utf-8").splitlines()]
    if (
        not entries
        or len({entry["request_sha256"] for entry in entries}) != len(entries)
        or any(
            entry["request"]["model"] != config["model"]
            or entry["request"]["max_tokens"] != config["max_new_tokens"]
            or digest(entry["request"]) != entry["request_sha256"]
            for entry in entries
        )
    ):
        raise ValueError("Public requests differ from the registered model or token cap")

    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True, use_safetensors=True
    ).eval()
    python_files = ("config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja")
    json_out(
        out.with_suffix(".lock.json"),
        {
            "registration_sha256": file_sha(REGISTRATION),
            "requests_sha256": file_sha(requests),
            "model_sources": {name: file_sha(model_path / name) for name in python_files},
            **config,
        },
    )
    with torch.inference_mode(), out.open("xb") as handle:
        for entry in entries:
            encoded = tokenizer.apply_chat_template(
                entry["request"]["messages"],
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_length = encoded["input_ids"].shape[-1]
            start = time.perf_counter()
            emitted = model.generate(**encoded, do_sample=False, max_new_tokens=config["max_new_tokens"])
            seconds = time.perf_counter() - start
            continuation = emitted[0, input_length:]
            reply = Reply(
                config["model"],
                entry["request_sha256"],
                tokenizer.decode(continuation, skip_special_tokens=True).strip(),
                seconds,
                input_length,
                len(continuation),
            )
            handle.write(canonical(asdict(reply)) + b"\n")
            handle.flush()
            print(
                f"{entry['skill']} {entry['request_sha256'][:12]}: {len(continuation)} tokens in {seconds:.2f}s",
                flush=True,
            )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--requests", type=Path, required=True)
    args.add_argument("--model-path", type=Path, required=True)
    args.add_argument("--out", type=Path, required=True)
    options = args.parse_args()
    generate(options.requests, options.model_path, options.out)
