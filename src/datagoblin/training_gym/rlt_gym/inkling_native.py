"""Convert native Q4 completions to the independently checked Inkling capture format."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured

from .inkling import MAX_RESPONSE_BYTES, _batch, _decode_candidate, _jsonl, _manifest, _new_dir, audit_stage
from .inkling_trace import MAX_PROMPTS, MAX_TOKENS, NATIVE_ANSWER, SHA256, boundary_sites, native_prompt
from .runner import read_jsonl


def _finished_rows(path: Path) -> list[dict]:
    lines = path.read_bytes().splitlines(keepends=True)
    # The executable publishes each completed prompt as a flushed TSV record.
    return list(csv.DictReader((line.decode("utf-8") for line in lines if line.endswith(b"\n")), delimiter="\t"))


def import_native(
    stage: Path, prompts: Path, run: Path, out: Path, *, passes: int, weights_revision: str,
    weights_sha256: str, runtime_sha256: str, program_sha256: str,
) -> dict:
    """Import the durable completed prefix from a guarded native run."""
    boundary_sites(passes)
    if (not weights_revision or any(SHA256.fullmatch(value) is None
            for value in (weights_sha256, runtime_sha256, program_sha256))):
        raise ValueError("native checkpoint and runtime provenance is required")
    root = audit_stage(stage)
    _, batch = _batch(stage, passes)
    prompt_manifest = prompts / "manifest.json"
    plan = _manifest(prompt_manifest)
    if (plan.get("schema_version") != "rlt_gym.inkling_trace_prompts.v1"
            or plan["input_format"] != NATIVE_ANSWER
            or plan["suite_manifest_sha256"] != root["suite_manifest_sha256"]
            or sha256_file(stage / "manifest.json") not in plan["source_stage_sha256"]):
        raise ValueError("native prompt plan does not match the staged request batch")
    native_config = run / "native-config.tsv"
    settings = _finished_rows(native_config)
    if len(settings) != 1 or int(settings[0]["passes"]) != passes or int(settings[0]["context"]) != MAX_TOKENS:
        raise ValueError("native runtime did not report the requested depth and context")
    budget = int(settings[0]["max_new_tokens"])
    if not 1 <= budget <= 384:
        raise ValueError("invalid native generation budget")

    samples = plan["samples"]
    keys = [row["sample_key"] for row in samples]
    if not 1 <= len(keys) <= MAX_PROMPTS or keys != [f"{i:03d}" for i in range(len(keys))]:
        raise ValueError("native prompt keys must be consecutive and bounded")
    requests = read_jsonl(stage / f"p{passes:02d}" / "requests.jsonl")
    selected = [row for row in samples if row["split"] == root["split"]]
    if not selected or len(selected) > len(requests):
        raise ValueError("no matching native prompts for this split")
    progress = run / "completed.tsv"
    completed = _finished_rows(progress)
    if [row["sample_key"] for row in completed] != keys[:len(completed)]:
        raise ValueError("native completions do not form an ordered prompt prefix")
    if (run / "native-complete.txt").exists() and len(completed) != len(samples):
        raise ValueError("native completion marker lacks full prompt coverage")
    by_key = {row["sample_key"]: row for row in completed}

    observations, replies = [], []
    for sample, request in zip(selected, requests, strict=False):
        key = sample["sample_key"]
        prompt = prompts / f"{key}.txt"
        if (sample["task_id"] != request["task_id"] or sample["task_digest"] != request["task_digest"]
                or sha256_file(prompt) != sample["input_sha256"]
                or prompt.read_bytes() != native_prompt(request).encode("utf-8")):
            raise ValueError("native prompt differs from its original task")
        observed = by_key.get(key)
        if observed is None:
            continue
        n_prompt = int(observed["prompt_tokens"])
        n_new = int(observed["generated_tokens"])
        elapsed = int(observed["elapsed_ms"])
        finish = observed["finish_reason"]
        if (not 1 <= n_prompt <= MAX_TOKENS - budget or not 1 <= n_new <= budget or elapsed < 0
                or finish not in {"stop", "length"} or (finish == "length" and n_new != budget)):
            raise ValueError("native completion has invalid usage or finish status")
        reply_path = run / f"{key}.reply"
        if reply_path.stat().st_size > MAX_RESPONSE_BYTES:
            raise ValueError("oversized native reply")
        try:
            raw = reply_path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            raw = ""
        parsed, candidate, steps = _decode_candidate(raw)
        identity = {field: request[field] for field in ("request_id", "task_digest")}
        usage = {"input_tokens": n_prompt, "output_tokens": n_new}
        observations.append({
            **identity, "usage": usage, "returned_model": root["model"], "content": raw, "parsed": parsed,
            "finish_reason": finish, "latency_seconds": elapsed / 1000, "raw_sha256": sha256_file(reply_path),
        })
        reply = {**identity, "usage": usage}
        if parsed:
            reply.update(candidate=candidate, public_steps=steps)
        replies.append(reply)

    _new_dir(out)
    _jsonl(out / "responses.jsonl", replies)
    _jsonl(out / "observations.jsonl", observations)
    result = {
        "schema_version": "rlt_gym.inkling_capture.v2", "source_kind": "local_native_declared", "passes": passes,
        "server_loop_passes_operator_declared": passes, "model_requested": root["model"],
        "share_cache_operator_declared": False, "server_configuration_verified": False,
        "runtime_operator_declared": "llama_cpp", "weights_revision_operator_declared": weights_revision,
        "temperature": 0, "max_tokens": budget, "native_input_format": NATIVE_ANSWER,
        "native_program_sha256": program_sha256, "native_runtime_sha256": runtime_sha256,
        "native_weights_sha256": weights_sha256, "native_config_sha256": sha256_file(native_config),
        "native_prompt_manifest_sha256": sha256_file(prompt_manifest),
        "native_progress_sha256": sha256_file(progress),
        "requests_sha256": batch["requests_sha256"], "requests_planned": len(selected),
        "network_calls": 0, "native_completions": len(observations), "responses": len(replies),
        "responses_sha256": sha256_file(out / "responses.jsonl"),
        "observations_sha256": sha256_file(out / "observations.jsonl"),
        "elapsed_seconds": sum(row["latency_seconds"] for row in observations),
    }
    dump_structured(out / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage", "prompts", "run", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--passes", type=int, required=True)
    parser.add_argument("--weights-revision", required=True)
    for name in ("weights", "runtime", "program"):
        parser.add_argument(f"--{name}-sha256", required=True)
    print(json.dumps(import_native(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
