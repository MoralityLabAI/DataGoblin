"""Stage and audit native Inkling prefill traces for lineage-isolated SAE fitting."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from datagoblin.hashing import sha256_file
from datagoblin.serialization import canonical_json, dump_structured

from .inkling import _completion_prompt, _jsonl, _new_dir, _suite_file, audit_stage
from .runner import read_jsonl
from .tasks import load_suite

WIDTH = 2304
MAX_TOKENS = 1024
MAX_PROMPTS = 64
SHA256 = re.compile(r"[0-9a-f]{64}")
RAW_PREFILL = "canonical_public_context_raw_prefill"
NATIVE_ANSWER = "inkling_answer_only_prefill.v1"


def boundary_sites(passes: int) -> dict[int, str]:
    if type(passes) is not int or not 1 <= passes <= 6:
        raise ValueError("the released variable-loop GGUF supports 1-6 passes")
    return {0: "l_out-5", 1: "l_out-17", **{p: f"l_out-{35 + (p - 2)*12}" for p in range(2, passes + 1)}}


def native_prompt(request: dict) -> str:
    """Render the pinned template's text-only user turn and prefill its answer channel."""
    return (
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
        "<|message_user|><|content_text|>" + _completion_prompt(request)
        + "<|end_message|><|message_model|><|content_text|>"
    )


def stage_traces(
    suite: Path, sources: list[Path], out: Path, *, per_stage: int = 4, completion: bool = False,
) -> dict:
    """Freeze a common prefill for each requested root."""
    if not sources or not 1 <= per_stage <= MAX_PROMPTS // len(sources):
        raise ValueError("invalid trace prompt budget")
    rows = []
    roots = set()
    suite_path = _suite_file(suite)
    suite_sha = sha256_file(suite_path)
    _, suite_tasks = load_suite(suite_path)
    tasks = {task["task_id"]: task for task in suite_tasks}
    for source in sources:
        manifest = audit_stage(source)
        if manifest["split"] not in {"train", "calibration"}:
            raise ValueError("trace staging requires a training or calibration split")
        if manifest["suite_manifest_sha256"] != suite_sha:
            raise ValueError("source stage does not belong to the declared suite")
        first_pass = manifest["passes"][0]
        requests = read_jsonl(source / f"p{first_pass:02d}" / "requests.jsonl")
        for request in requests[:per_stage]:
            task = tasks[request["task_id"]]
            if task["split"] != manifest["split"] or task["task_digest"] != request["task_digest"]:
                raise ValueError("prompt is outside its declared split or task binding")
            content = native_prompt(request) if completion else canonical_json(request["context"])
            rows.append(({
                "sample_key": f"{len(rows):03d}", "split": manifest["split"],
                "task_id": request["task_id"], "task_digest": request["task_digest"],
                "family": task["family"],
                "lineage_id": task["lineage_id"],
            }, content.encode("utf-8")))
            if task["lineage_id"] in roots:
                raise ValueError("duplicate trace lineage")
            roots.add(task["lineage_id"])
    _new_dir(out)
    samples = []
    for row, content in rows:
        path = out / f"{row['sample_key']}.txt"
        path.write_bytes(content)
        samples.append({**row, "input_sha256": sha256_file(path)})
    report = {
        "schema_version": "rlt_gym.inkling_trace_prompts.v1",
        "input_format": NATIVE_ANSWER if completion else RAW_PREFILL,
        "samples": samples,
        "suite_manifest_sha256": suite_sha,
        "source_stage_sha256": [sha256_file(source / "manifest.json") for source in sources],
    }
    dump_structured(out / "manifest.json", report)
    return report


def seal_trace(
    prompts: Path, trace: Path, *, passes: int, weights_sha256: str, runtime_sha256: str,
) -> dict:
    """Check exact prompt, token and site coverage, then seal the bounded raw matrix."""
    sites = boundary_sites(passes)
    if any(SHA256.fullmatch(value) is None for value in (weights_sha256, runtime_sha256)):
        raise ValueError("weights and runtime SHA-256 must be supplied")
    prompt_path = prompts / "manifest.json"
    plan = json.loads(prompt_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "rlt_gym.inkling_trace_prompts.v1":
        raise ValueError("invalid trace prompt schema")
    samples = plan["samples"]
    keys = {row["sample_key"] for row in samples}
    if not 1 <= len(samples) <= MAX_PROMPTS or len(keys) != len(samples):
        raise ValueError("invalid or repeated prompt keys")
    if (len({row["lineage_id"] for row in samples}) != len(samples)
            or any(row["split"] not in {"train", "calibration"} for row in samples)):
        raise ValueError("invalid lineage custody in trace prompts")
    if any(not re.fullmatch(r"[0-9]{3}", row["sample_key"]) or
           sha256_file(prompts / f"{row['sample_key']}.txt") != row["input_sha256"] for row in samples):
        raise ValueError("prompt file changed or has an invalid key")
    covered: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    rows = 0
    index = trace / "rows.tsv"
    with index.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            key, depth, token = row["sample_key"], int(row["pass_index"]), int(row["token_index"])
            if key not in keys or sites.get(depth) != row["tensor"] or not 0 <= token < MAX_TOKENS:
                raise ValueError("trace row does not match the declared prompt or pass boundary")
            seen = covered[key][depth]
            if token in seen:
                raise ValueError("duplicate token/site trace row")
            seen.add(token)
            rows += 1
    if covered.keys() != keys:
        raise ValueError("trace did not cover every prompt")
    counts = {}
    for key, found in covered.items():
        count = len(found[0])
        positions = set(range(count))
        if not count or found.keys() != sites.keys() or any(value != positions for value in found.values()):
            raise ValueError("a trace lacks contiguous matched token positions at every pass")
        counts[key] = count
    data = trace / "activations.f32"
    if data.stat().st_size != 4 * WIDTH * rows or int((trace / "native-complete.txt").read_text()) != rows:
        raise ValueError("native completion and activation sizes disagree")
    activations = np.memmap(data, mode="r", dtype="<f4", shape=(rows, WIDTH))
    for start in range(0, rows, 256):
        if not np.isfinite(activations[start:start + 256]).all():
            raise ValueError("native trace contains a nonfinite activation")
    report = {
        "schema_version": "rlt_gym.inkling_boundary_trace.v1", "rows": rows, "width": WIDTH,
        "dtype": "float32-le", "passes": passes, "sites": sites, "tokens_by_sample": counts,
        "source_prompt_manifest_sha256": sha256_file(prompt_path), "input_format": plan["input_format"],
        "weights_sha256": weights_sha256, "runtime_sha256": runtime_sha256,
        "files": {file.name: sha256_file(file) for file in (data, index, trace / "native-complete.txt")},
        "intervention_applied": False, "bpd_mechanism_verified": False,
    }
    output = trace / "manifest.json"
    if output.exists():
        raise ValueError("the trace was already sealed")
    dump_structured(output, report)
    return report


def export_sae_inputs(prompts: Path, trace: Path, out: Path, *, max_rows: int = 4096) -> dict:
    """Export lineage-disjoint matrices consumable by the VPD top-k SAE trainer."""
    if type(max_rows) is not int or not 1 <= max_rows <= 8192:
        raise ValueError("invalid per-split SAE row budget")
    prompt_path = prompts / "manifest.json"
    manifest_path = trace / "manifest.json"
    sealed = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = ("activations.f32", "rows.tsv", "native-complete.txt")
    if (sealed.get("schema_version") != "rlt_gym.inkling_boundary_trace.v1"
            or sealed["source_prompt_manifest_sha256"] != sha256_file(prompt_path)
            or any(sealed["files"][name] != sha256_file(trace / name) for name in files)):
        raise ValueError("trace files no longer match the sealed receipt")
    plan = json.loads(prompt_path.read_text(encoding="utf-8"))
    samples = {row["sample_key"]: row for row in plan["samples"]}
    selected = defaultdict(list)
    with (trace / "rows.tsv").open(encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream, delimiter="\t")):
            if row["pass_index"] == "0":
                continue
            sample = samples[row["sample_key"]]
            selected[sample["split"]].append((index, {
                **sample, "pass_index": int(row["pass_index"]), "tensor": row["tensor"],
                "token_index": int(row["token_index"]),
            }))
    data = np.memmap(trace / "activations.f32", mode="r", dtype="<f4", shape=(sealed["rows"], WIDTH))
    _new_dir(out)
    streams = {}
    for split, entries in sorted(selected.items()):
        offsets = np.linspace(0, len(entries) - 1, min(max_rows, len(entries)), dtype=np.int64)
        rows = [entries[i] for i in offsets]
        matrix = data[[index for index, _ in rows]]
        activation_path = out / f"{split}.npz"
        row_path = out / f"{split}.jsonl"
        np.savez(activation_path, inkling_core_prefill=matrix)
        _jsonl(row_path, [meta for _, meta in rows])
        streams[split] = {
            "rows": len(rows), "roots": len({meta["lineage_id"] for _, meta in rows}),
            "activation_sha256": sha256_file(activation_path), "rows_sha256": sha256_file(row_path),
        }
    report = {
        "schema_version": "rlt_gym.inkling_sae_inputs.v1",
        "key": "inkling_core_prefill", "trace_manifest_sha256": sha256_file(manifest_path),
        "weights_sha256": sealed["weights_sha256"], "streams": streams,
        "fit_split": "train", "heldout_split": "calibration",
    }
    dump_structured(out / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage")
    stage.add_argument("--suite", type=Path, required=True)
    stage.add_argument("--stage", type=Path, nargs="+", required=True)
    stage.add_argument("--out", type=Path, required=True)
    stage.add_argument("--per-stage", type=int, default=4)
    stage.add_argument("--completion", action="store_true")
    seal = commands.add_parser("seal")
    seal.add_argument("--prompts", type=Path, required=True)
    seal.add_argument("--trace", type=Path, required=True)
    seal.add_argument("--passes", type=int, required=True)
    seal.add_argument("--weights-sha256", required=True)
    seal.add_argument("--runtime-sha256", required=True)
    export = commands.add_parser("export-sae")
    export.add_argument("--prompts", type=Path, required=True)
    export.add_argument("--trace", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--max-rows", type=int, default=4096)
    args = parser.parse_args()
    if args.command == "stage":
        result = stage_traces(args.suite, args.stage, args.out, per_stage=args.per_stage, completion=args.completion)
    elif args.command == "seal":
        result = seal_trace(args.prompts, args.trace, passes=args.passes,
                            weights_sha256=args.weights_sha256, runtime_sha256=args.runtime_sha256)
    else:
        result = export_sae_inputs(args.prompts, args.trace, args.out, max_rows=args.max_rows)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
