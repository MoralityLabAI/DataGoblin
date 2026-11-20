"""Replay the Hermes Campsite repair curriculum with a pinned small Qwen proposer."""

from __future__ import annotations

import importlib
import json
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from datagoblin.hashing import sha256_file, sha256_json, sha256_text
from datagoblin.serialization import load_structured

from .rlt_lean_labels import atomic_record

SOURCE_FILES = ("common.py", "campsite.py", "repair_cells.py", "repair_trm.py", "metta_schema.py")
FRAMING = "atoms"
FRESH_CASES = 36
TRAIN_PUZZLES = 256
MODES = ("direct", "rlt", "matched_damping")
SEEDS = (117, 163)
STEPS = 320
BATCH = 16


def original(study, hermes_lite):
    source = str(study / "src")
    sys.path.insert(0, source)
    common = importlib.import_module("common")
    common.HERMES_LITE = hermes_lite
    modules = [importlib.import_module(name) for name in ("campsite", "repair_cells", "repair_trm")]
    if any(Path(module.__file__).parent != Path(source) for module in (common, *modules)):
        raise ValueError("unexpected Hermes study module")
    return common, *modules


def source_receipt(study, common, campsite, cells):
    return {
        "hermes_skills_head": common.git_head(), "hermes_lite_head": common.git_head(common.HERMES_LITE),
        "committed_operator_sha256": sha256_text(common.git_show(campsite.REPAIR_SCRIPT)),
        "verifier_sha256": sha256_file(Path(campsite.camp().__file__)),
        "study_files": {name: sha256_file(study / "src" / name) for name in SOURCE_FILES},
        "cell_declaration_sha256": sha256_file(cells.PACKAGE),
    }


def stable_key(*parts):
    return sha256_json(["hermes-rlt-repair-v1", *parts])


def record_set(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def prepare(root, study, hermes_lite, guard):
    import numpy as np

    common, campsite, cells, _ = original(study, hermes_lite)
    source = source_receipt(study, common, campsite, cells)
    files = {name: study / "results" / "proposals" / f"{name}.jsonl" for name in ("Qwen2.5-3B", "Bonsai-8B")}
    manifest = root / "prepared.json"
    paths = (root / "candidates.json", root / "training.npz")
    registration = {"source": source, "proposal_digests": {name: sha256_file(file) for name, file in files.items()},
                    "source_sha256": sha256_file(Path(__file__)), "model_framing": FRAMING,
                    "fresh_cases": FRESH_CASES, "training_puzzles": TRAIN_PUZZLES}
    if manifest.exists():
        old = load_structured(manifest)
        if old["registration"] != registration or old["files"] != {p.name: sha256_file(p) for p in paths}:
            raise ValueError("prepared attempt differs from the registered source")
        return
    proposals = {name: record_set(file) for name, file in files.items()}
    guards = ("train", "train2", "val", "test")
    tasks = {}
    for pool in guards:
        guard()
        tasks[pool] = campsite.task_by_id(pool)
    training_ids = ((pool, pid) for pool in guards[:2] for pid in tasks[pool])
    train = sorted(training_ids, key=lambda item: stable_key("training", *item))
    train_ids = set(train[:TRAIN_PUZZLES])
    encodings, counts = [], {}
    for name, rows in proposals.items():
        picked = [r for r in rows if r["k"] == 0 and (r["pool"], r["puzzle_id"]) in train_ids]
        if len({(r["pool"], r["puzzle_id"]) for r in picked}) != TRAIN_PUZZLES or len(picked) != TRAIN_PUZZLES:
            raise ValueError("training proposals must cover the selected puzzle IDs exactly once")
        counts[name] = len(picked)
        for row in picked:
            guard()
            task, _ = tasks[row["pool"]][row["puzzle_id"]]
            cand = cells.normalise(task.grid, row["grid"])
            label = cells.nearest_solution(row["pool"], row["puzzle_id"], cand)
            if not campsite.passes(task, label):
                raise ValueError("unverified training target")
            encodings.append(cells.encode(FRAMING, task.grid, task.row_constraints, task.col_constraints, cand, label))
    pending = paths[1].with_suffix(".part.npz")
    names = ("features", "labels", "loss_mask", "valid")
    np.savez(pending, **{name: np.asarray(items, dtype=np.float32)
                         for name, items in zip(names, zip(*encodings, strict=True), strict=True)})
    pending.replace(paths[1])
    archival = {r["puzzle_id"]: r["grid"] for r in proposals["Qwen2.5-3B"] if r["pool"] == "test" and r["k"] == 0}
    if archival.keys() != tasks["test"].keys() or len(archival) != 148:
        raise ValueError("archival rollout IDs disagree with the October test cohort")
    buckets = {big: sorted((pid for pid, (task, _) in tasks["test"].items() if (len(task.grid) == 6) == big),
                           key=lambda pid: stable_key("fresh-rollout", pid)) for big in (False, True)}
    selected = buckets[False][:24] + buckets[True][:12]
    if len(selected) != FRESH_CASES:
        raise ValueError("selected rollout is missing a size cohort")
    atomic_record(paths[0], {
        "archival": archival, "fresh": {pid: campsite.propose_prompt(tasks["test"][pid][0]) for pid in selected},
        "fresh_budget": {pid: 2 * len(tasks["test"][pid][0].grid) * len(tasks["test"][pid][0].grid[0]) + 16
                         for pid in selected},
    })
    atomic_record(manifest, {"registration": registration, "files": {p.name: sha256_file(p) for p in paths},
                             "counts": counts, "training_targets_verified": len(encodings)})
    print(f"Prepared {len(encodings)} verified training targets and {len(selected)} rollout prompts", flush=True)


def checked_inputs(root):
    receipt = load_structured(root / "prepared.json")
    for name, digest in receipt["files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"prepared input changed: {name}")
    return receipt, load_structured(root / "candidates.json")


def collect_rollouts(root, registration, proposer, guard):
    receipt, candidates = checked_inputs(root)
    protocol = root / "rollout-protocol.json"
    registration.update({"prepared_sha256": sha256_file(root / "prepared.json"), "greedy": True,
                         "rollout_code_sha256": sha256_file(Path(__file__))})
    if protocol.exists() and load_structured(protocol) != registration:
        raise ValueError("rollout protocol changed")
    atomic_record(protocol, registration)
    directory = root / "rollouts"
    directory.mkdir(exist_ok=True)
    prompts = candidates["fresh"]
    remaining = [pid for pid in prompts if not (directory / f"{pid}.json").exists()]
    if remaining:
        guard()
        with proposer() as complete:
            for index, pid in enumerate(remaining, 1):
                guard()
                text = prompts[pid]
                start = time.monotonic()
                raw, tokens = complete(text, candidates["fresh_budget"][pid])
                trimmed = raw.split("\n\n", 1)[0].split("Puzzle:", 1)[0]
                atomic_record(directory / f"{pid}.json", {"id": pid, "prompt_sha256": sha256_text(text),
                              "completion": trimmed, "new_tokens": tokens, "wall_s": time.monotonic() - start})
                if index == 1 or index % 6 == 0:
                    print(f"Collected {index}/{len(remaining)} remaining Qwen rollouts", flush=True)
    rows = {pid: sha256_file(directory / f"{pid}.json") for pid in prompts}
    atomic_record(root / "rollout-receipt.json", {"count": len(rows), "rollouts": rows,
                  "protocol_sha256": sha256_file(protocol), "prepared_sha256": sha256_file(root / "prepared.json"),
                  "training_targets": receipt["training_targets_verified"]})


def rollout(root, model_path, device, guard):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

    paths = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json")
    registration = {"model_files": {name: sha256_file(model_path / name) for name in paths},
                    "model_weights": {file.name: sha256_file(file) for file in model_path.glob("*.safetensors")},
                    "dtype": "bfloat16", "device": device, "transformers": __version__, "torch": str(torch.__version__)}
    if not registration["model_weights"]:
        raise ValueError("no local Qwen weights")

    @contextmanager
    def proposer():
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model, loading = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16, device_map={"": device},
            attn_implementation="sdpa", output_loading_info=True,
        )
        if loading["missing_keys"] or loading["mismatched_keys"]:
            raise ValueError("Qwen checkpoint did not load all required weights")
        model.eval()
        print(f"Loaded local Qwen on {device}", flush=True)

        def complete(text, budget):
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            with torch.inference_mode():
                out = model.generate(
                    input_ids=input_ids, attention_mask=torch.ones_like(input_ids), do_sample=False,
                    max_new_tokens=budget, use_cache=True, pad_token_id=tokenizer.eos_token_id,
                    stop_strings=["\n\n", "Puzzle:"], tokenizer=tokenizer,
                )
            tokens = out[0, input_ids.shape[1]:].tolist()
            return tokenizer.decode(tokens, skip_special_tokens=True), len(tokens)

        yield complete

    collect_rollouts(root, registration, proposer, guard)


def rollout_gguf(root, model_path, server, guard):
    registration = {"model": model_path.name, "model_sha256": sha256_file(model_path), "device": "cpu",
                    "server_sha256": sha256_file(server), "context": 2048, "threads": 1}

    @contextmanager
    def proposer():
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        args = [str(server), "-m", str(model_path), "-ngl", "0", "-t", "1", "-tb", "1", "-c", "2048",
                "-np", "1", "--no-webui", "--alias", model_path.name, "--host", "127.0.0.1", "--port", str(port)]
        with (root / "llama-server.log").open("ab") as log, subprocess.Popen(
            args, cwd=server.parent, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW,
        ) as process:
            try:
                for _ in range(45):
                    guard()
                    if process.poll() is not None:
                        raise RuntimeError("llama-server exited during startup. See llama-server.log.")
                    try:
                        with urlopen(base + "/v1/models", timeout=2) as response:
                            if [item["id"] for item in json.load(response)["data"]] != [model_path.name]:
                                raise RuntimeError("local server loaded a different checkpoint")
                        break
                    except URLError:
                        time.sleep(1)
                else:
                    raise TimeoutError("local llama-server did not load Qwen")

                def complete(prompt, budget):
                    body = {"prompt": prompt, "n_predict": budget, "temperature": 0, "top_k": 1, "seed": 0,
                            "cache_prompt": False, "stop": ["\n\n", "Puzzle:"]}
                    request = Request(base + "/completion", data=json.dumps(body).encode(),
                                      headers={"Content-Type": "application/json"})
                    with urlopen(request, timeout=45) as response:
                        result = json.load(response)
                    return result["content"], result["timings"]["predicted_n"]

                yield complete
            finally:
                if process.poll() is None:
                    process.terminate()

    collect_rollouts(root, registration, proposer, guard)
