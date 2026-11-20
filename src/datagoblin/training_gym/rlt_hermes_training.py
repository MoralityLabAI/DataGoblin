"""Rebuild the verified train-only portion of the pinned Hermes repair pilot."""

from __future__ import annotations

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import canonical_json, load_structured

from . import rlt_hermes_repair as pilot
from .rlt_lean_labels import atomic_record

SCHEMA = "datagoblin.campsite-repair-training.v1"
RULES = (
    "Keep every tree fixed. Pair each tree with exactly one orthogonally adjacent tent. "
    "Tents cannot share an edge or a corner. Match the given row and column tent totals. "
    "In the proposal, M marks a missing or invalid cell. Return a full grid using T, C, and X."
)
ARRAYS = ("features", "labels", "loss_mask", "valid")
TRAIN_POOLS = ("train", "train2")


def collect_row(rows, receipts, puzzle, proposal_grid, target, source, source_pass):
    root = sha256_json(["campsite", puzzle])
    inputs = {**puzzle, "proposed_grid": proposal_grid}
    record_id = sha256_json([root, inputs])
    row = {"id": record_id, "root": root, "split": "train", "task": "campsite_grid_repair",
           "input": inputs, "target": {"repaired_grid": target}}
    if record_id in rows:
        if rows[record_id] != row:
            raise ValueError("conflicting verified targets for an identical repair input")
    else:
        rows[record_id] = row
        receipts[record_id] = {"id": record_id, "root": root, "input_sha256": sha256_json(inputs),
                               "target_sha256": sha256_json(row["target"]), "official_pass": True,
                               "sources": []}
    receipts[record_id]["sources"].append({**source, "original_proposal_pass": source_pass})


def puzzle_input(task):
    return {"tree_grid": task.grid, "row_tent_totals": task.row_constraints, "col_tent_totals": task.col_constraints}


def export(run, out, study, hermes_lite, guard):
    import numpy as np

    prepared_path = run / "prepared.json"
    prepared = load_structured(prepared_path)
    registration = prepared["registration"]
    train_path = run / "training.npz"
    if sha256_file(train_path) != prepared["files"]["training.npz"] or registration["model_framing"] != pilot.FRAMING:
        raise ValueError("pinned training tensors or framing changed")
    common, campsite, cells, _ = pilot.original(study, hermes_lite)
    source = pilot.source_receipt(study, common, campsite, cells)
    if source != registration["source"]:
        raise ValueError("the upstream training source or independent verifier changed")

    tasks = {pool: campsite.task_by_id(pool) for pool in TRAIN_POOLS}
    available = ((pool, pid) for pool in TRAIN_POOLS for pid in tasks[pool])
    selected = set(sorted(available, key=lambda item: pilot.stable_key("training", *item))[
                   :registration["training_puzzles"]])
    roots = {sha256_json(["campsite", puzzle_input(tasks[pool][pid][0])]) for pool, pid in selected}
    if len(roots) != registration["training_puzzles"]:
        raise ValueError("the prepared selection repeats or lacks train roots")
    for pool in campsite.POOLS.keys() - set(TRAIN_POOLS):
        guard()
        reserved = {sha256_json(["campsite", puzzle_input(task)]) for task, _ in campsite.task_by_id(pool).values()}
        if roots & reserved:
            raise ValueError(f"a train root occurs in the reserved {pool} pool")

    rows, receipts, index = {}, {}, 0
    # Match the original preparation order. The receipt sorts proposer keys alphabetically.
    proposers = ("Qwen2.5-3B", "Bonsai-8B")
    with np.load(train_path, allow_pickle=False) as archive:
        expected = [archive[name] for name in ARRAYS]
        if not all(len(array) == prepared["training_targets_verified"] for array in expected):
            raise ValueError("train tensor count differs from the verifier receipt")
        for proposer in proposers:
            path = study / "results" / "proposals" / f"{proposer}.jsonl"
            if sha256_file(path) != registration["proposal_digests"][proposer]:
                raise ValueError("the pinned proposer archive changed")
            seen = set()
            for row in pilot.record_set(path):
                key = (row["pool"], row["puzzle_id"])
                if row["k"] != 0 or key not in selected:
                    continue
                guard()
                if key in seen:
                    raise ValueError("duplicate original proposer row")
                seen.add(key)
                task, _ = tasks[key[0]][key[1]]
                candidate = cells.normalise(task.grid, row["grid"])
                target = cells.nearest_solution(*key, candidate)
                if not campsite.passes(task, target):
                    raise ValueError("the training target failed independent replay")
                encoded = cells.encode(pilot.FRAMING, task.grid, task.row_constraints, task.col_constraints,
                                       candidate, target)
                if index >= len(expected[0]) or any(
                    not np.array_equal(np.asarray(values, dtype=np.float32), array[index])
                    for values, array in zip(encoded, expected, strict=True)
                ):
                    raise ValueError("reconstructed training row differs from the pinned tensor row")
                original = {"proposer": proposer, "pool": key[0], "puzzle_id": key[1],
                            "source_row_sha256": sha256_json(row), "tensor_row": index}
                collect_row(rows, receipts, puzzle_input(task), candidate, target, original,
                            campsite.passes(task, row["grid"]))
                index += 1
            if seen != selected or len(seen) != prepared["counts"][proposer]:
                raise ValueError("the proposer archive does not cover the frozen training roots")
    if index != prepared["training_targets_verified"]:
        raise ValueError("reconstructed training count differs from the verifier receipt")

    out.mkdir(parents=True, exist_ok=False)
    for name, records in (("train.jsonl", rows), ("train.receipts.jsonl", receipts)):
        with (out / name).open("x", encoding="utf-8", newline="\n") as stream:
            for row in records.values():
                stream.write(canonical_json(row) + "\n")
    atomic_record(out / "manifest.json", {
        "schema": SCHEMA, "split": "train", "task_rules": RULES, "objective": "response_only_supervision",
        "records": len(rows), "roots": len(roots), "verified_source_rows": index,
        "train_pools": list(TRAIN_POOLS), "files": {
            name: sha256_file(out / name) for name in ("train.jsonl", "train.receipts.jsonl")
        },
        "prepared_sha256": sha256_file(prepared_path), "training_tensors_sha256": sha256_file(train_path),
        "proposal_digests": registration["proposal_digests"], "verifier_source": source,
    })
    return len(rows)
