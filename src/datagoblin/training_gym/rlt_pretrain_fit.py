"""Train matched placements in the shared text, Lean and revision cell."""

from __future__ import annotations

import shutil
import time
from collections import deque
from pathlib import Path

import autograd.numpy as anp
import numpy as np
from autograd import value_and_grad

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import load_structured

from . import rlt_lean_grounded
from . import rlt_pretrain_cell as cell
from . import rlt_pretrain_data as data
from .rlt_lean_labels import atomic_record
from .rlt_lean_train_rank import features, masked_logsumexp, vocabulary
from .rlt_pilot import restore_optimizer, save_optimizer, target_nll
from .rlt_pilot_data import initialization_digest
from .rlt_revision import NODES

SPEC = {
    "seeds": (137, 163), "width": 20, "depth": 6, "steps": 192, "learning_rate": 0.004,
    "lanes": ("text", "lean", "revision"), "lean_availability_weight": 0.15,
}
RECIPES = {
    "mixed": {"batch_examples": (3, 4, 3), "phases": ((0.5, 0.25, 0.25),)},
    "text_first": {
        "batch_examples": (3, 4, 3),
        "phases": ((0.75, 0.125, 0.125), (0.5, 0.25, 0.25), (0.25, 0.375, 0.375)),
    },
    "text_heavier": {"batch_examples": (5, 2, 3), "phases": ((0.5, 0.25, 0.25),)},
}
NODE_ID = {node: index for index, node in enumerate(NODES)}


def frozen_inputs(root):
    manifest = load_structured(root / "dataset.json")
    if manifest["schema"] != "datagoblin.mixed-pretrain-pilot.v1":
        raise ValueError("unexpected mixed pilot dataset")
    for file, digest in manifest["files"].items():
        path = root / file
        if not path.resolve().is_relative_to(root.resolve()) or sha256_file(path) != digest:
            raise ValueError(f"mixed dataset member changed: {path}")
    return manifest


def lean_words():
    entries = [
        {"name": module.split("/")[0] + "." + name}
        for module, names in rlt_lean_grounded.LIBRARY.items() for name in names
    ]
    return vocabulary(entries)


def proof_examples(root, split):
    inputs = root / f"lean-{split}.public.jsonl"
    receipts = root / f"lean-{split}.receipts.jsonl"
    for row, grades in zip(data.json_lines(inputs), data.json_lines(receipts), strict=True):
        if (row["id"], row["epoch"]) != (grades["id"], grades["epoch"]):
            raise ValueError("Lean example and verifier receipt differ")
        lookup = {candidate["id"]: candidate["tactic"] for candidate in row["candidates"]}
        ids = row["orders"]["compose"]
        x = features(row["context"], [lookup[k] for k in ids], row["retrieved"]["compose"], lean_words())
        y = np.array([-1 if grades["outcomes"][k]["label"] is None else grades["outcomes"][k]["label"] for k in ids])
        yield row, grades, x, y


def pad_proofs(rows):
    size, dim = max(len(x) for _, _, x, _ in rows), rows[0][2].shape[1]
    encoded = np.zeros((len(rows), size, dim))
    present = np.zeros((len(rows), size), dtype=bool)
    labels = np.full((len(rows), size), -1)
    for i, (_, _, x, y) in enumerate(rows):
        encoded[i, : len(x)] = x
        present[i, : len(x)] = True
        labels[i, : len(y)] = y
    return encoded, present, labels


def route_input(task, graph, epoch):
    edges = np.zeros((len(NODES), len(NODES)))
    for source, destination in graph.items():
        edges[NODE_ID[destination], NODE_ID[source]] = 1.0
    x = np.zeros((len(NODES), cell.ROUTE_FEATURES))
    x[NODE_ID[task["start"]], 0] = 1
    x[:, 1] = task["steps"] / 10
    x[:, 2] = epoch
    return x, edges


def revision_examples(root, split):
    for task, expected in data.rule_rows(root / "revision", split):
        graph = dict(task["links"])
        patch = task["patch"]
        after = graph | {patch["source"]: patch["destination"]}
        x0, edges = route_input(task, graph, 0)
        x1, altered = route_input(task, after, 1)
        yield task, expected, (x0, edges, x1, altered, NODE_ID[expected["before"]], NODE_ID[expected["after"]])


def train_data(root):
    examples = np.array([row["tokens"] for row in data.json_lines(root / "text-train.jsonl")])
    proof = list(proof_examples(root, "train"))
    routes = list(revision_examples(root, "train"))
    text = (examples[:, :-1], examples[:, 1:])
    graph = tuple(np.stack(x) for x in zip(*(model_input for _, _, model_input in routes), strict=True))
    return text, pad_proofs(proof), graph


def lean_loss(logits, present, labels):
    known = (labels >= 0) & present
    positive = labels == 1
    has_proof = anp.any(positive, axis=1)
    listwise = masked_logsumexp(logits, known) - masked_logsumexp(logits, positive)
    rank_loss = anp.sum(anp.where(has_proof, listwise, 0)) / anp.maximum(1, anp.sum(has_proof))
    know_availability = has_proof | anp.all(known | ~present, axis=1)
    peak = anp.max(anp.where(present, logits, -10000), axis=1)
    binary = anp.logaddexp(0.0, peak) - peak * has_proof
    route_loss = anp.sum(anp.where(know_availability, binary, 0)) / anp.maximum(1, anp.sum(know_availability))
    return rank_loss + SPEC["lean_availability_weight"] * route_loss


def losses(weights, batches, arm, depth):
    (tokens, targets), (encoded, present, labels), (x0, edges, x1, altered, before, after) = batches
    text_loss = 0.0
    for logits in cell.text_visits(weights, tokens, np.ones(tokens.shape, dtype=bool), arm, depth):
        text_loss += anp.mean(target_nll(logits.reshape(-1, cell.VOCAB), targets.ravel())) / depth
    proof_loss = 0.0
    for logits in cell.lean_visits(weights, encoded, present, arm, depth):
        proof_loss += lean_loss(logits, present, labels) / depth
    first, memory = deque(cell.route_visits(weights, x0, edges, arm, depth), maxlen=1).pop()
    visits = cell.route_visits(weights, x1, altered, arm, depth, prior_answer=memory)
    second, _ = deque(visits, maxlen=1).pop()
    revision_loss = anp.mean(target_nll(first, before) + target_nll(second, after)) / 2
    return anp.stack((text_loss, proof_loss, revision_loss))


def objective(weights, batches, arm, depth, loss_weights):
    return anp.dot(losses(weights, batches, arm, depth), anp.array(loss_weights))


def recipe_weights(recipe, step, steps):
    phases = RECIPES[recipe]["phases"]
    return phases[(step - 1) * len(phases) // steps]


def fit_arm(out, arrays, arm, seed, plans, guard, *, steps=SPEC["steps"], recipe="mixed"):
    out.mkdir(parents=True, exist_ok=True)
    weights = cell.initialize(seed, SPEC["width"], arrays[1][0].shape[-1])
    init_hash = initialization_digest(weights)
    plan_hash = initialization_digest(dict(zip(SPEC["lanes"], plans, strict=True)))
    path = out / "fit.json"
    checkpoint = out / "weights.npz"
    if path.exists():
        receipt = load_structured(path)
        if (
            receipt["weights_sha256"] != sha256_file(checkpoint)
            or receipt["initialization_sha256"] != init_hash or receipt["batch_plan_sha256"] != plan_hash
            or receipt["steps"] != steps or receipt["recipe"] != recipe
        ):
            raise ValueError("completed fitted arm differs from its frozen inputs")
        return receipt
    first = {key: np.zeros_like(value) for key, value in weights.items()}
    second = {key: np.zeros_like(value) for key, value in weights.items()}
    resume = out / "optimizer.npz"
    done, elapsed = restore_optimizer(resume, weights, first, second) if resume.exists() else (0, 0.0)
    current = done
    started = time.monotonic()
    differentiate = value_and_grad(objective)
    try:
        for step in range(done + 1, steps + 1):
            guard()
            batches = [tuple(item[plan[step - 1]] for item in lane)
                       for lane, plan in zip(arrays, plans, strict=True)]
            loss, grads = differentiate(weights, batches, arm, SPEC["depth"], recipe_weights(recipe, step, steps))
            if not np.isfinite(loss) or any(not np.isfinite(grad).all() for grad in grads.values()):
                raise FloatingPointError("mixed-objective loss or gradient became nonfinite")
            for key, grad in grads.items():
                first[key] = 0.9 * first[key] + 0.1 * grad
                second[key] = 0.999 * second[key] + 0.001 * grad**2
                gradient = first[key] / (1 - 0.9**step)
                variance = second[key] / (1 - 0.999**step)
                weights[key] -= SPEC["learning_rate"] * gradient / (np.sqrt(variance) + 1e-8)
            current = step
            if step % 24 == 0:
                save_optimizer(resume, current, weights, first, second, elapsed + time.monotonic() - started)
    finally:
        if current > done:
            save_optimizer(resume, current, weights, first, second, elapsed + time.monotonic() - started)
    temp = checkpoint.with_suffix(".part.npz")
    np.savez(temp, **weights)
    temp.replace(checkpoint)
    sample = tuple(tuple(item[:8] for item in lane) for lane in arrays)
    sample_losses = losses(weights, sample, arm, SPEC["depth"]).tolist()
    batch_sizes = RECIPES[recipe]["batch_examples"]
    receipt = {
        "arm": arm, "seed": seed, "steps": current, "fit_wall_s": elapsed + time.monotonic() - started,
        "recipe": recipe, "batch_examples": batch_sizes,
        "initialization_sha256": init_hash, "batch_plan_sha256": plan_hash,
        "weights_sha256": sha256_file(checkpoint), "parameters": sum(weights[key].size for key in weights),
        "training_sample_loss": dict(zip(SPEC["lanes"], sample_losses, strict=True)),
        "resume_started_at": done,
        "text_target_tokens": current * batch_sizes[0] * arrays[0][1].shape[1],
        "lean_examples": current * batch_sizes[1],
        "revision_examples": current * batch_sizes[2],
        "cell_calls_per_training_step": 2 * SPEC["depth"] * (sum(batch_sizes) + batch_sizes[2]),
    }
    atomic_record(path, receipt)
    print(f"Fit seed {seed} {arm} on {recipe}, {current} updates", flush=True)
    return receipt


def resume_probe(root, arrays, guard):
    path = root / "resume-audit.json"
    if path.exists():
        return
    folder = root / "resume-probe"
    plans = [np.tile(np.arange(batch), (4, 1)) for batch in RECIPES["mixed"]["batch_examples"]]
    partial = fit_arm(folder / "partial", arrays, "direct", 123, plans, guard, steps=2)
    straight = fit_arm(folder / "straight", arrays, "direct", 123, plans, guard, steps=4)
    continued = folder / "continued"
    continued.mkdir(exist_ok=True)
    resume_state = continued / "optimizer.npz"
    if not resume_state.exists():
        shutil.copyfile(folder / "partial" / "optimizer.npz", resume_state)
    final = fit_arm(continued, arrays, "direct", 123, plans, guard, steps=4)
    with np.load(folder / "straight" / "weights.npz", allow_pickle=False) as a, np.load(
        continued / "weights.npz", allow_pickle=False,
    ) as b:
        equal = a.files == b.files and all(np.array_equal(a[key], b[key]) for key in a.files)
    if not equal or partial["steps"] != 2 or final["resume_started_at"] < 2:
        raise ValueError("checkpoint continuation diverged from uninterrupted fitting")
    atomic_record(path, {
        "identical_weight_tensors": equal, "partial": partial, "continued": final, "straight": straight,
    })


def plans_for(arrays, seed, recipe):
    batches = RECIPES[recipe]["batch_examples"]
    return [
        np.random.default_rng(seed + 998441 + i * 9989).integers(len(lane[0]), size=(SPEC["steps"], batch))
        for i, (lane, batch) in enumerate(zip(arrays, batches, strict=True))
    ]


def fit_directory(root, seed, arm, recipe):
    if recipe == "mixed":
        return root / "fits" / str(seed) / arm
    return root / "recipe-fits" / str(seed) / recipe


def fit(root, guard):
    frozen_inputs(root)
    arrays = train_data(root)
    protocol = {
        "dataset_sha256": sha256_file(root / "dataset.json"), "fit_source_sha256": sha256_file(Path(__file__)),
        "cell_source_sha256": sha256_file(Path(cell.__file__)), "data_source_sha256": sha256_file(Path(data.__file__)),
        "feedback_source_sha256": sha256_file(Path(__file__).with_name("rlt_pilot.py")),
        "spec": SPEC, "arms": cell.ARMS, "recipes": RECIPES, "lean_feature_vocabulary": lean_words(),
    }
    path = root / "fit-protocol.json"
    if path.exists() and sha256_json(protocol) != sha256_json(load_structured(path)):
        raise ValueError("mixed pilot training protocol changed")
    atomic_record(path, protocol)
    resume_probe(root, arrays, guard)
    all_fits = []
    for seed in SPEC["seeds"]:
        plans = plans_for(arrays, seed, "mixed")
        fits = [fit_arm(root / "fits" / str(seed) / arm, arrays, arm, seed, plans, guard) for arm in cell.ARMS]
        if len({(r["initialization_sha256"], r["batch_plan_sha256"], r["parameters"]) for r in fits}) != 1:
            raise ValueError("mixed pilot fits do not share matched construction")
        all_fits.extend(fits)
        direct = next(receipt for receipt in fits if receipt["arm"] == "direct")
        for recipe in ("text_first", "text_heavier"):
            plan = plans_for(arrays, seed, recipe)
            arm = fit_arm(
                fit_directory(root, seed, "direct", recipe), arrays, "direct", seed, plan, guard, recipe=recipe,
            )
            if (
                arm["initialization_sha256"] != direct["initialization_sha256"]
                or arm["cell_calls_per_training_step"] != direct["cell_calls_per_training_step"]
                or (recipe == "text_first" and arm["batch_plan_sha256"] != direct["batch_plan_sha256"])
                or not np.array_equal(plan[2], plans[2])
            ):
                raise ValueError("data recipe fits are not matched")
            all_fits.append(arm)
    atomic_record(root / "fit-receipts.json", {"protocol_sha256": sha256_file(path), "arms": all_fits})
