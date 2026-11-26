"""Small, bounded research experiments with real NumPy parameter updates.

``trm_style`` is a graph-conditioned teaching model: a shared MLP controls
reasoning transport and answer carry. It is a task-specific TRM-style learner,
not an implementation or checkpoint of an external TRM architecture. The LDT
learners are additive three-class heads over explicit tags. Their exact critical
veto is reported separately from the learned head's predictions.

The caller owns evaluation seeds and resource isolation. No seed is generated,
persisted or downloaded here. Split assignment groups identical input roots;
this is an exact-input split, not a claim of semantic novelty between graphs.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

SCHEMA = "datagoblin.research-v2.learner.v1"
KINDS = ("trm_style", "ldt_filter", "ldt_persona")
SPLITS = ("train", "calibration", "evaluation", "ood")
CLASSES = ("accept", "hold", "reject")
NODES = 6
TEMPERATURE = 8.0
FILTER_TAGS = (
    "rights_denied", "held_evaluation_root", "unredacted_private_data", "forged_receipt",
    "rights_unresolved", "outcome_unverified", "stale_receipt", "subject_mismatch",
    "high_quality", "fluent", "popular", "novel", "short", "well_formatted",
    "teacher_confident", "relevant", "diverse", "cheap", "readable", "well_cited",
    "rare_skill", "complete", "helpful", "positive_review",
)
PERSONA_TAGS = (
    "owner_scope_mismatch", "authorization_revoked", "hard_policy_breach", "bystander_private_data",
    "owner_ambiguous", "preference_stale", "mixed_character_scope", "inferred_preference",
    "user_style_match", "character_style_match", "warm", "concise", "playful", "formal",
    "helpful", "confident", "familiar", "novel", "convenient", "empathetic",
    "fast", "consistent", "well_written", "positive_feedback",
)
_BUCKETS = {
    "train": set(range(10)), "calibration": {10, 11}, "evaluation": {12, 13}, "ood": {14, 15},
}


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _integer(value: Any, low: int, high: int, name: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _seed(seed: Any) -> int:
    return _integer(seed, 0, 2**63 - 1, "seed")


def _rng(seed: int, namespace: str) -> np.random.Generator:
    derived = int(_digest([_seed(seed), namespace])[:16], 16)
    return np.random.default_rng(derived)


def normalize_config(kind: str, config: dict[str, Any]) -> dict[str, Any]:
    """Validate every user-controlled fit dimension before creating arrays."""
    if kind not in KINDS or not isinstance(config, dict):
        raise ValueError("unknown learner kind or invalid config")
    common = {"steps", "learning_rate", "train_examples"}
    specific = {"width", "recursions"} if kind == "trm_style" else {"critical_veto"}
    if set(config) - common - specific:
        raise ValueError("unknown or inapplicable learner config field")
    result = {
        "steps": _integer(config.get("steps", 64), 0, 128, "steps"),
        "train_examples": _integer(config.get("train_examples", 96), 16, 256, "train_examples"),
    }
    rate = config.get("learning_rate", 0.03)
    if type(rate) not in {int, float} or not math.isfinite(rate) or not 0.0001 <= rate <= 0.5:
        raise ValueError("learning_rate must be finite in [0.0001, 0.5]")
    result["learning_rate"] = float(rate)
    if kind == "trm_style":
        if result["train_examples"] % 2:
            raise ValueError("TRM examples must form complete before/after pairs")
        result["width"] = _integer(config.get("width", 16), 4, 32, "width")
        result["recursions"] = _integer(config.get("recursions", 4), 1, 8, "recursions")
    else:
        veto = config.get("critical_veto", True)
        if type(veto) is not bool:
            raise ValueError("critical_veto must be a boolean")
        result["critical_veto"] = veto
    return result


def _split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError("split must be train, calibration, evaluation or ood")


def _eligible(root: str, split: str) -> bool:
    return int(root[:8], 16) % 16 in _BUCKETS[split]


def _walk(edges: list[int], start: int, hops: int) -> int:
    # Independent Python target calculation; the learner uses matrix transport.
    node = start
    for _ in range(hops):
        node = edges[node]
    return node


def _graph_data(seed: int, split: str, count: int) -> dict[str, Any]:
    """Keep each original graph and its before/after examples in one split."""
    rng = _rng(seed, "graphs/" + split)
    roots: list[str] = []
    seen: set[str] = set()
    examples = []
    eye = np.eye(NODES, dtype=np.float64)
    for _ in range(40_000):
        edges = rng.integers(0, NODES, size=NODES).tolist()
        root = _digest(["graph-family", edges])
        if root in seen or not _eligible(root, split):
            continue
        start = int(rng.integers(NODES))
        hops = int(rng.integers(5, 8) if split == "ood" else rng.integers(1, 5))
        before = _walk(edges, start, hops)
        patch_source = _walk(edges, start, int(rng.integers(hops)))
        changed = edges.copy()
        # Accept a paired repair only when the endpoint really changes.
        alternatives = rng.permutation(NODES).tolist()
        after = before
        for destination in alternatives:
            changed[patch_source] = destination
            after = _walk(changed, start, hops)
            if after != before:
                break
        if after == before:
            continue
        seen.add(root)
        roots.append(root)
        examples.append((eye[edges], eye[start], hops, np.zeros(NODES), before))
        examples.append((eye[changed], eye[start], hops, eye[before], after))
        if len(examples) >= count:
            break
    if len(examples) != count:
        raise RuntimeError("bounded graph generator exhausted")
    return {
        "edges": np.stack([x[0] for x in examples]),
        "start": np.stack([x[1] for x in examples]),
        "hops": np.array([x[2] for x in examples], dtype=np.int64),
        "prior": np.stack([x[3] for x in examples]),
        "targets": np.array([x[4] for x in examples], dtype=np.int64),
        "roots": roots,
    }


def _tag_target(tags: list[int]) -> int:
    # Authority depends on critical and unresolved conditions, never style.
    if any(tags[i] == 1 for i in range(4)):
        return 2
    if any(tags[i] == 1 for i in range(4, 8)):
        return 1
    return 0


def _tag_data(kind: str, seed: int, split: str, count: int) -> dict[str, Any]:
    rng = _rng(seed, kind + "/" + split)
    roots: list[str] = []
    seen: set[str] = set()
    examples: list[list[int]] = []
    targets: list[int] = []
    for _ in range(40_000):
        tags = (rng.random(24) < 0.55).astype(np.int64)
        tags[:8] = (rng.random(8) < 0.1).astype(np.int64)
        if split == "ood":
            # Highly attractive style cannot override a critical condition.
            tags[8:] = (rng.random(16) < 0.9).astype(np.int64)
            tags[int(rng.integers(4))] = 1
        values = tags.tolist()
        root = _digest([kind, values])
        if root in seen or not _eligible(root, split):
            continue
        seen.add(root)
        roots.append(root)
        examples.append(values)
        targets.append(_tag_target(values))
        if len(examples) >= count:
            break
    if len(examples) != count:
        raise RuntimeError("bounded tag generator exhausted")
    return {"features": np.array(examples, dtype=np.float64), "targets": np.array(targets), "roots": roots}


def _data(kind: str, seed: int, split: str, count: int) -> dict[str, Any]:
    _seed(seed)
    _split(split)
    return _graph_data(seed, split, count) if kind == "trm_style" else _tag_data(kind, seed, split, count)


def _weights(kind: str, seed: int, config: dict[str, Any]) -> dict[str, np.ndarray]:
    rng = _rng(seed, kind + "/initialization")
    if kind == "trm_style":
        width = config["width"]
        return {
            "up": rng.normal(0, 0.08, (6, width)), "bias": np.zeros(width),
            "down": rng.normal(0, 0.08, width), "offset": np.array([-1.0]),
        }
    return {"up": rng.normal(0, 0.02, (24, 3)), "bias": np.zeros(3)}


def _json_weights(weights: dict[str, np.ndarray]) -> dict[str, Any]:
    return {name: values.tolist() for name, values in weights.items()}


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def _loss(probabilities: np.ndarray, targets: np.ndarray) -> float:
    chosen = probabilities[np.arange(len(targets)), targets]
    return float(-np.log(np.clip(chosen, 1e-300, 1)).mean())


def _gate(weights: dict[str, np.ndarray], features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.tanh(features @ weights["up"] + weights["bias"])
    logits = hidden @ weights["down"] + weights["offset"][0]
    gate = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    return gate, hidden


def _gate_backward(
    weights: dict[str, np.ndarray], features: np.ndarray, hidden: np.ndarray,
    gate: np.ndarray, upstream: np.ndarray, grads: dict[str, np.ndarray],
) -> np.ndarray:
    local = upstream * gate * (1 - gate)
    grads["offset"][0] += local.sum()
    grads["down"] += np.sum(hidden * local[..., None], axis=(0, 1))
    hidden_grad = local[..., None] * weights["down"] * (1 - hidden**2)
    grads["bias"] += hidden_grad.sum(axis=(0, 1))
    grads["up"] += features.reshape(-1, 6).T @ hidden_grad.reshape(-1, len(weights["bias"]))
    return hidden_grad @ weights["up"].T


def _trm_forward(
    weights: dict[str, np.ndarray], data: dict[str, Any], recursions: int,
) -> tuple[np.ndarray, list[Any]]:
    z, y = data["start"].copy(), data["prior"].copy()
    cache = []
    for visit in range(recursions):
        message = np.einsum("bi,bij->bj", z, data["edges"])
        clock = np.broadcast_to(np.array(visit / 8), z.shape)
        active = (visit < data["hops"])[:, None].astype(np.float64)
        features = np.stack((z, message, y, data["start"], clock, -np.ones_like(z)), axis=-1)
        gate, hidden = _gate(weights, features)
        next_z = z + active * gate * (message - z)
        answer_features = np.stack(
            (y, next_z, next_z - y, data["start"], clock, np.ones_like(z)), axis=-1,
        )
        answer_gate, answer_hidden = _gate(weights, answer_features)
        next_y = y + answer_gate * (next_z - y)
        cache.append((z, y, message, active, features, gate, hidden, next_z,
                      answer_features, answer_gate, answer_hidden))
        z, y = next_z, next_y
    return _softmax(TEMPERATURE * y), cache


def _trm_gradients(
    weights: dict[str, np.ndarray], data: dict[str, Any], config: dict[str, Any],
) -> tuple[float, dict[str, np.ndarray]]:
    probabilities, cache = _trm_forward(weights, data, config["recursions"])
    grads = {name: np.zeros_like(value) for name, value in weights.items()}
    dy = probabilities.copy()
    dy[np.arange(len(dy)), data["targets"]] -= 1
    dy *= TEMPERATURE / len(dy)
    dz = np.zeros_like(dy)
    for (z, y, message, active, features, gate, hidden, next_z,
         answer_features, answer_gate, answer_hidden) in reversed(cache):
        old_dy = dy * (1 - answer_gate)
        next_dz = dz + dy * answer_gate
        answer_features_grad = _gate_backward(
            weights, answer_features, answer_hidden, answer_gate, dy * (next_z - y), grads,
        )
        old_dy += answer_features_grad[..., 0] - answer_features_grad[..., 2]
        next_dz += answer_features_grad[..., 1] + answer_features_grad[..., 2]
        old_dz = next_dz * (1 - active * gate)
        dmessage = next_dz * active * gate
        features_grad = _gate_backward(weights, features, hidden, gate, next_dz * active * (message - z), grads)
        old_dz += features_grad[..., 0]
        dmessage += features_grad[..., 1]
        old_dy += features_grad[..., 2]
        old_dz += np.einsum("bj,bij->bi", dmessage, data["edges"])
        dz, dy = old_dz, old_dy
    return _loss(probabilities, data["targets"]), grads


def _ldt_gradients(
    weights: dict[str, np.ndarray], data: dict[str, Any],
) -> tuple[float, dict[str, np.ndarray]]:
    probabilities = _softmax(data["features"] @ weights["up"] + weights["bias"])
    delta = probabilities.copy()
    delta[np.arange(len(delta)), data["targets"]] -= 1
    delta /= len(delta)
    return _loss(probabilities, data["targets"]), {
        "up": data["features"].T @ delta, "bias": delta.sum(axis=0),
    }


def _probabilities(
    kind: str, weights: dict[str, np.ndarray], data: dict[str, Any], config: dict[str, Any],
) -> np.ndarray:
    if kind == "trm_style":
        return _trm_forward(weights, data, config["recursions"])[0]
    return _softmax(data["features"] @ weights["up"] + weights["bias"])


def _metric_values(
    kind: str, probabilities: np.ndarray, data: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    raw = probabilities.argmax(axis=1)
    predicted = raw.copy()
    result: dict[str, Any] = {}
    if kind != "trm_style":
        critical = data["features"][:, :4].any(axis=1)
        if config["critical_veto"]:
            predicted[critical] = 2
        critical_count = int(critical.sum())
        raw_false_accepts = int(np.sum(critical & (raw == 0)))
        false_accepts = int(np.sum(critical & (predicted == 0)))
        result.update(
            critical_examples=critical_count,
            raw_critical_false_accepts=raw_false_accepts,
            critical_false_accepts=false_accepts,
            critical_false_accept_rate=false_accepts / critical_count if critical_count else 0.0,
            raw_accuracy=float(np.mean(raw == data["targets"])),
            critical_veto=config["critical_veto"],
            unresolved_false_accepts=int(np.sum((data["targets"] == 1) & (predicted == 0))),
        )
    correct = int(np.sum(predicted == data["targets"]))
    result.update(
        example_count=len(predicted), correct=correct, accuracy=correct / len(predicted),
        loss=_loss(probabilities, data["targets"]),
    )
    if kind == "trm_style":
        result.update(
            before_accuracy=float(np.mean(predicted[::2] == data["targets"][::2])),
            revision_accuracy=float(np.mean(predicted[1::2] == data["targets"][1::2])),
            paired_correct=int(np.sum((predicted[::2] == data["targets"][::2])
                                      & (predicted[1::2] == data["targets"][1::2]))),
        )
    return result


def fit(kind: str, train_seed: int, config: dict[str, Any]) -> dict[str, Any]:
    """Fit a tiny learner on generated TRAIN roots and return its bound artifact."""
    selected = normalize_config(kind, config)
    train_seed = _seed(train_seed)
    data = _data(kind, train_seed, "train", selected["train_examples"])
    weights = _weights(kind, train_seed, selected)
    initial_weights_sha = _digest(_json_weights(weights))
    initial_loss = _loss(_probabilities(kind, weights, data, selected), data["targets"])
    first = {key: np.zeros_like(value) for key, value in weights.items()}
    second = {key: np.zeros_like(value) for key, value in weights.items()}
    updates = []
    for step in range(1, selected["steps"] + 1):
        loss, grads = (
            _trm_gradients(weights, data, selected) if kind == "trm_style" else _ldt_gradients(weights, data)
        )
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        if not math.isfinite(loss) or not math.isfinite(norm):
            raise FloatingPointError("nonfinite loss or gradient")
        for key, param in weights.items():
            grad = grads[key] / max(norm, 1)
            first[key] = 0.9 * first[key] + 0.1 * grad
            second[key] = 0.999 * second[key] + 0.001 * grad**2
            corrected = (first[key] / (1 - 0.9**step)) / (np.sqrt(second[key] / (1 - 0.999**step)) + 1e-8)
            param -= selected["learning_rate"] * (corrected + 0.0001 * param)
        updates.append({"step": step, "loss": loss, "gradient_norm": norm})
    final_weights = _json_weights(weights)
    metrics = _metric_values(kind, _probabilities(kind, weights, data, selected), data, selected)
    artifact = {
        "schema": SCHEMA, "kind": kind, "config": selected, "weights": final_weights,
        "receipt": {
            "training_split": "train", "train_seed": train_seed,
            "training_roots_sha256": _digest(data["roots"]), "training_root_count": len(data["roots"]),
            "init_weights_sha256": initial_weights_sha, "weights_sha256": _digest(final_weights),
            "updates_sha256": _digest(updates), "optimizer_updates": selected["steps"],
            "optimizer": "AdamW", "weight_decay": 0.0001,
            "parameters": sum(value.size for value in weights.values()),
            "loss_before": initial_loss, "loss_after": metrics["loss"],
            "train_metrics": metrics, "training_examples": selected["train_examples"],
            "cell_calls": selected["steps"] * selected["train_examples"]
            * (2 * NODES * selected["recursions"] if kind == "trm_style" else 1),
            "classes": list(range(NODES)) if kind == "trm_style" else list(CLASSES),
            "feature_names": list(FILTER_TAGS if kind == "ldt_filter" else PERSONA_TAGS)
            if kind != "trm_style" else ["reasoning", "message", "answer", "start", "clock", "phase"],
        },
    }
    artifact["artifact_sha256"] = _digest(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Check bounds, finite values and content bindings before evaluating weights.

    A hash proves byte identity. Source authority comes from the environment's
    artifact store, never from a submitted hash alone.
    """
    keys = {"schema", "kind", "config", "weights", "receipt", "artifact_sha256"}
    if not isinstance(artifact, dict) or set(artifact) != keys or artifact["schema"] != SCHEMA:
        raise ValueError("invalid learner artifact schema")
    kind = artifact["kind"]
    config = normalize_config(kind, artifact["config"])
    if config != artifact["config"]:
        raise ValueError("artifact config must contain normalized fields")
    # Bound plain JSON before materializing NumPy arrays from untrusted weights.
    try:
        encoded = json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("artifact must contain finite plain JSON") from error
    if len(encoded) > 100_000:
        raise ValueError("learner artifact exceeds byte bound")
    body = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if _digest(body) != artifact["artifact_sha256"]:
        raise ValueError("artifact content digest mismatch")
    receipt = artifact["receipt"]
    if not isinstance(receipt, dict) or receipt.get("training_split") != "train":
        raise ValueError("learner artifact requires a TRAIN receipt")
    seed = _seed(receipt.get("train_seed"))
    expected = _weights(kind, seed, config)
    if not isinstance(artifact["weights"], dict) or set(artifact["weights"]) != set(expected):
        raise ValueError("unexpected weight tensors")
    for key, initial in expected.items():
        try:
            values = np.asarray(artifact["weights"][key], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid weight values") from error
        if values.shape != initial.shape or not np.isfinite(values).all() or np.max(np.abs(values)) > 1_000_000:
            raise ValueError("invalid weight shape or values")
    if (
        receipt.get("init_weights_sha256") != _digest(_json_weights(expected))
        or receipt.get("weights_sha256") != _digest(artifact["weights"])
        or receipt.get("optimizer_updates") != config["steps"]
        or receipt.get("training_examples") != config["train_examples"]
        or receipt.get("parameters") != sum(value.size for value in expected.values())
    ):
        raise ValueError("learner receipt and weights disagree")
    return config


def evaluate(artifact: dict[str, Any], seed: int, split: str, *, examples: int = 128) -> dict[str, Any]:
    """Evaluate fresh roots; the caller supplies a steward-owned seed for evidence."""
    config = validate_artifact(artifact)
    kind = artifact["kind"]
    _integer(examples, 16, 256, "examples")
    if kind == "trm_style" and examples % 2:
        raise ValueError("TRM examples must form complete before/after pairs")
    data = _data(kind, seed, split, examples)
    weights = {key: np.array(value, dtype=np.float64) for key, value in artifact["weights"].items()}
    result = _metric_values(kind, _probabilities(kind, weights, data, config), data, config)
    result.update(
        kind=kind, split=split, dataset_roots_sha256=_digest(data["roots"]),
        root_count=len(data["roots"]), artifact_sha256=artifact["artifact_sha256"],
        cell_calls=examples * (2 * NODES * config["recursions"] if kind == "trm_style" else 1),
    )
    return result
