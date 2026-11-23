"""DG-SGRT-03: sequential storyworld evidence and holonomy-history CPU pilot.

Each root has two episodes containing the identical events in opposite order.
The trusted simulator owns expected states. Models only receive the initial
public state and one event at a time. All models choose from the same 16
candidate outcomes, with one exact verifier check after each event.

This is synthetic mechanism development, not a trained Aleph integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .sgrt_holoarc import BASE_DIM, FIBER_DIM, spiro_path, trace_transport
from .sgrt_holoarc_gym import apply_operator

EVENTS = ("xor1", "xor2", "xor4", "add1", "add3", "rol1", "swap01", "mask7", "or8")
EVENT_STORY = {
    "xor1": "courier reverses the witness flag",
    "xor2": "council swaps the registry parity",
    "xor4": "archive inverts the signal lamp",
    "add1": "clock advances the public state",
    "add3": "festival advances three calendar ticks",
    "rol1": "messenger rotates the four seals",
    "swap01": "scribe exchanges two witness seals",
    "mask7": "archive seals the highest register",
    "or8": "archive opens the highest register",
}
FAMILIES = ("courier", "council", "archive", "heldout_mixed")
TRAINED = ("bag", "direct", "flat", "curved", "random")
EVALUATED = (*TRAINED, "curved_transport_swap", "exact_oracle")
D = FIBER_DIM
CHOICES = tuple(range(16))
BATCH_SEED = 31003
EVAL_SEED = 61003


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _world_key(state: int, a: str, b: str) -> str:
    return f"{state}|{a}|{b}"


def _is_train_lineage(state: int, a: str, b: str) -> bool:
    # Same unordered event pair and starting state can never enter both splits.
    share = int(hashlib.sha256(_world_key(state, a, b).encode()).hexdigest()[:8], 16) / (2**32)
    return share < 0.60


@dataclass(frozen=True)
class PublicEpisode:
    root_id: str
    family: str
    initial_state: int
    events: tuple[str, str]
    candidate_states: tuple[int, ...] = CHOICES

    def public(self) -> dict[str, Any]:
        return {
            "schema_version": "dg-sgrt-03-public-episode-v1",
            "root_id": self.root_id,
            "family": self.family,
            "initial_state": self.initial_state,
            "event_stream": [
                {"step": idx + 1, "type": "fictional_rule_update", "event": event, "description": EVENT_STORY[event]}
                for idx, event in enumerate(self.events)
            ],
            "candidate_states": list(self.candidate_states),
        }


@dataclass(frozen=True)
class WorldRoot:
    root_id: str
    family: str
    initial_state: int
    event_a: str
    event_b: str

    def episode(self, reverse: bool = False) -> PublicEpisode:
        events = (self.event_b, self.event_a) if reverse else (self.event_a, self.event_b)
        return PublicEpisode(self.root_id, self.family, self.initial_state, events)

    def truth(self, reverse: bool = False) -> tuple[int, int]:
        """Evaluator only. The two exact intermediate states are private labels."""
        ep = self.episode(reverse)
        first = apply_operator(ep.events[0], ep.initial_state)
        second = apply_operator(ep.events[1], first)
        return first, second


def make_roots(*, split: Literal["train", "eval"], seed: int, per_family: int) -> tuple[WorldRoot, ...]:
    """Deterministic family/lineage-disjoint, noncommuting two-event world pairs."""
    if split not in ("train", "eval") or not 1 <= per_family <= 24:
        raise ValueError("split must be train/eval and per_family 1..24")
    families = range(3) if split == "train" else range(4)
    roots: list[WorldRoot] = []
    for family in families:
        eligible = []
        for i, a in enumerate(EVENTS):
            for j in range(i + 1, len(EVENTS)):
                b = EVENTS[j]
                if (i + j) % 4 != family:
                    continue
                for state in CHOICES:
                    direct = apply_operator(b, apply_operator(a, state))
                    reversed_outcome = apply_operator(a, apply_operator(b, state))
                    if direct == reversed_outcome:
                        continue
                    training = _is_train_lineage(state, a, b)
                    if family != 3 and (training != (split == "train")):
                        continue
                    eligible.append((state, a, b))
        rng = np.random.default_rng(seed + 1213 * family)
        order = rng.permutation(len(eligible))
        if per_family > len(eligible):
            raise ValueError(f"family {family} only has {len(eligible)} roots in {split}")
        for pos in order[:per_family]:
            state, a, b = eligible[int(pos)]
            root_id = _digest({"state": state, "a": a, "b": b})[:20]
            roots.append(WorldRoot(root_id, FAMILIES[family], state, a, b))
    if len({r.root_id for r in roots}) != len(roots):
        raise AssertionError("root collision")
    return tuple(roots)


def _rotation(angles: list[float]) -> np.ndarray:
    out = np.eye(D, dtype=np.float64)
    for i, angle in enumerate(angles):
        a, b = 2 * i, 2 * i + 1
        co, si = math.cos(angle), math.sin(angle)
        out[a, a] = co
        out[a, b] = -si
        out[b, a] = si
        out[b, b] = co
    return out


@lru_cache(maxsize=None)
def event_transport(mode: str, op: str) -> np.ndarray:
    """Spectrally matched flat, curved and random orthogonal connections."""
    if mode not in TRAINED or op not in EVENTS:
        raise ValueError("unknown mode or event")
    if mode in ("bag", "direct"):
        return np.eye(D, dtype=np.float64)
    ix = EVENTS.index(op)
    path = 3.0 * spiro_path(np.zeros(BASE_DIM), seed=901 + 97 * ix)
    curved = trace_transport(path, mode="curved").holonomy
    if mode == "curved":
        return curved
    phases = np.angle(np.linalg.eigvals(curved))
    positive = sorted(float(v) for v in phases if v > 1e-9)
    positive = (positive + [0.0] * (D // 2))[: D // 2]
    flat = _rotation(positive)
    if mode == "flat":
        return flat
    rng = np.random.default_rng(3047 + 97 * ix)
    orthogonal, _ = np.linalg.qr(rng.standard_normal((D, D)))
    return orthogonal @ flat @ orthogonal.T


def transport_diagnostics() -> dict[str, Any]:
    identity = np.eye(D, dtype=np.float64)
    commute_flat = 0.0
    commute_curved = 0.0
    max_spec = 0.0
    max_orthogonal = 0.0
    for a in EVENTS:
        curved = event_transport("curved", a)
        flat = event_transport("flat", a)
        max_spec = max(max_spec, abs(float(np.trace(curved - flat))))
        for mode in ("curved", "flat", "random"):
            q = event_transport(mode, a)
            max_orthogonal = max(max_orthogonal, float(np.linalg.norm(q.T @ q - identity)))
        for b in EVENTS:
            f2 = event_transport("flat", b)
            c2 = event_transport("curved", b)
            commute_flat = max(commute_flat, float(np.linalg.norm(flat @ f2 - f2 @ flat)))
            commute_curved = max(commute_curved, float(np.linalg.norm(curved @ c2 - c2 @ curved)))
    return {
        "flat_commutator_fro_max": commute_flat,
        "curved_commutator_fro_max": commute_curved,
        "flat_curved_trace_mismatch_max": max_spec,
        "orthogonality_error_max": max_orthogonal,
        "event_count": len(EVENTS),
        "fiber_dim": D,
        "gear_base_dim": BASE_DIM,
        "gear_microsteps": 8,
    }


@dataclass
class HistoryHead:
    initial: np.ndarray
    recurrent: np.ndarray
    evidence: np.ndarray
    output: np.ndarray
    bias: np.ndarray
    gates: np.ndarray

    @classmethod
    def initialize(cls, seed: int) -> HistoryHead:
        rng = np.random.default_rng(seed)
        return cls(
            rng.normal(0, 0.14, (16, D)),
            0.32 * np.eye(D) + rng.normal(0, 0.04, (D, D)),
            rng.normal(0, 0.12, (len(EVENTS), D)),
            rng.normal(0, 0.11, (D, 16)),
            np.zeros(16, dtype=np.float64),
            np.full(len(EVENTS), -0.6, dtype=np.float64),
        )

    @property
    def weights(self) -> tuple[np.ndarray, ...]:
        return self.initial, self.recurrent, self.evidence, self.output, self.bias, self.gates

    @property
    def parameter_count(self) -> int:
        return sum(v.size for v in self.weights)

    def digest(self) -> str:
        sha = hashlib.sha256()
        for v in self.weights:
            sha.update(v.astype("<f8", copy=False).tobytes())
        return sha.hexdigest()


def _softmax_loss(logits: np.ndarray, target: int) -> tuple[float, np.ndarray]:
    shifted = logits - logits.max()
    exponent = np.exp(shifted)
    probs = exponent / exponent.sum()
    loss = float(np.log(exponent.sum()) - shifted[target])
    probs[target] -= 1.0
    return loss, probs


def forward(
    head: HistoryHead,
    ep: PublicEpisode,
    *,
    mode: str,
    swap_transport: bool = False,
) -> tuple[tuple[np.ndarray, np.ndarray], list[dict[str, Any]], np.ndarray]:
    """Public-only forward, with retained states for the CPU training derivative."""
    h0 = np.tanh(head.initial[ep.initial_state])
    events = tuple(EVENTS.index(e) for e in ep.events)
    records: list[dict[str, Any]] = []
    if mode == "bag":
        first = np.tanh(h0 @ head.recurrent + head.evidence[events[0]])
        last = np.tanh(h0 @ head.recurrent + head.evidence[events[0]] + head.evidence[events[1]])
        logits = (first @ head.output + head.bias, last @ head.output + head.bias)
        return logits, [
            {"prev": h0, "h": first, "event": events[0]},
            {"prev": h0, "h": last, "event": events[1]},
        ], h0
    state = h0
    for t, event_id in enumerate(events):
        transport_id = events[1 - t] if swap_transport else event_id
        matrix = event_transport(mode, EVENTS[transport_id])
        gate = float(1.0 / (1.0 + math.exp(-float(np.clip(head.gates[event_id], -30, 30)))))
        difference = (matrix - np.eye(D)) @ state
        updated = np.tanh(state @ head.recurrent + head.evidence[event_id] + gate * difference)
        records.append({
            "prev": state,
            "h": updated,
            "event": event_id,
            "gate": gate,
            "difference": difference,
            "matrix": matrix,
        })
        state = updated
    logits = tuple(r["h"] @ head.output + head.bias for r in records)
    return (logits[0], logits[1]), records, h0


def gradients(
    head: HistoryHead,
    ep: PublicEpisode,
    truth: tuple[int, int],
    *,
    mode: str,
) -> tuple[float, tuple[np.ndarray, ...]]:
    logits, records, h0 = forward(head, ep, mode=mode)
    derivative = [np.zeros_like(p) for p in head.weights]
    weights = (0.35, 1.0)
    dlogits = []
    loss = 0.0
    for t in range(2):
        item_loss, probs = _softmax_loss(logits[t], truth[t])
        loss += weights[t] * item_loss
        dlogits.append(weights[t] * probs)
        derivative[3] += np.outer(records[t]["h"], dlogits[t])
        derivative[4] += dlogits[t]
    if mode == "bag":
        total_initial = np.zeros(D)
        for t in range(2):
            dpre = (head.output @ dlogits[t]) * (1 - records[t]["h"] ** 2)
            derivative[1] += np.outer(h0, dpre)
            derivative[2][records[0]["event"]] += dpre
            if t == 1:
                derivative[2][records[1]["event"]] += dpre
            total_initial += head.recurrent @ dpre
    else:
        carry = np.zeros(D, dtype=np.float64)
        for t in (1, 0):
            rec = records[t]
            dpre = (head.output @ dlogits[t] + carry) * (1 - rec["h"] ** 2)
            derivative[1] += np.outer(rec["prev"], dpre)
            derivative[2][rec["event"]] += dpre
            gate = rec["gate"]
            derivative[5][rec["event"]] += float(dpre @ rec["difference"]) * gate * (1 - gate)
            carry = head.recurrent @ dpre + gate * ((rec["matrix"] - np.eye(D)).T @ dpre)
        total_initial = carry
    derivative[0][ep.initial_state] += total_initial * (1 - h0**2)
    return loss / 1.35, tuple(gradient / 1.35 for gradient in derivative)


def train(
    mode: str,
    *,
    roots: tuple[WorldRoot, ...],
    seed: int = 23,
    updates: int = 384,
    lr: float = 0.014,
) -> tuple[HistoryHead, dict[str, Any]]:
    if mode not in TRAINED or not 16 <= updates <= 2000 or lr <= 0:
        raise ValueError("invalid mode or training budget")
    rng = np.random.default_rng(seed + 101)
    head = HistoryHead.initialize(seed)
    order = rng.permutation(len(roots))
    m = [np.zeros_like(p) for p in head.weights]
    v = [np.zeros_like(p) for p in head.weights]
    losses = []
    for step in range(1, updates + 1):
        if (step - 1) % len(order) == 0:
            order = rng.permutation(len(roots))
        root = roots[int(order[(step - 1) % len(order)])]
        loss_a, grad_a = gradients(head, root.episode(), root.truth(), mode=mode)
        loss_b, grad_b = gradients(head, root.episode(reverse=True), root.truth(reverse=True), mode=mode)
        losses.append((loss_a + loss_b) / 2)
        combined = [(a + b) / 2 for a, b in zip(grad_a, grad_b, strict=True)]
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in combined))
        scale = min(1.0, 5.0 / max(norm, 1e-12))
        for idx, (p, g) in enumerate(zip(head.weights, combined, strict=True)):
            gradient = scale * g + 1e-5 * p
            m[idx] = 0.9 * m[idx] + 0.1 * gradient
            v[idx] = 0.999 * v[idx] + 0.001 * gradient * gradient
            p -= lr * (m[idx] / (1 - 0.9**step)) / (
                np.sqrt(v[idx] / (1 - 0.999**step)) + 1e-8
            )
    return head, {
        "mode": mode,
        "train_roots": len(roots),
        "training_episodes": len(roots) * 2,
        "updates": updates,
        "trainable_parameters": head.parameter_count,
        "weights_sha256": head.digest(),
        "initial32_loss": float(np.mean(losses[:32])),
        "final32_loss": float(np.mean(losses[-32:])),
    }


def _prediction(
    model: HistoryHead,
    ep: PublicEpisode,
    mode: str,
    *,
    swap_transport: bool = False,
) -> tuple[int, int]:
    logits, _, _ = forward(model, ep, mode=mode, swap_transport=swap_transport)
    return int(np.argmax(logits[0])), int(np.argmax(logits[1]))


def evaluate(
    models: dict[str, HistoryHead],
    *,
    roots: tuple[WorldRoot, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(models) != set(TRAINED):
        raise ValueError("all matched learned arms are required")
    rows = []
    for root in roots:
        outcomes: dict[str, Any] = {}
        truth = (root.truth(), root.truth(reverse=True))
        for arm in EVALUATED:
            predictions = []
            for reverse in (False, True):
                if arm == "exact_oracle":
                    predicted = root.truth(reverse)
                elif arm == "curved_transport_swap":
                    predicted = _prediction(models["curved"], root.episode(reverse), "curved", swap_transport=True)
                else:
                    predicted = _prediction(models[arm], root.episode(reverse), arm)
                predictions.append(predicted)
            per_episode = [
                {
                    "first_correct": predictions[i][0] == truth[i][0],
                    "last_correct": predictions[i][1] == truth[i][1],
                    "joint_correct": predictions[i] == truth[i],
                }
                for i in (0, 1)
            ]
            outcomes[arm] = {
                "first_correct_count": sum(int(x["first_correct"]) for x in per_episode),
                "last_correct_count": sum(int(x["last_correct"]) for x in per_episode),
                "joint_correct_count": sum(int(x["joint_correct"]) for x in per_episode),
                "both_order_last_correct": all(x["last_correct"] for x in per_episode),
                "distinguished_order": predictions[0][1] != predictions[1][1],
                "predictions": predictions,
                "verifier_calls": 4,
                "candidate_pool_sha256": _digest(CHOICES),
            }
        rows.append({
            "root_id": root.root_id,
            "family": root.family,
            "initial_state": root.initial_state,
            "events": [root.event_a, root.event_b],
            "truth": truth,
            "arms": outcomes,
            "evaluator_only": True,
        })
    results: dict[str, Any] = {}
    for group in (*FAMILIES, "all"):
        selection = [row for row in rows if group == "all" or row["family"] == group]
        results[group] = {}
        for arm in EVALUATED:
            stats = [r["arms"][arm] for r in selection]
            results[group][arm] = {
                "roots": len(stats),
                "episodes": len(stats) * 2,
                "first_correct": sum(s["first_correct_count"] for s in stats),
                "last_correct": sum(s["last_correct_count"] for s in stats),
                "both_orders_correct": sum(int(s["both_order_last_correct"]) for s in stats),
                "order_distinguished": sum(int(s["distinguished_order"]) for s in stats),
                "verifier_calls": sum(s["verifier_calls"] for s in stats),
            }
    return rows, results


def paired_contrasts(rows: list[dict[str, Any]], *, replicates: int = 512) -> dict[str, Any]:
    """Root-clustered contrasts, no pseudo-replication of the swapped episode."""
    output: dict[str, Any] = {}
    for arm in TRAINED:
        if arm == "direct":
            continue
        diffs = np.asarray([
            int(r["arms"][arm]["both_order_last_correct"]) -
            int(r["arms"]["direct"]["both_order_last_correct"])
            for r in rows
        ], dtype=np.float64)
        rng = np.random.default_rng(int(_digest(arm)[:12], 16))
        samples = rng.integers(0, len(diffs), (replicates, len(diffs)))
        ci = np.quantile(100 * diffs[samples].mean(axis=1), (0.025, 0.975))
        output[arm] = {
            "roots": len(diffs),
            "delta_pp": float(100 * diffs.mean()),
            "root_bootstrap_95_pp": [float(ci[0]), float(ci[1])],
            "wins": int(np.sum(diffs > 0)),
            "losses": int(np.sum(diffs < 0)),
            "ties": int(np.sum(diffs == 0)),
        }
    return output


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in rows:
            stream.write(_canonical(item).decode("utf-8") + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--updates", type=int, default=384)
    parser.add_argument("--train-per-family", type=int, default=24)
    parser.add_argument("--eval-per-family", type=int, default=12)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be absent or empty")
    training = make_roots(split="train", seed=BATCH_SEED, per_family=args.train_per_family)
    evaluation = make_roots(split="eval", seed=EVAL_SEED, per_family=args.eval_per_family)
    if set(r.root_id for r in training) & set(r.root_id for r in evaluation):
        raise AssertionError("data lineage crossed train/eval split")
    models, train_logs = {}, {}
    for arm in TRAINED:
        models[arm], train_logs[arm] = train(arm, roots=training, seed=args.seed, updates=args.updates)
    rows, scores = evaluate(models, roots=evaluation)
    args.out.mkdir(parents=True, exist_ok=True)
    public_hash = _write_jsonl(
        args.out / "public_eval.jsonl",
        [r.episode(reverse=rev).public() for r in evaluation for rev in (False, True)],
    )
    scored_hash = _write_jsonl(args.out / "scorer_only.jsonl", rows)
    report = {
        "schema_version": "dg-sgrt-03-public-sequential-v1",
        "status": "synthetic_cpu_development_only",
        "optimizer_seed": args.seed,
        "train_root_count": len(training),
        "eval_root_count": len(evaluation),
        "per_family_train": args.train_per_family,
        "per_family_eval": args.eval_per_family,
        "updates_per_arm": args.updates,
        "candidate_set_size": len(CHOICES),
        "verifier_calls_per_root": 4,
        "training": train_logs,
        "geometry": transport_diagnostics(),
        "scores": scores,
        "paired_vs_direct": paired_contrasts(rows),
        "public_eval_sha256": public_hash,
        "scorer_only_sha256": scored_hash,
        "claim_boundary": (
            "Fictional 4-bit sequential event simulator; evaluates chronological"
            " learning and frozen transport priors, not general creativity or"
            " an Aleph-1B architecture gain."
        ),
    }
    (args.out / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "train_roots": len(training),
        "eval_roots": len(evaluation),
        "geometry": report["geometry"],
        "scores_all": scores["all"],
        "paired_vs_direct": report["paired_vs_direct"],
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
