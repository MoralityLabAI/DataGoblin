"""Small action-shared NumPy policies for trajectory supervision."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .airis02_stop import StopHead
from .airis02_student import MAX_ITEMS, _clip, _cross_entropy
from .ram_hillclimb import digest
from .ram_trajectory import STOP, VIOLATIONS


def features(world: dict, selected: list[str], feedback: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Encode public facts, one row per membership edit, plus STOP features."""
    chosen = set(selected)
    items = world["items"]
    by_id = {item["id"]: item for item in items}
    current = [by_id[key] for key in selected]
    cost = sum(item["cost"] for item in current)
    quality = sum(item["quality"] for item in current)
    covered = {topic for item in current for topic in item["topics"]}
    slack = world["token_budget"] - cost
    free_slots = world["max_items"] - len(current)
    rejected = feedback["action"]["id"] if feedback else None
    reasons = feedback["new_violations"] if feedback else ()
    rows = []
    feasible_adds, bad_selected = [], []
    for i, item in enumerate(items):
        key = item["id"]
        present = key in chosen
        other = [member for member in current if member["id"] != key]
        other_topics = {topic for member in other for topic in member["topics"]}
        unique = len(set(item["topics"]) - other_topics)
        dependency = len(set(item["requires"]) - chosen)
        needed = sum(key in member["requires"] for member in other)
        lineage = any(item["lineage"] == member["lineage"] for member in other)
        duplicate_key = any(item["key"] == member["key"] for member in other)
        stale = any(
            member["eligible"] and member["key"] == item["key"] and member["epoch"] > item["epoch"]
            for member in items
        )
        conflict = any(key in pair and bool((set(pair) - {key}) & chosen) for pair in world["conflicts"])
        invalid = not item["eligible"] or dependency or lineage or duplicate_key or stale or conflict
        fits = item["cost"] <= slack and free_slots > 0
        if not present and fits and not invalid:
            feasible_adds.append((unique, item["quality"]))
        if present and invalid:
            bad_selected.append(key)
        rows.append([
            1.0, float(present), float(item["eligible"]), i / 9,
            item["quality"] / 12, item["cost"] / 17,
            item["quality"] / item["cost"] / 3, unique / 4, len(item["topics"]) / 4,
            float(lineage), float(duplicate_key), float(stale), float(conflict),
            float(bool(dependency)), float(bool(needed)),
            float(fits), float(bool(invalid)),
            float(present and unique == 0), float(not present and unique > 0),
            slack / 30, free_slots / 4, len(current) / 5, len(covered) / len(world["goals"]),
            float(present and (slack < 0 or free_slots < 0)),
            float(rejected in item["requires"]), float(rejected == key),
            *(float(reason in reasons) for reason in VIOLATIONS),
        ])
    stop = np.array([
        len(current) / 5, len(covered) / len(world["goals"]), quality / 44,
        slack / 30, free_slots / 4, float(slack < 0), float(free_slots < 0),
        len(bad_selected) / 4, float(bool(bad_selected)), float(bool(feasible_adds)),
        len(feasible_adds) / 5, max((i[0] for i in feasible_adds), default=0) / 4,
        max((i[1] for i in feasible_adds), default=0) / 12,
        min((i["quality"] for i in current), default=0) / 12,
    ])
    return np.pad(np.asarray(rows), ((0, MAX_ITEMS - len(rows)), (0, 0))), stop


def slot(action: dict) -> int:
    return int(action["id"][1:])


class ActionPolicy:
    """Shared scorer with an optional tied tanh recurrent transition."""

    def __init__(self, seed: int, width: int, *, recurrent: bool, hidden: int = 12):
        rng = np.random.default_rng(seed)
        self.wx = rng.normal(0, 0.5 / np.sqrt(width), size=(width, hidden))
        self.wo = rng.normal(0, 0.5 / np.sqrt(hidden), size=hidden)
        self.b = np.zeros(hidden)
        self.wh = rng.normal(0, 0.3 / np.sqrt(hidden), size=(hidden, hidden)) if recurrent else None

    @property
    def parameters(self) -> tuple[np.ndarray, ...]:
        base = (self.wx, self.wo, self.b)
        return base if self.wh is None else (*base, self.wh)

    def states(self, x: np.ndarray, depth: int) -> Iterator[np.ndarray]:
        if not 1 <= depth <= 8 or (depth > 1 and self.wh is None):
            raise ValueError("Depth outside the registered action policy")
        drive = x @ self.wx + self.b
        state = np.tanh(drive)
        yield state
        for _ in range(1, depth):
            state = np.tanh(drive + state @ self.wh)
            yield state

    def fit(
        self, x: np.ndarray, y: np.ndarray, mask: np.ndarray, *, steps: int, lr: float, depth: int,
    ) -> list[float]:
        history = []
        flat = x.reshape(-1, x.shape[-1])
        for step in range(steps):
            states = list(self.states(x, depth))
            loss, grad = _cross_entropy(states[-1] @ self.wo, y, mask)
            dwo = states[-1].reshape(-1, len(self.wo)).T @ grad.ravel()
            dh = grad[..., None] * self.wo
            dwh = np.zeros_like(self.wh) if self.wh is not None else None
            ddrive = np.zeros_like(dh)
            for i in range(depth - 1, -1, -1):
                dz = dh * (1 - states[i] ** 2)
                ddrive += dz
                if i:
                    dwh += states[i - 1].reshape(-1, len(self.wo)).T @ dz.reshape(-1, len(self.wo))
                    dh = dz @ self.wh.T
            dz_flat = ddrive.reshape(-1, len(self.wo))
            gradients = [flat.T @ dz_flat, dwo, dz_flat.sum(axis=0)]
            if dwh is not None:
                gradients.append(dwh)
            eta = lr / (1 + step / steps)
            for parameter, delta in zip(self.parameters, _clip(gradients), strict=True):
                parameter -= eta * delta
            history.append(float(loss))
        return history

    def work_per_update(self, x: np.ndarray, depth: int) -> int:
        """Dense multiply-accumulate proxy for forward and backward matmuls."""
        hidden = len(self.wo)
        return len(x) * x.shape[1] * (2 * x.shape[2] * hidden + 2 * hidden + 3 * (depth - 1) * hidden**2)

    def digest(self) -> str:
        return digest(b"".join(p.astype("<f8", copy=False).tobytes() for p in self.parameters))


def make_training_batch(trajectories: list[dict], config: dict) -> tuple[np.ndarray, ...]:
    x, y, masks, stop_x, stop_y = [], [], [], [], []
    for trajectory in trajectories:
        world = trajectory["world"]
        frames = trajectory["transitions"]
        for frame in frames:
            selected = frame["selected"]
            _, stop = features(world, selected)
            terminal = frame["next_edit"] == STOP
            stop_x.append(stop)
            stop_y.append(terminal)
            if terminal:
                continue
            target = frames[0] if config["supervision"] == "initial" else frame
            selected = target["selected"]
            counterfactual = target["counterfactual"]
            feedback = None
            # Keep the same clean/counterfactual balance for each paired fit.
            if not counterfactual["receipt"]["accepted"]:
                feedback = {
                    "action": counterfactual["action"],
                    "new_violations": counterfactual["receipt"]["veto_violations"],
                }
            for context in (None, feedback):
                action_features, _ = features(world, selected, context if config["feedback"] else None)
                x.append(action_features)
                label = slot(target["next_edit"])
                mask = np.arange(MAX_ITEMS) < len(world["items"])
                if context:
                    mask[slot(context["action"])] = False
                masks.append(mask)
                y.append(label)
    return np.stack(x), np.array(y), np.stack(masks), np.stack(stop_x), np.array(stop_y)


def allocation_features(scores: np.ndarray, allowed: np.ndarray, stop: np.ndarray) -> np.ndarray:
    ranked = np.sort(scores[allowed])
    gap = ranked[-1] - ranked[-2] if len(ranked) > 1 else 0.0
    return np.append(stop, [ranked[-1], gap, len(ranked) / MAX_ITEMS])


def fit_allocator(model: ActionPolicy, x: np.ndarray, y: np.ndarray, masks: np.ndarray, stop_x: np.ndarray) -> tuple:
    """At depth two, learn when continuing to depth eight fixes the action."""
    early, final = None, None
    for depth, state in enumerate(model.states(x, 8), 1):
        if depth == 2:
            early = state @ model.wo
        if depth == 8:
            final = state @ model.wo
    early_choice = np.where(masks, early, -np.inf).argmax(axis=1)
    late_choice = np.where(masks, final, -np.inf).argmax(axis=1)
    benefit = (early_choice != y) & (late_choice == y)
    inputs = np.stack([
        allocation_features(scores, mask, stop)
        for scores, mask, stop in zip(early, masks, stop_x, strict=True)
    ])
    receipt = {"train_deeper_repairs": int(benefit.sum()), "examples": len(y), "calibration_passes": 8 * len(y)}
    if benefit.any() and not benefit.all():
        head = StopHead(101, input_dim=inputs.shape[1])
        receipt.update(head.fit(inputs, benefit, steps=32, lr=0.25, max_false_stop_rate=0.2))
        probabilities = head.probabilities(inputs)
        receipt.update(
            updates=32, initialization_seed=101, checkpoint_sha256=head.digest(), parameters=head.params,
            optimizer_matmul_macs=2 * 32 * inputs.size,
            final_loss=float(-np.log(np.where(benefit, probabilities, 1 - probabilities)).mean()),
        )
        return head, receipt
    receipt.update(updates=0, parameters=0, constant_extend=bool(benefit.all()))
    return bool(benefit.all()), receipt
