"""Frozen finite research games with actor-relative observations and legal interventions."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

from .hypergame_taste import _digest

FAMILIES = (
    "idea_generation", "experimental_design", "literature_selection",
    "algorithm_improvement", "scientific_replication", "failure_diagnosis",
)
DIMENSIONS = (140, 512, 1024)
HORIZONS = (2, 3, 4)
DEPTHS = (0, 1, 2, 4, 8, 12, 16)
DENOMINATOR = 16
MODES = (0, 1, 2)
ROLES = ("scout", "reviewer", "replicator", "funder")
TERMINAL = ("abstain", "publish:0", "publish:1", "publish:2")


@dataclass(frozen=True)
class State:
    credits: int
    turns: int
    history: tuple[tuple[str, int], ...] = ()


def _message_distribution(actor, upstream, mode):
    """Integrate the actor's private binary observation conditional on its predecessor."""
    total = 0
    for signal in (0, 1):
        stake = actor["incentive"] + actor["trust"] * (2 * upstream - 1)
        announce = stake + actor["sensor_weight"] * (2 * signal - 1) > 0
        if announce:
            p = actor["signal_p16"][mode]
            total += p if signal else DENOMINATOR - p
    return total


def message_likelihood(public):
    """Unnormalised likelihood of the public endorsement under each objective mode."""
    masses = list(public["origin_p16"])
    denominator = DENOMINATOR
    for actor in public["views"]:
        masses = [
            (denominator - mass) * _message_distribution(actor, 0, mode)
            + mass * _message_distribution(actor, 1, mode)
            for mode, mass in enumerate(masses)
        ]
        denominator *= DENOMINATOR
    return tuple(masses if public["endorsement"] else (denominator - mass for mass in masses))


def enumerated_likelihood(public):
    """Independent small-K oracle that explicitly enumerates private signal paths."""
    if len(public["views"]) > 4:
        raise ValueError("signal-path enumeration is limited to K <= 4")
    result = []
    for mode in MODES:
        total = 0
        for origin, *signals in product((0, 1), repeat=1 + len(public["views"])):
            message = origin
            weight = public["origin_p16"][mode] if origin else DENOMINATOR - public["origin_p16"][mode]
            for actor, signal in zip(public["views"], signals, strict=True):
                chance = actor["signal_p16"][mode]
                weight *= chance if signal else DENOMINATOR - chance
                perceived = actor["trust"] if message else -actor["trust"]
                observed = actor["sensor_weight"] if signal else -actor["sensor_weight"]
                message = int(perceived + observed > -actor["incentive"])
            if message == public["endorsement"]:
                total += weight
        result.append(total)
    return tuple(result)


def generate(depth, split, rng, ordinal):
    """Create paired public/sealed records. The full transition tape lives with the evaluator."""
    if depth not in DEPTHS:
        raise ValueError("unregistered epistemic depth")
    index = ordinal % (len(FAMILIES) * len(DIMENSIONS) * len(HORIZONS))
    category = FAMILIES[index % len(FAMILIES)]
    dimension = DIMENSIONS[(index // len(FAMILIES)) % len(DIMENSIONS)]
    horizon = HORIZONS[index // (len(FAMILIES) * len(DIMENSIONS))]
    condition = rng.choice(("incentive_shift", "confounded_sensor", "observation_alias")) if split == "ood_test" else (
        "canary" if split == "sealed_poison" else "ordinary"
    )
    bits = [rng.getrandbits(1) for _ in range(dimension)]
    indices = [rng.sample(range(dimension), width) for width in (7, 13, 19)]
    gates = [sum(bits[i] for i in group) % 2 for group in indices]
    excluded = rng.randrange(len(MODES))
    modes = [mode for mode in MODES if mode != excluded]
    prior = [rng.randrange(2, 7) for _ in MODES]
    if ordinal % 4 == 0:
        prior[excluded] += 6
    truth = rng.choices(MODES, weights=prior, k=1)[0]
    origin = [4, 7, 12]
    rng.shuffle(origin)
    report = int(rng.randrange(DENOMINATOR) < origin[truth])
    views = []
    for i in range(depth):
        sensor = [3, 7, 13]
        rng.shuffle(sensor)
        if condition == "observation_alias":
            sensor[2] = sensor[1]
        trust = rng.choice((1, 2, 3))
        if condition == "incentive_shift":
            trust *= -1
        actor = {
            "role": f"{ROLES[i % len(ROLES)]}_{i + 1}",
            "models": views[-1]["role"] if views else "origin",
            "observes": ("predecessor_endorsement", "private_sensor"),
            "actions": ("endorse", "oppose"),
            "objective": "fund_publication" if i % 2 == 0 else "avoid_false_publication",
            "trust": trust, "sensor_weight": rng.choice((2, 3)),
            "signal_p16": sensor, "incentive": rng.choice((-1, 0, 1)) + (1 if i % 2 == 0 else -1),
        }
        chance = _message_distribution(actor, report, truth)
        report = int(rng.randrange(DENOMINATOR) < chance)
        views.append(actor)
    target = modes[gates[0]]
    false_signal = 11 if condition == "confounded_sensor" else 5
    experiments = {
        "pilot": {"cost": 1, "p16": [12 if m == target else false_signal for m in MODES]},
        "replicate": {"cost": 2, "p16": [14 if m == modes[1] else 2 for m in MODES]},
        f"intervene:{excluded}": {
            "cost": 1 + gates[1], "p16": [15 if m == excluded else 1 + gates[2] for m in MODES],
        },
    }
    # The research category changes the experiment contract as well as the words in the task.
    shift = FAMILIES.index(category)
    experiments["pilot"]["p16"][target] -= shift % 3
    experiments["replicate"]["p16"][excluded] += shift // 2
    if condition == "observation_alias":
        origin[2] = origin[1]
        # Generate the public report again under the declared aliased origin.
        report = int(rng.randrange(DENOMINATOR) < origin[truth])
        for actor in views:
            report = int(rng.randrange(DENOMINATOR) < _message_distribution(actor, report, truth))
    public = {
        "K": depth, "T": horizon, "dimension": dimension, "family": category, "condition": condition,
        "world_bits": "".join(map(str, bits)), "gate_indices": indices,
        "prior_weights": prior, "origin_p16": origin, "views": views, "endorsement": report,
        "credits": rng.choice((2, 3, 4)), "native_hypotheses": modes, "experiments": experiments,
        "early_rewards": [10 + shift % 3, -14 - gates[0]],
        "late_rewards": [8 + gates[2], -17 - gates[1]],
        "goal_change": 1 + gates[0],
        "extension": {"rule": "intervene on or publish the missing causal mode", "mode": excluded},
    }
    root = _digest([public, split, ordinal])
    tape = {
        name: [int(rng.randrange(DENOMINATOR) < spec["p16"][truth]) for _ in range(horizon)]
        for name, spec in experiments.items()
    }
    return {"root": root, "split": split, "public": public}, {
        "root": root, "truth": truth, "experiment_tape": tape,
    }


class Game:
    """Public decision process and exact finite-horizon expected utility.

    The recursion propagates integer masses and multiplies terminal payoffs by
    16**(turns-1) so every action at a fork has the same scale.
    """

    def __init__(self, public):
        self.public = public
        self.experiments = public["experiments"]
        self.initial = State(public["credits"], public["T"])
        self.prior = tuple(public["prior_weights"])
        self.initial_masses = tuple(a * b for a, b in zip(self.prior, message_likelihood(public), strict=True))
        self.native = {"abstain", "pilot", "replicate", *(f"publish:{m}" for m in public["native_hypotheses"])}
        self._solve = lru_cache(maxsize=4096)(self._values)

    def rewards(self, turns):
        elapsed = self.initial.turns - turns
        return self.public["late_rewards" if elapsed >= self.public["goal_change"] else "early_rewards"]

    def legal(self, state):
        return TERMINAL + tuple(sorted(
            action for action, spec in self.experiments.items() if state.turns > 1 and spec["cost"] <= state.credits
        ))

    def after(self, state, action, answer):
        if action not in self.legal(state) or action not in self.experiments or answer not in (0, 1):
            raise ValueError("illegal experiment successor")
        return State(state.credits - self.experiments[action]["cost"], state.turns - 1,
                     (*state.history, (action, answer)))

    def outcome(self, state, action, sealed):
        """Counterfactual successor for a declared legal action on the frozen tape."""
        if action not in self.legal(state):
            raise ValueError("illegal action")
        if action == "abstain":
            return None, 0
        if action in TERMINAL:
            reward = self.rewards(state.turns)[int(action != f"publish:{sealed['truth']}")]
            return None, reward
        occurrence = sum(old == action for old, _ in state.history)
        answer = sealed["experiment_tape"][action][occurrence]
        return self.after(state, action, answer), -self.experiments[action]["cost"]

    def observe(self, masses, action, answer):
        weights = self.experiments[action]["p16"]
        return tuple(mass * (p if answer else DENOMINATOR - p) for mass, p in zip(masses, weights, strict=True))

    def masses(self, state):
        masses = self.initial_masses
        for action, answer in state.history:
            masses = self.observe(masses, action, answer)
        return masses

    def _values(self, masses, credits, turns, planning, native_only):
        scale = DENOMINATOR ** (planning - 1)
        total = sum(masses)
        correct, wrong = self.rewards(turns)
        values = {"abstain": 0}
        for mode in MODES:
            name = f"publish:{mode}"
            if not native_only or name in self.native:
                values[name] = scale * (masses[mode] * correct + (total - masses[mode]) * wrong)
        if planning > 1:
            for action in self.legal(State(credits, turns))[len(TERMINAL):]:
                if native_only and action not in self.native:
                    continue
                cost = self.experiments[action]["cost"]
                future = sum(max(value for _, value in self._solve(
                    self.observe(masses, action, answer), credits - cost, turns - 1, planning - 1, native_only,
                )) for answer in (0, 1))
                values[action] = future - total * scale * cost
        return tuple(values.items())

    def values(self, state, *, lookahead=None, native_only=False, ignore_views=False):
        planning = min(state.turns, lookahead or state.turns)
        masses = self.masses(state)
        if ignore_views:
            masses = self.prior
            for action, answer in state.history:
                masses = self.observe(masses, action, answer)
        divisor = sum(masses) * DENOMINATOR ** (planning - 1)
        return {action: score / divisor for action, score in self._solve(
            masses, state.credits, state.turns, planning, native_only,
        )}
