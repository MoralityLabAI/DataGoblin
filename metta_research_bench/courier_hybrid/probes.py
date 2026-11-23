"""Native-backed paired state probes and causal activation replacement."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from statistics import mean

import torch

from ..looped.model import LOOPS
from .controller import Observation
from .learning import tokenize
from .native import DELIVER, PLACES, STOP, state_atom, state_digest
from .oracle import state_graph


def _pairs(graph):
    pairs = {}
    for state in sorted(graph):
        if not state.permit and state.cargo != 2:
            other = replace(state, permit=True)
            new = set(graph.get(other, ())) - graph[state].keys()
            gated = sorted(a for a in new if a < PLACES)
            if gated:
                pairs.setdefault("permit_route", (state, other, gated[0]))
        if DELIVER in graph[state]:
            for bit in (1, 2):
                other = replace(state, evidence=state.evidence ^ bit)
                if other in graph and DELIVER not in graph[other]:
                    pairs.setdefault("evidence_delivery", (other, state, DELIVER))
            pairs.setdefault("terminal_stop", (state, graph[state][DELIVER], STOP))
    return pairs


@torch.inference_mode()
def patch_effect(model, left: Observation, right: Observation, action: int) -> list[dict]:
    tokens = torch.stack([tokenize(obs)[0] for obs in (left, right)])
    initial = model.encoded(tokens)
    snapshots = [initial]
    for visit in range(LOOPS):
        snapshots.append(model.visit(snapshots[-1], initial, visit))
    unpatched = model.readout(snapshots[-1])[:, action]
    placebo_slot = (action + 1) % PLACES
    results = []
    for visit in range(1, LOOPS + 1):

        def replace_slot(slot: int, after_visit=visit):
            hidden = snapshots[after_visit][:1].clone()
            hidden[:, slot] = snapshots[after_visit][1, slot]
            for next_visit in range(after_visit, LOOPS):
                hidden = model.visit(hidden, initial[:1], next_visit)
            return float(model.readout(hidden)[0, action] - unpatched[0])

        results.append(
            {
                "after_visit": visit,
                "paired_logit_gap": float(unpatched[1] - unpatched[0]),
                "targeted_patch_delta": replace_slot(action),
                "placebo_patch_delta": replace_slot(placebo_slot),
            }
        )
    return results


def probe_world(native, world, models, root, seeds, horizon, ceiling):
    graph, _ = state_graph(world)
    probes = []
    for name, (before, after, action) in _pairs(graph).items():
        left = native.checked_successors(world, before)
        right = native.checked_successors(world, after)
        if action == STOP:
            stopped = [native.query(world, f"(HybridAcceptStop {state_atom(state)})") for state in (before, after)]
            if stopped != [set(), {f"(HybridDone {state_atom(after)})"}]:
                raise RuntimeError("Native STOP probe failed")
        elif action in left or action not in right:
            raise RuntimeError("Native obligation probe failed")
        obs_left = Observation(world, before, (*sorted(left), STOP), horizon, ceiling)
        obs_right = Observation(world, after, (*sorted(right), STOP), horizon, ceiling)
        for model_name in ("shortest", "obligation"):
            for seed in seeds:
                probes.extend(
                    {
                        "root": root.world_sha256,
                        "family": root.family,
                        "probe": name,
                        "action": action,
                        "before_state_sha256": state_digest(world, before),
                        "after_state_sha256": state_digest(world, after),
                        "model": model_name,
                        "seed": seed,
                        **row,
                    }
                    for row in patch_effect(models[model_name, seed], obs_left, obs_right, action)
                )
    return probes


def summarize_probes(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"], row["probe"], row["after_visit"]].append(row)
    return {
        f"{model}/{probe}/visit-{visit}": {
            "roots": len({row["root"] for row in group}),
            "paired_logit_gap": mean(r["paired_logit_gap"] for r in group),
            "targeted_patch_delta": mean(r["targeted_patch_delta"] for r in group),
            "placebo_patch_delta": mean(r["placebo_patch_delta"] for r in group),
        }
        for (model, probe, visit), group in sorted(groups.items())
    }
