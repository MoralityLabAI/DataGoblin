"""Evaluate live LDT stopping on fresh graph revisions with frozen TRM weights."""

from __future__ import annotations

import json
import secrets
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from datagoblin.hashing import sha256_file
from datagoblin.serialization import canonical_json, dump_structured, load_structured

from . import rlt_pilot, rlt_route_certificate
from . import rlt_revision as revision
from . import rlt_trm as model
from .rlt_pilot import pending_rows
from .rlt_pilot_data import jsonl
from .rlt_route_certificate import observe_route, patched_endpoint

DEPTHS = (6, 10)
POLICIES = ("fixed", "verified", "verified_no_op_reuse")


def freeze(root: Path, previous_packs: list[Path]):
    """Draw new lineages and balance the long holdout across query lengths."""
    seen = set()
    previous_hashes = {}
    for pack in previous_packs:
        manifest_path = pack / "manifest.json"
        manifest = load_structured(manifest_path)
        for split in revision.SPLITS:
            path = pack / f"{split}.tasks.jsonl"
            if sha256_file(path) != manifest["splits"][split]["tasks_sha256"]:
                raise ValueError(f"source pack task file changed: {path}")
            seen.update(row["lineage"] for row in jsonl(path))
        previous_hashes[sha256_file(manifest_path)] = manifest["version"]
    previous_lineages = len(seen)
    selected = []
    seed = secrets.randbits(128)
    for split, quota in (("revision_ood", {0: 64}), ("length_ood", dict.fromkeys((7, 8, 9), 32))):
        pairs = iter(revision.episodes(seed, split, 512))
        counts = Counter()
        for first, second in zip(pairs, pairs, strict=True):
            task = first[0]
            bucket = 0 if split == "revision_ood" else task["steps"]
            if counts[bucket] >= quota[bucket] or task["lineage"] in seen:
                continue
            seen.add(task["lineage"])
            counts[bucket] += 1
            selected.extend((dict(item, split=split), target) for item, target in (first, second))
            if counts == quota:
                break
        if counts != quota:
            raise ValueError("unable to fill the fresh holdout")
    tasks_path, targets_path = root / "tasks.jsonl", root / "targets.jsonl"
    for idx, path in enumerate((tasks_path, targets_path)):
        with pending_rows(path) as handle:
            for pair in selected:
                handle.write(canonical_json(pair[idx]) + "\n")
    return {
        "generator": revision.VERSION, "items": len(selected), "lineages": len(selected) // 2,
        "previous_lineages_excluded": previous_lineages, "previous_manifest_sha256": previous_hashes,
        "tasks_sha256": sha256_file(tasks_path), "targets_sha256": sha256_file(targets_path),
    }


class EndpointGate:
    """Expose only accept or abstain to the loop and keep the route in the verifier."""

    def __init__(self, membrane, links, start, steps, *, epoch=0, witness=None, patch=None):
        self.membrane = membrane
        self.links, self.start, self.steps = links, start, steps
        self.epoch, self.witness, self.patch = epoch, witness, patch
        self.state = membrane.CandidateState({"endpoint": frozenset(revision.NODES)})
        self.endpoint = None
        self.hops = 0
        self.attempts = []

    def advance(self, patch):
        if self.epoch != 0 or self.witness is None:
            raise ValueError("the first-turn verifier must observe its route before the patch")
        return EndpointGate(
            self.membrane, self.links, self.start, self.steps,
            epoch=1, witness=self.witness, patch=patch,
        )

    def _verify(self, proposal, _context):
        soundness = self.membrane.SoundnessType
        if proposal.metadata["epoch"] != self.epoch:
            return soundness.UNKNOWN
        if self.endpoint is None:
            if self.patch is None:
                self.witness = observe_route(self.links, self.start, self.steps)
                self.endpoint, self.hops = self.witness.nodes[-1], self.steps
            else:
                self.endpoint, self.hops = patched_endpoint(self.links, self.witness, self.patch)
        return (
            soundness.ENV_SOUND_DEAD
            if proposal.proposed_state.domains["endpoint"] == {self.endpoint} else soundness.UNKNOWN
        )

    def check(self, visit, candidate, *, epoch):
        proposal = self.membrane.LatticeProposal(
            self.membrane.CandidateState({"endpoint": frozenset({candidate})}),
            self.membrane.SoundnessType.MODEL_SOUND_DEAD, metadata={"epoch": epoch},
        )
        decision = self.membrane.certify_and_apply_verified(self.state, proposal, provenance_verifier=self._verify)
        self.attempts.append([visit, candidate, epoch, decision.accepted])
        self.state = decision.after
        return decision.accepted

    def stop(self, visit, answer):
        return self.check(visit, revision.NODES[int(answer.argmax())], epoch=self.epoch)

    def result(self):
        visit, candidate, _, accepted = self.attempts[-1]
        return {
            "prediction": candidate if accepted else None, "visits": visit,
            "checks": len(self.attempts), "checker_hops": self.hops, "attempts": self.attempts,
        }


def evaluate_case(weights, task, mode, depth, membrane, links):
    start, hops, before, after = model.prepare([task])
    zeros = np.zeros_like(start)
    source, target = task["patch"]["source"], task["patch"]["destination"]
    no_op = links[source] == target
    results = {}
    for policy in POLICIES:
        gate = None if policy == "fixed" else EndpointGate(membrane, links, task["start"], task["steps"])
        logits, carry = model.refine(
            weights, before, start, hops, zeros, mode, depth, placement="answer",
            stop_when=None if gate is None else gate.stop,
        )
        first = gate.result() if gate else fixed_result(logits, depth)
        # The patch enters only the second refinement turn and its verifier epoch.
        next_gate = gate.advance({source: target}) if gate else None
        reuse = policy == "verified_no_op_reuse" and no_op and first["prediction"] is not None
        if reuse:
            if not next_gate.check(0, first["prediction"], epoch=1):
                raise ValueError("current-epoch verification failed for an unchanged graph")
            second = next_gate.result()
        else:
            logits, _ = model.refine(
                weights, after, start, hops, carry, mode, depth, placement="answer",
                stop_when=None if next_gate is None else next_gate.stop,
            )
            second = next_gate.result() if next_gate else fixed_result(logits, depth)
        results[policy] = {"before": first, "after": second}
    return results


def fixed_result(logits, depth):
    return {"prediction": revision.NODES[int(logits.argmax())], "visits": depth,
            "checks": 0, "checker_hops": 0, "attempts": []}


def tally(path, targets_path, out, guard):
    """Score staged predictions against labels only after online evaluation."""
    targets = revision.read_rows(targets_path, {"id", "lineage", "before", "after"})
    totals = defaultdict(Counter)
    keys = set()
    with path.open(encoding="utf-8") as handle, pending_rows(out) as scored:
        for line in handle:
            guard()
            row = json.loads(line)
            target = targets[row["id"]]
            key = row["seed"], row["mode"], row["depth"], row["id"]
            cohort = row["cohort"]
            if key in keys or target["lineage"] != row["lineage"]:
                raise ValueError("duplicate item or incorrect lineage")
            keys.add(key)
            if (target["before"] == target["after"]) != cohort.endswith("no_op"):
                raise ValueError("fixture target does not match its patch variant")
            for policy, result in row["results"].items():
                scores = {}
                for turn, receipt in result.items():
                    prediction = receipt["prediction"]
                    scores[f"{turn}_exact"] = prediction == target[turn]
                    if policy != "fixed" and prediction is not None and not scores[f"{turn}_exact"]:
                        raise ValueError("the live verifier accepted an incorrect endpoint")
                    for metric in ("visits", "checks", "checker_hops"):
                        scores[f"{turn}_{metric}"] = receipt[metric]
                scores["joint_exact"] = scores["before_exact"] and scores["after_exact"]
                scores["zero_visit_after"] = result["after"]["visits"] == 0
                scores["items"] = 1
                result["scores"] = scores
                for steps in (0, row["steps"]):
                    totals[(row["depth"], cohort, steps, row["mode"], policy)].update(scores)
            scored.write(canonical_json(row) + "\n")
        if len(keys) != len(targets) * len(model.SEEDS) * len(model.MODES) * len(DEPTHS):
            raise ValueError("incomplete holdout coverage")
    return [
        dict(depth=depth, cohort=cohort, hops=hops or None, mode=mode, policy=policy, **counts)
        for (depth, cohort, hops, mode, policy), counts in sorted(totals.items())
    ]


def audit_epoch(membrane, task):
    links = dict(task["links"])
    source, target = task["patch"]["source"], task["patch"]["destination"]
    endpoint = revision.follow(links, task["start"], task["steps"])
    gate = EndpointGate(membrane, links, task["start"], task["steps"])
    if links[source] != target or not gate.check(1, endpoint, epoch=0):
        raise ValueError("invalid no-op audit fixture")
    next_gate = gate.advance({source: target})
    top = next_gate.state
    if next_gate.check(0, endpoint, epoch=0) or next_gate.state != top:
        raise ValueError("stale no-op provenance was accepted")
    if not next_gate.check(0, endpoint, epoch=1) or next_gate.hops:
        raise ValueError("the current no-op certificate could not be independently validated")
    return {"id": task["id"], "attempts": next_gate.attempts, "checker_hops": next_gate.hops}


def run(root, checkpoints, previous_packs, membrane, guard):
    started = time.monotonic()
    source_protocol = load_structured(checkpoints / "protocol.json")
    source_manifest = source_protocol["rule_manifest_sha256"]
    if source_protocol["placement"] != "answer" or source_protocol["train_depth"] != model.TRAIN_DEPTH:
        raise ValueError("expected frozen answer-placement checkpoints")
    fixture_path = root / "fixture.json"
    if not fixture_path.exists():
        dump_structured(fixture_path, freeze(root, previous_packs))
    fixture = load_structured(fixture_path)
    if source_manifest not in fixture["previous_manifest_sha256"]:
        raise ValueError("the checkpoint training pack was not included in the exclusion audit")
    tasks_path, targets_path = root / "tasks.jsonl", root / "targets.jsonl"
    for key, path in (("tasks", tasks_path), ("targets", targets_path)):
        if sha256_file(path) != fixture[f"{key}_sha256"]:
            raise ValueError("frozen holdout changed")
    tasks = jsonl(tasks_path)
    weights = {}
    weight_hashes = {}
    for seed in model.SEEDS:
        matched = set()
        for mode in model.MODES:
            directory = checkpoints / str(seed) / mode
            path = directory / "weights.npz"
            receipt = load_structured(directory / "train.json")
            digest = sha256_file(path)
            expected = {
                "placement": "answer", "weights_sha256": digest, "seed": seed, "mode": mode,
                "steps": model.STEPS, "training_manifest_sha256": source_manifest,
            }
            if any(receipt[key] != value for key, value in expected.items()):
                raise ValueError("weights do not match their training receipt")
            with np.load(path, allow_pickle=False) as tensors:
                weights[seed, mode] = dict(tensors)
            weight_hashes[f"{seed}/{mode}"] = digest
            matched.add((receipt["init_sha256"], receipt["schedule_sha256"], receipt["cell_calls"]))
        if len(matched) != 1:
            raise ValueError("expected matched training compute, initialization and batch order")
    protocol = {
        "depths": DEPTHS, "modes": model.MODES, "policies": POLICIES, "seeds": model.SEEDS,
        "cell_calls_per_visit": 2, "weights_sha256": weight_hashes, "fixture_sha256": sha256_file(fixture_path),
        "checkpoint_protocol_sha256": sha256_file(checkpoints / "protocol.json"), "numpy": np.__version__,
        "source_sha256": {
            "loop": sha256_file(Path(model.__file__)), "evaluator": sha256_file(Path(__file__)),
            "membrane": sha256_file(Path(membrane.__file__)),
            "feedback": sha256_file(Path(rlt_pilot.__file__)), "scorer": sha256_file(Path(revision.__file__)),
            "route_certificate": sha256_file(Path(rlt_route_certificate.__file__)),
        },
    }
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and canonical_json(load_structured(protocol_path)) != canonical_json(protocol):
        raise ValueError("the frozen protocol changed")
    dump_structured(protocol_path, protocol)
    raw_path = root / "predictions.jsonl"
    if not raw_path.exists():
        with pending_rows(raw_path) as raw:
            for (seed, mode), w in weights.items():
                for depth in DEPTHS:
                    for task in tasks:
                        guard()
                        links = dict(task["links"])
                        no_op = links[task["patch"]["source"]] == task["patch"]["destination"]
                        horizon = "long" if task["split"] == "length_ood" else "short"
                        cohort = f"{horizon}_{'no_op' if no_op else 'changed'}"
                        row = {
                            "seed": seed, "mode": mode, "depth": depth, "id": task["id"],
                            "lineage": task["lineage"], "steps": task["steps"], "cohort": cohort,
                            "results": evaluate_case(w, task, mode, depth, membrane, links),
                        }
                        raw.write(canonical_json(row) + "\n")
    scored_path = root / "item-outcomes.jsonl"
    totals = tally(raw_path, targets_path, scored_path, guard)
    no_op = next(task for task in tasks if dict(task["links"])[task["patch"]["source"]] == task["patch"]["destination"])
    dump_structured(root / "evidence.json", {
        "schema": "datagoblin.rlt-trm-ldt-stopping.v1", "fixture": fixture, "protocol": protocol,
        "cells": totals, "scored_items_sha256": sha256_file(scored_path),
        "epoch_audit": audit_epoch(membrane, no_op),
        "predictions_sha256": sha256_file(raw_path), "evaluation_wall_s": time.monotonic() - started,
    })
