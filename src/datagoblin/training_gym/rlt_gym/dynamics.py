"""Small trainable CPU control with a frozen recurrent reservoir.

Only the ridge readout is fitted. The shared tanh block, feature encoding and
time inputs stay frozen. These runs test feedback and rollout plumbing; they
do not establish a trained transformer or BPD component decomposition.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections.abc import Iterable, Iterator
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.serialization import canonical_json, dump_structured, load_structured
from datagoblin.training_gym.rlt_feedback import feedback_weight

READOUT_SITES = ("endpoint", "carried_answer")
MODEL_SCHEMA = "datagoblin-rlt-native-ranker-v1"
MAX_CANDIDATES = 512
MAX_DEPTH = 256


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _public_input(task: dict[str, Any]) -> dict[str, Any]:
    return {key: task[key] for key in ("family", "prompt", "input")}


def _integer(value: Any, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return int(value)


def _finite_number(value: Any, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must lie in [{lower}, {upper}]")
    return value


def _summary(array: np.ndarray) -> dict[str, Any]:
    if not np.isfinite(array).all():
        raise ValueError("nonfinite recurrent state")
    return {
        "shape": list(array.shape),
        "rms": float(np.sqrt(np.mean(array * array))),
        "l2": float(np.linalg.norm(array)),
        "mean": float(array.mean()),
        "max_abs": float(np.max(np.abs(array))),
        "sha256": _digest({"shape": list(array.shape), "values": array.tolist()}),
    }


def _readout_site(site: str) -> None:
    if site not in READOUT_SITES:
        raise ValueError(f"unknown readout site {site!r}")


class NativeRanker:
    """Ridge-trained ranker using the same frozen block at each recurrent visit."""

    def __init__(self, seed: int, width: int = 16, features: int = 64, ridge: float = 0.1):
        self.seed = _integer(seed, "seed", 0, 2**63 - 1)
        self.width = _integer(width, "width", 1, 256)
        self.features = _integer(features, "features", 8, 4096)
        self.ridge = _finite_number(ridge, "ridge", 1e-12, 1e6)
        rng = np.random.default_rng(self.seed)
        recurrent = rng.normal(size=(self.width, self.width))
        recurrent *= 0.6 / max(float(np.linalg.norm(recurrent, ord=2)), 1e-12)
        self.weights = {
            "input": rng.normal(0, 1.0, (self.features, self.width)),
            "recurrent": recurrent,
            "time": rng.normal(0, 0.15, (2, self.width)),
            "bias": rng.normal(0, 0.05, self.width),
            "readout": rng.normal(0, 0.05, self.width + 1),
        }
        self._metadata: dict[str, Any] = {
            "schema_version": MODEL_SCHEMA,
            "backend": "numpy_cpu",
            "model_kind": "frozen_random_reservoir",
            "training_kind": "trained_readout",
            "fitted": False,
            "seed": self.seed,
            "width": self.width,
            "features": self.features,
            "ridge": self.ridge,
            "initialization_digest": _digest(self._weight_lists()),
            "trainable_parameters": self.width + 1,
            "total_parameters": sum(w.size for w in self.weights.values()),
            "readout_kind": "linear_state_ridge",
            "state_placement": "answer",
            "architecture_promotion_eligible": False,
            "bpd_mechanism_verified": False,
            "training_splits": [],
            "training_lineages": [],
            "training_task_digests": [],
            "training_input_fingerprints": [],
        }

    def _weight_lists(self) -> dict[str, Any]:
        return {name: value.tolist() for name, value in sorted(self.weights.items())}

    def _payload(self) -> dict[str, Any]:
        return {"schema_version": MODEL_SCHEMA, "metadata": copy.deepcopy(self._metadata),
                "weights": self._weight_lists()}

    @property
    def model_digest(self) -> str:
        return _digest(self._payload())

    @property
    def metadata(self) -> dict[str, Any]:
        return {**copy.deepcopy(self._metadata), "model_digest": self.model_digest}

    def snapshot(self) -> dict[str, Any]:
        payload = self._payload()
        return {**payload, "model_digest": _digest(payload)}

    def save(self, path: Path | str) -> None:
        """Write a checkpoint to an operator-owned, ignored run directory."""
        dump_structured(Path(path), self.snapshot())

    @classmethod
    def load(cls, path: Path | str) -> NativeRanker:
        snapshot = load_structured(Path(path))
        if set(snapshot) != {"schema_version", "metadata", "weights", "model_digest"}:
            raise ValueError("invalid ranker snapshot fields")
        payload = {k: v for k, v in snapshot.items() if k != "model_digest"}
        if snapshot["schema_version"] != MODEL_SCHEMA or _digest(payload) != snapshot["model_digest"]:
            raise ValueError("ranker snapshot digest or version mismatch")
        metadata = snapshot["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("invalid ranker metadata")
        model = cls(metadata["seed"], metadata["width"], metadata["features"], metadata["ridge"])
        for key in ("schema_version", "backend", "model_kind", "training_kind", "initialization_digest",
                    "trainable_parameters", "total_parameters", "readout_kind", "state_placement",
                    "architecture_promotion_eligible", "bpd_mechanism_verified"):
            if metadata.get(key) != model._metadata[key]:
                raise ValueError(f"invalid ranker metadata: {key}")
        if metadata.get("training_splits") not in ([], ["train"]):
            raise ValueError("ranker metadata contains an evidence split")
        if type(metadata.get("fitted")) is not bool:
            raise ValueError("fitted must be boolean")
        if metadata["fitted"] != (metadata["training_splits"] == ["train"]):
            raise ValueError("fitted status and training split disagree")
        if set(snapshot["weights"]) != set(model.weights):
            raise ValueError("invalid ranker weights")
        for name, reference in model.weights.items():
            value = np.asarray(snapshot["weights"][name], dtype=np.float64)
            if value.shape != reference.shape or not np.isfinite(value).all():
                raise ValueError(f"invalid ranker weight {name}")
            if name != "readout" and not np.array_equal(value, reference):
                raise ValueError("frozen reservoir differs from its initialization")
            model.weights[name] = value
        model._metadata = copy.deepcopy(metadata)
        model._validate_training_metadata()
        return model

    def _validate_training_metadata(self) -> None:
        if not self._metadata["fitted"]:
            if any(self._metadata[key] for key in ("training_lineages", "training_task_digests",
                                                   "training_input_fingerprints")):
                raise ValueError("unfitted model carries training lineages")
            return
        manifest = self._metadata.get("training_manifest")
        if not isinstance(manifest, list) or not manifest:
            raise ValueError("fitted model is missing its training manifest")
        for row in manifest:
            if not isinstance(row, dict) or row.get("split") != "train":
                raise ValueError("ranker training manifest contains an evidence split")
            if not all(isinstance(row.get(key), str) and row[key]
                       for key in ("task_id", "lineage_id", "task_digest", "public_input_digest")):
                raise ValueError("invalid training manifest identity")
        if len({row["task_id"] for row in manifest}) != len(manifest):
            raise ValueError("duplicate training task")
        if sorted({row["lineage_id"] for row in manifest}) != self._metadata["training_lineages"]:
            raise ValueError("training lineage manifest mismatch")
        if sorted(row["task_digest"] for row in manifest) != self._metadata["training_task_digests"]:
            raise ValueError("training task manifest mismatch")
        if sorted({row["public_input_digest"] for row in manifest}) != self._metadata["training_input_fingerprints"]:
            raise ValueError("training input manifest mismatch")
        if _digest(manifest) != self._metadata.get("training_fingerprint"):
            raise ValueError("training fingerprint mismatch")

    def _encode(self, task: dict[str, Any]) -> np.ndarray:
        from .tasks import validate_task

        validate_task(task)
        candidates = task["candidates"]
        if not 1 <= len(candidates) <= MAX_CANDIDATES:
            raise ValueError("candidate count exceeds the bounded ranker contract")
        public = canonical_json(_public_input(task))
        if len(public) > 262144:
            raise ValueError("public task is too large for the CPU control")
        task_tokens = re.findall(r"[\w.+-]+", public.lower())[:512]
        rows = np.zeros((len(candidates), self.features), dtype=np.float64)
        for index, candidate in enumerate(candidates):
            candidate_tokens = re.findall(r"[\w.+-]+", canonical_json(candidate).lower())[:128]
            tokens = [f"task:{token}" for token in task_tokens]
            tokens.extend(f"candidate:{token}" for token in candidate_tokens)
            # Public task/candidate interactions; identifiers, labels and split never enter.
            tokens.extend(f"match:{token}" for token in set(task_tokens).intersection(candidate_tokens))
            for token in tokens:
                hashed = hashlib.sha256(token.encode("utf-8")).digest()
                column = int.from_bytes(hashed[:8], "little") % self.features
                rows[index, column] += 1.0 if hashed[8] % 2 else -1.0
            rows[index] /= max(float(np.linalg.norm(rows[index])), 1.0)
        return rows

    @staticmethod
    def _validate_intervention(intervention: dict[str, Any] | None, depth: int) -> dict[str, Any] | None:
        if intervention is None:
            return None
        if not isinstance(intervention, dict):
            raise ValueError("intervention must be an object")
        allowed = {"kind", "boundary", "visit", "relative_dose", "absolute_dose", "seed", "direction", "donor_id"}
        if set(intervention) - allowed:
            raise ValueError("unknown intervention fields")
        if intervention.get("kind") not in ("random_pulse", "direction_pulse"):
            raise ValueError("unknown intervention kind")
        if intervention.get("boundary") != "carried_answer":
            raise ValueError("intervention boundary must be carried_answer")
        _integer(intervention.get("visit"), "intervention visit", 0, depth - 1)
        if ("relative_dose" in intervention) == ("absolute_dose" in intervention):
            raise ValueError("supply exactly one relative or absolute intervention dose")
        if "relative_dose" in intervention:
            _finite_number(intervention["relative_dose"], "relative dose", 0, 2)
        else:
            _finite_number(intervention["absolute_dose"], "absolute dose", 0, 100)
        _integer(intervention.get("seed", 0), "intervention seed", 0, 2**63 - 1)
        if "donor_id" in intervention and not isinstance(intervention["donor_id"], str):
            raise ValueError("donor_id must be a string")
        return copy.deepcopy(intervention)

    def _apply_intervention(self, state: np.ndarray, spec: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        if spec["kind"] == "random_pulse":
            direction = np.random.default_rng(spec.get("seed", 0)).normal(size=state.shape)
        else:
            try:
                direction = np.asarray(spec["direction"], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("direction_pulse requires a finite direction") from exc
            if direction.shape == (self.width,):
                direction = np.broadcast_to(direction, state.shape)
        if direction.shape != state.shape or not np.isfinite(direction).all():
            raise ValueError("intervention direction shape or values are invalid")
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise ValueError("intervention direction must be nonzero")
        state_norm = float(np.linalg.norm(state))
        dose = float(spec.get("absolute_dose", spec.get("relative_dose", 0) * state_norm))
        delta = direction * (dose / norm)
        patched = state + delta
        record = {
            "kind": spec["kind"], "boundary": spec["boundary"], "visit": spec["visit"],
            "spec_digest": _digest(spec), "requested_absolute_dose": dose,
            "realized_absolute_dose": float(np.linalg.norm(patched - state)),
            "realized_relative_dose": float(np.linalg.norm(patched - state)) / max(state_norm, 1e-12),
            "pre_patch": _summary(state), "post_patch": _summary(patched),
            "direction_digest": _summary(direction)["sha256"],
            "bpd_mechanism_verified": False,
        }
        if "donor_id" in spec:
            record["donor_id"] = spec["donor_id"]
        return patched, record

    def _states(self, encoded: np.ndarray, mode: str, depth: int, intervention=None):
        answer = np.zeros((encoded.shape[0], self.width), dtype=np.float64)
        projected = encoded @ self.weights["input"]
        for visit in range(depth):
            before = answer
            clock = np.asarray([visit / depth, 1 / depth])
            endpoint = np.tanh(projected + before @ self.weights["recurrent"]
                               + clock @ self.weights["time"] + self.weights["bias"])
            eta = feedback_weight(mode, visit, depth)
            answer = endpoint.copy() if eta == 1 else before + eta * (endpoint - before)
            feedback_update = answer - before
            patch = None
            if intervention is not None and visit == intervention["visit"]:
                answer, patch = self._apply_intervention(answer, intervention)
            yield visit, eta, before, endpoint, answer, feedback_update, patch

    @staticmethod
    def _design(state: np.ndarray) -> np.ndarray:
        return np.column_stack((state, np.ones(state.shape[0])))

    def fit(self, tasks: Iterable[dict[str, Any]], training_mode: str = "direct", planned_depth: int = 4,
            readout_site: str = "carried_answer") -> dict[str, Any]:
        """Fit all-visit readouts from independently checked train candidates."""
        from .tasks import validate_task, verify

        depth = _integer(planned_depth, "planned depth", 1, MAX_DEPTH)
        feedback_weight(training_mode, 0, depth)
        _readout_site(readout_site)
        if self._metadata["fitted"]:
            raise ValueError("fitted ranker is immutable; create a new candidate for another fit")
        rows = list(tasks)
        if not rows:
            raise ValueError("training requires tasks")
        # Preflight the entire cohort before any verifier label is requested.
        for task in rows:
            validate_task(task)
            if task["split"] != "train":
                raise ValueError("ranker may fit only the train split")
        if len({task["task_id"] for task in rows}) != len(rows):
            raise ValueError("duplicate training task")
        rows.sort(key=lambda task: task["task_id"])
        size = self.width + 1
        gram = np.zeros((size, size), dtype=np.float64)
        rhs = np.zeros(size, dtype=np.float64)
        receipts, manifest = [], []
        samples = positive_candidates = candidate_count = 0
        for task in rows:
            encoded = self._encode(task)
            labels = []
            for candidate in task["candidates"]:
                receipt = verify(task, candidate)
                if (receipt.get("task_digest") != task["task_digest"]
                        or receipt.get("candidate_digest") != _digest(candidate)
                        or receipt.get("evidence_epoch") != task["evidence_epoch"]):
                    raise ValueError("verifier receipt does not bind the candidate to its task")
                passed = receipt.get("passed")
                if type(passed) is not bool or receipt.get("status") != ("passed" if passed else "failed"):
                    raise ValueError("verifier did not return a definite checked label")
                labels.append(1.0 if passed else -1.0)
                receipts.append(_digest(receipt))
                positive_candidates += int(passed)
            target = np.asarray(labels)
            candidate_count += len(labels)
            for _, _, _, endpoint, answer, _, _ in self._states(encoded, training_mode, depth):
                state = endpoint if readout_site == "endpoint" else answer
                design = self._design(state)
                gram += design.T @ design
                rhs += design.T @ target
                samples += len(target)
            manifest.append({**{key: task[key] for key in ("task_id", "lineage_id", "split", "task_digest")},
                             "public_input_digest": _digest(_public_input(task))})
        fitted = np.linalg.solve(gram / samples + self.ridge * np.eye(size), rhs / samples)
        if not np.isfinite(fitted).all():
            raise ValueError("nonfinite fitted readout")
        self.weights["readout"] = fitted
        self._metadata.update({
            "fitted": True, "training_mode": training_mode, "training_depth": depth,
            "training_readout_site": readout_site, "supervision": "all_visit_verifier_outcomes",
            "training_splits": ["train"], "training_tasks": len(rows),
            "training_candidates": candidate_count, "training_positive_candidates": positive_candidates,
            "training_verifier_calls": candidate_count, "training_readout_examples": samples,
            "training_block_calls": len(rows) * depth, "training_manifest": manifest,
            "training_lineages": sorted({task["lineage_id"] for task in rows}),
            "training_task_digests": sorted(task["task_digest"] for task in rows),
            "training_input_fingerprints": sorted({_digest(_public_input(task)) for task in rows}),
            "training_fingerprint": _digest(manifest), "training_receipts_digest": _digest(receipts),
        })
        self._validate_training_metadata()
        return self.metadata

    def visits(self, task: dict[str, Any], mode: str, planned_depth: int,
               readout_site: str = "carried_answer", intervention: dict[str, Any] | None = None
               ) -> Iterator[dict[str, Any]]:
        """Yield bounded, observable visit records; the caller owns verification and stop."""
        depth = _integer(planned_depth, "planned depth", 1, MAX_DEPTH)
        feedback_weight(mode, 0, depth)
        _readout_site(readout_site)
        encoded = self._encode(task)
        if (task["split"] != "train" and task["lineage_id"] in self._metadata["training_lineages"]):
            raise ValueError("evaluation task shares a training lineage")
        if (task["split"] != "train"
                and _digest(_public_input(task)) in self._metadata["training_input_fingerprints"]):
            raise ValueError("evaluation task repeats a training input")
        patch_spec = self._validate_intervention(intervention, depth)
        segment_id = _digest({"task_digest": task["task_digest"], "model_digest": self.model_digest,
                              "mode": mode, "depth": depth, "epoch": task["evidence_epoch"],
                              "readout_site": readout_site, "intervention": patch_spec})
        for visit, eta, before, endpoint, answer, update, patch in self._states(encoded, mode, depth, patch_spec):
            if patch is not None:
                patch["readout_site"] = readout_site
                patch["first_affected_readout_visit"] = (
                    visit if readout_site == "carried_answer" else visit + 1 if visit + 1 < depth else None
                )
                patch["next_block_visit"] = visit + 1 if visit + 1 < depth else None
            readout = endpoint if readout_site == "endpoint" else answer
            scores = self._design(readout) @ self.weights["readout"]
            if not np.isfinite(scores).all():
                raise ValueError("nonfinite ranker output")
            chosen = int(np.argmax(scores))
            ranked = np.sort(scores)
            yield {
                "visit": visit, "planned_depth": depth, "eta": eta, "segment_id": segment_id,
                "evidence_epoch": task["evidence_epoch"], "state_reset": "zero_at_segment_start",
                "dynamics_kind": "continuous_answer_state", "model_kind": "frozen_random_reservoir",
                "feedback_mode": mode, "state_placement": "answer", "readout_site": readout_site,
                "time": visit / depth, "step_size": 1 / depth,
                "candidate": copy.deepcopy(task["candidates"][chosen]), "candidate_index": chosen,
                "candidate_scores": scores.tolist(), "selected_score": float(scores[chosen]),
                "score_margin": float(ranked[-1] - ranked[-2]) if len(ranked) > 1 else 0.0,
                "state_summaries": {"before": _summary(before), "endpoint": _summary(endpoint),
                                    "carried_answer": _summary(answer), "feedback_update": _summary(update)},
                "block_calls": 1, "cumulative_block_calls": visit + 1,
                "intervention": patch, "bpd_mechanism_verified": False,
            }
