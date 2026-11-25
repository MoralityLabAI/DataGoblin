"""Loader and supervision views for the DCF-16 verifier-grounded decision-trace dataset.

Records are synthetic parity queries on binary factor graphs with exact, independently verified reference posteriors.
The loader validates every record against ``schemas/dcf16_decision_trace.schema.json``, enforces split hygiene, and
exports per-target supervision views for a compact recurrent model. Held-out records never enter a training view.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_TAG = "dcf16.decision_trace.v1"
SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "dcf16_decision_trace.schema.json"

SPLITS = ("train", "val", "heldout")
TRAINING_SPLITS = ("train", "val")
HELDOUT_ONLY_FAMILIES = frozenset({"star", "dense", "hyper", "ktree", "xor_noiseless", "weak_dense"})
MAX_TRAINING_SIZE = 32

TARGETS = (
    "dependency_edges",
    "relevant_variables",
    "complexity",
    "algorithm",
    "recurrent_depth",
    "calibration",
    "abstention",
)


class SplitHygieneError(ValueError):
    """Raised when a split contains records that violate the held-out protocol."""


def _validator() -> Any:
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("jsonschema is required to validate DCF-16 records; install data-goblin[dcf16]") from exc
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a gzip JSONL file (plain JSONL when the name does not end in .gz) without validating."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_records(records: Iterable[Mapping[str, Any]]) -> None:
    """Raise ``ValueError`` naming the record index and path of the first schema violation."""
    validator = _validator()
    for index, record in enumerate(records):
        error = next(iter(validator.iter_errors(record)), None)
        if error is not None:
            location = "/".join(str(part) for part in error.absolute_path) or "<record>"
            raise ValueError(f"record {index} violates {SCHEMA_TAG} at {location}: {error.message}")


def check_split(split: str, records: Sequence[Mapping[str, Any]]) -> None:
    """Enforce per-split hygiene: split tags, held-out-only families, training size range, unique ids."""
    if split not in SPLITS:
        raise SplitHygieneError(f"unknown split {split!r}")
    seen: set[str] = set()
    for record in records:
        rid = record["id"]
        if record["split"] != split:
            raise SplitHygieneError(f"record {rid} is tagged {record['split']!r} but was loaded as {split!r}")
        if rid in seen:
            raise SplitHygieneError(f"canonical id {rid} repeats inside split {split!r}")
        seen.add(rid)
        if split in TRAINING_SPLITS:
            if record["family"] in HELDOUT_ONLY_FAMILIES:
                raise SplitHygieneError(f"held-out-only family {record['family']!r} appears in {split!r} (id {rid})")
            if record["n"] > MAX_TRAINING_SIZE:
                raise SplitHygieneError(f"size {record['n']} exceeds the training range in {split!r} (id {rid})")


def check_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    """Raise if any canonical id occurs in more than one split."""
    owner: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            previous = owner.setdefault(record["id"], name)
            if previous != name:
                raise SplitHygieneError(f"canonical id {record['id']} appears in both {previous!r} and {name!r}")


def load_split(path: Path, split: str) -> list[dict[str, Any]]:
    records = read_records(path)
    validate_records(records)
    check_split(split, records)
    return records


def load_dataset(root: Path, splits: Sequence[str] = SPLITS) -> dict[str, list[dict[str, Any]]]:
    """Load ``<root>/<split>.jsonl.gz`` for each split, validating schema and hygiene across all of them."""
    loaded = {split: load_split(root / f"{split}.jsonl.gz", split) for split in splits}
    check_disjoint(loaded)
    return loaded


def _inputs(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "problem": record["problem"],
        "conditions": record["conditions"],
        "decision_rule": record["decision_rule"],
        "budget_log2_ops": record["budget_log2_ops"],
        "observation_history": record["observation_history"],
    }


def _target_value(record: Mapping[str, Any], target: str) -> dict[str, Any]:
    t = record["targets"]
    if target == "dependency_edges":
        return {"edges": [list(edge) for edge in t["dependency_edges"]]}
    if target == "relevant_variables":
        return {"relevant_variables": list(t["relevant_variables"])}
    if target == "complexity":
        return {"log2_exact_cost": t["log2_exact_cost"]}
    if target == "algorithm":
        return {"label": record["selected"]["option"], "operator": t["algorithm"]}
    if target == "recurrent_depth":
        return {"recurrent_depth": t["recurrent_depth"]}
    if target == "calibration":
        return {"p_reference": t["p_reference"], "exact_action": record["verification"]["exact_action"]}
    if target == "abstention":
        return {"abstain": t["abstain"], "verified_by": list(record["verification"]["verified_by"])}
    raise KeyError(f"unknown target {target!r}; expected one of {TARGETS}")


def supervision_view(
    records: Iterable[Mapping[str, Any]], target: str, *, allow_heldout: bool = False
) -> Iterator[dict[str, Any]]:
    """Yield ``{id, split, family, n, input, target}`` rows for one supervision target.

    Held-out records raise unless ``allow_heldout`` is set, which only evaluation code may do.
    """
    if target not in TARGETS:
        raise KeyError(f"unknown target {target!r}; expected one of {TARGETS}")
    for record in records:
        if record["split"] == "heldout" and not allow_heldout:
            raise SplitHygieneError(f"held-out record {record['id']} refused by a training view")
        yield {
            "id": record["id"],
            "split": record["split"],
            "family": record["family"],
            "n": record["n"],
            "input": _inputs(record),
            "target": _target_value(record, target),
        }


def training_views(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Build every target view for each training split; the ``heldout`` split is never read."""
    return {
        split: {target: list(supervision_view(splits[split], target)) for target in TARGETS}
        for split in TRAINING_SPLITS
        if split in splits
    }


def summarize(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    return {
        split: {
            "records": len(records),
            "families": dict(sorted(Counter(r["family"] for r in records).items())),
            "labels": dict(sorted(Counter(r["selected"]["option"] for r in records).items())),
            "abstain": sum(bool(r["targets"]["abstain"]) for r in records),
        }
        for split, records in splits.items()
    }
