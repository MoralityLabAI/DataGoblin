from __future__ import annotations

import argparse
import hashlib
import hmac
import random
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_set_metric_robustness_v10 import (
    EXACT_SELECTED_SET_ACCURACY,
    EXACT_STATUS_VECTOR_ACCURACY,
    SetMetricContractError,
    score_exact_set_metrics,
)
from datagoblin.serialization import (
    canonical_json,
    load_structured,
    pretty_json,
)

ROOT = Path(__file__).resolve().parents[1]
METRIC_SOURCE = (
    ROOT
    / "src"
    / "datagoblin"
    / "experiments"
    / "ldt_set_metric_robustness_v10.py"
)
METRIC_IDS = (
    EXACT_STATUS_VECTOR_ACCURACY,
    EXACT_SELECTED_SET_ACCURACY,
)
REFERENCE_STATUSES = {
    "SELECTED",
    "DOMINATED",
    "UNRESOLVED",
    "UNSAFE",
}
Assignment = tuple[tuple[str, str], ...]


class StewardReferenceError(ValueError):
    """Raised by the independently coded steward contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_index(
    assignment: Sequence[tuple[str, str]],
    *,
    role: str,
) -> dict[str, str]:
    if len(assignment) == 0:
        raise StewardReferenceError(f"{role} assignment is empty")
    result: dict[str, str] = {}
    for candidate_id, status in assignment:
        if not isinstance(candidate_id, str) or candidate_id == "":
            raise StewardReferenceError(f"{role} candidate ID is empty")
        if candidate_id in result:
            raise StewardReferenceError(
                f"{role} duplicate candidate ID: {candidate_id}"
            )
        if status not in REFERENCE_STATUSES:
            raise StewardReferenceError(
                f"{role} invalid status for {candidate_id}: {status}"
            )
        result[candidate_id] = status
    return result


def _reference_score(
    oracle: Sequence[tuple[str, str]],
    predicted: Sequence[tuple[str, str]],
) -> dict[str, float]:
    oracle_index = _reference_index(oracle, role="oracle")
    prediction_index = _reference_index(predicted, role="predicted")
    if oracle_index.keys() != prediction_index.keys():
        raise StewardReferenceError("candidate inventories differ")
    status_exact = all(
        prediction_index[candidate_id] == oracle_status
        for candidate_id, oracle_status in oracle_index.items()
    )
    oracle_selected = frozenset(
        candidate_id
        for candidate_id, status in oracle_index.items()
        if status == "SELECTED"
    )
    prediction_selected = frozenset(
        candidate_id
        for candidate_id, status in prediction_index.items()
        if status == "SELECTED"
    )
    return {
        EXACT_STATUS_VECTOR_ACCURACY: 1.0 if status_exact else 0.0,
        EXACT_SELECTED_SET_ACCURACY: (
            1.0 if oracle_selected == prediction_selected else 0.0
        ),
    }


def _rng_for(seed: bytes, family_id: str) -> random.Random:
    derived = hmac.new(
        seed,
        family_id.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return random.Random(int.from_bytes(derived, byteorder="big"))


def _base_assignment(seed: bytes, family_id: str) -> Assignment:
    rng = _rng_for(seed, family_id)
    size = rng.randint(9, 15)
    candidate_ids = [
        f"heldout-{rng.getrandbits(96):024x}" for _ in range(size)
    ]
    if len(set(candidate_ids)) != size:
        raise AssertionError("seeded candidate-ID collision")
    statuses = [
        "SELECTED",
        "SELECTED",
        "DOMINATED",
        "DOMINATED",
        "UNRESOLVED",
        "UNSAFE",
    ]
    statuses.extend(
        rng.choice(tuple(sorted(REFERENCE_STATUSES)))
        for _ in range(size - len(statuses))
    )
    rng.shuffle(statuses)
    assignment = list(zip(candidate_ids, statuses, strict=True))
    rng.shuffle(assignment)
    return tuple(assignment)


def _replace(
    assignment: Assignment,
    replacements: Mapping[str, str],
) -> Assignment:
    return tuple(
        (candidate_id, replacements.get(candidate_id, status))
        for candidate_id, status in assignment
    )


def _ids_with_status(
    assignment: Assignment,
    status: str,
) -> list[str]:
    return [
        candidate_id
        for candidate_id, candidate_status in assignment
        if candidate_status == status
    ]


def _selected_and_nonselected(
    assignment: Assignment,
) -> tuple[list[str], list[str]]:
    selected = _ids_with_status(assignment, "SELECTED")
    nonselected = [
        candidate_id
        for candidate_id, status in assignment
        if status != "SELECTED"
    ]
    if len(selected) < 2 or len(nonselected) < 2:
        raise AssertionError("holdout base lacks membership support")
    return selected, nonselected


def _remap(
    assignment: Assignment,
    mapping: Mapping[str, str],
) -> Assignment:
    return tuple(
        (mapping[candidate_id], status)
        for candidate_id, status in assignment
    )


def _directional_fixture(
    seed: bytes,
    family_id: str,
) -> dict[str, Any]:
    oracle = _base_assignment(seed, family_id)
    rng = _rng_for(seed, family_id + ":transform")
    selected, nonselected = _selected_and_nonselected(oracle)
    if family_id == "seeded-cyclic-order":
        offset = rng.randint(1, len(oracle) - 1)
        staged_prediction = oracle[offset:] + oracle[:offset]
        return {
            "baseline_oracle": oracle,
            "baseline_prediction": oracle,
            "staged_oracle": oracle,
            "staged_prediction": staged_prediction,
        }
    if family_id == "seeded-bijective-id-remap":
        mapped_ids = [
            f"remapped-{rng.getrandbits(96):024x}" for _ in oracle
        ]
        mapping = {
            candidate_id: mapped_id
            for (candidate_id, _), mapped_id in zip(
                oracle,
                mapped_ids,
                strict=True,
            )
        }
        remapped = _remap(oracle, mapping)
        return {
            "baseline_oracle": oracle,
            "baseline_prediction": oracle,
            "staged_oracle": remapped,
            "staged_prediction": remapped,
        }
    if family_id == "seeded-selected-to-unsafe":
        staged = _replace(oracle, {rng.choice(selected): "UNSAFE"})
        return {
            "baseline_oracle": oracle,
            "baseline_prediction": oracle,
            "staged_oracle": oracle,
            "staged_prediction": staged,
        }
    if family_id == "seeded-unresolved-to-selected":
        unresolved = _ids_with_status(oracle, "UNRESOLVED")
        staged = _replace(oracle, {rng.choice(unresolved): "SELECTED"})
        return {
            "baseline_oracle": oracle,
            "baseline_prediction": oracle,
            "staged_oracle": oracle,
            "staged_prediction": staged,
        }
    if family_id == "seeded-compounded-mixed-membership":
        selected_id = rng.choice(selected)
        nonselected_id = rng.choice(nonselected)
        baseline = _replace(oracle, {selected_id: "DOMINATED"})
        staged = _replace(
            baseline,
            {nonselected_id: "SELECTED"},
        )
        return {
            "baseline_oracle": oracle,
            "baseline_prediction": baseline,
            "staged_oracle": oracle,
            "staged_prediction": staged,
        }
    if family_id == "seeded-expanded-selection-swap":
        removed = rng.sample(selected, 2)
        inserted = rng.sample(nonselected, 2)
        baseline = _replace(
            oracle,
            {removed[0]: "DOMINATED", inserted[0]: "SELECTED"},
        )
        staged = _replace(
            baseline,
            {removed[1]: "DOMINATED", inserted[1]: "SELECTED"},
        )
        return {
            "baseline_oracle": oracle,
            "baseline_prediction": baseline,
            "staged_oracle": oracle,
            "staged_prediction": staged,
        }
    raise ValueError(f"unsupported directional family: {family_id}")


def _violation_fixture(
    seed: bytes,
    family_id: str,
) -> dict[str, Any]:
    oracle = _base_assignment(seed, family_id)
    rng = _rng_for(seed, family_id + ":transform")
    selected, _ = _selected_and_nonselected(oracle)
    if family_id == "seeded-high-cardinality-decoy":
        prediction = _replace(
            oracle,
            {rng.choice(selected): "DOMINATED"},
        )
    elif family_id == "seeded-inventory-substitution":
        replacement_id = f"substitute-{rng.getrandbits(96):024x}"
        original_id, original_status = oracle[-1]
        if replacement_id == original_id:
            raise AssertionError("seeded substitution collision")
        prediction = (*oracle[:-1], (replacement_id, original_status))
    elif family_id == "seeded-exact-shuffle-control":
        shuffled = list(oracle)
        rng.shuffle(shuffled)
        prediction = tuple(shuffled)
    elif family_id == "seeded-exact-remap-control":
        mapped_ids = [
            f"clean-{rng.getrandbits(96):024x}" for _ in oracle
        ]
        mapping = {
            candidate_id: mapped_id
            for (candidate_id, _), mapped_id in zip(
                oracle,
                mapped_ids,
                strict=True,
            )
        }
        oracle = _remap(oracle, mapping)
        shuffled = list(oracle)
        rng.shuffle(shuffled)
        prediction = tuple(shuffled)
    else:
        raise ValueError(f"unsupported violation family: {family_id}")
    return {"oracle": oracle, "prediction": prediction}


def _assert_scores_agree(
    oracle: Assignment,
    prediction: Assignment,
) -> dict[str, float]:
    reference = _reference_score(oracle, prediction)
    candidate = score_exact_set_metrics(oracle, prediction)
    if reference != candidate:
        raise AssertionError(
            f"candidate and steward reference disagree: {reference} != {candidate}"
        )
    return candidate


def _detect_violation(
    oracle: Assignment,
    prediction: Assignment,
) -> dict[str, bool]:
    reference_error = False
    candidate_error = False
    try:
        reference = _reference_score(oracle, prediction)
    except StewardReferenceError:
        reference_error = True
        reference = {}
    try:
        candidate = score_exact_set_metrics(oracle, prediction)
    except SetMetricContractError:
        candidate_error = True
        candidate = {}
    if reference_error != candidate_error:
        raise AssertionError("candidate and steward validity checks disagree")
    if reference_error:
        return {metric_id: True for metric_id in METRIC_IDS}
    if reference != candidate:
        raise AssertionError("candidate and steward metric scores disagree")
    return {
        metric_id: candidate[metric_id] < 1.0
        for metric_id in METRIC_IDS
    }


def _private_fixture_row(
    probe: Mapping[str, Any],
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "datagoblin.metric-holdout-fixture.v1",
        "probe_id": probe["probe_id"],
        "probe_family_id": probe["probe_family_id"],
        "source_ref": probe["source_ref"],
        "fixture": fixture,
    }


def build_holdout_evidence(
    spec: Mapping[str, Any],
    seed: bytes,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(seed) < 32:
        raise ValueError("holdout seed must contain at least 256 bits")
    if spec.get("schema_version") != "1.0":
        raise ValueError("unsupported robustness specification")
    if spec.get("evidence_scope") != "held_out":
        raise ValueError("steward requires a held-out specification")
    declared_metrics = {
        str(metric["id"]) for metric in spec.get("metrics", ())
    }
    if declared_metrics != set(METRIC_IDS):
        raise ValueError("held-out metric declaration differs from implementation")
    observations: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for probe in spec.get("probes", ()):
        family_id = str(probe["probe_family_id"])
        expected = str(probe["expected_behavior"])
        if expected in {"stable", "increase", "decrease"}:
            fixture = _directional_fixture(seed, family_id)
            baseline = _assert_scores_agree(
                fixture["baseline_oracle"],
                fixture["baseline_prediction"],
            )
            staged = _assert_scores_agree(
                fixture["staged_oracle"],
                fixture["staged_prediction"],
            )
            outcome_by_metric = {
                metric_id: {
                    "baseline_value": baseline[metric_id],
                    "staged_value": staged[metric_id],
                }
                for metric_id in METRIC_IDS
            }
        else:
            fixture = _violation_fixture(seed, family_id)
            detected = _detect_violation(
                fixture["oracle"],
                fixture["prediction"],
            )
            outcome_by_metric = {
                metric_id: {"observed_violation": detected[metric_id]}
                for metric_id in METRIC_IDS
            }
        private_rows.append(_private_fixture_row(probe, fixture))
        for raw_metric_id in probe["metric_ids"]:
            metric_id = str(raw_metric_id)
            key = (str(probe["probe_id"]), metric_id)
            if key in seen_keys:
                raise ValueError(f"duplicate probe/metric key: {key}")
            seen_keys.add(key)
            observation = {
                "schema_version": "1.0",
                "probe_id": probe["probe_id"],
                "probe_family_id": family_id,
                "metric_id": metric_id,
                "probe_type": probe["probe_type"],
                "split": probe["split"],
                "expected_behavior": expected,
                **outcome_by_metric[metric_id],
            }
            for field in ("tolerance", "required_delta"):
                if field in probe:
                    observation[field] = probe[field]
            observations.append(observation)
    return observations, private_rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(f"{canonical_json(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _require_external(path: Path, *, role: str) -> Path:
    resolved = path.resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError(f"{role} must remain outside the repository")
    return resolved


def create_private_seed(path: Path) -> str:
    resolved = _require_external(path, role="private seed")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("xb") as handle:
        handle.write(secrets.token_bytes(32))
    return _sha256_file(resolved)


def run_holdout(
    *,
    spec_path: Path,
    seed_path: Path,
    output_dir: Path,
    expected_spec_sha256: str,
    expected_metric_sha256: str,
) -> dict[str, Any]:
    seed_path = _require_external(seed_path, role="private seed")
    output_dir = _require_external(output_dir, role="steward output")
    actual_spec_sha256 = _sha256_file(spec_path)
    actual_metric_sha256 = _sha256_file(METRIC_SOURCE)
    if actual_spec_sha256 != expected_spec_sha256:
        raise ValueError("held-out specification hash changed after staging")
    if actual_metric_sha256 != expected_metric_sha256:
        raise ValueError("metric implementation changed before holdout")
    output_dir.mkdir(parents=True, exist_ok=False)
    seed = seed_path.read_bytes()
    observations, private_rows = build_holdout_evidence(
        load_structured(spec_path),
        seed,
    )
    private_pack_path = output_dir / "private-probe-pack.jsonl"
    observations_path = output_dir / "observations.jsonl"
    _write_jsonl(private_pack_path, private_rows)
    _write_jsonl(observations_path, observations)
    receipt = {
        "schema_version": "datagoblin.metric-holdout-custody.v1",
        "authority": "ENVIRONMENT_OWNED_VERIFIER",
        "review_id": "datagoblin-ldt-set-metric-robustness-v10-holdout",
        "evidence_scope": "held_out",
        "independence": {
            "candidate_metric_frozen_before_concrete_fixtures": True,
            "concrete_assignments_exposed_before_metric_freeze": False,
            "candidate_author_received_private_probe_pack": False,
            "generation_seed_exposed_to_candidate": False,
            "separate_reference_implementation": True,
            "third_party_identity_authentication": False,
            "classification": "environment_owned_artifact_bound",
        },
        "coverage": {
            "probe_rows": len(observations),
            "probe_families": len(private_rows),
            "metric_ids": list(METRIC_IDS),
        },
        "bindings": {
            "spec_sha256": actual_spec_sha256,
            "metric_implementation_sha256": actual_metric_sha256,
            "private_seed_commitment_sha256": hashlib.sha256(seed).hexdigest(),
            "private_probe_pack_sha256": _sha256_file(private_pack_path),
            "observations_sha256": _sha256_file(observations_path),
        },
        "custody": {
            "seed_retained_outside_repository": True,
            "private_probe_pack_retained_outside_repository": True,
            "raw_assignments_publication_authorized": False,
        },
    }
    (output_dir / "custody-receipt.json").write_text(
        pretty_json(receipt),
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the environment-owned v10 metric holdout."
    )
    parser.add_argument("--create-private-seed", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--seed-file", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-spec-sha256")
    parser.add_argument("--expected-metric-sha256")
    args = parser.parse_args()
    if args.create_private_seed is not None:
        if any(
            value is not None
            for value in (
                args.spec,
                args.seed_file,
                args.out,
                args.expected_spec_sha256,
                args.expected_metric_sha256,
            )
        ):
            parser.error("seed creation cannot be combined with a holdout run")
        create_private_seed(args.create_private_seed)
        print("created a 256-bit private seed outside the repository")
        return 0
    required = {
        "--spec": args.spec,
        "--seed-file": args.seed_file,
        "--out": args.out,
        "--expected-spec-sha256": args.expected_spec_sha256,
        "--expected-metric-sha256": args.expected_metric_sha256,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    run_holdout(
        spec_path=args.spec,
        seed_path=args.seed_file,
        output_dir=args.out,
        expected_spec_sha256=args.expected_spec_sha256,
        expected_metric_sha256=args.expected_metric_sha256,
    )
    print("wrote sequestered probe evidence and a public-safe custody receipt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
