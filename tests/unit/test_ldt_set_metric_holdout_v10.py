from __future__ import annotations

import hashlib
import runpy
from pathlib import Path
from typing import Any

import pytest

from datagoblin.serialization import load_structured

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = (
    ROOT
    / "configs"
    / "preregistrations"
    / "ldt_set_metric_robustness_v10_holdout.yaml"
)
METRIC_PATH = (
    ROOT
    / "src"
    / "datagoblin"
    / "experiments"
    / "ldt_set_metric_robustness_v10.py"
)


def _namespace() -> dict[str, Any]:
    return runpy.run_path(
        str(ROOT / "scripts" / "steward_ldt_set_metric_holdout_v10.py")
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_seeded_holdout_covers_frozen_probe_plan() -> None:
    namespace = _namespace()
    observations, private_rows = namespace["build_holdout_evidence"](
        load_structured(SPEC_PATH),
        bytes(range(32)),
    )
    assert len(observations) == 20
    assert len(private_rows) == 10
    assert len(
        {(row["probe_id"], row["metric_id"]) for row in observations}
    ) == 20
    assert all(row["split"] == "held_out" for row in observations)
    for row in observations:
        expected = row["expected_behavior"]
        if expected == "stable":
            assert row["baseline_value"] == row["staged_value"]
        elif expected == "decrease":
            assert row["staged_value"] <= row["baseline_value"]
            assert (
                row["baseline_value"] - row["staged_value"]
                >= row["required_delta"]
            )
        elif expected == "detect_violation":
            assert row["observed_violation"] is True
        else:
            assert expected == "no_violation"
            assert row["observed_violation"] is False


def test_steward_run_binds_private_pack_without_publishing_seed(
    tmp_path: Path,
) -> None:
    namespace = _namespace()
    seed_path = tmp_path / "custody" / "seed.bin"
    seed_path.parent.mkdir()
    seed_path.write_bytes(bytes(range(32)))
    output_dir = tmp_path / "steward-run"
    receipt = namespace["run_holdout"](
        spec_path=SPEC_PATH,
        seed_path=seed_path,
        output_dir=output_dir,
        expected_spec_sha256=_sha256(SPEC_PATH),
        expected_metric_sha256=_sha256(METRIC_PATH),
    )
    assert receipt["authority"] == "ENVIRONMENT_OWNED_VERIFIER"
    assert receipt["coverage"] == {
        "probe_rows": 20,
        "probe_families": 10,
        "metric_ids": [
            "exact_status_vector_accuracy",
            "exact_selected_set_accuracy",
        ],
    }
    assert receipt["bindings"]["private_seed_commitment_sha256"] == hashlib.sha256(
        bytes(range(32))
    ).hexdigest()
    receipt_text = (output_dir / "custody-receipt.json").read_text(
        encoding="utf-8"
    )
    assert bytes(range(32)).hex() not in receipt_text
    assert (output_dir / "private-probe-pack.jsonl").is_file()
    assert (output_dir / "observations.jsonl").is_file()


def test_steward_refuses_repository_custody_paths() -> None:
    namespace = _namespace()
    with pytest.raises(ValueError, match="outside the repository"):
        namespace["_require_external"](
            ROOT / "runs" / "forbidden-holdout",
            role="steward output",
        )
