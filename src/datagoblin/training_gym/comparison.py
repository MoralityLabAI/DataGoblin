from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from datagoblin.serialization import dump_structured, load_structured

from .contract import load_gym_contract


def compare_candidates(
    contract_path: Path,
    report_paths: Sequence[Path],
    baseline_candidate_id: str,
    out_dir: Path,
) -> dict[str, Any]:
    if len(report_paths) < 2:
        raise ValueError("candidate comparison requires at least two reports")
    contract = load_gym_contract(contract_path)
    reports = [load_structured(path) for path in report_paths]
    _validate_comparable(reports)
    by_id = {str(report["candidate_id"]): report for report in reports}
    if len(by_id) != len(reports):
        raise ValueError("candidate IDs must be unique")
    if baseline_candidate_id not in by_id:
        raise ValueError("baseline candidate is absent from reports")
    baseline = by_id[baseline_candidate_id]
    baseline_ordinary = baseline["evidence_results"]["ordinary_test"]
    baseline_ood = baseline["evidence_results"]["ood_test"]
    rows = [
        _comparison_row(contract, report, baseline_ordinary, baseline_ood, baseline_candidate_id)
        for report in reports
    ]
    for row in rows:
        row["pareto_dominated"] = _is_dominated(row, rows)
        row["promotion_eligible"] = all(row["promotion_checks"].values()) and not row["pareto_dominated"]
    comparison = {
        "schema_version": "1.0",
        "gym_id": contract["gym_id"],
        "baseline_candidate_id": baseline_candidate_id,
        "comparison_kind": "matched_deltas_and_pareto_no_universal_scalar",
        "rows": rows,
        "promotion_eligible_candidates": [row["candidate_id"] for row in rows if row["promotion_eligible"]],
        "winner": None,
        "claim_boundary": "Eligibility is a gate for fresh confirmatory runs, not proof of architecture superiority.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_structured(out_dir / "comparison.json", comparison)
    (out_dir / "comparison.md").write_text(_render_comparison(comparison), encoding="utf-8")
    return comparison


def _validate_comparable(reports: Sequence[dict[str, Any]]) -> None:
    gym_ids = {report["gym_id"] for report in reports}
    contracts = {report["reproducibility"]["contract_sha256"] for report in reports}
    datasets = {report["reproducibility"]["dataset_manifest_sha256"] for report in reports}
    if len(gym_ids) != 1 or len(contracts) != 1 or len(datasets) != 1:
        raise ValueError("candidate reports do not share an exact gym contract and dataset")
    if any(report["evidence_results"]["ordinary_test"] is None for report in reports):
        raise ValueError("ordinary-test evidence is required for comparison")


def _comparison_row(
    contract: dict[str, Any],
    report: dict[str, Any],
    baseline_ordinary: dict[str, Any],
    baseline_ood: dict[str, Any] | None,
    baseline_candidate_id: str,
) -> dict[str, Any]:
    ordinary = report["evidence_results"]["ordinary_test"]
    ood = report["evidence_results"]["ood_test"]
    sealed = report["evidence_results"]["sealed_poison"]
    budget = report["candidate_budget"]
    resource = report["resource_summary"]
    gates = contract["promotion_gates"]
    is_baseline = report["candidate_id"] == baseline_candidate_id
    ordinary_gain = ordinary["exact_record_accuracy"] - baseline_ordinary["exact_record_accuracy"]
    ood_regression = (
        baseline_ood["exact_record_accuracy"] - ood["exact_record_accuracy"]
        if baseline_ood is not None and ood is not None
        else None
    )
    checks = {
        "candidate_not_baseline": not is_baseline,
        "not_mock": not budget["mock"],
        "all_evaluation_tasks_executable": all(
            task.get("status") == "executable" for task in contract.get("evaluation_tasks", {}).values()
        ),
        "hard_caps_enforced": bool(resource.get("caps_enforced")),
        "cleanup_passed": bool(resource.get("cleanup_passed")),
        "sealed_available": sealed is not None,
        "schema_parse": ordinary["schema_parse_rate"] >= float(gates["schema_parse_rate"]),
        "critical_false_acceptance": ordinary["critical_false_acceptance_rate"]
        <= float(gates["maximum_critical_false_acceptance_rate"]),
        "ordinary_gain": ordinary_gain >= float(gates["minimum_ordinary_exact_record_gain"]),
        "ood_regression": ood_regression is not None
        and ood_regression <= float(gates["maximum_ood_exact_record_regression"]),
        "termination": ordinary["nontermination_rate"] <= float(gates["maximum_nontermination_rate"]),
    }
    return {
        "candidate_id": report["candidate_id"],
        "architecture_family": report["architecture_family"],
        "ordinary_exact_record_accuracy": ordinary["exact_record_accuracy"],
        "ordinary_exact_record_gain": ordinary_gain,
        "ood_exact_record_accuracy": ood["exact_record_accuracy"] if ood else None,
        "ood_exact_record_regression": ood_regression,
        "sealed_exact_record_accuracy": sealed["exact_record_accuracy"] if sealed else None,
        "ordinary_aurc": ordinary["area_under_risk_coverage_curve"],
        "critical_false_acceptance_rate": ordinary["critical_false_acceptance_rate"],
        "nontermination_rate": ordinary["nontermination_rate"],
        "trainable_parameters": budget["trainable_parameters"],
        "inference_update_budget": budget["inference_update_budget"],
        "promotion_checks": checks,
    }


def _is_dominated(candidate: dict[str, Any], rows: Sequence[dict[str, Any]]) -> bool:
    if not candidate["promotion_checks"]["candidate_not_baseline"]:
        return False
    for other in rows:
        if other is candidate:
            continue
        no_worse = (
            other["ordinary_exact_record_accuracy"] >= candidate["ordinary_exact_record_accuracy"]
            and other["ordinary_aurc"] <= candidate["ordinary_aurc"]
            and other["critical_false_acceptance_rate"] <= candidate["critical_false_acceptance_rate"]
            and other["nontermination_rate"] <= candidate["nontermination_rate"]
            and other["trainable_parameters"] <= candidate["trainable_parameters"]
            and other["inference_update_budget"] <= candidate["inference_update_budget"]
        )
        strictly_better = any(
            (
                other["ordinary_exact_record_accuracy"] > candidate["ordinary_exact_record_accuracy"],
                other["ordinary_aurc"] < candidate["ordinary_aurc"],
                other["critical_false_acceptance_rate"] < candidate["critical_false_acceptance_rate"],
                other["nontermination_rate"] < candidate["nontermination_rate"],
                other["trainable_parameters"] < candidate["trainable_parameters"],
                other["inference_update_budget"] < candidate["inference_update_budget"],
            )
        )
        if no_worse and strictly_better:
            return True
    return False


def _render_comparison(comparison: dict[str, Any]) -> str:
    lines = [
        "# Training-gym candidate comparison",
        "",
        f"Baseline: `{comparison['baseline_candidate_id']}`. No universal scalar winner is selected.",
        "",
        "| Candidate | Family | Ordinary exact | Gain | OOD regression | AURC | "
        "Critical false accept | Dominated | Eligible |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison["rows"]:
        ood_regression = row["ood_exact_record_regression"]
        lines.append(
            f"| {row['candidate_id']} | {row['architecture_family']} | "
            f"{row['ordinary_exact_record_accuracy']:.3f} | {row['ordinary_exact_record_gain']:.3f} | "
            f"{ood_regression if ood_regression is not None else 'unavailable'} | {row['ordinary_aurc']:.3f} | "
            f"{row['critical_false_acceptance_rate']:.3f} | {row['pareto_dominated']} | "
            f"{row['promotion_eligible']} |"
        )
    lines.extend(("", str(comparison["claim_boundary"]), ""))
    return "\n".join(lines)
