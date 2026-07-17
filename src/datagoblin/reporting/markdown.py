from __future__ import annotations

from datagoblin.domain.experiments import ExperimentRecord


def experiment_markdown(record: ExperimentRecord) -> str:
    metrics = "\n".join(f"| `{name}` | {value:.6f} |" for name, value in sorted(record.actual_metrics.items()))
    predictions = "\n".join(
        f"- `{item.get('metric')}` expected in [{item.get('lower')}, {item.get('upper')}] "
        f"at confidence {item.get('confidence')}"
        for item in record.predicted_outcomes
    )
    violations = ", ".join(record.violations) if record.violations else "none"
    return f"""# DataGoblin experiment report

- Experiment: `{record.experiment_id}`
- Status: `{record.run_status.value}`
- Evidence: `{record.evidence_type.value}`
- Violations: {violations}

## Preregistered prediction

{predictions}

## Held-out actual metrics

| Metric | Value |
|---|---:|
{metrics}

## Interpretation

{record.interpretation}

## Next experiment

{chr(10).join(f"- {item}" for item in record.next_experiment_suggestions)}

## Claim boundary

{record.claim_boundary}
"""
