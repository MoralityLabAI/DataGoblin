from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

TRACE_ORDER = ("L1", "L2", "L3", "H1", "L4", "L5", "L6", "H2")


@dataclass(frozen=True)
class ForwardOutput:
    final_logits: np.ndarray
    final_output: str
    verdict_probabilities: Mapping[str, float]
    sequence_log_probability: float
    prompt_token_count: int
    output_token_count: int


@dataclass(frozen=True)
class RecurrentStep:
    name: str
    state: np.ndarray
    logits: np.ndarray | None = None
    attention_gate_summary: Mapping[str, float] | None = None


StepObserver = Callable[[RecurrentStep], None]


class TraceableRecurrentModel(Protocol):
    model_id: str

    def ordinary_forward(self, prompt: str) -> ForwardOutput: ...

    def trace_forward(self, prompt: str, observer: StepObserver) -> ForwardOutput: ...

    def close(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class TraceSummary:
    trace_id: str
    schema_version: str
    model_id: str
    trace_order: tuple[str, ...]
    per_step_relative_update_norms: Mapping[str, float]
    pooled_update_norms: Mapping[str, float]
    selected_token_update_norms: Mapping[str, Mapping[str, float]]
    residual_norms: Mapping[str, float]
    attention_gate_summaries: Mapping[str, Mapping[str, float]]
    recurrent_features: Mapping[str, float]
    final_verdict_probabilities: Mapping[str, float]
    final_output: str
    schema_parse_success: bool
    parsed_decision: str | None
    parsed_violations: tuple[str, ...]
    authoritative_decision: str | None
    authoritative_violations: tuple[str, ...]
    correctness_label: bool | None
    prompt_hash: str
    artifact_hashes: tuple[str, ...]
    sequence_log_probability: float
    prompt_length: int
    output_length: int
    schema_complexity: float
    final_verdict_entropy: float
    maximum_verdict_probability: float
    verdict_token_margin: float
    instrumented_logits_max_abs_error: float
    instrumented_logits_tolerance: float
    sampled_activations: Mapping[str, tuple[float, ...]] | None = None

    def to_jsonable(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "trace_id": self.trace_id,
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "trace_order": list(self.trace_order),
            "per_step_relative_update_norms": dict(self.per_step_relative_update_norms),
            "pooled_update_norms": dict(self.pooled_update_norms),
            "selected_token_update_norms": {
                step: dict(values) for step, values in self.selected_token_update_norms.items()
            },
            "residual_norms": dict(self.residual_norms),
            "attention_gate_summaries": {
                step: dict(values) for step, values in self.attention_gate_summaries.items()
            },
            "recurrent_features": dict(self.recurrent_features),
            "final_verdict_probabilities": dict(self.final_verdict_probabilities),
            "final_output": self.final_output,
            "schema_parse_success": self.schema_parse_success,
            "parsed_decision": self.parsed_decision,
            "parsed_violations": list(self.parsed_violations),
            "authoritative_decision": self.authoritative_decision,
            "authoritative_violations": list(self.authoritative_violations),
            "correctness_label": self.correctness_label,
            "prompt_hash": self.prompt_hash,
            "artifact_hashes": list(self.artifact_hashes),
            "sequence_log_probability": self.sequence_log_probability,
            "prompt_length": self.prompt_length,
            "output_length": self.output_length,
            "schema_complexity": self.schema_complexity,
            "final_verdict_entropy": self.final_verdict_entropy,
            "maximum_verdict_probability": self.maximum_verdict_probability,
            "verdict_token_margin": self.verdict_token_margin,
            "instrumented_logits_max_abs_error": self.instrumented_logits_max_abs_error,
            "instrumented_logits_tolerance": self.instrumented_logits_tolerance,
        }
        if self.sampled_activations is not None:
            value["sampled_activations"] = {
                step: list(values) for step, values in self.sampled_activations.items()
            }
        return value

