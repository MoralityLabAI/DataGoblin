from __future__ import annotations

import gc
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.hashing import sha256_text

from .cycle_metrics import (
    CycleFeatureValues,
    entropy,
    exponential_decay_estimate,
    jensen_shannon_from_logits,
    pooled_update,
    safe_relative_update,
    token_update_norms,
    top_probability_margin,
)
from .prompting import VerdictDecision, parse_verdict_output
from .trace_protocol import TRACE_ORDER, ForwardOutput, RecurrentStep, TraceableRecurrentModel, TraceSummary


@dataclass(frozen=True)
class TraceOptions:
    selected_token_indices: tuple[int, ...] = (0, -1)
    logits_tolerance: float = 1e-5
    output_byte_budget: int = 1_000_000
    store_sampled_activations: bool = False
    activation_byte_budget: int = 0

    def __post_init__(self) -> None:
        if self.logits_tolerance < 0 or self.output_byte_budget < 1 or self.activation_byte_budget < 0:
            raise ValueError("trace budgets and tolerance must be nonnegative")
        if self.store_sampled_activations and self.activation_byte_budget < 8:
            raise ValueError("sampled activation storage requires an explicit byte budget of at least 8")


class _StreamingTraceAccumulator:
    def __init__(self, options: TraceOptions) -> None:
        self.options = options
        self.order: list[str] = []
        self.per_step: dict[str, float] = {}
        self.pooled: dict[str, float] = {}
        self.selected: dict[str, dict[str, float]] = {}
        self.residuals: dict[str, float] = {}
        self.gates: dict[str, dict[str, float]] = {}
        self.samples: dict[str, tuple[float, ...]] | None = {} if options.store_sampled_activations else None
        self._sample_bytes = 0
        self._previous: np.ndarray | None = None
        self._h1_state: np.ndarray | None = None
        self._h1_logits: np.ndarray | None = None
        self._h2_logits: np.ndarray | None = None
        self._final_l_updates: list[float] = []
        self._relative_h_update: float | None = None
        self._maximum_late_token_update: float | None = None
        self._residual_growth: float | None = None

    def observe(self, step: RecurrentStep) -> None:
        expected = TRACE_ORDER[len(self.order)] if len(self.order) < len(TRACE_ORDER) else None
        if step.name != expected:
            raise ValueError(f"trace order mismatch: expected {expected}, observed {step.name}")
        state = np.asarray(step.state, dtype=np.float64)
        if state.size == 0 or not np.all(np.isfinite(state)):
            raise ValueError(f"invalid state at {step.name}")
        self.order.append(step.name)
        previous = self._previous
        if previous is None:
            self.per_step[step.name] = 0.0
            self.pooled[step.name] = 0.0
            updates = np.zeros(state.reshape(-1, state.shape[-1]).shape[0], dtype=np.float64)
        else:
            self.per_step[step.name] = safe_relative_update(previous, state)
            self.pooled[step.name] = pooled_update(previous, state)
            updates = token_update_norms(previous, state)
        self.selected[step.name] = self._select_updates(updates)
        self.residuals[step.name] = float(np.linalg.norm(state))
        self.gates[step.name] = {str(key): float(value) for key, value in (step.attention_gate_summary or {}).items()}
        if step.name in {"L5", "L6"} and previous is not None:
            self._final_l_updates.append(safe_relative_update(previous, state))
        if step.name == "H1":
            self._h1_state = state
            if step.logits is not None:
                self._h1_logits = np.asarray(step.logits, dtype=np.float64).copy()
        elif step.name == "H2":
            if self._h1_state is None:
                raise ValueError("H2 observed before H1")
            self._relative_h_update = safe_relative_update(self._h1_state, state)
            late_updates = token_update_norms(self._h1_state, state)
            self._maximum_late_token_update = float(np.max(late_updates)) if len(late_updates) else 0.0
            h1_norm = float(np.linalg.norm(self._h1_state))
            self._residual_growth = (float(np.linalg.norm(state)) - h1_norm) / max(h1_norm, 1e-12)
            if step.logits is not None:
                self._h2_logits = np.asarray(step.logits, dtype=np.float64).copy()
        self._sample(step.name, state)
        self._previous = state

    def cycle_features(self) -> CycleFeatureValues:
        required = (
            self._relative_h_update,
            self._maximum_late_token_update,
            self._residual_growth,
        )
        if any(value is None for value in required) or len(self._final_l_updates) != 2:
            raise ValueError("trace did not provide the required late-cycle boundaries")
        js_divergence = (
            jensen_shannon_from_logits(self._h1_logits, self._h2_logits)
            if self._h1_logits is not None and self._h2_logits is not None
            else 0.0
        )
        assert self._relative_h_update is not None
        assert self._maximum_late_token_update is not None
        assert self._residual_growth is not None
        return CycleFeatureValues(
            relative_h1_h2_update_norm=self._relative_h_update,
            mean_final_l_update_norm=float(np.mean(self._final_l_updates)),
            maximum_token_level_late_update=self._maximum_late_token_update,
            exponential_update_decay_estimate=exponential_decay_estimate(tuple(self.per_step.values())),
            h1_h2_js_logit_divergence=js_divergence,
            residual_norm_growth=self._residual_growth if math.isfinite(self._residual_growth) else 0.0,
        )

    def _select_updates(self, values: np.ndarray) -> dict[str, float]:
        result: dict[str, float] = {}
        for raw_index in self.options.selected_token_indices:
            index = raw_index if raw_index >= 0 else len(values) + raw_index
            if 0 <= index < len(values):
                result[str(raw_index)] = float(values[index])
        return result

    def _sample(self, name: str, state: np.ndarray) -> None:
        if self.samples is None:
            return
        available = (self.options.activation_byte_budget - self._sample_bytes) // 8
        flattened = state.reshape(-1)
        count = min(int(available), len(flattened), 16)
        if count <= 0:
            return
        values = tuple(float(value) for value in flattened[:count])
        self.samples[name] = values
        self._sample_bytes += count * 8


def run_instrumented_forward(
    model: TraceableRecurrentModel,
    prompt: str,
    *,
    authoritative_decision: str | None,
    authoritative_violations: Sequence[str] = (),
    schema_complexity: float,
    artifact_hashes: Sequence[str] = (),
    options: TraceOptions | None = None,
) -> TraceSummary:
    selected_options = options or TraceOptions()
    ordinary = model.ordinary_forward(prompt)
    accumulator = _StreamingTraceAccumulator(selected_options)
    traced = model.trace_forward(prompt, accumulator.observe)
    if tuple(accumulator.order) != TRACE_ORDER:
        raise ValueError(f"incomplete recurrent trace: {accumulator.order}")
    if ordinary.final_output != traced.final_output:
        raise ValueError("instrumentation changed the model's final output")
    if ordinary.final_logits.shape != traced.final_logits.shape:
        raise ValueError("instrumentation changed the final-logit shape")
    maximum_error = float(np.max(np.abs(ordinary.final_logits - traced.final_logits)))
    if not np.allclose(
        ordinary.final_logits,
        traced.final_logits,
        rtol=selected_options.logits_tolerance,
        atol=selected_options.logits_tolerance,
    ):
        raise ValueError(
            f"instrumented logits differ from ordinary logits: {maximum_error} > {selected_options.logits_tolerance}"
        )
    parsed = parse_verdict_output(traced.final_output)
    verdict = parsed.verdict
    parsed_decision = verdict.decision.value if verdict else None
    parsed_violations = verdict.violations if verdict else ()
    correctness = (
        parsed_decision == authoritative_decision
        and set(parsed_violations) == set(authoritative_violations)
        if authoritative_decision is not None
        else None
    )
    features = accumulator.cycle_features()
    maximum_probability, margin = top_probability_margin(traced.verdict_probabilities)
    summary = TraceSummary(
        trace_id=f"trace-{sha256_text(prompt + model.model_id)[:20]}",
        schema_version="1.0",
        model_id=model.model_id,
        trace_order=tuple(accumulator.order),
        per_step_relative_update_norms=accumulator.per_step,
        pooled_update_norms=accumulator.pooled,
        selected_token_update_norms=accumulator.selected,
        residual_norms=accumulator.residuals,
        attention_gate_summaries=accumulator.gates,
        recurrent_features=features.to_jsonable(),
        final_verdict_probabilities=traced.verdict_probabilities,
        final_output=traced.final_output,
        schema_parse_success=parsed.parsed,
        parsed_decision=parsed_decision,
        parsed_violations=parsed_violations,
        authoritative_decision=authoritative_decision,
        authoritative_violations=tuple(authoritative_violations),
        correctness_label=correctness,
        prompt_hash=sha256_text(prompt),
        artifact_hashes=tuple(artifact_hashes),
        sequence_log_probability=traced.sequence_log_probability,
        prompt_length=traced.prompt_token_count,
        output_length=traced.output_token_count,
        schema_complexity=schema_complexity,
        final_verdict_entropy=entropy(traced.verdict_probabilities),
        maximum_verdict_probability=maximum_probability,
        verdict_token_margin=margin,
        instrumented_logits_max_abs_error=maximum_error,
        instrumented_logits_tolerance=selected_options.logits_tolerance,
        sampled_activations=accumulator.samples,
    )
    encoded = json.dumps(summary.to_jsonable(), sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) > selected_options.output_byte_budget:
        raise ValueError(f"trace summary exceeds output byte budget: {len(encoded)}")
    return summary


class DeterministicMockRecurrentModel:
    """Tiny deterministic test seam; its traces are not evidence about HRM-Text."""

    model_id = "deterministic-mock-recurrent-v1"

    def ordinary_forward(self, prompt: str) -> ForwardOutput:
        return self._output(prompt)

    def trace_forward(self, prompt: str, observer: Any) -> ForwardOutput:
        seed = int(sha256_text(prompt)[:16], 16)
        rng = np.random.default_rng(seed)
        state = rng.normal(0.0, 0.15, size=(8, 12))
        for index, name in enumerate(TRACE_ORDER):
            decay = 0.82 + 0.01 * index
            state = decay * state + rng.normal(0.0, 0.025, size=state.shape)
            logits = np.asarray([state[-1, 0], state[-1, 1], state[-1, 2], -state[-1, 0], 0.0])
            observer(
                RecurrentStep(
                    name=name,
                    state=state.copy(),
                    logits=logits,
                    attention_gate_summary={"mean": float(0.5 + 0.01 * math.sin(index))},
                )
            )
        return self._output(prompt)

    def _output(self, prompt: str) -> ForwardOutput:
        recipe_text = prompt.split("RECIPE_JSON=", 1)[-1]
        violations: list[str] = []
        checks = {
            "evaluation_leakage": '"evaluation_seeds_exposed":true',
            "missing_independent_verifier": '"required":false',
            "unbounded_budget": '"wall_clock_seconds":0',
            "missing_falsifier": '"falsification_criteria":[]',
            "metric_mutated_after_freeze": '"metric_changed_after_run":true',
            "curriculum_diversity_collapse": '"diversity_collapsed":true',
            "cherry_picked_seeds": '"cherry_picked_seeds":true',
            "hidden_result_exposed": '"hidden_result_exposed":true',
        }
        for rule, marker in checks.items():
            if marker in recipe_text:
                violations.append(rule)
        decision = "ACCEPT" if not violations else ("REJECT" if violations[0] in {
            "evaluation_leakage", "missing_independent_verifier", "unbounded_budget", "metric_mutated_after_freeze"
        } else "REVISE")
        confidence = 0.76 if not violations else 0.68
        output = json.dumps(
            {"decision": decision, "violations": violations, "confidence": confidence},
            sort_keys=True,
            separators=(",", ":"),
        )
        probability = {
            VerdictDecision.ACCEPT.value: 0.72 if decision == "ACCEPT" else 0.07,
            VerdictDecision.REVISE.value: 0.72 if decision == "REVISE" else 0.07,
            VerdictDecision.AUDIT.value: 0.07,
            VerdictDecision.ABSTAIN.value: 0.07,
            VerdictDecision.REJECT.value: 0.72 if decision == "REJECT" else 0.07,
        }
        total = sum(probability.values())
        probability = {key: value / total for key, value in probability.items()}
        logits = np.log(np.asarray(list(probability.values()), dtype=np.float64))
        return ForwardOutput(
            final_logits=logits,
            final_output=output,
            verdict_probabilities=probability,
            sequence_log_probability=-0.2 * len(output.split()),
            prompt_token_count=len(prompt.split()),
            output_token_count=len(output.split()),
        )

    def close(self) -> Mapping[str, Any]:
        gc.collect()
        return {"model_id": self.model_id, "closed": True, "cuda_cleanup": "not_applicable"}


class TransformersHRMTextModel:
    """Optional HRM-Text adapter using structural L/H hooks.

    The mapping of the released implementation's L/H modules to cognitive roles is
    deliberately not asserted here. This adapter only observes call boundaries.
    """

    def __init__(self, model: Any, tokenizer: Any, reference: str, max_new_tokens: int) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.model_id = reference
        self.max_new_tokens = max_new_tokens
        self._recurrent_owner = self._find_recurrent_owner(model)

    @classmethod
    def load(
        cls,
        *,
        local_model_path: Path | None = None,
        model_id: str | None = None,
        allow_download: bool = False,
        trust_remote_code: bool = False,
        max_new_tokens: int = 96,
        device: str = "cpu",
    ) -> TransformersHRMTextModel:
        from .availability import validate_model_reference

        reference, local_files_only = validate_model_reference(
            model_path=local_model_path,
            model_id=model_id,
            allow_download=allow_download,
        )
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install the hrm-text optional extra to load HRM-Text") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            reference,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            reference,
            local_files_only=local_files_only,
            torch_dtype="auto",
            trust_remote_code=trust_remote_code,
        )
        if device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda, or auto")
        selected_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if selected_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            torch.cuda.set_per_process_memory_fraction(0.9)
        model.to(selected_device)
        model.eval()
        torch.set_grad_enabled(False)
        try:
            return cls(model, tokenizer, reference, max_new_tokens)
        except Exception:
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            raise

    @staticmethod
    def _find_recurrent_owner(model: Any) -> Any:
        candidates = [
            module for module in model.modules() if hasattr(module, "H_module") and hasattr(module, "L_module")
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"expected one HRM recurrent owner, found {len(candidates)}")
        owner = candidates[0]
        h_cycles = int(getattr(getattr(owner, "config", owner), "H_cycles", getattr(owner, "H_cycles", -1)))
        l_cycles = int(getattr(getattr(owner, "config", owner), "L_cycles", getattr(owner, "L_cycles", -1)))
        if (h_cycles, l_cycles) != (2, 3):
            raise RuntimeError(f"trace contract requires H_cycles=2,L_cycles=3; observed {h_cycles},{l_cycles}")
        return owner

    def ordinary_forward(self, prompt: str) -> ForwardOutput:
        return self._generate_and_score(prompt)

    def trace_forward(self, prompt: str, observer: Any) -> ForwardOutput:
        inputs, generated = self._generate(prompt)
        observed: list[str] = []
        pending_gate = {"sum": 0.0, "count": 0, "minimum": math.inf, "maximum": -math.inf}

        def begin_stack(_module: Any, _inputs: Any) -> None:
            pending_gate.update({"sum": 0.0, "count": 0, "minimum": math.inf, "maximum": -math.inf})

        def capture_gate(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if getattr(tensor, "shape", ()) and int(tensor.shape[-1]) % 4 == 0:
                gate = tensor.chunk(4, dim=-1)[3].sigmoid().detach().float()
                pending_gate["sum"] += float(gate.sum().cpu())
                pending_gate["count"] += gate.numel()
                pending_gate["minimum"] = min(pending_gate["minimum"], float(gate.min().cpu()))
                pending_gate["maximum"] = max(pending_gate["maximum"], float(gate.max().cpu()))

        def hook(kind: str) -> Any:
            def capture(_module: Any, _inputs: Any, output: Any) -> None:
                index = len(observed)
                expected = TRACE_ORDER[index] if index < len(TRACE_ORDER) else None
                name = expected if expected and expected.startswith(kind) else f"unexpected_{kind}_{index}"
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                state = tensor.detach().float().cpu().numpy()
                logits = self._project_state(tensor) if kind == "H" else None
                gate_summary = None
                if pending_gate["count"]:
                    gate_summary = {
                        "mean": pending_gate["sum"] / pending_gate["count"],
                        "minimum": pending_gate["minimum"],
                        "maximum": pending_gate["maximum"],
                    }
                observer(
                    RecurrentStep(
                        name=name,
                        state=state,
                        logits=logits,
                        attention_gate_summary=gate_summary,
                    )
                )
                observed.append(name)

            return capture

        l_module = self._recurrent_owner.L_module
        h_module = self._recurrent_owner.H_module
        handles = [
            l_module.register_forward_pre_hook(begin_stack),
            h_module.register_forward_pre_hook(begin_stack),
            l_module.register_forward_hook(hook("L")),
            h_module.register_forward_hook(hook("H")),
        ]
        for stack in (l_module, h_module):
            for layer in getattr(stack, "layers", ()):
                projection = getattr(getattr(layer, "attn", None), "gqkv_proj", None)
                if projection is not None:
                    handles.append(projection.register_forward_hook(capture_gate))
        try:
            result = self._score_generated(prompt, inputs, generated)
        finally:
            for handle in handles:
                handle.remove()
        if tuple(observed) != TRACE_ORDER:
            raise RuntimeError(f"released-source call boundary did not match trace contract: {observed}")
        return result

    def _project_state(self, state: Any) -> np.ndarray | None:
        head = self._model.get_output_embeddings()
        if head is None:
            return None
        try:
            projected_state = state[..., -1:, :] if getattr(state, "ndim", 0) >= 3 else state
            return head(projected_state).detach().float().cpu().numpy()
        except (RuntimeError, TypeError, ValueError):
            return None

    def _generate_and_score(self, prompt: str) -> ForwardOutput:
        inputs, generated = self._generate(prompt)
        return self._score_generated(prompt, inputs, generated)

    def _generate(self, prompt: str) -> tuple[dict[str, Any], Any]:
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt")
        if getattr(self._model.config, "prefix_lm", False) and "token_type_ids" not in inputs:
            inputs["token_type_ids"] = torch.ones_like(inputs["input_ids"])
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return inputs, generated

    def _score_generated(self, prompt: str, inputs: Mapping[str, Any], generated: Any) -> ForwardOutput:
        import torch

        prompt_length = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            scored = self._model(input_ids=generated)
        output_ids = generated[:, prompt_length:]
        text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        final_logits_tensor = scored.logits[:, -1, :]
        decision_ids = {
            decision.value: self._tokenizer.encode(decision.value, add_special_tokens=False)[0]
            for decision in VerdictDecision
        }
        selected = torch.stack([final_logits_tensor[0, token] for token in decision_ids.values()])
        probabilities = torch.softmax(selected, dim=-1).detach().float().cpu().numpy()
        token_logits = scored.logits[:, prompt_length - 1 : -1, :]
        if output_ids.numel():
            log_probabilities = torch.log_softmax(token_logits, dim=-1)
            selected_log_probabilities = log_probabilities.gather(-1, output_ids.unsqueeze(-1)).squeeze(-1)
            sequence_log_probability = float(selected_log_probabilities.sum().detach().cpu())
        else:
            sequence_log_probability = 0.0
        return ForwardOutput(
            final_logits=final_logits_tensor.detach().float().cpu().numpy(),
            final_output=text,
            verdict_probabilities={key: float(value) for key, value in zip(decision_ids, probabilities, strict=True)},
            sequence_log_probability=sequence_log_probability,
            prompt_token_count=prompt_length,
            output_token_count=int(output_ids.shape[-1]),
        )

    def close(self) -> Mapping[str, Any]:
        reference = self.model_id
        del self._model
        del self._tokenizer
        gc.collect()
        cuda_cleanup = "not_available"
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                cuda_cleanup = "synchronize_empty_cache_ipc_collect"
        except ImportError:
            pass
        return {"model_id": reference, "closed": True, "cuda_cleanup": cuda_cleanup}
