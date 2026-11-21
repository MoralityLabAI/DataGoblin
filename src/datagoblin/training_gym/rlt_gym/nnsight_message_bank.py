"""Pass-aware taps on Inkling's previous-pass message bank.

Official NNsight guide: llm.message_input.inputs[0][:2] yields the two
completed core-block sums. The module fires once per core layer for passes
2..N, so there are 12*(N-1) calls in a single token forward.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any

from .nnsight_probe import (
    Kind,
    _configuration_passes,
    _package_version,
    _resolve,
    _trace_context,
    sample_random_features,
)

BANK_SCHEMA = "rlt_gym.nnsight_message_bank.v1"
CORE_START = 6
CORE_LENGTH = 12


@dataclass(frozen=True)
class BankPlan:
    model_id: str
    revision: str
    task_digest: str
    site_path: str = "model.language_model.message_input"
    logits_path: str = "output.logits"
    expected_passes: int = 3
    intervention_pass: int = 1  # zero-based pass 2; pass 1 has no message bank
    layer_index: int = 12
    feature_ids: tuple[int, ...] = (2027,)
    kind: Kind = "observe"
    dose: float = 1.0
    seed: int = 7
    target_token_id: int | None = None
    donor_task_digest: str | None = None

    def __post_init__(self) -> None:
        if not all(isinstance(x, str) and 0 < len(x) <= 256 for x in
                   (self.model_id, self.revision, self.task_digest, self.site_path, self.logits_path)):
            raise ValueError("model, task and site identities required")
        if type(self.expected_passes) is not int or not 2 <= self.expected_passes <= 16:
            raise ValueError("message-bank probes require 2 to 16 passes")
        if type(self.intervention_pass) is not int or not 1 <= self.intervention_pass < self.expected_passes:
            raise ValueError("message-bank intervention pass must be 1..expected_passes-1")
        if type(self.layer_index) is not int or not 6 <= self.layer_index <= 17:
            raise ValueError("message-bank layer must be in 6..17")
        if not isinstance(self.feature_ids, tuple) or not 1 <= len(self.feature_ids) <= 32 or (
            any(type(f) is not int or f < 0 for f in self.feature_ids)
            or len(set(self.feature_ids)) != len(self.feature_ids)
        ):
            raise ValueError("invalid message feature indices")
        if self.kind not in {"observe", "sham", "lesion", "random", "donor_patch"}:
            raise ValueError("invalid intervention kind")
        if not math.isfinite(self.dose) or not 0 <= self.dose <= 1:
            raise ValueError("invalid intervention dose")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("invalid seed")
        if self.target_token_id is not None and (
            type(self.target_token_id) is not int or self.target_token_id < 0
        ):
            raise ValueError("invalid target token")
        if self.kind == "donor_patch" and not self.donor_task_digest:
            raise ValueError("donor patch requires donor task digest")
        if self.kind != "donor_patch" and self.donor_task_digest is not None:
            raise ValueError("unexpected donor task digest")


def message_call_index(pass_index: int, layer_index: int) -> int:
    """Pass 2 layer 6 maps to call 0; Pass 3 layer 12 maps to call 18."""
    if type(pass_index) is not int or not 1 <= pass_index <= 15:
        raise ValueError("message bank exists only after the first pass")
    if type(layer_index) is not int or not CORE_START <= layer_index < CORE_START + CORE_LENGTH:
        raise ValueError("layer is outside the loop core")
    return (pass_index - 1) * CORE_LENGTH + layer_index - CORE_START


def check_message_calls(rows: list[dict[str, Any]], passes: int, features: tuple[int, ...]) -> None:
    expected = CORE_LENGTH * (passes - 1)
    if len(rows) != expected:
        raise RuntimeError(f"missing message calls: expected {expected}, observed {len(rows)}")
    widths: set[int] = set()
    for i, row in enumerate(rows):
        width = row.get("width")
        if (row.get("call_index") != i or row.get("pass_index") != 1 + i // CORE_LENGTH
                or row.get("layer_index") != CORE_START + i % CORE_LENGTH
                or type(width) is not int or width <= max(features)):
            raise RuntimeError("invalid message call identity or feature dimension")
        widths.add(width)
        for bank in ("bank_0", "bank_1"):
            measurements = row.get(bank)
            if not isinstance(measurements, dict):
                raise RuntimeError("missing message bank")
            if (not isinstance(measurements.get("l2"), (float, int))
                    or not math.isfinite(measurements["l2"])):
                raise RuntimeError("nonfinite message-bank norm")
            values = measurements.get("selected_features")
            if (not isinstance(values, list) or len(values) != len(features)
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)):
                raise RuntimeError("invalid message-bank feature values")
    if len(widths) != 1:
        raise RuntimeError("message bank width changed across calls")


def trace_message_bank(
    model: Any, prompt: Any, plan: BankPlan, *, remote: bool = False,
    donor_values: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
) -> dict[str, Any]:
    """Read or patch the two previous-pass core blocks before message mixing."""
    import nnsight
    import torch

    configured = _configuration_passes(model)
    if configured is not None and configured != plan.expected_passes:
        raise ValueError(f"configured {configured} passes, requested {plan.expected_passes}")
    if plan.kind == "donor_patch":
        if (donor_values is None or len(donor_values) != 2
                or any(len(values) != len(plan.feature_ids) for values in donor_values)
                or any(not math.isfinite(v) for values in donor_values for v in values)):
            raise ValueError("a donor feature vector is needed for each message block")
    elif donor_values is not None:
        raise ValueError("donor values require a donor patch")

    message_input = _resolve(model, plan.site_path)
    target_call = message_call_index(plan.intervention_pass, plan.layer_index)
    total_calls = CORE_LENGTH * (plan.expected_passes - 1)
    with _trace_context(model, prompt, remote) as tracer:
        observations = nnsight.save([])
        for call_index in tracer.iter[:total_calls]:
            bank0, bank1 = message_input.inputs[0][:2]
            if call_index == target_call:
                if plan.kind in {"lesion", "random"}:
                    bank0[0, -1, list(plan.feature_ids)] *= 1 - plan.dose
                    bank1[0, -1, list(plan.feature_ids)] *= 1 - plan.dose
                elif plan.kind == "donor_patch":
                    for bank, donor_coordinates in zip((bank0, bank1), donor_values):
                        previous = bank[0, -1, list(plan.feature_ids)]
                        replacement = torch.as_tensor(donor_coordinates, device=previous.device,
                                                      dtype=previous.dtype)
                        bank[0, -1, list(plan.feature_ids)] = (
                            (1 - plan.dose) * previous + plan.dose * replacement
                        )
            observations.append({
                "call_index": call_index,
                "pass_index": 1 + call_index // CORE_LENGTH,
                "layer_index": CORE_START + call_index % CORE_LENGTH,
                "width": int(bank0.shape[-1]),
                "bank_0": {
                    "l2": bank0[0, -1].detach().float().norm().item(),
                    "selected_features": bank0[0, -1, list(plan.feature_ids)].detach().float().cpu().tolist(),
                },
                "bank_1": {
                    "l2": bank1[0, -1].detach().float().norm().item(),
                    "selected_features": bank1[0, -1, list(plan.feature_ids)].detach().float().cpu().tolist(),
                },
            })
        logits = _resolve(model, plan.logits_path)[0, -1]
        top_token_id = nnsight.save(logits.argmax().item())
        if plan.target_token_id is not None:
            target_logprob = nnsight.save(logits.float().log_softmax(-1)[plan.target_token_id].item())

    check_message_calls(observations, plan.expected_passes, plan.feature_ids)
    logprob = float(target_logprob) if plan.target_token_id is not None else None
    if type(top_token_id) is not int or top_token_id < 0 or (logprob is not None and not math.isfinite(logprob)):
        raise RuntimeError("invalid next-token readout")
    return {
        "schema_version": BANK_SCHEMA, "plan": asdict(plan),
        "provider": "NDIF" if remote else "local",
        "configured_passes": configured,
        "exact_pass_count_supported_by_config": configured == plan.expected_passes,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest() if isinstance(prompt, str) else None,
        "nnsight_version": _package_version("nnsight"),
        "torch_version": _package_version("torch"),
        "observed_message_calls": len(observations),
        "observations": observations,
        "readout": {"top_token_id": top_token_id, "target_logprob": logprob},
        "verifier_status": "unverified", "student_training_eligible": False,
    }


def run_bank_control_suite(
    model: Any, prompt: Any, plan: BankPlan, *, remote: bool = False,
    donor_prompt: Any | None = None, donor_task_digest: str | None = None,
    trace_fn: Any = trace_message_bank,
) -> dict[str, Any]:
    if plan.kind != "observe":
        raise ValueError("suite requires an observe plan")
    if (donor_prompt is None) != (donor_task_digest is None):
        raise ValueError("donor_prompt and donor digest must be paired")
    target = message_call_index(plan.intervention_pass, plan.layer_index)
    arms = {"observe": trace_fn(model, prompt, plan, remote=remote)}
    reference = arms["observe"]
    check_message_calls(reference["observations"], plan.expected_passes, plan.feature_ids)
    width = reference["observations"][target]["width"]
    random_ids = sample_random_features(width, plan.feature_ids, plan.seed)
    for kind, features in (("sham", plan.feature_ids), ("lesion", plan.feature_ids), ("random", random_ids)):
        candidate = replace(plan, kind=kind, feature_ids=features, dose=0.0 if kind == "sham" else plan.dose)
        arms[kind] = trace_fn(model, prompt, candidate, remote=remote)
        check_message_calls(arms[kind]["observations"], plan.expected_passes, features)
    if donor_prompt is not None:
        observed = trace_fn(model, donor_prompt, replace(plan, task_digest=donor_task_digest), remote=remote)
        check_message_calls(observed["observations"], plan.expected_passes, plan.feature_ids)
        if observed["observations"][target]["width"] != width:
            raise RuntimeError("mismatched donor message width")
        donor_values = tuple(tuple(observed["observations"][target][bank]["selected_features"])
                             for bank in ("bank_0", "bank_1"))
        patch = replace(plan, kind="donor_patch", donor_task_digest=donor_task_digest)
        arms["donor_patch"] = trace_fn(model, prompt, patch, remote=remote, donor_values=donor_values)
        check_message_calls(arms["donor_patch"]["observations"], plan.expected_passes, plan.feature_ids)
    sham = arms["sham"]["readout"]
    contrasts = {}
    for kind, result in arms.items():
        if (result.get("plan", {}).get("kind") != kind or result.get("verifier_status") != "unverified"
                or result.get("student_training_eligible") is not False):
            raise RuntimeError("unbound or unverified message-bank receipt")
        value = result["readout"]
        contrasts[kind] = {
            "top_token_changed_from_sham": value["top_token_id"] != sham["top_token_id"],
            "target_logprob_minus_sham": (
                value["target_logprob"] - sham["target_logprob"]
                if value["target_logprob"] is not None and sham["target_logprob"] is not None else None
            ),
        }
    report = {
        "schema_version": BANK_SCHEMA, "model_id": plan.model_id,
        "revision": plan.revision, "task_digest": plan.task_digest,
        "target_message_call": target, "target_layer": plan.layer_index,
        "target_pass_index": plan.intervention_pass,
        "feature_ids": list(plan.feature_ids), "random_feature_ids": list(random_ids),
        "arms": arms, "comparisons": contrasts,
        "verifier_status": "unverified", "student_training_eligible": False,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return report
