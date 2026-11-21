"""Bounded, pass-aware NNsight probes for a looped teacher.

These readouts are mechanistic observations, not verifier decisions. The caller
must separately bind a completed answer to the environment-owned verifier.
NNsight and torch are optional dependencies, imported only on trace execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

Kind = Literal["observe", "sham", "lesion", "random", "donor_patch"]
SCHEMA = "rlt_gym.nnsight_loop_probe.v1"


@dataclass(frozen=True)
class ProbePlan:
    model_id: str
    revision: str
    task_digest: str
    site_path: str = "model.language_model.layers.17"
    logits_path: str = "output.logits"
    expected_passes: int = 3
    intervention_pass: int = 1  # Zero-based: 1 is the second architectural pass.
    feature_ids: tuple[int, ...] = (2027,)
    kind: Kind = "observe"
    dose: float = 1.0
    seed: int = 7
    target_token_id: int | None = None
    donor_task_digest: str | None = None

    def __post_init__(self) -> None:
        if not all(isinstance(x, str) and x and len(x) <= 256 for x in
                   (self.model_id, self.revision, self.task_digest)):
            raise ValueError("model, revision and task digest are required")
        for path in (self.site_path, self.logits_path):
            segments = path.split(".")
            if not segments or any(not s or not s.replace("_", "a").isalnum() for s in segments):
                raise ValueError("module paths must be dotted attributes or numeric indices")
        if type(self.expected_passes) is not int or not 1 <= self.expected_passes <= 16:
            raise ValueError("expected_passes must be in [1, 16]")
        if type(self.intervention_pass) is not int or not 0 <= self.intervention_pass < self.expected_passes:
            raise ValueError("intervention_pass is zero-based and must be inside expected_passes")
        if (not isinstance(self.feature_ids, tuple) or not 1 <= len(self.feature_ids) <= 32
                or any(type(i) is not int or i < 0 for i in self.feature_ids)
                or len(set(self.feature_ids)) != len(self.feature_ids)):
            raise ValueError("1 to 32 distinct, nonnegative feature indices required")
        if self.kind not in {"observe", "sham", "lesion", "random", "donor_patch"}:
            raise ValueError("unknown probe kind")
        if not math.isfinite(self.dose) or not 0 <= self.dose <= 1:
            raise ValueError("dose must be finite and between 0 and 1")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a uint32")
        if self.target_token_id is not None and (
            type(self.target_token_id) is not int or self.target_token_id < 0
        ):
            raise ValueError("target_token_id must be nonnegative")
        if self.kind == "donor_patch" and not self.donor_task_digest:
            raise ValueError("donor_patch requires donor_task_digest")
        if self.kind != "donor_patch" and self.donor_task_digest is not None:
            raise ValueError("only donor_patch may carry a donor digest")


def _resolve(model: Any, path: str) -> Any:
    value = model
    for part in path.split("."):
        value = value[int(part)] if part.isdecimal() else getattr(value, part)
    return value


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _configuration_passes(model: Any) -> int | None:
    """Read Inkling's actual runtime config, before fallback wrapper metadata.

    The official Inkling implementation reads language_model.config.loop_num_passes
    at the start of each forward. TransformersModel.config can differ when a
    hosted backend changes the active loop depth.
    """
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    configs = (getattr(language_model, "config", None), getattr(model, "config", None))
    for config in configs:
        for candidate in (config, getattr(config, "text_config", None)):
            value = getattr(candidate, "loop_num_passes", None)
            if type(value) is int and 1 <= value <= 16:
                return value
    return None


def _trace_context(model: Any, prompt: Any, remote: bool) -> Any:
    """Only remote-capable wrappers accept the remote keyword.

    A plain NNsight(torch.nn.Module) uses Envoy.trace, whose extra keywords are
    forwarded to the module forward. Supplying remote=False there is a bug.
    """
    return model.trace(prompt, remote=True) if remote else model.trace(prompt)


def sample_random_features(width: int, avoid: tuple[int, ...], seed: int) -> tuple[int, ...]:
    """A size-matched negative control with disjoint coordinate indices."""
    if type(width) is not int or width < 1 or max(avoid, default=-1) >= width:
        raise ValueError("invalid activation width or selected coordinate")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("invalid random seed")
    pool = [i for i in range(width) if i not in set(avoid)]
    if len(pool) < len(avoid):
        raise ValueError("too few coordinates for disjoint random features")
    return tuple(sorted(random.Random(seed).sample(pool, len(avoid))))


def check_passes(rows: list[dict[str, Any]], expected_passes: int, features: tuple[int, ...]) -> None:
    """Reject silently truncated traces and malformed or nonfinite measurements."""
    if len(rows) != expected_passes:
        raise RuntimeError(f"incomplete loop trace: expected {expected_passes}, observed {len(rows)}")
    widths: set[int] = set()
    for index, row in enumerate(rows):
        if row.get("pass_index") != index or type(row.get("width")) is not int:
            raise RuntimeError("loop occurrences are missing or reordered")
        width = row["width"]
        widths.add(width)
        coords = row.get("selected_features")
        if (max(features) >= width or not isinstance(coords, list) or len(coords) != len(features)
                or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in coords)):
            raise RuntimeError("bad feature observation")
        if not isinstance(row.get("activation_l2"), (int, float)) or not math.isfinite(row["activation_l2"]):
            raise RuntimeError("nonfinite activation norm")
    if len(widths) != 1:
        raise RuntimeError("activation width changed across architectural passes")


def trace_once(
    model: Any, prompt: Any, plan: ProbePlan, *, remote: bool = False,
    donor_values: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Trace one prefill through the chosen shared module.

    tracer.iter counts that module's occurrences in one forward. This is not
    a generation-step trace. Do not assume more passes than the remote config.
    NNsight's returned tensors are reduced server-side to bounded scalars.
    """
    import nnsight
    import torch

    configured = _configuration_passes(model)
    if configured is not None and configured != plan.expected_passes:
        raise ValueError(f"configured {configured} passes, requested {plan.expected_passes}")
    if plan.kind == "donor_patch":
        if (donor_values is None or len(donor_values) != len(plan.feature_ids)
                or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in donor_values)):
            raise ValueError("donor_patch needs one finite donor value per feature")
    elif donor_values is not None:
        raise ValueError("donor values were provided for a non-donor intervention")

    site = _resolve(model, plan.site_path)
    with _trace_context(model, prompt, remote) as tracer:
        observations = nnsight.save([])
        for pass_index in tracer.iter[:plan.expected_passes]:
            activation = site.output  # Inkling decoder layer returns [B, S, H].
            if pass_index == plan.intervention_pass:
                if plan.kind in {"lesion", "random"}:
                    activation[0, -1, list(plan.feature_ids)] *= 1 - plan.dose
                elif plan.kind == "donor_patch":
                    old = activation[0, -1, list(plan.feature_ids)]
                    donor = torch.as_tensor(donor_values, device=old.device, dtype=old.dtype)
                    activation[0, -1, list(plan.feature_ids)] = (1 - plan.dose) * old + plan.dose * donor
            state = activation[0, -1]
            observations.append({
                "pass_index": pass_index,
                "width": int(state.shape[-1]),
                "activation_l2": state.detach().float().norm().item(),
                "selected_features": state[list(plan.feature_ids)].detach().float().cpu().tolist(),
            })
        logits = _resolve(model, plan.logits_path)[0, -1]
        top_token_id = nnsight.save(logits.argmax().item())
        if plan.target_token_id is not None:
            target_logprob = nnsight.save(logits.float().log_softmax(-1)[plan.target_token_id].item())

    check_passes(observations, plan.expected_passes, plan.feature_ids)
    if type(top_token_id) is not int or top_token_id < 0:
        raise RuntimeError("invalid top-token readout")
    logprob = float(target_logprob) if plan.target_token_id is not None else None
    if logprob is not None and not math.isfinite(logprob):
        raise RuntimeError("invalid target token logprob")
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest() if isinstance(prompt, str) else None
    return {
        "schema_version": SCHEMA,
        "plan": asdict(plan),
        "provider": "NDIF" if remote else "local",
        "configured_passes": configured,
        "observed_passes": len(observations),
        "client_config_passes_match": configured == plan.expected_passes,
        "exact_pass_count_supported_by_config": not remote and configured == plan.expected_passes,
        "remote_server_configuration_verified": False if remote else None,
        "prompt_sha256": prompt_sha256,
        "nnsight_version": _package_version("nnsight"),
        "torch_version": _package_version("torch"),
        "observations": observations,
        "readout": {"top_token_id": top_token_id, "target_logprob": logprob},
        "verifier_status": "unverified",
        "student_training_eligible": False,
    }


def run_control_suite(
    model: Any, prompt: Any, plan: ProbePlan, *, remote: bool = False,
    donor_prompt: Any | None = None, donor_task_digest: str | None = None,
    trace_fn: Any = trace_once,
) -> dict[str, Any]:
    """Run clean, same-site sham, lesion and random ablation; optionally donor patch.

    trace_fn can be injected for deterministic offline contract tests.
    The verifier is deliberately out-of-band: logprobs are mechanistic probes.
    """
    if plan.kind != "observe":
        raise ValueError("suite starts from an observe plan")
    if (donor_prompt is None) != (donor_task_digest is None):
        raise ValueError("donor_prompt and donor_task_digest must occur together")
    arms = {"observe": trace_fn(model, prompt, plan, remote=remote)}
    reference = arms["observe"]
    check_passes(reference["observations"], plan.expected_passes, plan.feature_ids)
    width = reference["observations"][plan.intervention_pass]["width"]
    random_ids = sample_random_features(width, plan.feature_ids, plan.seed)
    for kind, features in (("sham", plan.feature_ids), ("lesion", plan.feature_ids), ("random", random_ids)):
        arm = replace(plan, kind=kind, feature_ids=features, dose=0.0 if kind == "sham" else plan.dose)
        arms[kind] = trace_fn(model, prompt, arm, remote=remote)
        check_passes(arms[kind]["observations"], plan.expected_passes, features)
    if donor_prompt is not None:
        donor_plan = replace(plan, task_digest=donor_task_digest)
        donor_obs = trace_fn(model, donor_prompt, donor_plan, remote=remote)
        check_passes(donor_obs["observations"], plan.expected_passes, plan.feature_ids)
        if donor_obs["observations"][0]["width"] != width:
            raise RuntimeError("donor activation width does not match target")
        donor_features = tuple(donor_obs["observations"][plan.intervention_pass]["selected_features"])
        patch = replace(plan, kind="donor_patch", donor_task_digest=donor_task_digest)
        arms["donor_patch"] = trace_fn(model, prompt, patch, remote=remote, donor_values=donor_features)
        check_passes(arms["donor_patch"]["observations"], plan.expected_passes, plan.feature_ids)

    for name, result in arms.items():
        if (result.get("plan", {}).get("kind") != name or result.get("verifier_status") != "unverified"
                or result.get("student_training_eligible") is not False):
            raise RuntimeError("inconsistent intervention provenance")

    reference_readout = arms["sham"]["readout"]
    comparisons: dict[str, Any] = {}
    for name, result in arms.items():
        score = result["readout"]
        comparisons[name] = {
            "top_token_changed_from_sham": score["top_token_id"] != reference_readout["top_token_id"],
            "target_logprob_minus_sham": (
                score["target_logprob"] - reference_readout["target_logprob"]
                if score["target_logprob"] is not None and reference_readout["target_logprob"] is not None
                else None
            ),
        }
    result = {
        "schema_version": SCHEMA,
        "model_id": plan.model_id,
        "revision": plan.revision,
        "task_digest": plan.task_digest,
        "expected_passes": plan.expected_passes,
        "intervention_pass": plan.intervention_pass,
        "feature_ids": list(plan.feature_ids),
        "random_feature_ids": list(random_ids),
        "arms": arms,
        "comparisons": comparisons,
        "verifier_status": "unverified",
        "student_training_eligible": False,
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    result["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return result
