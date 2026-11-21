"""Stage and verify a matched Inkling SAE readout intervention on fresh roots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured

from .bpd_contract import validate_comparison
from .inkling import MAX_RESPONSE_BYTES, _decode_candidate, _jsonl, _manifest, _new_dir, _suite_file
from .inkling_native import _finished_rows
from .inkling_trace import MAX_TOKENS, SHA256, WIDTH, boundary_sites, native_prompt
from .tasks import load_suite, public_task, verify

ARMS = ("untreated", "sham", "matched_random", "component")
ACTIVATION_OPERATOR = "additive_sae_direction"
PARAMETER_OPERATOR = "rank_one_q_gain"
GAIN_TENSOR = "blk.6.loop_q_ctrl_gain.weight"
GAIN_RANK = 64


def _parameter_directions(
    preflight: Path, feature: int, weights_sha256: str, dictionary_sha256: str, trace_sha256: str, coefficient: int,
):
    manifest = preflight / "manifest.json"
    factors = preflight / "rank-one-factors.npz"
    report = _manifest(manifest)
    if (report["schema_version"] != "rlt_gym.inkling_native_parameter_preflight.v1"
            or report["feature"] != feature or report["parameter_target"] != GAIN_TENSOR
            or report["checkpoint_sha256"] != weights_sha256 or report["factors_sha256"] != sha256_file(factors)
            or report["dictionary_sha256"] != dictionary_sha256 or report["trace_manifest_sha256"] != trace_sha256):
        raise ValueError("query-gain factors do not match the frozen weights and SAE feature")
    with np.load(factors, allow_pickle=False) as data:
        left, random, right = (data[name].astype(np.float64) for name in
                               ("selected_output", "matched_random_output", "shared_input"))
        dose = float(data["scale"])
    if (left.shape != (WIDTH,) or random.shape != (WIDTH,) or right.shape != (GAIN_RANK,)
            or any(not np.isclose(np.linalg.norm(vector), 1) for vector in (left, random, right))
            or not np.isclose(left @ random, 0, atol=1e-6)
            or not math.isfinite(dose) or dose <= 0 or dose != report["factor_frobenius_dose"]):
        raise ValueError("invalid matched query-gain factors")
    return {
        "sham": np.zeros(WIDTH + GAIN_RANK),
        "matched_random": np.concatenate((coefficient * dose * random, right)),
        "component": np.concatenate((coefficient * dose * left, right)),
    }, dose, {
        "parameter_preflight_sha256": sha256_file(manifest), "parameter_target": GAIN_TENSOR,
        "parameter_coefficient": coefficient,
    }


def _final_pass_states(run: Path, completed: list[dict]) -> dict[str, np.ndarray]:
    tokens = {row["sample_key"]: str(int(row["prompt_tokens"]) - 1) for row in completed}
    states = {}
    if not tokens:
        return states
    with (run / "rows.tsv").open(encoding="utf-8", newline="") as index, (
        run / "activations.f32"
    ).open("rb") as vectors:
        for offset, row in enumerate(csv.DictReader(index, delimiter="\t")):
            key = row["sample_key"]
            if key not in tokens or row["pass_index"] != "3" or row["token_index"] != tokens[key]:
                continue
            if key in states or row["tensor"] != boundary_sites(3)[3]:
                raise ValueError("invalid final-pass readout trace")
            vectors.seek(offset * WIDTH * 4)
            state = np.frombuffer(vectors.read(WIDTH * 4), dtype="<f4")
            if state.shape != (WIDTH,) or not np.isfinite(state).all():
                raise ValueError("invalid final-pass activation vector")
            states[key] = state.astype(np.float64)
            if states.keys() == tokens.keys():
                break
    if states.keys() != tokens.keys():
        raise ValueError("completed middle-pass trials require final-pass readout states")
    return states


def stage_trial(
    suite: Path, sae: Path, screened: Path, out: Path, *, roots: int = 4, seed: int = 7,
    parameter_preflight: Path | None = None, parameter_coefficient: int = 1,
) -> dict:
    if (not 1 <= roots <= 12 or parameter_coefficient not in (-1, 1)
            or (parameter_preflight is None and parameter_coefficient != 1)):
        raise ValueError("invalid intervention root budget or parameter coefficient")
    shortlist_path = screened / "manifest.json"
    shortlist = _manifest(shortlist_path)
    if (shortlist.get("schema_version") != "rlt_gym.inkling_vpd_feature_screen.v1"
            or shortlist["contrast"] not in ([1, 2], [1, 3])):
        raise ValueError("expected a screened feature contrast ending at core pass two or three")
    target_pass = shortlist["contrast"][1]
    if target_pass == 2 and shortlist.get("token_selection") != "last_prefill_token":
        raise ValueError("middle-pass trials require screening at the intervention token")
    dictionary = sae / "sae_dictionary.npz"
    if sha256_file(dictionary) != shortlist["dictionary_sha256"] or not shortlist["admitted_features"]:
        raise ValueError("intervention needs an admitted feature and its original dictionary")
    feature = shortlist["admitted_features"][0]
    selected = next(row for row in shortlist["features"] if row["feature"] == feature)
    fitted = _manifest(sae / "manifest.json")
    delta = -0.25 * float(selected["training_mean_difference"])
    parameter_receipt = {}
    if parameter_preflight is None:
        with np.load(dictionary, allow_pickle=False) as data:
            axis = data["decoder"][feature].astype(np.float64)
        norm = np.linalg.norm(axis)
        if axis.shape != (WIDTH,) or not math.isfinite(norm) or norm <= 0 or not math.isfinite(delta) or delta == 0:
            raise ValueError("invalid feature direction or dose")
        axis /= norm
        vector = np.random.default_rng(seed).standard_normal(WIDTH)
        vector -= axis * (vector @ axis)
        dose = abs(delta)
        vector *= dose / np.linalg.norm(vector)
        directions = {"sham": np.zeros(WIDTH), "matched_random": vector, "component": axis * delta}
        operator = ACTIVATION_OPERATOR
    else:
        if target_pass != 2:
            raise ValueError("query-gain factors are fitted for the second core pass")
        directions, dose, parameter_receipt = _parameter_directions(
            parameter_preflight, feature, fitted["weights_sha256"], shortlist["dictionary_sha256"],
            shortlist["position_trace_sha256"], parameter_coefficient,
        )
        operator = PARAMETER_OPERATOR

    suite_file = _suite_file(suite)
    _, tasks = load_suite(suite_file)
    fitted_roots = shortlist["train_pairs_by_root"].keys() | shortlist["calibration_pairs_by_root"].keys()
    first_by_family = {}
    for task in sorted(tasks, key=lambda task: (task["family"], task["task_id"])):
        if task["split"] == "ordinary_test" and task["lineage_id"] not in fitted_roots:
            first_by_family.setdefault(task["family"], task)
    chosen = list(first_by_family.values())[:roots]
    if len(chosen) != roots:
        raise ValueError("insufficient fresh test families for a matched trial")

    _new_dir(out)
    prompt_dir, vector_dir = out / "prompts", out / "deltas"
    prompt_dir.mkdir()
    vector_dir.mkdir()
    if operator == PARAMETER_OPERATOR:
        (vector_dir / "q-gain-rank-one").write_text(GAIN_TENSOR + "\n", encoding="ascii")
    rows = []
    for task in chosen:
        content = native_prompt({"context": {"task": public_task(task), "history": []}})
        for arm in ARMS:
            key = f"{len(rows):03d}"
            prompt = prompt_dir / f"{key}.txt"
            prompt.write_bytes(content.encode("utf-8"))
            vector_hash = None
            if arm in directions:
                path = vector_dir / f"{key}.delta.f32"
                directions[arm].astype("<f4").tofile(path)
                vector_hash = sha256_file(path)
            rows.append({
                "sample_key": key, "arm": arm, "task_id": task["task_id"], "task_digest": task["task_digest"],
                "lineage_id": task["lineage_id"], "family": task["family"], "evidence_epoch": task["evidence_epoch"],
                "prompt_sha256": sha256_file(prompt), "delta_sha256": vector_hash,
            })
    donor = out / "training-contrast.json"
    coefficient = parameter_coefficient if operator == PARAMETER_OPERATOR else delta
    dump_structured(donor, {"feature": feature, "coefficient": coefficient, "source": selected,
                            "training_pairs": shortlist["train_pairs_by_root"], **parameter_receipt})
    result = {
        "schema_version": "rlt_gym.inkling_feature_trial.v1", "feature": feature, "seed": seed,
        "intervention_pass": target_pass, "intervention_operator": operator,
        "suite_sha256": sha256_file(suite_file), "selection_receipt_sha256": sha256_file(shortlist_path),
        "donor_receipt_sha256": sha256_file(donor), "dictionary_sha256": sha256_file(dictionary),
        "requested_dose": dose, "rows": rows, "weights_sha256": fitted["weights_sha256"],
        **parameter_receipt,
    }
    dump_structured(out / "manifest.json", result)
    return result


def score_trial(suite: Path, trial: Path, run: Path, out: Path, *, native_program_sha256: str) -> dict:
    plan = _manifest(trial / "manifest.json")
    suite_file = _suite_file(suite)
    if (plan["schema_version"] != "rlt_gym.inkling_feature_trial.v1"
            or plan["suite_sha256"] != sha256_file(suite_file)
            or SHA256.fullmatch(native_program_sha256) is None
            or plan["donor_receipt_sha256"] != sha256_file(trial / "training-contrast.json")):
        raise ValueError("invalid intervention provenance")
    config = _finished_rows(run / "native-config.tsv")
    target_pass = plan.get("intervention_pass", 3)
    operator = plan.get("intervention_operator", ACTIVATION_OPERATOR)
    if (target_pass not in (2, 3) or len(config) != 1 or config[0]["passes"] != "3"
            or int(config[0].get("intervention_pass", "3")) != target_pass
            or operator not in (ACTIVATION_OPERATOR, PARAMETER_OPERATOR)
            or config[0].get("intervention_operator", ACTIVATION_OPERATOR) != operator
            or int(config[0]["context"]) != MAX_TOKENS
            or not 1 <= int(config[0]["max_new_tokens"]) <= 384):
        raise ValueError("intervention run did not use the registered depth and intervention pass")
    target_site = boundary_sites(3)[target_pass]
    component = f"sae:{plan['feature']}"
    if operator == PARAMETER_OPERATOR:
        if target_pass != 2 or plan["parameter_target"] != GAIN_TENSOR:
            raise ValueError("invalid query-gain target")
        component = f"{GAIN_TENSOR}/{component}"
    budget = int(config[0]["max_new_tokens"])
    completed = _finished_rows(run / "completed.tsv")
    patched = _finished_rows(run / "interventions.tsv")
    keys = [row["sample_key"] for row in plan["rows"]]
    observed = [row["sample_key"] for row in completed]
    if (len(set(keys)) != len(keys) or observed != keys[:len(observed)]
            or [row["sample_key"] for row in patched][:len(observed)] != observed):
        raise ValueError("intervention receipts do not match completed prompts")
    final_states = _final_pass_states(run, completed) if target_pass == 2 else {}
    _, suite_tasks = load_suite(suite_file)
    tasks = {task["task_id"]: task for task in suite_tasks}
    rows, grouped = [], defaultdict(dict)
    for expected, completion, receipt in zip(plan["rows"], completed, patched, strict=False):
        key, arm = expected["sample_key"], expected["arm"]
        task = tasks[expected["task_id"]]
        delta = trial / "deltas" / f"{key}.delta.f32"
        dose = float(receipt["realized_l2"])
        projected = float(receipt.get("projected_delta_l2", "nan" if operator == PARAMETER_OPERATOR else dose))
        input_tokens, output_tokens = int(completion["prompt_tokens"]), int(completion["generated_tokens"])
        if (arm not in ARMS or receipt["applied"] != ("0" if arm == "untreated" else "1")
                or int(receipt["prefill_token"]) != input_tokens - 1
                or not 1 <= input_tokens <= MAX_TOKENS - budget or not 1 <= output_tokens <= budget
                or completion["finish_reason"] not in {"stop", "length"}
                or arm in grouped[task["task_id"]]
                or task["split"] != "ordinary_test" or task["task_digest"] != expected["task_digest"]
                or sha256_file(trial / "prompts" / f"{key}.txt") != expected["prompt_sha256"]
                or (sha256_file(delta) if delta.is_file() else None) != expected["delta_sha256"]
                or not math.isfinite(dose)
                or not math.isfinite(projected) or projected < 0
                or (arm in ARMS[:2] and projected != 0)
                or not math.isclose(dose, plan["requested_dose"] if arm in ARMS[2:] else 0, rel_tol=1e-5)):
            raise ValueError("the realized intervention differs from its frozen arm")
        reply = run / f"{key}.reply"
        if reply.stat().st_size > MAX_RESPONSE_BYTES:
            raise ValueError("oversized native reply")
        try:
            content = reply.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            content = ""
        parsed, answer, _ = _decode_candidate(content)
        verdict = verify(task, answer) if parsed else {"passed": False, "status": "parse_failure"}
        row = {
            **expected, "verdict": verdict, "reply_sha256": sha256_file(reply), "realized_absolute_dose": dose,
            "projected_delta_l2": projected,
            "prefill_token": int(receipt["prefill_token"]), "checkpoint_sha256": plan["weights_sha256"],
            "adapter_sha256": native_program_sha256, "readout_site": target_site,
            "operator": operator, "component": component,
            "placement": "last_prefill_token", "visit": target_pass - 1, "planned_depth": 3,
            "feedback_mode": "native_inkling", "seed": plan["seed"],
            "donor_receipt_sha256": plan["donor_receipt_sha256"],
        }
        grouped[task["task_id"]][arm] = row
        rows.append(row)
    comparisons = []
    for arms in grouped.values():
        if set(arms) != set(ARMS):
            continue
        if len({row["prompt_sha256"] for row in arms.values()}) != 1:
            raise ValueError("matched intervention arms use different prompts")
        comparison = {"arms": arms, "selection_receipt_sha256": plan["selection_receipt_sha256"],
                      "bpd_mechanism_verified": False}
        validate_comparison(comparison)
        if final_states:
            baseline = final_states[arms["untreated"]["sample_key"]]
            comparison["final_pass_state_l2_vs_untreated"] = {
                arm: float(np.linalg.norm(final_states[arms[arm]["sample_key"]] - baseline))
                for arm in ARMS[1:]
            }
        comparisons.append(comparison)
    _new_dir(out)
    _jsonl(out / "rollouts.jsonl", rows)
    _jsonl(out / "comparisons.jsonl", comparisons)
    result = {
        "schema_version": "rlt_gym.inkling_feature_trial_score.v1",
        "plan_sha256": sha256_file(trial / "manifest.json"), "feature": plan["feature"],
        "intervention_pass": target_pass, "intervention_operator": operator,
        "planned_roots": len(plan["rows"]) // len(ARMS), "matched_roots": len(comparisons), "answers": len(rows),
        "verified_solves": {arm: sum(row["arms"][arm]["verdict"]["passed"] for row in comparisons) for arm in ARMS},
        "reply_changes_vs_untreated": {
            arm: sum(row["arms"][arm]["reply_sha256"] != row["arms"]["untreated"]["reply_sha256"]
                     for row in comparisons)
            for arm in ARMS[1:]
        },
        "rollouts_sha256": sha256_file(out / "rollouts.jsonl"),
        "comparisons_sha256": sha256_file(out / "comparisons.jsonl"), "bpd_mechanism_verified": False,
    }
    dump_structured(out / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage")
    for option in ("suite", "sae", "screened", "out"):
        stage.add_argument(f"--{option}", type=Path, required=True)
    stage.add_argument("--roots", type=int, default=4)
    stage.add_argument("--seed", type=int, default=7)
    stage.add_argument("--parameter-preflight", type=Path)
    stage.add_argument("--parameter-coefficient", type=int, choices=(-1, 1), default=1)
    score = commands.add_parser("score")
    for option in ("suite", "trial", "run", "out"):
        score.add_argument(f"--{option}", type=Path, required=True)
    score.add_argument("--native-program-sha256", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = stage_trial(**args) if command == "stage" else score_trial(**args)
    print(json.dumps({"feature": result["feature"],
                      "planned_arms": len(result["rows"])} if command == "stage" else result, sort_keys=True))


if __name__ == "__main__":
    main()
