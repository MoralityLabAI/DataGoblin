import json
from pathlib import Path

import numpy as np
import pytest

from datagoblin.hashing import sha256_file
from datagoblin.serialization import canonical_json, dump_structured
from datagoblin.training_gym.rlt_gym.inkling_bpd_pair import audit_geometry_pair, score_geometry_pair
from datagoblin.training_gym.rlt_gym.inkling_feature_trial import ACTIVATION_OPERATOR, ARMS
from datagoblin.training_gym.rlt_gym.inkling_trace import WIDTH
from datagoblin.training_gym.rlt_gym.tasks import load_suite, reference_answer, write_suite


def _geometry_capture(tmp_path: Path, *, partial: bool = False):
    suite = tmp_path / "suite"
    write_suite(suite, seed=7, per_family=1)
    _, tasks = load_suite(suite / "manifest.json")
    tasks = [task for task in tasks if task["split"] == "ordinary_test"][:2]
    trials = tmp_path / "trials"
    trials.mkdir()
    selection = trials / "geometry.json"
    dump_structured(selection, {
        "schema_version": "rlt_gym.inkling_bpd_matched_geometry.v1",
        "requested_l2": 1.0, "feature": 0, "seed": 7, "native_mechanism_verified": False,
    })
    paths = []
    for geometry in ("aligned", "orthogonal"):
        trial = trials / geometry
        run = tmp_path / f"run-{geometry}"
        paths.append(run)
        raw = run / "raw"
        raw.mkdir(parents=True)
        (trial / "deltas").mkdir(parents=True)
        (trial / "prompts").mkdir()
        donor = trial / "training-contrast.json"
        dump_structured(donor, {"geometry": geometry})
        plan_rows, trace_rows, patches, completed = [], [], [], []
        states = []
        for i, task in enumerate(tasks):
            fields = ("task_id", "task_digest", "lineage_id", "family", "evidence_epoch")
            context = {field: task[field] for field in fields}
            for j, arm in enumerate(ARMS):
                key = f"{4 * i + j:03d}"
                prompt = trial / "prompts" / f"{key}.txt"
                prompt.write_text(task["task_id"], encoding="utf-8")
                dose = int(arm in ARMS[2:])
                delta_sha = None
                if arm != "untreated":
                    delta = trial / "deltas" / f"{key}.delta.f32"
                    direction = np.zeros(WIDTH, dtype="<f4")
                    direction[j + (4 if geometry == "orthogonal" else 0)] = dose
                    direction.tofile(delta)
                    delta_sha = sha256_file(delta)
                plan_rows.append({
                    **context,
                    "sample_key": key, "arm": arm, "prompt_sha256": sha256_file(prompt), "delta_sha256": delta_sha,
                })
                if partial and geometry == "orthogonal" and i == 1:
                    continue
                completed.append(f"{key}\t128\t8\tstop\t300\n")
                patches.append(f"{key}\t{int(arm != 'untreated')}\t127\t{dose}\t{dose}\n")
                trace_rows.append(f"{key}\t3\tl_out-47\t127\n")
                state = np.zeros(WIDTH, dtype="<f4")
                aligned = geometry == "aligned"
                state[0] = {"component": 6 if aligned else 3, "matched_random": 2 if aligned else 1}.get(arm, 0)
                states.append(state)
                answer = reference_answer(task) if arm in ARMS[:2] or (arm == "component" and aligned) else {}
                (raw / f"{key}.reply").write_text(canonical_json(answer), encoding="utf-8")
        plan = trial / "manifest.json"
        suite_sha = sha256_file(suite / "manifest.json")
        dump_structured(plan, {
            "schema_version": "rlt_gym.inkling_feature_trial.v1",
            "geometry": geometry, "intervention_pass": 2, "intervention_operator": ACTIVATION_OPERATOR,
            "weights_sha256": "1" * 64, "dictionary_sha256": "2" * 64,
            "selection_receipt_sha256": sha256_file(selection), "donor_receipt_sha256": sha256_file(donor),
            "suite_sha256": suite_sha, "requested_dose": 1.0, "feature": 0, "seed": 7, "rows": plan_rows,
        })
        dump_structured(run / "command.json", {
            "argv": ["trace", "runtime", "weights", "Vulkan1", "3", "85", "patterns",
                     str(trial / "prompts"), str(raw), "mmap", "192", str(trial / "deltas"), "2"],
            "trial_manifest_sha256": sha256_file(plan), "operator": ACTIVATION_OPERATOR,
            "native_program_sha256": "3" * 64, "placement_sha256": "4" * 64,
            "runtime_archive_sha256": "5" * 64, "runtime_llama_sha256": "6" * 64,
            "suite_sha256": suite_sha, "weight_sha256": "1" * 64,
        })
        dump_structured(run / "resource-receipt.json", {"native_returncode": int(partial and geometry == "orthogonal")})
        (raw / "native-config.tsv").write_text(
            "passes\tmax_new_tokens\tcontext\tintervention_pass\tintervention_operator\n"
            f"3\t192\t1024\t2\t{ACTIVATION_OPERATOR}\n", encoding="utf-8",
        )
        for file, header, rows in (
            ("completed.tsv", "sample_key\tprompt_tokens\tgenerated_tokens\tfinish_reason\telapsed_ms", completed),
            ("interventions.tsv", "sample_key\tapplied\tprefill_token\trealized_l2\tprojected_delta_l2", patches),
            ("rows.tsv", "sample_key\tpass_index\ttensor\ttoken_index", trace_rows),
        ):
            (raw / file).write_text(header + "\n" + "".join(rows), encoding="utf-8")
        np.stack(states).tofile(raw / "activations.f32")
    return suite, trials, *paths


def test_geometry_pair_uses_independent_verdicts_and_shared_complete_roots(tmp_path: Path) -> None:
    suite, trials, aligned, orthogonal = _geometry_capture(tmp_path, partial=True)
    out = tmp_path / "paired"
    result = score_geometry_pair(suite, trials, aligned, orthogonal, out)
    assert result["planned_roots"] == 2
    assert result["observed_answers"] == {"aligned": 8, "orthogonal": 4}
    assert result["complete_roots"] == {"aligned": 2, "orthogonal": 1}
    assert result["paired_roots"] == result["paired_roots_with_matching_control_replies"] == 1
    assert result["complete_pairing"] is False
    assert result["mean_solve_contrast"] == 1
    assert result["mean_final_state_l2_contrast"] == 2.0
    assert result["bpd_mechanism_verified"] is False
    assert result["pairs_sha256"] == sha256_file(out / "pairs.jsonl")
    assert result["verified_geometry_manifest_sha256"]["aligned"] == sha256_file(out / "aligned" / "manifest.json")


def test_geometry_pair_rejects_changed_prompts_or_native_runtime(tmp_path: Path) -> None:
    suite, trials, aligned, orthogonal = _geometry_capture(tmp_path)
    command_path = orthogonal / "command.json"
    command = json.loads(command_path.read_text(encoding="utf-8"))
    command["placement_sha256"] = "7" * 64
    dump_structured(command_path, command)
    with pytest.raises(ValueError, match="different runtime"):
        score_geometry_pair(suite, trials, aligned, orthogonal, tmp_path / "rejected")

    plan_path = trials / "orthogonal" / "manifest.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["rows"][0]["prompt_sha256"] = "8" * 64
    dump_structured(plan_path, plan)
    with pytest.raises(ValueError, match="different ordered roots, prompts, or arms"):
        audit_geometry_pair(trials)
