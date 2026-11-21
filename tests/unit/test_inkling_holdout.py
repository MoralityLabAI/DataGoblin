"""Check held-out native outcome scoring and full restart cost."""

from pathlib import Path

import numpy as np
import pytest

from datagoblin.hashing import sha256_file
from datagoblin.serialization import canonical_json, dump_structured
from datagoblin.training_gym.rlt_feedback import MODES
from datagoblin.training_gym.rlt_gym.inkling_holdout import score_holdout
from datagoblin.training_gym.rlt_gym.inkling_trace import NATIVE_ANSWER, WIDTH, native_prompt
from datagoblin.training_gym.rlt_gym.runner import read_jsonl
from datagoblin.training_gym.rlt_gym.tasks import load_suite, public_task, reference_answer, write_suite


def test_heldout_replay_checks_answers_and_charges_restarts(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    write_suite(suite, seed=7, per_family=1)
    suite_file = suite / "manifest.json"
    _, tasks = load_suite(suite_file)
    task = next(row for row in tasks if row["split"] == "ordinary_test")
    train = [row for row in tasks if row["split"] == "train"]
    holdout = tmp_path / "holdout"
    prompts = holdout / "batch-01"
    prompts.mkdir(parents=True)
    path = prompts / "000.txt"
    path.write_bytes(native_prompt({"context": {"task": public_task(task), "history": []}}).encode("utf-8"))
    row = {key: task[key] for key in ("task_id", "task_digest", "lineage_id", "family", "split")}
    row.update(batch="batch-01", sample_key="000", prompt_sha256=sha256_file(path))
    fit_prompts = tmp_path / "fit-prompts"
    fit_prompts.mkdir()
    dump_structured(fit_prompts / "manifest.json", {
        "suite_manifest_sha256": sha256_file(suite_file), "samples": train,
    })
    reference = tmp_path / "fit"
    for folder in ("p03-raw", "answer-sae", "answer-heads", "audited/routing"):
        (reference / folder).mkdir(parents=True)
    trace = reference / "p03-raw/manifest.json"
    dump_structured(trace, {
        "input_format": NATIVE_ANSWER, "passes": 3, "weights_sha256": "1" * 64, "runtime_sha256": "2" * 64,
        "source_prompt_manifest_sha256": sha256_file(fit_prompts / "manifest.json"),
    })
    dictionary = reference / "answer-sae/sae_dictionary.npz"
    np.savez(dictionary, pre_bias=np.zeros(WIDTH), encoder_weight=np.eye(1, WIDTH),
             encoder_bias=np.zeros(1), top_k=1)
    heads = reference / "answer-heads/heads.npz"
    np.savez(heads, **{mode: np.array([[0], [0], [0], [2]], dtype=np.float32) for mode in MODES})
    head_receipt = heads.parent / "manifest.json"
    dump_structured(head_receipt, {
        "dictionary_sha256": sha256_file(dictionary), "heads_sha256": sha256_file(heads),
        "trace_manifest_sha256": sha256_file(trace),
    })
    policy = reference / "audited/routing/manifest.json"
    dump_structured(policy, {
        "fitted": {mode: {"threshold": 1.5} for mode in MODES},
        "trace_feedback_sha256": sha256_file(head_receipt),
    })
    plan = holdout / "manifest.json"
    dump_structured(plan, {
        "schema_version": "rlt_gym.inkling_native_holdout_plan.v1", "rows": [row],
        "fixed_depths": [1, 3, 6], "model_new_token_cap": 192, "suite_sha256": sha256_file(suite_file),
        "prior_trace_sha256": sha256_file(trace), "frozen_depth_policy_sha256": sha256_file(policy),
    })
    runs = tmp_path / "runs"
    for depth in (1, 3, 6):
        arm = runs / "batch-01" / f"p{depth:02d}"
        raw = arm / "raw"
        raw.mkdir(parents=True)
        dump_structured(arm / "command.json", {
            "holdout_plan_sha256": sha256_file(plan), "batch": "batch-01", "passes": depth,
            "weights_sha256": "1" * 64, "runtime_archive_sha256": "2" * 64,
            "native_program_sha256": "3" * 64, "placement_sha256": "4" * 64, "runtime_llama_sha256": "5" * 64,
        })
        (raw / "native-config.tsv").write_text(
            f"passes\tmax_new_tokens\tcontext\trequested_load_mode\n{depth}\t192\t1024\tmmap\n",
            encoding="utf-8",
        )
        (raw / "completed.tsv").write_text(
            "sample_key\tprompt_tokens\tgenerated_tokens\tfinish_reason\telapsed_ms\n000\t10\t2\tstop\t100\n",
            encoding="utf-8",
        )
        content = canonical_json({"candidate": reference_answer(task) if depth == 3 else None})
        (raw / "000.reply").write_bytes((content if depth > 1 else "bad JSON").encode("utf-8"))
        if depth == 1:
            (raw / "rows.tsv").write_text("sample_key\tpass_index\ttensor\ttoken_index\n000\t0\tl_out-5\t9\n",
                                         encoding="utf-8")
            np.zeros(WIDTH, dtype="<f4").tofile(raw / "activations.f32")
    out = tmp_path / "score"
    report = score_holdout(suite, holdout, runs, reference, fit_prompts, out)
    got = report["results"]["ordinary_test/all"]
    assert report["observed_attempts"] == 3 and report["verifier_calls"] == 2
    assert got["paired_1_3_6"]["first_valid_depth"] == {"1": 0, "3": 1, "6": 0}
    assert got["paired_1_3_6"]["regressed_roots"] == 1
    rlt = got["paired_1_3"]["routing"]["rlt"]
    assert rlt["verified_correct"] == 1 and rlt["verifier_calls"] == 1 and rlt["model_calls"] == 2
    assert rlt["layer_visit_token_proxy"] == (24 + 48) * 12
    routing = read_jsonl(out / "routing.jsonl")[0]
    assert routing["prelude_sha256"] == sha256_file(runs / "batch-01/p01/raw/activations.f32")
    assert routing["prompt_sha256"] == row["prompt_sha256"]
    path.write_text("mutated prompt", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its original"):
        score_holdout(suite, holdout, runs, reference, fit_prompts, tmp_path / "rejected")
