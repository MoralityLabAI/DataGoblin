# GoblinBench Skill Factory bridge (GB-SF-09)

The eight HybridTRMLDT gyms now share a bounded research interface. The bridge
uses the existing Research Verifiers v2 endpoint binding and exact token
preflight, without modifying that suite's frozen implementation digest.
HybridTRMLDT source pin: `db5aef512bb0346e2cb60eded03984bfb7f84322`
([PR #48](https://github.com/MoralityLabAI/HybridTRMLDT/pull/48)).
Use a clean checkout at that commit, with CPU PyTorch and DataGoblin installed
in the same Python environment. All output must be outside the Hybrid checkout.

```sh
python -m datagoblin.skill_factory \
  --hybrid /path/to/HybridTRMLDT \
  --registry /private/registered-models.json \
  --model-id YOUR_REGISTERED_MODEL_ID \
  --binding /private/endpoint-binding.json \
  --role development --per-gym 1 --out /private/inky-gsf-dev
```

The model sees the task and refreshed public observation only. Each gym has
four turns, up to three experiment choices and a final host-bound submission.
The worker writes selected weights and final grading to a separate evaluator
directory. Each attempted case contributes to the denominator; failed transport,
format or evaluator sessions receive zero. The report identifies the exact
registered model binding and source pin and keeps transport/action receipts.
The local evaluator deadline is 150 seconds, run deadline 1,200 seconds,
model requests at most 30 seconds, generated tokens 512, rendered context
2,048. Use an independent local RAM cap wrapper; this bridge supplies wall
deadlines, not an OS memory sandbox. No tool can access evaluator files from
the model's action schema. Operator must control the local process environment.

For fixed-loop Inkling 2 Mini served by llama.cpp, register its actual revision,
GGUF digest, tokenizer/template, recurrence and precision separately. The
existing variable-loop research counterpart entry must not be relabeled as the
fixed-loop checkpoint. The tokenization endpoint must satisfy the exact
Research v2 transport contract (adapt the server if necessary). No provider
credentials, model endpoints or runtime allocations are included in this PR.

Use `--role trace_train` with a distinct output directory for candidate
post-training research traces. Public development results remain outside
training intake. Both roles are public deterministic tasks; this is not a
sealed model ranking or a claim that prior training never saw related examples.

The HybridTRMLDT workflow also stages the eight audited native gym sources
and scripted reference research traces into normalized envelopes. The source
audits, held lineage, root grouping, public/evaluator separation and artifact
hashes travel in the staging receipt. Every staged record remains
`training_ready=false`. The post-training collator must apply per-turn
observation boundaries, separate oracle references and raw failures from
verified corrections, exclude development lineages, and run tokenizer/mixture
admission before counting supervised tokens. No automatic corpus admission.

For Distill, bind a research form to the task root, chosen experiment and
CAL receipt. Ask for the failure hypothesis, matched control, measured change,
cost and reason for the selected artifact. Grade the actual structured actions
and claims using the host; explanatory prose is an unverified rationale until
reviewed. Keep final evaluator targets outside the teacher's input.
