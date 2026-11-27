# Research Verifiers v2.2: preregistered implementation pilot

Status: draft, implementation scaffold; no new empirical results.
Parent: PR #107 at 4b042e833c9325b70d103cc05c698c6dddc6e2d8.
Frozen v2.0/v2.1 tasks, scoring, and workers are inherited. The new controller
runner and implementation digest are separately versioned. No checkpoint ranking.

## RV22-A and C: selective RLT and causal ablations

The v2.1 RLT already consumes TRM logits. Compare autonomous, deterministic
stopping, learned LDT stopping, TRM, always-on RLT, selective RLT (predeclared
top-two margin < 0.20), narrow schema-only repair, exact synthetic routing-rule
baseline, and TRM/RLT/rule variants without the public persona projection.
All models use identical actor bindings, action caps, per-turn context, tool
budgets and timeout settings. The rule control invokes exactly the same
execution methods as TRM; verify it reproduces the synthetic labels.
The schema control renames supported aliases or converts a symbolic experiment
verb to a check; it does not fill configs, run extra fits, or synthesize answers.
The no-persona variants leave final persona submissions to the actor after a
matched pair, so report this as a policy ablation, including differences in turns.

First run fresh development cases and reference/noop controls. Freeze the
threshold, code digest, model binding, panel manifest and comparison plan.
Run a new sealed 64-case (8 per family) panel only after exclusion audit.
Exclude all previous held/prospective roots and all training/calibration roots;
private registry format: {"sources":{"v2.0-held":[root hashes],
"v2.1-prospective32":[root hashes],"training-and-calibration":[root hashes]}}.
The registry is operator supplied; verify each source against original manifests.
Keep sealed inputs, transcripts and raw evidence outside Git.

Primary endpoints: paired family-macro primary score and capability; report
per-family success, real fits, error/timeout counts, requests, wall time,
optimizer updates, controller passes, intervention types and persona effects.
Stratify paired bootstrap by family and resample whole task roots (10,000
replicates). Report absolute deltas and denominators. Separate capability from
transport failures; keep failed cases in primary denominators. Investigate
whether learned routing improves on the exact teacher-rule and narrow schema
controls. Selective RLT passes if primary score is within 0.02 of always-on RLT
and the one-sided paired lower 95% bound exceeds -0.02, with >=25% fewer RLT
passes; account for the extra TRM forward in selective correction. Test costs
at an identical CPU thread cap before any local GPU lease is disturbed.

Example (fresh development suite first; use actual registered paths):

    python -m datagoblin.research_v2_2 run --suite "$SUITE" --out "$OUT" \
      --registry "$REGISTRY" --binding "$BINDING" --model-id qwen3 \
      --controller-mode selective_rlt --max-episodes 64 \
      --exclusion-registry "$EXCLUSIONS"

A v2.1 task manifest is intentional: v2.2 compares policies on the frozen
environment contract. New report hashes identify the v2.2 implementation.
The paired evaluator now uses 10,000 joint root bootstrap replicates, stratified by family,
requires matched actor bindings and budget contracts, and emits the one-sided lower bound.
Selective RLT requires a heldout panel with at least eight roots per family.

## RV22-B: recurrent depth allocation

Train the six-feature additive exit head on train-only native Jinn/Beast
corrector traces, calibrate the threshold on separate roots. Freeze a K=24
damping horizon; early prefixes of this model are different from separately
trained K=3/6/12 heads. Native adapter executes only actual requested passes.
Threshold 1.0 always runs the full horizon. Record base and exit-head overhead
separately from 108 MACs per corrector pass. Compare fixed K=3/6/12/24,
confidence-only, shuffled exit and learned exit. Compare by actual executed
MACs including feature/exit costs, not just nominal passes.

Use a third, fresh 600-root panel outside Git, disjoint from archive 2,304,
spent 432, v2.1 confirmation 600, and new training/calibration roots. Keep the
two persona targets grouped by root. Report raw and guarded accuracy, raw
critical false accepts for each persona, mean and p95 depth, and paired
root-level uncertainty. Predeclare a target of mean depth <6 with guarded
accuracy within 0.25 percentage points of the same K=24 head; require no
increase in raw critical false accepts per persona. Treat this as a
noninferiority hypothesis, not an achieved result. Reconcile the reported
K=12 minus K=6 0.23pp displayed difference versus 0.33pp paired estimate from
the underlying seed/root aggregation before carrying it forward.

## RV22-D and E: Baby Goblin stage authorization

Recover the original durable 10M checkpoint including optimizer, scheduler,
RNG, data cursor, exact corpus and tokenizer. Hash and verify bytes before
creating a continuation receipt. The gate checks the receipt structure; it
cannot establish that operator attestations are true. Reconcile the v2.1
document's 1.25e-5 at 10M and stated 2.50e-4 at 20M (20x rather than 2x) against
the actual scheduler config. Do not infer a schedule from prose.
Full 2048-token prompts, zero truncation, explicit EOS supervision. No
weights-only optimizer restart. Preserve full-model pretraining lineage and
keep specialist forks separate. Record exact cumulative nonpadding positions,
loss-bearing positions, groups, padding and schedule at every stage.

20M is the only next gate. Use base, specialist-with-general-replay,
forward-matched base and separately loss-token-matched controls. Forward and
loss budgets are distinct; a single control is insufficient when masks differ.
Cover at least two complete specialist epochs (423,812 supervised tokens);
record the example coverage and ensure source membership and no held overlap.
Require the existing emergence gates, plus record actual generation validity
and termination separately from teacher-forced metrics. Repeated 16-case
emergence probes are development observations; use a fresh sealed panel for
confirmation. 50M/100M/200M remain conditional. No paid compute or occupied
GPU leases authorized here.

## RV22-F: research decision supervision

Use structured public packets with an experimentally testable hypothesis,
matched control, falsifier, expected observation, allowed evidence, budget,
failure recovery and brief rationale. Retain records only after independent
verification; teacher assertions alone cannot supply grades. The
research-decision.v1 bridge is an internal proposed adapter contract, not a
claim of compatibility with Distill's distill.response.v1. Inspect the
Compactification Theory Distill schema before implementing the conversion.
Fit controller targets for action routing, matched-control selection,
verification timing, repairing invalid evidence and allocating depth.
Hold out entire gym families and versions to test research transfer, and
include a deterministic template-only baseline.

## Execution gates

- Run ruff, v2.0/v2.1 unit/integration tests, new v2.2 tests, both frozen
  vf-validate runs and digest checks. Confirm original implementation bytes
  unchanged. Run a private path/size/secret audit.
- Add native paired-bootstrap campaign receipts, inspect root membership and
  run the local CPU actor pilot before the fresh sealed endpoint campaign.
- Train and calibrate the native adaptive head using private data; publish
  aggregate receipts only. The private native two-arm launcher and root-paired confidence intervals are implemented;
  separate trained fixed-depth controls and exit overhead comparisons are pending.
- Audit Baby Goblin durable checkpoint and resource lease; use gate functions
  to validate verified receipts before launch. No training has been launched.
- The native Distill v0.4 bridge replays the exact grader for response.v2 certificates;
  records remain reference-only, with prose review and training approval pending.
  No teacher calls or training have been made.

## RV22-G follow-up implementation

The v2.1 Jinn/Beast discrepancy is explained by the code: the displayed means average
five seeds, whereas `s0_fresh_g` and `s0_fresh_r` contain only seed-zero predictions.
The old bootstrap concatenates persona-level outcomes, so its confidence intervals are
seed-zero target-level intervals. They cannot establish five-seed root-level significance.
The new pilot averages the two targets and all five fixed seeds within each root, then
bootstraps roots. Seed-wise effects are reported separately; intervals do not estimate
uncertainty over newly trained seeds. The prior report and receipts are unchanged.

Run the two-arm native pilot on the CPU operator host with the actual private source
archive and the exact v2.1 600-root exclusion registry (SHA-256 checked). The fresh seed
must be selected in advance and the output directory must be fresh and outside Git:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/research_v2_2/jinn_adaptive_pilot.py \
      --source-root "$PRIVATE_SOURCE" --spent-registry "$OLD_REGISTRY" \
      --fresh-seed "$FRESH_SEED" --out "$PRIVATE_OUTPUT"

The pilot fits the native K24 corrector and exit heads before constructing the third
panel, then runs actual adaptive and fixed-horizon inference. This is the narrow
same-model comparison; it does not replace independently fitted K3/6/12 controls.

Distill's research branch is `MoralityLabAI/Compactification-Theory:research/distill-gym-v01-20261127`.
Its v0.4 research lane uses `distill.response.v2`, not the original v1 response contract.
`distill_native.reference_record` invokes the upstream `grade_research` and binds the
returned receipt to the canonical response hash. It explicitly leaves custody reference-only
and training_admission false. Provider identity remains claimed and prose remains unreviewed.
An optional integration test runs when the reviewed Distill checkout is on PYTHONPATH.
