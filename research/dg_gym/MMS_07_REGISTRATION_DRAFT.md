# MMS-07 preregistration (draft for freeze)

Title: Optimizer-space admission, unbiased replay and exact caching under an early-stopping-protected comparator.
Date drafted: 2026-11-27. Author: Fable. Theory and notation: `MMS_07_FABLE_REVIEW.md`.

**Status: DRAFT. Nothing in this document is registered until it is committed with every field marked `FREEZE`
filled, and that commit precedes generation of any final-development cohort.** Public development only. No sealed
cohort is opened by this plan. All MMS-01 to MMS-06 results, including negatives, stand as recorded and are not
re-analysed here.

## 1. Custody

| Field | Value |
| --- | --- |
| DataGoblin base revision | `FREEZE` (must include P0-1 to P0-5 of the implementation specification) |
| LoopedTransformers base revision | `FREEZE` |
| Registration commit | `FREEZE` (this file, byte-identical) |
| Source digests | `FREEZE`: campaign runner, `mms_utility.py`, `factor_maps.py`, `fast_gradient_bank.py`, gym generators |
| Host, threads, clock calibration | `FREEZE`; `torch.set_num_threads(2)`; tick sizes recorded |
| EvalMath preflight verdict | `FREEZE`; recorded verbatim. A verdict of `REDESIGN_OR_PILOT_ONLY` limits every claim to "prospective development evidence" |
| Burned cohorts | LBF-20D `dev6[0:32]` and `stress8`, LBF-20M roots 0 to 11, every seed in {101, 107, 113}. None may serve as monitor or final cohort |

Deviations after freeze are appended in §11 with a timestamp and never edited into the body.

## 2. Questions and estimands

Value $V$ is higher-is-better on a registered metric (negated regret; accuracy). A **world** $w$ is one training seed
together with its own freshly generated, mutually disjoint cohorts. For arm $\pi$ and compute budget $b$,
$\hat\theta_\pi(w,b)$ is the checkpoint the arm's registered monitor rule selects within budget $b$.

- **Primary estimand E1 (per compute).** $\Delta_\pi(b^*)=\mathbb E_w[V_{\text{final}}(\hat\theta_\pi(w,b^*))-V_{\text{final}}(\hat\theta_{\text{ordinary\_es}}(w,b^*))]$.
- **Specificity estimands E2.** The same difference against each control arm in §5.
- **Secondary E3 (per update).** The difference at a fixed update count, no monitor selection.
- **Secondary E4 (budget curve).** E1 at every budget on the registered grid.
- **Exact cache E5.** Paired per-root ratio of end-to-end pipeline CPU (setup, queries, accounting, verification),
  bounded exact cache against the truly disabled mode.
- **Controller E6.** Rank agreement between clone-audit estimates gathered inside training and world-level E1.

Hypotheses. H1: $\Delta_\pi(b^*)>0$ for a primary arm. H2: the arm also exceeds each control. H3: E5 ratio below 1.
Registered predictions from the review, to be scored whatever the outcome: dose-only is indistinguishable from
ordinary under AdamW (Proposition 1.7); replay shows no specific benefit when the median hit age exceeds the measured
decorrelation age (F5.8); hits follow Proposition 5.1.

## 3. Gyms, worlds and cohorts

### G1: LBF-20D-R (primary)

Model `recur3` (tied residual tanh block, hidden 64, three visits), AdamW, learning rate 0.002, weight decay 1e-4,
batch 128, site `blocks[0]` output at visit 1. CPU only.

- Generator: `tasks(seed)` with world seed $s_w=2026112700+w$. Per world: train 96 roots (128 sampled states each),
  feedback 16, monitor 16, final 32; all $n=6$; signatures mutually disjoint within the world and disjoint from every
  signature produced by seed `2026112509`. Feasibility of 160 disjoint signatures per world is confirmed by a
  source-only dry run before freeze; if infeasible the cohort sizes are reduced in this document before freeze.
- Worlds: 8 calibration worlds ($w=0..7$, burned after calibration); **32 confirmation worlds** ($w=100..131$), 16 at
  minimum where compute does not permit 32. A cohort digest manifest for all worlds is committed before any run.
- Final metric: exact policy regret on final roots (mean over roots). Monitor metric: mean local regret over monitor
  states (one forward pass; no dynamic programme).

### G2: RMP pointer_chase (replication)

LOOP-T, hidden 128, visits 8, sequence 64, batch 64 per family, AdamW 1e-3 with 200 warm-up steps, gradient clip 1.0,
registered depth curriculum. Site `blocks.0.mlp.2` at visit 4. CPU only.

- Worlds: 2 calibration, **8 confirmation**. World $w$ uses training pool slice $[2048w,2048(w+1))$ of the training
  stream per family and model seed $1000+w$.
- Monitor: first half of the validation split; final development: second half. Both are disjoint from training by
  construction; both have been exposed in earlier RMP phases, so G2 yields development evidence only.
- Final metric: exact-match accuracy at the registered target depth `FREEZE`.
- Bank addressing: direct index or `splitmix64_v2`. The pinned multiplicative hash is forbidden on this gym (review
  D1). One observe-only run under the pinned hash at 512 and 4,096 MiB is registered to score prediction F5.3.

### G3: LBF-20M-R (exact cache)

24 fresh roots not among roots 0 to 11; modes: truly disabled, bounded exact LRU at 4, 12 and 64 KiB. Five trials in
randomized mode order. Every timed repetition checks the independent output digest. The whole pipeline is timed by an
outer clock.

### G4: Mathlib TRM-XS

Blocked until the packed corpus is staged. If staged, engineering-only: observe mode on a 2,048-row pool, scoring
F5.6 and F5.7. No learning claim.

## 4. Competence gate

A gym enters the confirmatory analysis only if, on calibration worlds, `ordinary_es` at $b^*$:

- G1: has mean final-cohort-analogue regret (on calibration final roots) below both the untrained network and the
  `independent_dp` analytic policy, in at least 6 of 8 calibration worlds;
- G2: reaches interim accuracy at least 0.90 at the registered target depth on the monitor split in both calibration
  worlds.

A gym that fails the gate is run for telemetry only and reported as "not competent at the registered budget".

## 5. Arms

All replay arms use ratio $\rho=0.2$, `max_age` and bank size `FREEZE` (bank sized for full direct coverage), and
identical initialization and minibatch index stream within a world.

| Arm | Definition | Role |
| --- | --- | --- |
| `ordinary_es` | ordinary training with monitor access, checkpoint selection and early stopping | **primary comparator** |
| `dense_local_align` | physical cotangent replay, hook propagation, admission $\langle h,c\rangle>0$ (MMS-06 `dense` semantics) | primary family |
| `orth_subspace` | replay projected by $\Pi$ on a **learned non-orthogonal** map (sparse fit with invariant canonicalization), Gram receipts | primary family |
| `osa_dense` | physical replay, local injection, pushforward admission $\langle\delta_b,q_b\rangle>\tau\lVert\delta_b\rVert\lVert q_b\rVert$, $\tau=$ `FREEZE` | primary family |
| `cv_table` | control-variate table on the site weight, $\alpha$ from previous-step statistics | primary family |
| `component_outer` | same learned map as `orth_subspace`, operator $P=U_S^\top U_S$ | secondary (contrast with `orth_subspace`) |
| `svd_component` | canonical SVD map, frozen | secondary (continuity with MMS-06; $P=\Pi$ here) |
| `sham_key_permuted` | as `dense_local_align`, each hit sample receives the history of another hit sample (derangement), same admission rule and norm | control |
| `dose_only` | $\delta=\rho c$ on the same hit mask, no bank content used | control |
| `pushforward_only` | $\delta_b=\rho\lVert c_b\rVert q_b/\lVert q_b\rVert$, local injection, no bank | control |
| `schedule_only` | all bank, map and projection work executed, $\delta$ discarded | control (compute and update-count match) |
| `utility_controller` | repaired controller with clone audits, STOP and propensity logging | secondary (E6) |
| `yoked_schedule` | action sequence replayed from another world's `utility_controller` run | control for the controller |

G2 runs `ordinary_es`, `dense_local_align`, `osa_dense`, `cv_table`, `sham_key_permuted`, `dose_only`,
`schedule_only`. G3 arms are the cache modes; the hybrid arm is deferred to LBF-20H.

## 6. Budgets, monitor access and stopping

- **Calibrated update budgets.** On calibration worlds measure each arm's CPU per update, $\kappa_\pi c_0$, including
  bank, map, refresh and monitor costs, with the outer process clock. The compute budget $b$ is converted once to a
  fixed update count $U_\pi(b)=\lfloor b/(\kappa_\pi c_0)\rfloor$, frozen here. Confirmation runs are then
  deterministic. The measured CPU ledger of every confirmation run is reported and must lie within ±10% of $b$;
  otherwise the run is flagged and the arm's $U_\pi$ is not changed.
- **Budget grid.** $b\in\{0.5,1,2,4\}\times b_0$ with $b_0=$ `FREEZE`. The **primary budget $b^*$** is the smallest
  grid value at which `ordinary_es` mean monitor-selected value on calibration worlds is within one standard error of
  its best over the grid. $b^*$ and the grid are frozen before confirmation cohorts are generated.
- **Monitor access, identical for every arm.** Evaluate the monitor metric every $E$ updates ($E=20$ on G1,
  $E=250$ on G2). The evaluation and the best-checkpoint copy are charged to the arm.
- **Checkpoint selection.** Best monitor value; ties to the earliest.
- **Early stopping.** Stop after 5 consecutive monitor evaluations without improvement, or at $U_\pi(b)$.
- **Final evaluation.** The selected checkpoint is evaluated once on the final cohort, outside the budget, identically
  for all arms. Final-cohort outcomes are not read until all runs of a gym are complete.
- **Fixed-exposure runs (E3).** 400 updates on G1, `FREEZE` on G2, no monitor selection.

Under this protocol an arm cannot gain by executing fewer updates: `ordinary_es` has the same right to stop.
`schedule_only` verifies that claim empirically.

## 7. Cost and memory accounting

$C_{\text{tot}}=C_{\text{train}}+C_{\text{map}}+C_{\text{lookup}}+C_{\text{refresh}}+C_{\text{feedback}}+C_{\text{scheduler}}+C_{\text{audit}}+C_{\text{monitor}}$,
closed against outer `process_time` and `perf_counter` readings that bracket setup, timer bookkeeping and cleanup
(closure within 2% or the run is void). RAM: reserve before allocating, sum live buffers incrementally, refuse on
breach; report payload bytes and peak RSS. Total RAM cap `FREEZE`. Arms run sequentially, order randomized per world
with a recorded seed.

## 8. Analysis plan

- **Unit.** The world. Roots and states within a world are not independent units for the primary analysis.
- **Primary test.** For each primary arm, paired differences $D_w$ of world-mean final value against `ordinary_es` at
  $b^*$. One-sided sign-flip permutation test on the mean (exact enumeration for $n\le20$; $10^6$ Monte Carlo
  sign patterns with recorded seed for $n=32$). Family: the four primary arms on G1; Holm at $\alpha=0.05$.
- **Gatekeeping.** Specificity tests (E2) are evaluated only for arms that pass the primary test, each at
  $\alpha=0.05$, same statistic. G2 replication is evaluated only for arms passing on G1, one-sided $\alpha=0.05$.
- **Margin.** $\delta_{\min}$ on G1 is 5% of the `ordinary_es` mean final regret on calibration worlds, as an absolute
  number `FREEZE`; on G2, 2 percentage points of accuracy. Effects are reported with one-sided 95% bounds from the
  same permutation distribution (inverted) and a world-level bootstrap as a cross-check.
- **Secondary.** Root-level paired analysis within worlds; crossed bootstrap over worlds and roots; E3; E4.
- **Resolution, stated before data.** Planning SD of $D_w$ is 0.09 regret units (from three MMS-06 seeds; itself
  uncertain). At one-sided $\alpha=0.0125$ and 80% power the detectable mean effect is $0.278/\sqrt n$: 0.098 at
  $n=8$, 0.069 at $n=16$, 0.049 at $n=32$. The best MMS-06 mean effect at 1.5 s was 0.04. Eight worlds resolve only
  effects near 10% of regret. The exact sign-flip minimum $p$ is $2^{-n}$: 0.0039 at $n=8$, which supports a Holm
  family of at most 12.
- **Interim.** One futility look at 16 worlds when 32 are registered: if no primary arm has a positive mean $D_w$,
  neural arms stop. No efficacy stopping, so no alpha is spent.
- **G3.** Paired per-root pipeline differences, root bootstrap interval for the ratio; claim only if the 95% interval
  excludes 1 at 12 KiB.
- **E6.** Spearman correlation across arms and worlds between mean clone-audit estimate and $D_w$.
- **Missing or void runs.** A void run voids the world for all arms. Worlds are not replaced after final outcomes are
  read.

## 9. Mechanism telemetry (necessary, never sufficient)

Recorded per arm and world, reported in full, excluded from every promotion decision: hits, hit-age quantiles and
occupancy against the forecast; $\cos\varphi$ distribution and the parallel/orthogonal split of the injected norm;
agreement between local-alignment and pushforward admission; exact $\langle g_{\text{ref}},\Delta u\rangle$ under
AdamW; $r(\text{age})$ by age bin and by visit (observe mode at visits 1, 4, 7 on G2); gradient coverage $\chi$ and
$\lVert\Pi_{U_S}q\rVert/\lVert q\rVert$; Gram condition number, inflation index and refresh matching certificates;
control-variate $r^2$ and realized variance ratio; step-cost decomposition at each shape.

## 10. Falsification and promotion criteria

**Does not count as evidence of learning benefit, singly or jointly:** cache hit rate, reconstruction cosine, local
alignment, certificate pass rate, variance ratio, gradient coverage, probe accuracy, raw window progress.

**Promotion of a replay arm to "specific per-compute benefit, development scale" requires all of:**

1. primary test passed on G1 at $b^*$ under Holm;
2. specificity passed against `sham_key_permuted`, `dose_only` and `pushforward_only`;
3. one-sided 95% lower bound of $\Delta_\pi(b^*)$ above 0 and point estimate at least $\delta_{\min}$;
4. `schedule_only` not distinguishable from `ordinary_es` (otherwise the comparator protocol failed and the result is
   void);
5. same sign with one-sided $p<0.05$ on G2;
6. competence gate passed on both gyms; ledger closure and determinism checks passed;
7. EvalMath preflight `READY` on a confirmation plan before any sealed cohort; until then the label is "prospective
   development evidence".

**Outcomes that are recorded as negative results:**

- **N1, refuted at registered scale.** With at least 16 worlds, the one-sided 95% upper bound of $\Delta_\pi(b^*)$ is
  below $\delta_{\min}$ for every primary arm.
- **N2, non-specific.** An arm passes the primary test and fails any specificity test. It is reported as a dose,
  schedule or memoryless effect, not as a memory benefit.
- **N3, comparator artifact.** An arm is positive at fixed CPU without monitor rights in exploratory reruns and null
  under the registered protocol.
- **N4, exact cache.** The G3 interval for the end-to-end ratio includes 1; the query-wall claim keeps its scope.
- **N5, controller.** E6 correlation not above 0 with one-sided $p<0.05$, or the null simulation fails.

**Theory falsifiers scored regardless of outcome:** F5.1 to F5.8 of the review; `dose_only` differing from
`ordinary_es` by more than the sham SD (against Proposition 1.7 as an approximation of this setting); pushforward and
local-alignment admission agreeing on more than 95% of hit samples (`osa_dense` redundant); certificate sign
uncorrelated with clone-audit sign; a specific replay benefit with $r<0.2$ at the median hit age.

## 11. Deviations and amendments

None at draft time. Entries are appended below with UTC timestamp, commit, reason, and whether final outcomes had
been read.

## 12. Execution order

1. Land P0-1 to P0-5; run `mms07_math_checks.py`; record receipts.
2. Source-only dry run of world generation; commit cohort digest manifests for calibration worlds.
3. Calibration worlds: competence gate, per-arm step costs, $b_0$, $b^*$, $U_\pi(b)$, $\delta_{\min}$, $\tau$.
4. Fill every `FREEZE`; run EvalMath preflight; commit this document. That commit is the registration.
5. Generate confirmation cohorts; commit digests.
6. Observe-mode telemetry runs (no injection): hit laws, $r(\text{age})$, coverage. If the median hit age exceeds the
   decorrelation age on a gym, replay arms on that gym still run as registered and the prediction F5.8 is scored.
7. Confirmation runs, sequential, randomized arm order; ledgers and receipts written per run.
8. Final evaluation of selected checkpoints; analysis exactly as §8; results document preserving every arm.
