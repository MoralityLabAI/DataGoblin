# MMS-07 second-review addendum

Date: 2026-11-27. Status: proposed amendments, pre-freeze. No new gym training or learning-benefit evidence. Preserve the four Fable artifacts and all historical MMS results verbatim. This addendum is intended for `research/dg_gym/` alongside them.

## Custody and independent checks

- Uploaded `mms07_math_checks.py` SHA-256: `355662ca538f5fa50bbd8e0edc65821b7d0202608f2404e37ad9e48a59eb89fa`, matching the uploaded receipt. The receipt contains 29 PASS entries and no skipped result. Its recorded timings sum to 11.175 seconds.
- Independently reran all 18 checks with no repository-dependent arguments using NumPy 2.3.5; all passed. The other 11 were not rerun; their results remain attributable to the original receipt.
- Independently exercised two additional counterexamples against the uploaded `canonicalize_invariant` reference; see below. No repository modules were modified or retrieved for this second review.
- No commits, PRs, or training runs were made in this review.

## P0 amendments before freeze

### A. Factor geometry and finite precision

Approximate collinearity does not define an equivalence relation. The reference merge is greedy and order dependent. Use exact collinearity for algebraic identities; make approximate clustering deterministic from the multiset and record a reconstruction-error budget. Include ambiguous/tied clusters in the receipt. A relative norm threshold is a pruning policy, not proof of float32 unrepresentability. An exactly represented small coordinate can coexist with a large coordinate. A matrix-wide relative Frobenius guarantee is valid when the total residual is bounded; it does not imply elementwise preservation.

Adversarial regressions (run from the directory containing the original script):

```python
import numpy as np
from mms07_math_checks import canonicalize_invariant

angles = np.array([0., 1e-5, 2e-5])
u = np.stack([np.cos(angles), np.sin(angles)], axis=1)
v = np.eye(3)
a = canonicalize_invariant(v, u)
p = [1, 0, 2]
b = canonicalize_invariant(v[:, p], u[p])
# Existing reference: two active components versus one.
assert len(a['logm']) == 2 and len(b['logm']) == 1
for f in (a, b):
    approx = (f['v_hat'] * np.exp(f['logm'])) @ f['u_hat']
    print(np.linalg.norm(approx - v @ u))
# Existing errors: about 1e-5 and 1.414e-5.
f = canonicalize_invariant(np.eye(2), np.diag([1., 1e-9]))
assert f['status_out'] == ['active', 'unrepresentable']
assert np.float32(1e-9) != 0
```

After repair, replace the first assertion with permutation-equivalence (within the declared tolerance and clustering policy); require the second component to be active or explicitly pruned, not declared unrepresentable. Verify cancellation, near-tie ordering and cross-repository parity on actual implementations. Distinguish invariant subspace claims from identifiable rank-one components under full GL mixing; only a suitable restricted gauge or identifiable block permits component-specific claims.

### B. Actual AdamW admission

Let the objective gradient be G, the training gradient be g, and u(g; s) denote AdamW's adaptive update with the same pre-step state s. Define A as the Jacobian of the injected site (or A_loc for a local injection), d = A^T delta. If clipping is used, define u as the entire gradient-to-update pipeline including clipping.

The optimizer's differential certificate is q_Adam = A (D_g u)^T G. For small injection, objective improvement to first order in eta and delta is eta <q_Adam, delta>. For finite injection, compute Delta u = u(g+d; s) - u(g; s) exactly, without committing either optimizer state. Let theta_o be the full ordinary next parameter state, including decoupled weight decay. Then for an L_s-smooth objective J,

    J(theta_o - eta Delta u) - J(theta_o)
        <= -eta <grad J(theta_o), Delta u> + (L_s eta^2 / 2) ||Delta u||^2.

This certificate is exact for displacement; its predictive value for held-out loss still depends on the gradient estimate and the curvature bound. A lagged momentum or previous adaptive direction is a proxy, not the exact Adam sensitivity. For local injection, compute the differential pushforward with a layer matmul; use the full clipping transformation if clipping is active. Compute matched candidates from the same state and record both linear and exact update-difference scores. Track effects on optimizer moments as well as immediate parameter movement.

Scale invariance requires a constant positive gradient-history scaling, eps approaching zero, compatible optimizer state, and no scale-breaking operation. Partial hit-mask doses and changing global clipping factors do not generally satisfy it. Treat full-coverage constant-dose invariance as an engineering positive control; a hit-matched dose differing from ordinary is not a refutation of Proposition 1.7.

Orthogonality to q only implies zero linear gain. It does not imply a nonnegative second-order relative cost, even for a positive definite quadratic: take H=[[1,.9],[.9,1]], g=(1,0), d=(0,-1), so g.d=0 but g^T H d + .5 d^T H d = -.4. The oracle dominance statement holds for a fixed metric, objective gradient and injection norm, not for arbitrary multi-step effects.

### C. Bank addressing and occupancy

The direct-index mapping must be compact and injective on the full logical key tuple. For RMP keys 8i+4, raw key-minus-base spans 159,993 slots for 20,000 examples; this exceeds the 32,736 slots at 512 MiB. Use a registered dense tuple index rather than raw-key subtraction (for a single visit, i). Keep exact tags and include all stage/site identity in the mapping. If hashing is necessary, report exact forecast occupancy for the planned set and scheme. For independent uniform hashing with N keys and C slots, the finite expected occupied-slot fraction per key is C[1-(1-1/C)^N]/N; the Poisson formula is an approximation. For RMP at 512 MiB a random hash is expected to occupy about 14,966 slots, so an arbitrary 19,000-slot acceptance criterion cannot apply at every tier. Prefer direct indexing if all logical keys fit.

## P1: BPD-coordinate control-variate memory (MMS-07-CV)

Use the BPD map as a compressed gradient dictionary. Fix a dictionary D of physical parameter-space atoms for the pilot, including the basis version. For sample i, cache coefficients a_i computed from a previous visit; missing entries are zero. Define t_i = D a_i and maintain S = sum_i a_i. For a uniformly drawn batch of size B, use

    g_CV = g_batch - alpha D[(1/B) sum_{i in batch} a_i - S/N].

The table, coefficient alpha and dictionary must be fixed before the current draw. Read before write. Subtract an evicted entry from S and add its replacement exactly in the chosen accumulator precision. Handle duplicates and collision tags explicitly. For sampling with nonuniform probabilities p_i and a uniform target, apply 1/(N p_i) to the sampled terms, including the ordinary gradient, or use a correctly weighted stratified estimator for the declared objective. The table is unbiased conditionally on the pre-draw state regardless of entry staleness, compression or eviction, in exact arithmetic. It is a statement about the gradient estimator; nonlinear AdamW updates need their own evaluation.

For a site weight W with normalized atoms u_j v_j^T, record the local per-example gradient coefficient

    a_ij = sum_t (u_j^T c_it)(v_j^T x_it).

Use per-example rather than minibatch-averaged cotangent normalization; account for all token/position loss weighting. If atoms are nonorthogonal, use a declared Gram-aware codec, or explicitly interpret raw inner products as a sketch. The decoded approximation can be arbitrary for unbiasedness, but its quality determines variance. If W is shared across visits, aggregate the chosen visits consistently; a single-visit approximation may still reduce variance of the total tied gradient if its covariance is favorable. This method uses the ordinary backward pass; it does not need an extra full backward pass solely to populate the table.

A fixed rank-16 FP16 coefficient bank for 12,288 LBF examples is 384 KiB of payload; 20,000 RMP examples require 625 KiB per logical entry group. Include tags, age, FP32/FP64 sums, workspaces, dictionary and allocator costs in measured RAM. Contrast against a full direct gradient bank (LBF 64x64 FP16: 96 MiB payload at 12,288 entries) and the physical factor-pair table. The small table may already saturate; vary rank, active visits, example count and capacity independently to test what RAM buys. Never infer a scaling advantage merely from larger allocated RAM.

Estimate trace covariance and variance over the target distribution, with zeros for misses. Hit-only correlation is useful telemetry, but cannot substitute for all-draw covariance. Freeze alpha from earlier observations. Report the raw variance ratio and covariance, including negative covariance and miss rate. Add a matched zero-alpha arm (same table work) and a permuted-table arm that preserves the sum and marginal table distribution. Also compare with a random orthonormal dictionary at the same rank, bytes and update cost to isolate the BPD contribution. Freeze basis in the first pilot; for refresh, preserve physical sums using versioned dictionaries or recompute every affected stored entry and its aggregate consistently before future draws.

## Registration amendments

1. Remove the self-referential `Registration commit = FREEZE (this file, byte-identical)` requirement. Record the registration commit in a subsequent manifest or external receipt. Freeze source/config digests and the world-generation specification before materializing final-development cohorts. Resolve the ordering contradiction between the draft's manifest requirements and execution order.
2. Distinguish equal opportunity (same maximum budget with early stopping) from equal realized compute. A stopped arm may use far less than the budget. Require no overrun above the registered tolerance; report actual cost and reason for stopping. Separate fixed-exposure and equal-compute analyses. Define a monitor schedule that matches the estimand; update-based schedules imply different numbers of evaluations at equal compute. Charge mandatory evaluations consistently.
3. Match injection controls to the same intervention scope, hit mask, admitted dose distribution and propagation semantics. Compare OSA against local memoryless controls. CV requires its own zero-alpha, permuted-table and dictionary controls; dose-only replay is a different mechanism. The G2 arms currently omit pushforward-only. If cross-gym specificity is claimed, add matched controls there. Otherwise label G2 as replication of baseline contrast only.
4. `schedule_only` failing to differ significantly from ordinary is not evidence of equivalence. Register an equivalence margin and require an appropriate confidence interval inside it. At equal budget, overhead alone can change update count; use the schedule-only comparison to isolate treatment-specific gain, rather than void a result automatically because schedule-only differs from ordinary.
5. A sign-flip test is exact under suitable paired exchangeability/symmetry, not for an unrestricted null of mean zero. State the assumptions and choose an inference procedure appropriate to world sampling. The specificity promotion claim is conjunctive: for each arm form the maximum p-value across required contrasts, then control multiplicity across arm-level claims (e.g. Holm). Prespecify a separate replication analysis and claim hierarchy. Invert an appropriate interval procedure; distinguish positive effect, effect above the practical margin, and futility.
6. 32 worlds give approximately 80% normal-approximation power for a 0.049 effect, given SD 0.09 and one-sided alpha 0.0125. For 0.04 the same calculation needs 49 worlds rounded up; consider 64 worlds if the aim is to resolve the earlier effect scale. These are planning estimates, not guarantees. Burn calibration and earlier cohorts. G2 uses previously exposed and shared validation data, so it remains development evidence, conditional on that dataset; fresh generator-derived final worlds are needed for stronger cross-gym claims.
7. Repair receipt handling: explicitly record SKIP as distinct from PASS (historical-receipt check currently returns a skip dictionary that the driver would count as PASS); record informational timing separately; encode nonfinite values as tagged strings or null with reason rather than JSON Infinity. Preserve the original receipt bytes. Rerun legacy checks against the pinned source and add new tests against repaired modules; do not require obsolete defect reproductions to pass unchanged on fixed code.

## Execution gates

- Land P0 correctness changes and regression tests; preserve legacy scheme loading for historical replay. Register source digests and prove occupancy, local/full injection parity, and accumulator consistency.
- Run calibration only: ordinary competence, costs, gradient covariance across all draws, rank/bytes sweeps and staleness by visit. No final outcomes opened. Include an observe-only CV pilot before enabling correction; record estimates and cost with no outcome claim.
- Freeze the amended registration and run EvalMath preflight before prospective development cohorts. Stop any gym failing the ordinary competence gate. Keep LBF-20H separate, with exact cache correctness independent of learned retention.
- Report every arm, negative and null included. No promotion on the basis of mathematical or synthetic checks.
