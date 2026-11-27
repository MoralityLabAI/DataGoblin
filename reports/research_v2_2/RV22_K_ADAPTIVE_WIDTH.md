# RV22-K: adaptive width against optimized exact search

Status: completed local CPU synthetic campaign. No paid inference, GPU use, or
checkpoint promotion. Parent RV22-J commit 988b6c3 has both hosted workflows green.

## Exact state compression

A test observed along a path has a fixed deterministic bit on every surviving
hypothesis. Any such test necessarily fails the nonconstant-split condition on
the current mask. Hence among reachable states, the legal test inventory depends
only on the surviving mask and remaining budget, independent of the used-test
mask. For a depth-two planner the cached best one-step continuation is keyed by
(mask, budget). Weighted class masses use an incremental low-bit cache;
terminal values are integer mass-weighted rewards. Pure-class states can stop
immediately under positive test costs and the stated reward ordering.
The exact candidate comparison uses integer numerators and preserves the
legacy smallest-test-ID tie preference and stop-on-tie rule.

For a one-step continuation at mask M, let S be the best immediate stop reward,
W its mass, C the correct-class reward and c_min the cheapest legal next test.
Then U1=max(S,(C-c_min)W) bounds its value above; with no legal tests U1=S.
Apply this bound in both successor masks, subtract the first test cost times W,
and prune only when the resulting bound cannot beat the incumbent with the
registered tie rule. This is a bound for finite deterministic tests, not a
stochastic observation or repeatable-test claim.

Independent local regression: 128 new worlds, 103,650 reachable states,
129,013 exact Q2 comparisons and 129,013 upper-bound checks. Zero legal-action
mismatches and zero policy-tree mismatches for legacy/optimized/bounded.
Campaign: exact/bounded trees match and legacy utility matches at every root.

## Design

128 ordinary train roots (7,990 legal root and one-test-descendant training
actions), 48 mixed-size ordinary calibration roots, 96 roots per evaluation
cohort: ordinary12, parity3, ordinary16 and parity4. Split seeds 281–286,
trained head seeds 61–63. Ordinary16 uses the same test generator as parity4
but balanced shuffled class labels; this controls size without assuming
complete structural isomorphism disjointness. Serialized worlds disjoint
within campaign. Distill-compatible synthetic values, no native Distill
teacher-record admission.

Arms: legacy Q2, cached exact Q2, exact bounded Q2, random top-two, immediate
utility top-two, residual-ranked FF50 and TRM16 K3 top-two, adaptive TRM
shortlist. Active head MACs 750 versus 752 per candidate. The adaptive rule
uses width 1 if first/second score gap reaches g; width w if second/third
gap is below g; otherwise width 2. Fixed grid g in {.025,.05,.10}, w in
{3,8}. Calibrate by fewest expanded candidates among arms within .05 mean
utility of cached exact. Selected g=.025,w=3 (calibration utility 7.15567
versus exact 7.16605; 9.493 expanded candidates). This calibration criterion
is a point estimate, not a prospective noninferiority proof. Calibration
fallback to cached full exact if no grid member meets the tolerance.

Complete policy tree replay on all weighted hypotheses, charging path
credits. Independent full DP gives regret. CPU process clock includes
construction and policy tree only; excludes replay and oracle grading.
Within each root, arm order is randomized for each of three seeds/runs.
Average the three fixed seed outcomes per root, paired root bootstrap
10,000 replicates. CPU intervals are within this local campaign and hardware,
not cross-machine generalization. Legacy expansion fields in raw receipt
are zero placeholders (unmeasured), not comparable counts.

## Results

Net utility / CPU milliseconds:

| Arm | Ordinary12 | Parity3 | Ordinary16 | Parity4 |
|---|---|---|---|---|
| Legacy exact | 6.7322 / .4236 | 6.6447 / .4947 | 7.4811 / 1.1139 | 6.8119 / 1.1898 |
| Cached exact | 6.7322 / .2464 | 6.6447 / .2893 | 7.4811 / .5855 | 6.8119 / .6461 |
| Bounded exact | 6.7322 / .2916 | 6.6447 / .3214 | 7.4811 / .6141 | 6.8119 / .7428 |
| Random2 | 5.4058 / .2480 | 5.6447 / .2684 | 5.9190 / .4189 | 5.1423 / .4841 |
| Analytic2 | 6.5336 / .1680 | 6.3304 / .1775 | 7.3320 / .3081 | 6.5574 / .3395 |
| FF2 | 6.8065 / .3519 | 6.6329 / .3740 | 7.5790 / .5586 | 6.8818 / .6598 |
| TRM2 | 6.8029 / .3676 | 6.6059 / .3754 | 7.5796 / .5748 | 6.8904 / .6786 |
| Adaptive | 6.6423 / .3352 | 6.4744 / .3650 | 7.5406 / .5675 | 6.8638 / .6627 |

Cached exact reduced CPU time by 41.8%, 41.5%, 47.4%, 45.7%. Paired CPU
savings are positive on all four cohorts (legacy minus cached 95% intervals:
[.1525,.2044], [.1440,.2772], [.4985,.5584], [.5022,.5846] ms).
Bounded exact reduces expansions, but is slower due to bound evaluation.

Adaptive candidate reductions versus cached exact: 65.4%, 65.0%, 71.9%,
69.6%. Corresponding CPU changes: +36.0%, +26.2%, -3.1%, +2.6%.
Utility deltas and paired 95% intervals:

| Cohort | Adaptive minus cached exact | FF2 minus cached exact | TRM2 minus cached exact |
|---|---|---|---|
| Ordinary12 | -.0899 [-.2431,.0294] | +.0743 [-.0177,.1670] | +.0707 [-.0198,.1631] |
| Parity3 | -.1703 [-.4220,.0643] | -.0118 [-.2165,.1876] | -.0388 [-.2761,.1758] |
| Ordinary16 | +.0595 [-.0492,.1679] | +.0979 [.0118,.1847] | +.0985 [.0175,.1839] |
| Parity4 | +.0519 [-.0735,.1756] | +.0699 [-.0752,.2065] | +.0785 [-.0836,.2295] |

Neither learned architecture establishes a recurrence-specific advantage.
FF2 and TRM2 improve realized utility on ordinary16, but their CPU intervals
against cached exact include zero. The adaptive policy loses utility on the
smaller cohorts and has no robust end-to-end time win. No promotion. This
supersedes the previous hybrid speed claim against a slower baseline.
Use cached exact as the reference for subsequent optimization.

## Reproduce

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/research_v2_2/adaptive_width.py --out /tmp/rv22k.json

Measured source SHA-256 28569f1bb681eaceecc2f1fdf786453848c4c4b1b7c69117d41d52adc951619a.
Measured source Git blob 8e229f14a9e75afcbe44f50e05cb759f96dc1158.
Full local receipt SHA-256 260ae654ded5e321a9e3fb44b010b64929d7f877a5c0081c916eee20641a639a.

Next: extend mask/budget compression to full DP and derive reusable exact
certificates, test branch-cache reuse across structurally related worlds,
and compare learned ranking against integer-only exact search on larger
fresh problem families. All current evaluation roots are spent.
