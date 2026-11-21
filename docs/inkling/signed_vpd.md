# Signed VPD query-gain intervention on Inkling Q4

The native run evaluated both signs of a rank-one perturbation of
`blk.6.loop_q_ctrl_gain.weight`. The SAE screen selected feature 121.
The intervention targeted the final prefill token at pass two of a
three-pass run. Both signs used Frobenius dose 0.503565 and the same
checkpoint, Vulkan runtime, core placement, prompts, and decode budget.

Four test roots each ran untreated, sham, matched random, and selected
component arms for each sign. All **32 answers** reached the independent
checker. Every sign and arm solved **3 of 4 roots**, and the reply was
identical across arms on each root. The sign captures include durable
prefixes preserved across interruptions. The joined captures were scored
against the original frozen trial plans.

## Final-pass hidden-state displacement

Each value is the L2 distance from the untreated final-prefill-token
state at pass three.

| Root family | Positive component | Positive random | Negative component | Negative random |
| --- | ---: | ---: | ---: | ---: |
| Arithmetic program | 8.47 | 0.82 | 17.03 | 0.79 |
| Constraint assignment | 6.16 | 0.87 | 1.01 | 0.86 |
| Evidence retrieval | 6.37 | 0.79 | 0.76 | 0.92 |
| Experiment planning | 5.93 | 8.51 | 0.64 | 0.53 |

The selected component changed final-pass states with different
magnitudes across roots and signs. The answer-level checks stayed flat
on this sample. The local `parameter-core-offload-02-vpd-contrast-01`
manifest records the signed factor checks, matched roots, and score and
runtime digests. The separate BPD geometry trial compares aligned and
orthogonal activation interventions at matched dose.
