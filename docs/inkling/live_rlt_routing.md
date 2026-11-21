# Live RLT depth routing on Inkling Q4

The prompt router distilled a depth selector trained on Inkling rollout
traces with RLT feedback. It chose one pass for six test roots and three
passes for 22. The fixed policy ran all 28 roots at three passes.
Both policies used the same IQ4_NL checkpoint, streamed Vulkan runtime,
RTX 5080 placement, prompts, and decode budget.

| Metric | RLT routing | Fixed three passes |
| --- | ---: | ---: |
| Verified solves | 14/28 | 14/28 |
| Ordinary-test solves | 9/14 | 9/14 |
| Out-of-distribution solves | 5/14 | 5/14 |
| Independent verifier calls | 24 | 24 |
| Parse failures | 4 | 4 |
| Layer-visit-token work estimate | 529,056 | 618,912 |
| Native model wall time, including startup | 469.72 s | 478.58 s |

The policies agreed on the solve verdict for every root. Routing reduced
the work estimate by **14.5%** and measured model wall time by **1.9%**.
The prompt selector took 51 ms for all 28 routes. With that time included,
the observed wall-time reduction was 1.8%. The work estimate multiplies
prompt plus generated tokens by `12 + 12 × passes` for each model call.
The time comparison comes from one live run per policy.

The local `native-rlt-live-core-offload-02-scored` directory contains all
56 outcomes, paired metrics, and provenance for the three terminal
captures. The capture roots and fits are bound to their original receipts.
The campaign also contains the [six-pass SAE traces](six_pass_holdout.md),
the [signed query-gain VPD run](signed_vpd.md), and the
[paired BPD geometry run](bpd_geometry_pair.md).
