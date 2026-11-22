# AIRIS-02 v4: environment-owned ADD veto, safety and compute audit

**Measured source:** [CPU workflow 44993288177](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44993288177), PR #33. Source dataset version metta-airis02-bundle-v2; student report version metta-airis02-cpu-student-v4. The verifier-assisted arm reuses the **same trained v3 model weights**, source roots and public development tasks; it adds only the verifier wrapper at inference. The full source-root synthetic audit passed again.

## Measured 60-task shared public development results

| Student trained on | Learned STOP without veto: new-violation ADDs | Verifier vetoes | Environment verifier checks | Accepted new-violation ADDs with veto | Verified STOP solves |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform | 43 | 98 | 354 | **0** | 1/60 |
| Motif-guided | 52 | 106 | 394 | **0** | 0/60 |
| Budget-first | 40 | 93 | 341 | **0** | 0/60 |
| MOSES-gated | 43 | 93 | 349 | **0** | 1/60 |

The guard checks **public constraint violations**, comparing proposed and previous state. It rejects an ADD only if that edit introduces a new violation. Since some states already contain unresolved problems, this distinguishes an ADD that *leaves an invalid state* from one that *adds a new invalidity*. In the measured run all scored unguarded invalid ADDs happened to introduce new violations, so the guard reduced accepted invalid ADDs to zero. It is not a learned safety capability.

The policy may still fail, stall or stop too early. The learned STOP head is unchanged and does not see the hidden optimal selected set. The guard does not run any native Hyperon semantics or provide the exact-optimum solution. Across arms, the guarded run reached the true optimal set at least once in 15–16 of 60 episodes, while explicit verified STOP was completed in at most one case. This exposes a remaining model planning and stopping problem.

## Authority boundary

This is a local synthetic experiment with an environment-owned verifier, **not** a production admission policy. The guard's safety property follows directly from the set-difference check on violations in the same bounded Python curation world. It offers no proof of external-source correctness, completeness of real-world constraints, or robustness to malicious tool replies.

Every verifier check and rejected proposal counts toward computation. Added checks are substantially more numerous than model episodes: 341–394 per arm for 60 tasks. Any future benchmark should include these costs in a matched budget comparison rather than treating the safer policy as free.

No new model weights were trained for this guard. The existing research-only source and output restrictions remain unchanged: training_ready=false, maximum four presentations per synthetic root across intervention arms, no original v1 SFT root reuse, no public test or steward labels opened.

## Registered follow-on

Evaluate on a genuinely independent, steward-controlled fresh source family with the model and guard frozen. Increase task completion by teaching short verified repair trajectories and training a context-aware STOP policy, while maintaining root grouping and reporting total state/label exposures. Compare direct, fixed recurrent-depth and train-calibrated halting under a matched model size, data-token budget, verifier-call allowance and compute cap.

A safe veto wrapper is useful as a hard boundary, but the demonstrated learned student remains far from solving the curation task.
