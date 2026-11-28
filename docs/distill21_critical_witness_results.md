# DISTILL-21 hosted critical witness result

Tested commit f70b2356aebd77c184d184e9870029afc8bcd0d3. [Dedicated run 46103552037](https://github.com/MoralityLabAI/DataGoblin/actions/runs/46103552037) passed 30 focused tests, Ruff and the frozen campaign. [Artifact 15246257787](https://github.com/MoralityLabAI/DataGoblin/actions/runs/46103552037/artifacts/15246257787); raw rows digest 6518213187f127a70e96e6dca93712d1d43e55b48854072835d1b2517d749039. The initial run failed because the witness mode overwrote the inherited evidence-refresh mode; the corrected run retained the experimental settings and added three refresh regressions. All 576 native episodes matched reference traces and passed host verification/optimality; 26,496 public requests matched exact reference actions, feasibility and current evidence binding.

Aggregate CPU speedup over packed dense and winning cells (64 paired root cases per horizon):

| Workload | Delayed full atlas | Witness | Critical atlas | Full wins | Witness wins | Critical wins |
|---|---:|---:|---:|---:|---:|---:|
| stable/1 | 1.0072 | 0.9965 | 1.0000 | 34 | 20 | 22 |
| stable/4 | 0.6081 | 1.0024 | 1.0070 | 0 | 32 | 30 |
| stable/16 | 2.3947 | 1.5543 | 2.0523 | 64 | 64 | 64 |
| stable/32 | 4.7303 | 1.7127 | 3.9294 | 64 | 64 | 64 |
| churn/16 | 0.6736 | 1.0388 | 1.0423 | 0 | 45 | 44 |

32 stable revisions: critical atlas handles 1121 queries with the optimum witness, 654 via atlas, 18 memo hits, 255 direct solves, 64 extra fixed-scope solves and 58 builds. Full atlas builds 64 times. Peak temporary cells across registered cases 132225 full vs 84050 critical (36.43% lower), not measured RSS. Churn: delayed full atlas builds 192 times; critical atlas builds zero times; critical/witness win 44/45 of 64 cases with modest aggregate gains, and no evidence of a large CPU win at 128 medium. Full atlas faster in aggregate on long stable workloads; all three lose or roughly tie packed dense on short scopes. Keep default reference unchanged. Use opt-in packed solver or atlas on validated workload types. Repeated horizons share roots and timings are descriptive.

Next investigate inclusion-ordered reuse across changing scopes: keep exact numeric witnesses keyed by the set of forbidden candidates, use a relaxation witness only if its exact optimum remains feasible under the new exclusions, and compare a bounded lazy intersection-based variant with no speculative prefill. Receipt and frame authority remain current; no model updates, protected evaluation, provider calls or training admission.
