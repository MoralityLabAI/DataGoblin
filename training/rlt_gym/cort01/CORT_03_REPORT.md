# CORT-03 development findings

The public synthetic action-value pilot uses 650 training and 640 evaluation episodes per seed, three seeds, with paired caps 4 and 6. Every policy sees the same roots and pays the same action price schedule; actual spend varies. Tests executed successfully in a JS runtime. The Node module wrapper and repository-wide Python checks were not run. No native LLM inference.

| Cohort/cap | Policy | Correct | Spend | Utility |
|---|---|---:|---:|---:|
| ordinary_4 | fixed_query | 740/960 | 3840 | 0.6308 |
| ordinary_4 | fixed_audit | 823/960 | 3840 | 0.7173 |
| ordinary_4 | heuristic_full | 827/960 | 3840 | 0.7215 |
| ordinary_4 | heuristic_stop | 827/960 | 3328 | 0.7401 |
| ordinary_4 | learned_full | 848/960 | 3840 | 0.7433 |
| ordinary_4 | learned | 845/960 | 3074 | 0.7681 |
| ordinary_6 | fixed_query | 746/960 | 5760 | 0.5671 |
| ordinary_6 | fixed_audit | 859/960 | 5687 | 0.6875 |
| ordinary_6 | heuristic_full | 879/960 | 5760 | 0.7056 |
| ordinary_6 | heuristic_stop | 863/960 | 4005 | 0.7529 |
| ordinary_6 | learned_full | 884/960 | 5760 | 0.7108 |
| ordinary_6 | learned | 883/960 | 4409 | 0.7590 |
| shift_4 | fixed_query | 588/960 | 3840 | 0.4725 |
| shift_4 | fixed_audit | 723/960 | 3840 | 0.6131 |
| shift_4 | heuristic_full | 732/960 | 3840 | 0.6225 |
| shift_4 | heuristic_stop | 732/960 | 3421 | 0.6378 |
| shift_4 | learned_full | 758/960 | 3840 | 0.6496 |
| shift_4 | learned | 755/960 | 3117 | 0.6728 |
| shift_6 | fixed_query | 599/960 | 5760 | 0.4140 |
| shift_6 | fixed_audit | 775/960 | 5636 | 0.6018 |
| shift_6 | heuristic_full | 820/960 | 5760 | 0.6442 |
| shift_6 | heuristic_stop | 800/960 | 4316 | 0.6760 |
| shift_6 | learned_full | 829/960 | 5760 | 0.6535 |
| shift_6 | learned | 826/960 | 4499 | 0.6964 |

This is a reproducible development report, not a held-out result. The generated episode-level runs and learned weights are not committed. Check stronger heuristic, distribution-shift and realized-cost controls before any promotion.
