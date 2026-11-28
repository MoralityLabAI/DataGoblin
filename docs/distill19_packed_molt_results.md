# DISTILL-19 hosted packed MoLT result

Tested commit 0c72dbe00d92ea47c1a1f9ba2d7381f36aef38e4. [Dedicated workflow 46101399274](https://github.com/MoralityLabAI/DataGoblin/actions/runs/46101399274) passed focused tests, Ruff and the full campaign. [Artifact 15244749938](https://github.com/MoralityLabAI/DataGoblin/actions/runs/46101399274/artifacts/15244749938), raw rows digest e43d7f9994f8ebdad87584fd818b0e8adeb176d5913e01b64030c89bf5d15351. All 1,536 native episode executions matched reference traces and passed independent host verification and optimality. All 2,048 wider public-array selection requests matched native reference DP and feasibility/receipt checks.

Native policy CPU (including preparation and all decisions; summed train/calibration cells):

| Width | Reference ms | Packed dense ms | Speedup | Winning roots |
|---:|---:|---:|---:|---:|
| 5 | 1.515333 | 2.483664 | 0.610 | 0/48 |
| 8 | 2.292441 | 3.041622 | 0.754 | 0/48 |
| 12 | 5.027514 | 4.136909 | 1.215 | 47/48 |
| 16 | 9.444513 | 6.722446 | 1.405 | 47/48 |

Wider public objective CPU, including setup: packed dense speedup tight/medium at width 16: 1.56/2.61; 32: 2.95/3.44; 64: 3.42/3.77; 128: 3.89/6.00. Dense wins 127/128 wider roots, sparse 127/128. Timings descriptive, not strict matched-resource or native wider-host evidence. The host's enumeration was retained only through width 16; wider arrays use reference DP parity.

Keep reference default for existing five-candidate native gym. Packed dense is a validated optional implementation for larger public objectives. Do not promote a global crossover from these finite data. Next test exact single-source exclusion compilation: prefix/suffix at-most-capacity packed DP, bounded immutable numerical atlas, conservative fallback on changes to the fixed exclusions, no cached receipt authority. Charge all preprocessing; compare cold and repeated revisions with explicit cell budgets. No training or paid calls.
