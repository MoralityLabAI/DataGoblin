# MOLT-05 measured development results

Frozen source head: `3ebe28c00907cae8e59216408d80c09902078e8d`.
[Focused workflow](https://github.com/MoralityLabAI/DataGoblin/actions/runs/45844204382)
passed focused tests, Ruff, full development training/evaluation and artifact upload.
Four freshly initialized Tied3 heads (53,59,61,67) fit 1,024 TRAIN roots.
The same 128 short and 128 long fresh development worlds are crossed with all four
heads, each with clean/revised source variants. No sealed evaluation.

Rows: 28,672. Rows digest:
`83b2e6c4abd76b90c34d106300b91a246fda0b64fe9d53fb2be4b82709b3c0f2`.
Audit digest:
`bc8b49a449ee9257f374f1a2de64a5511cd499741f6074aa379be87fd696a689`.
Raw receipt artifact is retained by Actions for its configured seven-day period.
No generated runs committed.

## Instrument gate

All 26,888 active strong-sham recipient displacements matched; 1,784 entries had
zero dose and contribute no distinct direction evidence. No active unmatched
entries. Max errors: norm 2.665e-15, signed future-Jacobian displacement
1.860e-15, recipient endpoint radius 4.441e-15. Full recurrent-write
reconstruction 2.776e-15, reconstructed score 8.882e-16, full restore score
4.441e-16. Rounded host retention decisions were exactly preserved by full
reconstruction/restoration. This is a pass of the SVD instrument, not native BPD.

## Primary descriptive slice

Long revised-source worlds, 512 crossed world/model rows per condition.
Intact: 220/512 verified. Donor corruption: 205/512 verified.
45 intact-success/corrupt-failure rows were eligible for conditional recovery.
Clean damage uses the 220 intact successes.

| Intervention | Corrupt verified | Clean verified | Recovered / 45 | Clean damaged / 220 | Balanced success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Targeted | 217 | 221 | 29 | 31 | 438/1024 |
| Equal rank | 208 | 208 | 9 | 32 | 416/1024 |
| Norm sham | 206 | 223 | 12 | 25 | 429/1024 |
| Jacobian-radius sham | 217 | 222 | 25 | 29 | 439/1024 |

Full restoration returned 220/512 with all 45 eligible failures recovered.
In the corrupt condition the target's mean absolute score error was 0.05113
versus 0.05805 for the strongest sham. This small score-level difference did
not produce an aggregate balanced-task advantage.

## Paired contrasts

Targeted minus comparator in balanced success, percentage points:

| Split/source condition | Equal rank | Norm | Signed Jacobian + radius |
| --- | ---: | ---: | ---: |
| Short clean | 2.05078125 | 2.1484375 | -0.68359375 |
| Short revised | 2.44140625 | 2.05078125 | -0.48828125 |
| Long clean | 2.24609375 | 0.9765625 | -0.1953125 |
| Long revised | 2.1484375 | 0.87890625 | -0.09765625 |

Long revised per-model targeted-minus-strong-sham effects (seeds in declared
order): +1.171875, -0.78125, -0.390625, -0.390625 percentage points.
All 512 model/world pairs in each stratum passed the strong match gate.
These crossed observations are not independent Bernoulli trials.

Verdict: instrument PASS; donor-specific superiority against the strongest
registered sham is unsupported in this development panel. No promotion.
The target changes output-sensitive directions, which are controlled by
the strong sham. The result does not show a population equivalence or a
causal mechanism for native BPD. The corruption is artificial, and the
restorative vector has privileged access to the original write. The scorer
is separable across entries, with a scalar future output per entry.

## Next discriminator

Prioritize a task with interacting evidence entries and genuinely delayed
multi-output dependencies. Use a native prefix donor valid at the recipient
epoch, a dependency-aware factorization and a complete future-logit Jacobian.
Predeclare corruption headroom, matched-support feasibility and a fresh
development comparison before evaluation. A second-order diagnostic can
measure d^T H d versus s^T H s to account for nonlinear score changes left
after first-order matching. If the output dimension consumes all allowed
recipient degrees of freedom, report the distinct-match feasibility failure
rather than silently reducing the strong sham. Keep frozen intervention work
separate from any training-steering study; this result alone does not justify
promotion into task training. Use the pinned EvalMath strict preflight and
fresh independently assigned model/task populations for any confirmation.
