# ALEPH-06 matched scale frontier

21 fits completed on the public mixed-corpus development panel. The largest completed model had 14,179,584 unique parameters. The run measured two recurrent passes, 24 logical block visits and 32 matched optimizer updates. The paid gate remains on hold. [Registered protocol](SCALE_PROTOCOL.md)

The scale screen retains direct recurrence and global constant damping as calibration arms. Two-pass whole-pass RLT has the same two coefficients as its matched pass-damping control. At three passes the earlier 14M replication favored matched damping on text in all three seeds.

## Promotion outcomes

- Width 80, 1,414,480 unique parameters. The control competence gate failed. The next rung admitted direct recurrence, global constant damping, prelude reread.
- Width 256, 14,179,584 unique parameters. The control competence gate failed. The larger rungs were withheld.
- Width 480, 49,622,880 unique parameters, not promoted after failed competence gate.
- Width 680, 99,437,080 unique parameters, not promoted after failed competence gate.

## Development measurements

The training-byte unigram scored 3.27565 nats and lexical Lean succeeded on 50.0% of goals. Transparent source cues reached 75.0% on repair and 87.5% on counterexamples. The registered promotion rule uses repair chance of 25% or the lexical Lean comparator as its checked threshold.

Each pair of text or verified-macro numbers gives the seed mean at 16 and 32 updates. Verified macro equally weights Lean top-four, joint rule revision, repair and counterexamples. Throughput counts nonpadding positions during optimization. RSS is the largest process sample at an update boundary.

| Width | Arm | Text NLL, 16 / 32 | Checked macro, 16 / 32 | Median positions/s | Peak RSS, MiB |
|---:|---|---:|---:|---:|---:|
| 80 | Direct recurrence | 4.4349 / 3.8382 | 0.1771 / 0.1458 | 385.9 | 377.4 |
| 80 | Global constant damping | 4.3619 / 3.7673 | 0.1823 / 0.1562 | 224.4 | 370.4 |
| 80 | Prelude reread | 4.4405 / 3.8440 | 0.1667 / 0.1771 | 228.1 | 351.0 |
| 80 | Local reread control | 4.4308 / 3.8361 | 0.1875 / 0.1562 | 359.2 | 357.9 |
| 256 | Direct recurrence | 3.5996 / 3.4158 | 0.1562 / 0.1875 | 48.9 | 568.9 |
| 256 | Global constant damping | 3.5940 / 3.4209 | 0.1875 / 0.1771 | 51.2 | 568.7 |
| 256 | Prelude reread | 3.5981 / 3.4044 | 0.1458 / 0.1771 | 42.3 | 549.1 |

All arms and widths replayed their ordered training-position digests. Padding counts and dense projection, output head and full-square attention FLOPs were reconstructed independently at both checkpoints. Within each width, parameter count and logical block applications match across arms.

| Width | Logical block parameters visited per position | Nonpadding positions at 32, seed range | Padded positions at 32, seed range | Training FLOPs, seed range |
|---:|---:|---:|---:|---:|
| 80 | 2,375,040 | 11,471 to 11,523 | 16,310 to 16,440 | 2.552e+11 to 2.574e+11 |
| 256 | 24,293,376 | 11,471 to 11,523 | 16,310 to 16,440 | 2.450e+12 to 2.470e+12 |

## Seed-level scale slopes

The entries are changes in mean held text NLL divided by the natural-log change in unique parameters, at the same 32-update token prefix and physical recurrence. Negative slopes mean lower text loss at the larger width. A wider model spends more FLOPs on that prefix. In a law $L(P)=L_\infty+A P^{-b}$, each admissible choice of $L_\infty$ gives a different exponent through two points. These short-prefix secants cannot identify a 1B winner or an optimum at fixed compute.

| Widths | Arm | Seed 137 | Seed 163 | Seed 191 | Range |
|---|---|---:|---:|---:|---:|
| 80 → 256 | Direct recurrence | -0.1940 | -0.1734 | -0.1824 | [-0.1940, -0.1734] |
| 80 → 256 | Global constant damping | -0.1437 | -0.1409 | -0.1663 | [-0.1663, -0.1409] |
| 80 → 256 | Prelude reread | -0.1946 | -0.1844 | -0.1930 | [-0.1946, -0.1844] |

## Strict-1B work equivalent

The largest direct proxy used for this comparison had width 256. Dividing full-context strict-1B work by its three measured one-core arithmetic rates gives the following median and range of equivalent years. The strict model's ~15.99 GB optimizer inventory exceeds the 3 GiB experimental cap. The full-scale formula assumes ideal triangular attention whereas the proxy counts dense scores over padded short batches. [Credit stages and memory assumptions](STRICT_1B_BUDGET.md)

| Nonpadding positions | Two-pass one-core equivalent years | Three-pass one-core equivalent years |
|---:|---:|---:|
| 100,000,000 | 3.18 [2.89, 3.24] | 4.44 [4.03, 4.53] |
| 1,000,000,000 | 31.76 [28.85, 32.39] | 44.38 [40.32, 45.25] |
| 10,000,000,000 | 317.60 [288.53, 323.86] | 443.77 [403.16, 452.52] |
| 20,000,000,000 | 635.21 [577.07, 647.73] | 887.54 [806.31, 905.04] |

The source corpus is development evidence. Each worker used one allowed logical CPU while other experiments were active on the host. Shared physical resources can affect timing. Evaluation uses eight roots per source group and shares them across all arms. The raw seed metrics, gradients, archive and stream hashes, throughput and exact promotion checks are in the local machine-readable release.

## Reproduction

```text
python training/aleph_next/bounded.py --wall 21600 --core 4 scale_lab.py --out ../aleph-next/scale --pack runs/placement/mixed
python training/aleph_next/scale_release.py --out ../aleph-next/scale --pack runs/placement/mixed --figure ../aleph-next/scale/scale-frontier.svg --markdown ../aleph-next/scale/SCALE_RESULTS.md
python -m pytest -q training/aleph_next/test_scale_release.py
```
