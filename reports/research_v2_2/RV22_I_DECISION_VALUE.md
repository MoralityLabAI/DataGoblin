# RV22-I: synthetic research decision-value pilot

Status: exploratory local NumPy execution (2.3.5), not native Distill or a live
actor campaign. No provider calls, GPU leases, or paid compute.

Native RV22-H at 979fc94 has two green hosted workflows, 786 passed / 15 skipped
and clean lint. All five native router fits classified all 896 supported vectors,
with zero selective-RLT corrections at threshold 0.20.

## Method

Distill research v0.4 adaptive-evidence format is defined in
[research_cases.py](https://github.com/MoralityLabAI/Compactification-Theory/blob/research/distill-gym-v01-20261127/Distill/research_cases.py)
and [research_oracle.py](https://github.com/MoralityLabAI/Compactification-Theory/blob/research/distill-gym-v01-20261127/Distill/research_oracle.py).
This pilot independently generates compatible worlds rather than consuming the
native export. Each world has 12 weighted hypotheses, six binary tests, path
budget 7, rewards +12/-16, and defer reward zero. Ordinary worlds have randomly
assigned balanced classes. The structural cohort contains three planted parity
bits plus distractor tests and nonuniform prior masses.

128 train, 32 calibration, 96 IID and 96 parity worlds; exact serialized
cross-split world disjointness (not a complete isomorphism audit). Train on 7,972
legal test actions at the root and one-test descendants. Target is
(exact optimal continuation after test minus best current stop utility)/12.
The actor sees 14 public scalar features, including one-step utility.
No oracle continuation in the learned rollout: independently replay each
complete chosen policy tree over every weighted hypothesis and subtract path
cost. Check exact DP against the oracle tree replay. Calibrate the stopping
threshold on the 32 calibration worlds from {0,.025,.05,.1,.2}. Fixed model seeds
21/22/23. Stop, cost-weighted split balance, greedy one-step, receding two-step
and exact DP are controls. Split balance per cost is a heuristic, not actual
Shannon entropy.

## Local result

| Arm | IID net utility | IID regret | Parity net utility | Parity regret |
|---|---:|---:|---:|---:|
| Stop | 0.2874 | 6.7917 | 0.1970 | 7.2336 |
| Balance per cost | 4.8850 | 2.1941 | 5.6143 | 1.8162 |
| One-step | 5.4269 | 1.6522 | 4.4419 | 2.9887 |
| FF16 | 6.3491 | 0.7301 | 6.7026 | 0.7280 |
| TRM16 K3 | 6.3954 | 0.6838 | 6.7285 | 0.7021 |
| Wide FF48 | 6.3319 | 0.7472 | 6.7184 | 0.7121 |
| Two-step | 6.7470 | 0.3322 | 7.0559 | 0.3746 |
| Exact DP | 7.0792 | 0 | 7.4306 | 0 |

10,000 root-paired percentile bootstrap replicates after averaging the three
fixed training seeds within each root. No uncertainty over new training seeds
or multiplicity adjustment. Both test cohorts are now spent.

| TRM minus control | IID delta [95% CI] | Parity delta [95% CI] |
|---|---|---|
| One-step | +0.9684 [0.6527,1.2994] | +2.2866 [1.8202,2.7400] |
| Two-step | -0.3516 [-0.5318,-0.1810] | -0.3274 [-0.5463,-0.1263] |
| FF16 | +0.0463 [0.0065,0.0943] | +0.0259 [-0.0130,0.0649] |
| Wide FF48 | +0.0634 [0.0038,0.1406] | +0.0101 [-0.0299,0.0471] |

TRM mean credits: 3.7424 IID / 4.1319 parity; two-step: 3.8671 / 4.2606.
TRM calls per full policy tree: 26.2326 / 29.1771. Two-step DP cache misses:
172.2083 / 188.5000. These are different units, not evidence of a wall-time
or compute win. Active head MACs per candidate are 240, 752 and 720 respectively;
FF1's allocated recurrent weights are inactive. Head MACs exclude feature
construction, tree replay, memory and activation costs. Width48 is near-matched
to K3, not exactly compute-matched. No promotion or checkpoint ranking.

## Reproduction

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/research_v2_2/decision_value.py --out /tmp/rv22i.json

Local receipt SHA-256 70923cf706928f9a135fd58712fd9bd0f4ee4e36d1973791b546aa482d29f524.
Local measured source SHA-256 908f115d96e4882223e5195fd071b508d009c63bef096d390b36565c1d64e1ce.
Repository source adds three cache-lint suppression comments only; hosted
reproduction remains to be verified. This result is independently generated
synthetic supervision, not admission of Distill reference-only teacher records.

## Follow-up gate

Train on native Distill public train worlds with exact referee values only after
explicit admission; validate upstream manifest and exact oracle parity first.
Expose candidate-test interaction features (conditional remaining-test
discriminability) and compare with an exact two-step residual hybrid. Predeclare
new seeds, stronger structural shifts, actual feature and neural/search compute
costs, and a fresh confirmation panel. Require noninferiority against two-step
at matched CPU work before a learned controller is promoted.
