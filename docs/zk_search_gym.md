# ZKL-13: learned and graph-prioritized certificate search

The proposal state lives in SatsHedgeZero at c4a80400cc901f9528e735ac882292f973ea24da.
The independent acceptance checker remains poly_certificate.verify. MathCan's
ZKL-12 kernel is the intended replay gate, not claimed to have replayed these
candidate certificates. Never authorize production use.

The synthetic fixture uses 3 core wires plus 6 nuisance wires defined from
earlier wires; wire IDs, row order and coefficients are randomized. The
training families and test families are disjoint. Training roots 13001–13008
use F7/F11; evaluation roots 13101–13108 and 13201–13208 use F11/BN254 Fr.
The train set has 24 tasks, 32 positive reference-certificate column labels,
9578 other columns, and 1536 balanced sampled SGD updates for a 12-feature
logistic ranker. The support labels are nonunique and are not ground truth
necessity labels. No family label enters scoring.

The initial seven-arm pilot compares degree order, uniform random, support
heuristic, residual-adaptive heuristic, learned ranker, residual-adaptive
learned ranker and shuffled learned scores. Each uses a matched offered-column
budget of 24, 48 or 96. After observing that all seven missed the degree-two
square transport tasks on the first block, we added peel_local as an
exploratory eighth control. It prioritizes the surviving neighborhood after
iteratively peeling rows with an unshared nonzero linear nuisance wire.
This changes only candidate order; the exact replacement matrix is unchanged.
The second block uses new seeds and the frozen control, with the same fixture
topologies. This is development evidence, not a blinded topology holdout
for the new graph heuristic.

Local results (Python proposal and gym sources matched to Git blobs; exact
audit.py and ideal_certificate.py blobs; copied verifier functions from
poly_certificate.py, not its entire blob). The scoped CI uses the actual
pinned full checker files and reruns both blocks:

| 24-column budget | First block (32) | Fresh block (32) |
|---|---:|---:|
| degree | 8 | 8 |
| random | 0 | 0 |
| heuristic | 16 | 16 |
| adaptive heuristic | 16 | 17 |
| learned | 16 | 16 |
| adaptive learned | 16 | 16 |
| shuffled | 0 | 0 |
| peel_local | 24 | 24 |

The full degree-ordered reference certifies 24 per block. All 16 essential
controls have explicit counterexample assignments checked by audit.check;
no arm certified them. For the 48 certified tasks, the graph control uses
435 offered columns in total vs 6603 for full degree order. For all 64
tasks including the essential controls, the 24-column graph run uses
819 columns vs 12763 in the full reference. The 16 degree-two tasks
are all solved by the graph control. The adaptive scans can be costly:
on the first block at 24 columns, static learned scores 13640 candidates
and adaptive learned scores 183868. Each arm still builds the full pool.
These numbers are not matched elapsed CPU times or speedup claims.

Run from a reviewed checkout with the pinned SatsHedgeZero research/zk_lean
directory on PYTHONPATH:

```sh
python -m datagoblin.zk_search_gym --out runs/zkl/search-first.json
python -m datagoblin.zk_search_gym --test-start 13201 --out runs/zkl/search-fresh.json
```

Run under the existing CPU-only RAM watchdog for larger campaigns. The
research implication is to learn search after deterministic graph pruning,
on connected nonlinear decoys where the graph heuristic is insufficient.
This pilot trains a logistic ranker only. No BPD, TRM or LDT model training
or inference has been run.
