# MeTTa Research Bench

The [paired Looped Transformer study](looped/README.md) tests two native
MeTTa curation policies with the same model architecture and training budget.

MeTTa Research Bench grades bounded evidence-selection repairs. The
world, initial selection, and action grammar are public. Each submission
is a JSON array of up to ten decisions, including the terminal STOP. The scorer regenerates the public source,
checks its hash, computes the exact target, and independently replays the
actions. A reward of one requires the canonical optimal selection, an
explicit STOP, and zero unsafe additions. Other signals remain metrics.

The taskset targets Prime Intellect Verifiers **0.4.1** and uses its public
`verifiers.v1` API. The CPU harness submits actual action sequences to
`Task.score` and saves the resulting Prime Verifiers traces. No model API
or hosted runtime is needed for the paired CPU experiment.
[Prime Intellect Verifiers 0.4.1](https://github.com/PrimeIntellect-ai/verifiers/releases/tag/v0.4.1)

## Run locally

Use Python 3.11–3.14 on Linux or WSL and run from the DataGoblin checkout.
The pinned Verifiers runtime imports the POSIX file-locking module.

```sh
python -m pip install -r metta_research_bench/requirements.txt
PYTHONPATH="$PWD" vf-validate metta-research-bench --runtime.type subprocess -n 5
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 \
  python -m metta_research_bench --out runs/metta-research-bench
```

The Verifiers taskset ID is `metta-research-bench`. Its configuration accepts
`development`, `frozen`, or `conflict` and a per-family root count.
The paired runner saves its development report before loading either
frozen split. It supports `--development-only` while preparing the study.

## Paired training comparison

The chronological control selects the first two available CONTINUE
frames of each exact repair trajectory. The MeTTa condition queries the
verified trajectory in native Hyperon for removals and additions, then
selects an early removal and a late addition when available. Both retain
the same root identities, per-root edit quota, public input encoding,
and supervised objective. All arms train the STOP head on the full set
of checked states. The native query produces action
phase membership used for data selection. The exact Python oracle
independently supplies all target edits and STOP labels.

The TRM-style model is a four-pass tied recurrent action policy. The
LDT-style proxy is a separately trained feedforward action scorer that
projects membership proposals through a typed lattice membrane. The
models have different parameter counts and use the same reference action
matmul budget. The STOP head uses the same number of optimizer updates in
all arms. The LDT policy uses the existing public typed-checking fallback
at inference. Public lookahead is disabled for both architectures.

The report gives seed-paired wins and losses for each architecture, along
with a root-clustered bootstrap interval for the reward difference and
for the difference between the two architecture effects. The curation
receipts identify the chosen TRAIN steps without adding new source
presentations. The separate frozen reports bind their Prime traces and
independent replay receipts by hash.

## Recorded benchmark

The clean implementation at 51fa346 trained two seeds on 100 shared
non-conflict roots. Chronological selection retained 168 removal and
27 addition targets. Native MeTTa selection retained 98 removal and
97 addition targets. Both used 481 checked STOP/CONTINUE states with
100 terminal targets. The seed-paired STOP checkpoints matched exactly.
The TRM-style fit used 591 parameters and 96 action updates. The LDT-style
fit used 447 parameters and 244 action updates, spending approximately
the same action matmul budget. Each fit used 96 STOP updates.

| Verifiers split | Architecture | Chronological | MeTTa-curated | Paired reward change |
| --- | --- | ---: | ---: | ---: |
| Development | TRM-style | 9/80 | 5/80 | −5.0 pp |
| Development | LDT-style proxy | 23/80 | 20/80 | −3.75 pp |
| Fresh curation | TRM-style | 7/160 | 24/160 | +10.63 pp |
| Fresh curation | LDT-style proxy | 18/160 | 43/160 | +15.63 pp |
| Fresh conflict | TRM-style | 0/48 | 7/48 | +14.58 pp |
| Fresh conflict | LDT-style proxy | 0/48 | 15/48 | +31.25 pp |

The paired curation effect on fresh conflict was 16.67 percentage points
larger for the LDT-style policy. Its descriptive root-clustered bootstrap
95% interval ran from +4.17 to +31.25 percentage points. On fresh curation
the architecture interaction was +5.0 points, with an interval from
−1.88 to +12.5 points. These estimates describe the paired synthetic roots
and the two fitted seeds.

The intervention increased premature STOP on both frozen splits, even
as verified safe completion rose. On fresh curation, TRM unsafe additions
fell from 160 to 62. The LDT-style controller recorded zero unsafe
additions throughout. The frozen results concern these compact CPU
models and their stated controllers.

Prime's CLI passed gold and noop validation on all 40 development tasks.
The paired CPU run produced 1,152 Prime Verifiers traces and independently
checked receipts. Peak process memory was 709 MiB and wall time was
59.58 seconds in Ubuntu WSL. All conflict worlds were excluded from
training and development, and frozen conflict sources were disjoint
from the prior MeTTa-11 through MeTTa-13 evaluations.
