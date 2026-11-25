# COL-DG-01: verified arithmetic trajectories

This source supplies bounded synthetic supervision for eventual-cycle reasoning.
It connects MathCan's arithmetic-dynamics program with EvalMath's readout-ceiling
benchmark. Each accepted example carries a complete finite trajectory and a
fresh receipt from an independent full-map replay.

## What enters the curriculum

The input specifies positive `d = 1 (mod 4)`, `gcd(d,6)=1`, and an odd seed.
The map sends an odd `n = 1 (mod 4)` to `3n+d`, and an odd `n = 3 (mod 4)`
to `n+d`; even states are halved. The target names the minimum odd integer
on the eventual cycle. The certificate records odd states, branch choices,
exact counts of halvings and the first repeated state.

The checker replays every original branch and halving, verifies that the first
repeat closes the claimed cycle, recomputes all cycle statistics, and checks
`2^A = 3^u (mod d/gcd(n,d))`. The generator uses bit arithmetic to accelerate
halving; the checker uses explicit repeated division and shares no transition
routine with it. The returned evidence scope is `finite_unaccelerated_replay`.

Supervision can cover the next odd state, complete trajectory arithmetic,
cycle closure, eventual basin, and cycle branch counts. General theorems,
parameter-wide basin censuses and ordinary Collatz convergence remain outside
this certificate's scope. A valid finite trace provides a proof witness for
its stated seed. It does not establish a learned student's ability.

## Admission and split custody

The proposer returns a candidate certificate. `verify_certificate` reconstructs
the evidence. `audit_records` replays that step again before checking inputs,
targets, immutable IDs and partition custody. A supplied `verified` label has
no authority. Altered branches, truncation, false closure, forged targets,
boolean values in integer fields and duplicate task identities are rejected.

The parameter split uses SHA-256 with the shared literal salt
`COL-parameter-split-v1`. Every seed at the same `d` has one split. The default
generator emits train/dev and skips holdout parameters. It chooses unit seeds,
so normalizing a trace by `gcd(n,d)` leaves its parameter intact. This prevents
scaled copies of the same trajectory from crossing d-based splits. The general
auditor also rejects identical normalized trajectories assigned to different
splits, including externally supplied nonunit examples.

These deterministic partitions are public. Prior mathematical audits may have
revealed some labels. Future model confirmation requires fresh roots under
separate custody. Repeated steps, seeds and rotations at one parameter never
count as independent evaluation roots.

## Run locally

```bash
PYTHONPATH=src python -m datagoblin.training_gym.collatz_corpus generate \
  --max-d 1021 --per-parameter 8 --out runs/collatz-curriculum
PYTHONPATH=src python -m datagoblin.training_gym.collatz_corpus audit \
  --out runs/collatz-curriculum
PYTHONPATH=src python -m pytest tests/unit/test_collatz_corpus.py
```

The hard limits are `max_d <= 4093` and at most 32 unit seeds per parameter.
The per-certificate checker bounds both edge count and integer size. Generated
JSONL and run manifests stay under `runs/`; this PR contains the verifier and
reproduction instructions. No weights, network requests or provider calls occur.
An identical rerun leaves the export intact. Changed content or source hashes
require a new output directory, preserving evaluated records.

## Curation use

Admit these records into a tagged synthetic-math lane after replay. Keep their
root IDs through tokenization and sampling. The existing presentation caps and
mixture controls still apply. Exports count verified arithmetic records and
microsteps; token quota remains zero until the actual tokenizer and mixture
admission measure it. A student comparison should report parameter-macro scores
alongside case-micro scores and keep certificate access in the evaluator.

The companion EvalMath package includes constant, gcd and full-modular-orbit
readout ceilings. If a student's predictor receives only a modular orbit,
several eventual cycles can share its input. This fixture supports discovering
that ambiguity with explicit witnesses and verifying a proposed resolution.

MathCan reference: `patrickdugan/MathCan/research/collatz_resonance`.
EvalMath reference: `MoralityLabAI/EvalMath/paper/collatz_basin_readout.md`.

## Replay completed on 25 November 2026

Running the generator with `--max-d 4093 --per-parameter 8` produced 4,235
accepted records over 531 parameter roots. The full-map checker verified
250,121 odd edges, accounting for 1,000,827 original map steps. Train contains
3,153 records and dev contains 1,082. The 152 parameters assigned to holdout
were skipped by this export.

The audit found zero repeated IDs, zero parameters spanning splits and zero
scaled copies spanning splits. Its canonical dataset SHA-256 is
`ef91328604502c0d9c1e53fce82038f6204de0fa53145c515227ebc2d95dbd0c`.
The runtime manifest records generator and verifier source hashes and the
auditor checks them against the installed code. Generated runs are excluded
from the repository according to its contributor guide.

The 25 focused tests include all 2,860 odd-core seed cases for admissible
`d <= 257`, twelve kinds of malformed certificate, six altered admission
records, scaled-copy leakage, explicit resource limits and deterministic
round trips. These checks establish finite arithmetic integrity. Model
training and learned performance remain unmeasured.

The full repository suite passed with 589 tests and 32 optional-dependency
skips. Repository-wide `ruff check .` also passed. The focused tests were
rerun after adding the immutable-export guard.
