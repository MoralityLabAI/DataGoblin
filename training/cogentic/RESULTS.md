# COG-02 development run, 26 November 2026

The first run generated 256 finite Horn proof-search roots and replayed 2,965
attempts through the environment checker. Lean 4.35.0 compiled all 256 completed
teacher derivations. Each theorem's axiom report was empty, and a deliberately
wrong proof term failed compilation.

| Split | Roots | Attempted rules | Newly proved lemmas |
| --- | ---: | ---: | ---: |
| Train | 64 | 642 | 514 |
| Calibration | 64 | 618 | 490 |
| Ordinary test | 64 | 615 | 487 |
| Deeper OOD | 64 | 1,090 | 962 |
| Total | 256 | 2,965 | 2,453 |

The collection policy deliberately inserted one failed rule and one duplicate
per root. This yields 256 rejections and 256 duplicate calls for testing the
export and cost rules. Those failures were retained in the 642 training outcome
records. The SFT stream contains 514 newly accepted training lemmas; every row
is bound to a replayed step. Calibration and evaluation roots were excluded
from all exported supervision streams.

The ranking dataset covers 2,453 distinct reached states. Evaluating all
candidate rules at those states required 66,893 local checker calls. This cost
is recorded as data-collection work and should accompany the subsequent model
training cost. Online policy evaluation charges its own attempted actions.

The largest generated world has 52 atoms and 50 rules. Train, calibration and
ordinary-test depths range from two to five; the deeper split ranges from six
to nine. Root hashes are unique across the four splits. Their shared generator
families and public seed make this a development experiment.

## Verification

The focused test suite passed 17 tests. Adversarial cases recomputed the hash
chain after forging a success, injecting a state fact, changing provenance or
erasing a verifier cost. Replay rejected every altered trace. The tests also
checked negative-loss boundaries, split custody, duplicate accounting and
public-input projection.

The full DataGoblin suite completed with 605 passed and 33 skipped tests before
the final provenance test was added; that additional test passed in the final
focused run. Repository-wide `ruff check .` passed after the implementation was
complete.

Native Lean receipt:

| Field | Value |
| --- | --- |
| Toolchain | Lean 4.35.0 |
| Compiler commit | `eb6cafb7290e8bd5f2937522d8d804f58f02bafc` |
| Binary SHA-256 | `bf8d54e4714cc4b03d3f6bb34c83b7202b87e49c8bfcbff6895c085bb90ceb38` |
| Proof source SHA-256 | `f3c3cd37bdb8e7848d9bbe928cee18f394f8ec05327058e40cdced2a765b56af` |
| Proofs compiled | 256 / 256 |
| Theorems with empty axiom lists | 256 / 256 |
| Wrong-term sentinel | Rejected |

## Replay fingerprints

Run the command in the README with `--per-split 64 --seed 20261126`. The emitted
manifest records every file hash, the exact generator source hash and the native
compiler receipt. These data hashes identify the first corpus:

| Artifact | SHA-256 |
| --- | --- |
| Tasks | `aef151e3f7de099e9b5d7b51c92ad94aa3bab20a85b4f73b8f4fb5ac8ce7905a` |
| Teacher traces | `b250a58f31d37709242c7da1eb1d68aa01e75b2cc190ae54d9da05822cf331af` |
| Ranking rows | `4df7ae5b17d7a37fae306168d95b05de687a0ee67fa7c3a2832581069efdad6c` |
| Train SFT | `53c9b21cf522477c426ef39c6fe35cc7188b6516890672965655afdfa080db39` |
| Train outcomes | `bf5e51b0a6fd5fcb270b347011747e0331a56e972d9c231ea0c7bed1d7e7a71f` |

The conclusions concern corpus custody and proof replay. COG-03 and COG-04
measure learned routing with separate executed rollouts. The Lean proofs here
establish conditional propositional consequences inside the generated worlds.
The Cogentic paper collection remains outside this run's proof scope.
