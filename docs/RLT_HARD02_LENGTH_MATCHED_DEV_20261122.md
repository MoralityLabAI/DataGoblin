# RLT-HARD-02: script-length confound, independent re-audit and matched-menu follow-up

Date: 2026-11-22. Development research; [issue #24](https://github.com/MoralityLabAI/DataGoblin/issues/24).

## Frozen native evidence: source and checksum custody

The original native Mathlib experiment executed in
[Actions #44983534801](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44983534801)
at source head `4ac40827c06053ab6b89ce02feecbf95bbe4d9d2`.
It produced 32 source-root episodes, 24 retained certified Lean proof sources,
16 independent learned-policy attempt replays, and 397 recorded Lean compiler calls.
The source was pinned to Mathlib commit
`809c3fb3b5c8f5d7dace56e200b426187516535a`
and Lean 4.12.0.

An independently implemented auditor re-downloaded the historical source
artifact from GitHub Actions (rather than fitting on or resampling any held-out
examples) in
[Actions #44989615504](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44989615504).
All **59** retained file SHA-256 entries matched, and **277** candidate
receipts passed provenance/type/allowed-kernel-axiom checks.
The audit confirmed **zero** learned-policy false stops against fixed-full
search on 16 development roots.

| Native development arm | Ordinary certified / attempts | OOD certified / attempts |
| --- | ---: | ---: |
| Learned ranker and CAL-frozen stopping | 6/8, 8 | 6/8, 8 |
| Fixed full menu | 6/8, 31 | 6/8, 28 |
| Deterministic random full menu | 6/8, 25 | 6/8, 23 |
| Trivial-first full menu | 6/8, 36 | 6/8, 38 |
| **Post-hoc longest-script-first control** | **6/8, 8** | **6/8, 8** |

In every original root with a registered successful script, that script was
the **longest candidate**. Its smallest advantage over the next-longest
candidate was 16 UTF-8 bytes. Thus the learned policy has **no measured value
above the cheap length-only heuristic** in this original design. The compiler
attempt and wall-time savings are authentic observations about those menus
but are scientifically explained by a constructed candidate-length cue.

The length-first rule was identified **after** observing the original
evaluation. It is a diagnostic confound test, never a preregistered success
metric or independent confirmation.

## RLT-HARD-02B: fresh matched-menu development protocol

New source seed: `2026112203`; Mathlib/Lean pins are unchanged.
This source profile is `length_matched_near_miss_v2`, selected explicitly
using `--matched-menus`. The original `trivial_controls_v1`
generation remains available for archive reproduction.

All 32 new source roots retain four source-module-disjoint splits
(Set, Finset, List, Nat divisibility) with eight derived theorems per split.
The theorem generator changes the numeric salt, and the Nat divisibility
cohort adds an independent salted divisibility premise. Source statements,
identities, and source-file hashes change across successive development seeds.

Each five-action menu contains plausible-looking **near-miss proofs**.
An answerable root has one teacher-constructed witness and four mutations.
The two preregistered exhausted-menu roots per split have five mutations.
The generator pads inert trailing whitespace so **all five actions have
identical UTF-8 byte length** within each root. The printed target theorem
and public task carry no witness flag, private correct-candidate ID, or
first-success label. Only independent Lean kernel compilation determines
which scripts certify.

This controls the simple length fingerprint. The menus are still synthetic,
the same family-specific witness source pattern is reused within a split,
and there may be other lexical cues. Candidate effects, ranking, and
continuation must be assessed against the **real compiler** and all new
native receipts retained.

The policy is fitted on newly executed TRAIN candidate receipts only. CAL
selects the continuation threshold using its prespecified utility. Ordinary
and OOD development episodes run the frozen learned, fixed-full,
random-full, and trivial-first policies with actual bounded native calls.
Every learned attempt is independently replayed, with explicit Lean
axiom output and proof sources saved by SHA-256.

**Promotion conditions:** no loss of independent source/compilation integrity;
measured kernel-certified success and real call/wall costs must be inspected,
and failure to beat cheap source-text baselines must be preserved.
The episode family remains Mathlib-dependent *parametric derived theorems*,
not an established difficult Mathlib benchmark or genuine free-form
tactic-state transition sequence. This cohort is **development only**.
Sealed confirmation and the full hard-theorem acceptance in issue #24
remain open.

## Reproduction

Source contracts:

```bash
python -m pytest -q tests/unit/test_rlt_hard02_matched.py
ruff check src/datagoblin/training_gym/rlt_gym/hard02_mathlib.py tests/unit/test_rlt_hard02_matched.py
```

The native [RLT-HARD-02B workflow](../.github/workflows/rlt-hard02-matched.yml)
uses Mathlib v4.12.0, source seed `2026112203`,
`--matched-menus`, eight roots per split and create-only output.
The workflow retains the fresh source roots and kernel-certified proofs
without committing generated data or trained weights to the repository.

## RLT-HARD-02B actual native outcome and fixture failure

[Completed native Actions #44989821104](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44989821104) passed every native Lean / Mathlib / independent audit step. It used source seed `2026112203` and the frozen grounded ranker; CAL selected stopping threshold `0.30`. The archive records **372 actual compiler calls**, **40 independent learned attempt replays**, **39 certified Lean proof source files** and **997.91 cumulative compiler seconds**. The independent audit verified all **74 file hashes** and **218 candidate receipts**, with no learned false STOP relative to full search.

| Arm | Ordinary development | OOD development |
| --- | ---: | ---: |
| Learned / CAL-frozen stop | 8/8 certified, 8 candidate checks | 6/8, **32 checks**, 104.08 candidate seconds |
| Fixed exhaustive order | 8/8, 8 checks | 6/8, **24 checks**, 82.11 seconds |
| Matched random exhaustive order | 8/8, 8 checks | 6/8, 26 checks, 87.38 seconds |
| Trivial-first full order | 8/8, 8 checks | 6/8, 24 checks, 82.02 seconds |

**Failure verdict:** the model has no gain over fixed full search and spends **33.3% more checks on OOD**. The OOD learned candidate time is 21.97 seconds greater, about 26.8% more, with equal 6/8 certified coverage.

The ordinary development cohort violates the intended exhaustion construction. Every one of its eight roots passed on the first candidate, including the two roots whose registered witness was omitted. A mutated `simpa [List.length_append] using congrArg ...` tactic still proves the statement because its alternative congruence produces a simplifier-compatible equality. Thus there are **zero truly exhausted ordinary menus under native Lean**, despite two intentionally teacher-witness-omitted menus. Future records must count *native menu-exhaustion labels* rather than infer them from generator flags.

## Feature capacity audit and next representation

A separate source-only [feature capacity audit #44990729864](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44990729864) reproduced the current source seed and original feature extractor. **All 32 roots have only one distinct candidate feature signature after excluding the menu-position column.** That establishes why the matched-length policy cannot recognize tactical differences. The original source-length shortcut is removed, but the old representation then becomes position-dependent.

The registered follow-up is `docs/RLT_HARD02_SEMANTIC_ABLATION_20261122.md`: fresh source seed `2026112211`, a position-invariant full-script hashing feature bank, and a same-root paired ablation against the old grounded features. This is a development iteration prompted by the preceding failures. Its ordinary negative-control deficiency must be repaired before any hard-theorem promotion; source-level feature distinguishability alone is not evidence of reasoning gain.
