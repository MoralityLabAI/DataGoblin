# ALEPH-STATE-12 | proof-producing normalization and two-frontier search

2026-11-25. Free CPU diagnostic following STATE-11. Direct Aleph A remains the reference. No native model training, no accelerator credits, no authorization for either 10B slot. No Lean binary on PATH; kernel replay remains outstanding. Every accepted certificate was checked by a restricted rewrite verifier and every intermediate expression by a separate denotational evaluator.

## Design

Fresh public seed 2026112518. Reconstruct all 480 training and 64 held STATE-11 equation pairs and exclude those exact pairs in either orientation. Generate 48 Nat and 48 List mixed rewrite trajectories; evaluate each equation in both directions, plus one negative mutation (Nat target +1, List target appended with an independent variable). Total 288 unique ordered obligations, 192 true and 96 false. The forward and reverse versions are paired and share underlying mathematical structure. The grammar, symbols and subterms can overlap previous stages. No prospective sealed registration or actual held mathlib theorem families.

The original goal-distance search uses STATE-11's 19 oriented rewrite schemas; enhanced search adds five List schemas (reverse/map of nil and commuting reverse with map, with two map symbols). All search arms use a 64 expanded-expression limit and successor size <=55. The two-frontier arm advances the same forward rules from both endpoints in alternating turns until the frontiers meet; it is a common-descendant search, not a complete reverse-edge enumeration. Reverse steps in the resulting proof are checked by applying the forward rule to the explicitly provided predecessor; this avoids guessing erased metavariables.

The normalizer expands Nat multiplication across sums, eliminates zeros/units, right-associates and sorts multiplicative and additive spines using adjacent swaps certified by associativity and commutativity. List normalization distributes maps and reverse across append, removes empty blocks, cancels double reversal and commutes reverse inward through maps. It produces an explicit path from each endpoint to the same syntactic representative, with the second path reversed. Caps: 4,096 steps per endpoint and 4,096 AST nodes. The initial hybrid caps normalization at 16 steps per endpoint, then runs two-frontier search at 64 expansions if the cap was reached. If distinct normal forms are produced, it declines to emit a proof. Every successful path is replayed against the declared rule and checked against polynomial/list-block denotation at each step. These are certificates in the synthetic grammar, not kernel-checked Lean proofs.

## Results

|Policy|Solved /192|Forward /96|Reverse /96|False accepts /96|Search expansions|Proof steps total|Max proof steps|
|---|---:|---:|---:|---:|---:|---:|---:|
|Original|93|93|0|0|7498|357|6|
|Enhanced|93|93|0|0|9013|357|6|
|Two-frontier|188|94|94|0|7254|768|7|
|Full normalization|192|96|96|0|0|20302|2244|
|Bounded normalization then search|190|95|95|0|2999|1708|26|
|Semantic gate, search then normalization (post hoc)|192|96|96|0|2251|928|61|

Search expansions include failed searches and negatives. Proof steps count successful proofs only; compare lengths directly only between policies solving the same panel. Full normalization executed 31,373 local rewrites across all obligations, inspecting 901,820 nodes and attempting 8,383,319 schema matches. Its success is accompanied by severe proof growth. The bounded-normalization hybrid used 3,908 normalization rewrites, 194,749 match tests and 2,999 search expansions; two true obligations remained unsolved.

## Post-hoc portfolio

On the same panel (not an independent confirmation), compute the semantic normal forms using the separate STATE-11 polynomial and List-word evaluator. Abstain on unequal normal forms. For equal forms, run two-frontier search first (64 expansions) and use full explicit normalization only on failures. The semantic gate discards 96 negative mutations; search alone proves 188 and full normalization proves the remaining four. All 192 accepted proof paths pass replay, intermediate denotational checking and round-trip serialization of compact certificates. In the tested grammar the gate relies on known semantics; it is not a general decision procedure for Lean or arbitrary opaque definitions.

Certificate steps fall from 20,302 to 928 (21.877x) at identical positive coverage. Median proof length falls from 15 to 4; maximum from 2,244 to 61. Search expansions fall from 7,254 to 2,251 relative to ungated two-frontier search, partly because the gate avoids searching negatives. The portfolio also performs 160 normalization rewrites. Do not compare normalization rewrite counts directly with expanded search expressions or claim these are Lean kernel calls.

Measured CPU process seconds in the initial run: original 2.4207, enhanced 1.6671, two-frontier 1.3838, normal 12.5955, hybrid 1.0533. Timings include certificate checking and, for normal/hybrid fallback, Lean source rendering. Cache/order effects remain; these are not controlled runtime comparisons. Initial recorded total 21.0090 CPU s excluding final receipt serialization; portfolio 1.0546 s for search/check/compact serialization, .0214 s for Lean rendering. Total recorded CPU 22.085 s excluding other setup and file I/O. Zero paid credits.

## Mathematical note

Let an oriented rewrite system R consist only of sound equations and contextual congruence. If finite derivations x ->* c and y ->* c are witnessed, then the concatenation of the first path with the reverse of the second proves x=y. Neither confluence nor termination of R is required for soundness of this proof. Completeness would need additional hypotheses and is not implied by the 64-expansion search.

A deterministic semantic normal form N can be used as a gate when its meaning is independently established. Equal forms give a semantic reason to seek a certificate; a positive training label still requires the target proof checker. For a general grammar, unequal computed forms need not imply non-equivalence unless N is complete for that equivalence. In this experiment the negatives have independently constructed distinguishing denotations; out-of-grammar use must abstain or escalate.

The portfolio minimizes observed proof effort by placing search before expensive canonicalization. A next experiment should learn a cost-sensitive choice among direct normalization, two-frontier search, specialized tactics and abstention on fresh operator/theorem families. Charge actual checker time, proof length, acquisition and model inference, and compare a static rule-based portfolio before any learned promotion.

## Reproduction

From training/aleph_next/state01:
    python state12.py --out /tmp/state12
    python -c "from pathlib import Path; from state12_portfolio import run; run(Path('/tmp/state12'))"

Both files use state11.py and state11_normal_forms.py. The first run deliberately emits a large full proof receipt (approximately 224 MB) and generated Lean file (approximately 30 MB). The second run emits a compact numerical receipt and a gzip-compressed certificate archive (192 certificates, 928 local proof steps, approximately 7.3 KiB), plus 192 generated Lean declarations. The compact certificate stores rule/position/direction, and the needed replacement witness for reversed steps; each certificate was round-trip replayed. The decoder still depends on the shared rule definitions. Hashes: panel eaaeb9e75887d826134c77b7665933bcaec2f4fc090ed3b38e3aff568798c09e; state12.py SHA256 629d318ca3330c9ed788cce593cd618e580c4aff29617fd8a192b71679534a11; state12_portfolio.py SHA256 33f35e141a7328490d72f14c9cb9e9467bc6e41d2f3ebf8dbcf70f915d61b8ee. Committed Git blobs 80189a42d5621dda3c842a708c7933f814f1ac48 and 61a735d31f634964eb783b97c0ae223a044bc90c match the executed files byte-for-byte. Portfolio Lean source SHA256 282e09f6abc02cc0d0bc2fd7e5f60c2f44e87b17e10e23bb8bb522ac1c8302; compact certificates SHA256 846eaf7e41cceeec54b005bfbf3ffbe3923f0831db952e79a40b4f1b5d143e0f; compact receipt SHA256 9581d3cdbbb2a7d1d61bc4c8cb6de74b92ffce22b0b7ab2444d290c71a3e4bdb. Timing-dependent receipt hashes change on rerun.

Generated Lean declarations contain explicit calc steps, congrArg, symmetry and named lemmas, with axiom audit commands. Their elaboration and lemma names have not been checked by Lean. No kernel-certified labels are admitted to the corpus by this experiment. Historical Aleph checkpoints are still unavailable. Retain A and preserve paid slots.
