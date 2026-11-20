# Owned mathematics intake: first verified source-level observations

**Date:** 20 November 2026. **Scope:** the default-branch commit SHAs pinned in training/corpus_intake/audit_owned_math.py. This note is an inventory and source scan, not a retokenized Aleph corpus and not a kernel replay.

## Cross-repository source rather than Lean alone

The four MoralityLabAI repositories (Compactification-Theory, EvalMath, LoopedTransformers, BPD) contain:

| Source type | Tracked bytes | Bytes after exact-file SHA deduplication |
|---|---:|---:|
| Lean | 1,045,444 | 1,035,586 |
| Python | 26,948,239 | 25,772,227 |
| Markdown | 12,402,830 | 12,005,973 |
| **Subtotal** | **40,396,513** | **38,813,786** |
| TeX, counted separately | 570,001 | 476,324 |

There are 146 exact duplicate **occurrences** across these four source categories, including substantial source reused in BPD and LoopedTransformers. This deduplication is only by identical Git blobs; partial copies, theorem ancestry, reproduced abstracts and repeated prompts remain to be resolved.

Prior Lean-only estimates around 560,000 tokens therefore cannot be used as a total for original mathematics, experimental reasoning and programming. This note makes no claim about accepted unique training tokens: the Aleph tokenizer, proof verification, source licenses, source-to-target projection and family deduplication are still pending. The proposed o200k_base CI scanner reports its surrogate tokenizer explicitly.

## Semaev: measured core Lean declarations

A direct 20 November source scan, using GitHub's contents API at commit 3c630e179ae837dad839949dcb03cf4fdb41815e, inspected the 30 ordinary formal-development Lean files totaling 463,656 Git source bytes. InstanceD, ChainStandardD, MinDegreeM, RefuteD and obvious check/axiom entrypoint names were excluded from this *core* subset. The scan matched top-level declaration introductions after removing source comments.

| Core source declaration kind | Count |
|---|---:|
| theorem + lemma | **553** |
| def | **208** |
| abbrev, structure, inductive, instance, etc. | **35** |
| **All matched declarations** | **796** |
| source sorry tokens after simple comment removal | **0** |

These are source-scan counts, subject to Lean parsing and elaboration details. They do not claim 553 independent mathematical theorems or 553 kernel-verified proofs. A theorem may merely repackage a nearby lemma, depend on an imported or assumed fact, or recur across generated variants. No proof state or proof-step count has yet been extracted.

Semaev has 124 Lean files and 77,074,795 tracked Lean bytes overall. Seven giant generated certificate files account for 74,515,825 of those bytes. Those files belong in a verifier/evidence lane rather than being treated as millions of independent reasoning tokens.

## Training interpretation

The immediately promising Semaev roots include the generic-group lower bound and its explicit scope, the conditional S-unit ramp argument, the free-extension induction under a finite base obligation, rank versus acquired online relations, and the Strategy IV cost/finite-size corrections. Their research value comes from proposal, objection, countermodel or exact check, and a revised claim, together with the source of the verdict.

The six tasks in training/gyms/semaev_redteam/tasks_dev.json are **development only**. Their answer obligations are public and must not be reported as a sealed benchmark, independent proof checks or evidence of cryptographic key recovery.

## Execution status

The PR supplies a reproducible pinned source scanner and a 25-minute bounded GitHub Actions job that would produce exact o200k_base counts and source-level declaration rows for all five repositories. On 20 November the repository's existing CI and the new intake job both failed **before any job steps executed**, including on the pre-change baseline commit. No completed CI token report or Lean kernel verification is asserted here. The next executable stage is the workflow artifact from a functioning runner, followed by an Aleph-tokenizer recount and independent proof/split verification.
