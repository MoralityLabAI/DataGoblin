# Semaev core: kernel verification gate

Status: PREPARED; execution has not been recorded in DataGoblin. Source ref: patrickdugan/semaev-index-calculus at 3c630e179ae837dad839949dcb03cf4fdb41815e.

## Reproducible environment

The Semaev source pins Lean v4.34.1 in lean/lean-toolchain and Mathlib at d13f23b723b8a846827a245b89c10fc7d3f11612 in lean/lakefile.toml. Build in a native Linux/WSL filesystem with an installed compatible elan/lake environment. Mathlib cache installation can consume substantial storage and network bandwidth; keep that resource approval separate from source-only DataGoblin intake.

Read-only checkout of that exact commit, then enter its lean/ directory. The initial scope is four high-value core modules:

- SemaevIC.GenericDLog
- SemaevIC.RampGap
- SemaevIC.FreeExtension
- SemaevIC.LocalCertificate

First resolve and cache the pinned Mathlib revision, then use lake build with one module target at a time, e.g.:

    lake exe cache get
    lake build SemaevIC.GenericDLog
    lake build SemaevIC.RampGap
    lake build SemaevIC.FreeExtension
    lake build SemaevIC.LocalCertificate

The complete library build and seven megabyte-scale generated instance arrays are separate, later jobs.

## Proof soundness contract

For each module record repository commit, toolchain, Mathlib SHA, command, exit code, wall-time, max RSS, stdout/stderr SHA-256, and all declarations that compiled. Check the actual axiom dependencies with the existing lean/axioms_check.lean source or explicit print commands, for example:

    lake env lean axioms_check.lean

A build result supports kernel-level verification only for the declarations built. An axiom inventory must distinguish built-in propext, Classical.choice and Quot.sound from added axioms, sorryAx, and explicitly conditional hypotheses. The S-unit gap theorem in RampGap.lean appears in the source as a theorem hypothesis; its external mathematical applicability must be audited separately. Source comments claiming no sorry do not substitute for compilation and axiom inspection.

Failure to compile, unknown imports, a missing toolchain, memory exhaustion, and unverifiable generated certificates must remain distinct statuses. Record errors rather than silently dropping the corresponding theorem roots.

## Ancestry and training extraction

Attach each accepted theorem to the name-free statement fingerprint in the DataGoblin intake, its namespace/import context, declaration name and source file. Declarations using the same proof template or numerical specialization are assigned one generator-family parent even if their concrete theorem statements differ. The 628 EvalMath PatchWitness declarations, 288 curriculum fixtures and 197 small LoopedTransformers certificates require that family grouping before any model split.

Preserve training, development and test splits. The six Semaev-RedTeam development tasks and their answer-key obligations are permanently out of the training projection. Each training trajectory requires an independent check of the exact attempted claim and its result. A theorem source plus a synthetic explanation is insufficient to claim an observed research episode.

Kernel evidence from this stage can establish the mathematical statement in Lean's formal model. Claims about deployed ECDSA security additionally require a valid bridge from the formal assumptions to the concrete algorithmic and computational model.
