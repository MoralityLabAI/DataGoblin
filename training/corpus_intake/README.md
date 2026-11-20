# Owned mathematics corpus census

This is a reproducible **source-level intake**, pinned to five GitHub commits on 20 November 2026. It investigates the difference between Lean-only counts and the larger mathematical research corpus held in prose, Python, proofs, tests, and experimental critique.

| Repository | Pinned revision |
|---|---|
| MoralityLabAI/Compactification-Theory | 071cf29e27a71fff858cc3bbb4dc3ea4193c9b53 |
| MoralityLabAI/EvalMath | bed334f032d31d777bad0e5152e732f57252092b |
| MoralityLabAI/LoopedTransformers | 39db0ab08a2c7c69b3b0bb04e74b13c7b59bdc7a |
| MoralityLabAI/BPD | 60dd9ed19aeddd8dddc40a6facad36140a1250ed |
| patrickdugan/semaev-index-calculus | 3c630e179ae837dad839949dcb03cf4fdb41815e |

## Run

Use a temporary directory for upstream checkouts. This reads the repositories at fixed SHAs using Git's partial-clone sparse checkout and leaves the upstreams unchanged. The workflow runs on PRs touching the audit and produces downloadable audit files.

    python -m pip install tiktoken==0.12.0
    python training/corpus_intake/audit_owned_math.py --clone-pinned --root /tmp/owned_math_sources
    python training/corpus_intake/audit_owned_math.py --root /tmp/owned_math_sources --out /tmp/owned_math_audit --tokenizer o200k_base

Once a tokenizer has been frozen for Aleph, supply an **existing local** Hugging Face Tokenizers JSON, with its SHA-256 recorded automatically:

    python -m pip install tokenizers
    python training/corpus_intake/audit_owned_math.py --root /tmp/owned_math_sources --out /tmp/aleph_audit --tokenizer-json /path/to/frozen/tokenizer.json

If GitHub Actions runners are unavailable, there is a Windows PowerShell CPU-only runner using the same pinned inputs. It requires Python with tiktoken and an installed Git command:

    python -m pip install tiktoken==0.12.0
    pwsh -File training/corpus_intake/run_local_windows.ps1

The Windows runner leaves the checkout and reports under the system temporary directory, avoids GPU and Lean builds, and emits both the interim token census and proof-root/split-overlap candidates. See SEMAEV_KERNEL_GATE.md for the separate pinned Lean proof verification contract.

The current o200k_base counts are interim measurements, not native Aleph token counts. The tokenizer, model loss mask and context packing need distinct accounting. Keep these interim counts separate from the 20B/30B training manifest.

## Produced files

- files.jsonl: source path, SHA-256, role, size, optional token count, cross-repo exact duplicate and explicit exclusions.
- lean_declarations.jsonl: declaration name, source line, statement fingerprint, source token span, sorry and explicit-axiom source flags. **No declaration receives a verified status.**
- reasoning_episode_candidates.jsonl: Markdown sections with assumption/failure cues. These are review pointers, not independent trajectories.
- summary.json and report.md: exact token counts by repo for admitted text plus exclusions and source-only declaration counts.
- proof_root_candidates.json: repeated name-free theorem statement syntax clusters, source lanes, split roles and train/evaluation collision candidates. These are review-only proposals rather than independently verified roots.

Source scope: Lean, Python, Markdown, TeX only. JSON/JSONL experiment receipts, source outside these extensions, large external archives, upstream branches other than the pinned refs and local uncommitted traces remain outside this pass. Any file over 2 MB is recorded only by metadata. Semaev's generated instance arrays and chain-standard templates are recorded as proof artifacts, not tokenized as natural-language or tactic trajectories.

A second source tool groups candidate theorem statement fingerprints after removing declared theorem names. Its CLI can also run independently on a previously generated declaration manifest:

    python training/corpus_intake/proof_roots.py --declarations /tmp/owned_math_audit/lean_declarations.jsonl --output /tmp/owned_math_audit/proof_root_candidates.json

Statement hashes detect literal syntax overlap without claiming semantic identity. Group by the recorded original repository/commit and inspect namespace/import assumptions before deduplicating. If the same statement appears in a training file and a protected development/test file, the candidate is flagged and must be withheld pending a split decision.

The source regex scans declarations for an **inventory estimate**, without parsing Lean elaborator state. Inline axioms, theorem provenance, native_decide, duplicated reasoning ancestry and kernel success require a later structured extractor and pinned compilation. The research-section detector merely flags text to examine. The output is excluded from training by default (training_eligible=false).

## Training roles and leakage controls

Treat Semaev's general theorems, mathematical papers, verifier programs, and explicit correction dialogues as distinct lanes. Keep its generated instances in an immutable evidence/certificate store; a model may learn to invoke their checks without predicting megabytes of coefficients. The seven giant Semaev Lean instances were independently counted at 74,515,825 bytes in the 19 November repository inventory. The Semaev results are valuable for the quality of the mathematical hypotheses, counterexamples and revisions rather than that numeric bulk.

Before admission:

1. Freeze the Aleph tokenizer and loss policy, then recount accepted text.
2. Clear source licenses, imported mathlib snippets, private logs and document-level redistribution.
3. Resolve roots across copied files, reused research trials, near duplicates, theorem statements, and generated variants. Keep family-level eval holdouts excluded.
4. Pin Lean toolchains and run kernel checks, including imported axioms, sorry, external assumptions and finite instance certificates. Record individual verdicts rather than inferring them from README claims.
5. Join proposal, declared assumptions, test, counterexample or proof, verdict and next action using independently recorded outcomes. Count target loss tokens and candidate roots separately.

## Semaev-RedTeam

See the development gym at training/gyms/semaev_redteam/README.md. The transparent six-task set is a development diagnostic only. Its gold obligations appear in the repository and cannot serve as sealed or independent testing of a trained model.
