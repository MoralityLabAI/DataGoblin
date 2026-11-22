# RLT-HARD-02C: position-invariant tactic content under matched Mathlib menus

Date: 2026-11-22. **Development registration, before native policy outcomes.**
Related: DataGoblin PR #28, issue #24.

## Motivation and preflight evidence

The first Mathlib native experiment showed the learned policy and a one-check
longest-script rule both certify 6/8 ordinary and 6/8 OOD roots with eight
candidate checks per split. The independent audit on Actions #44989615504
verified every retained source/proof digest and kernel receipt.

A second cohort (RLT-HARD-02B) eliminated the script-length difference using
five length-matched near-miss scripts per root, source seed \`2026112203\`.
Its source-only feature identifiability audit, Actions #44990729864, found
**32/32 position-only roots and 0/32 roots where the old grounded features
distinguished candidate semantics**. All candidate feature vectors are
identical after removing menu index. This result is independent of Lean
kernel outcomes.

The original grounded model recognizes the first tactic command and
qualified List/Nat lemmas, and places normalized script byte length and
candidate index in its feature matrix. Every RLT-HARD-02B menu has equal
byte lengths, the same first-command form and the same recognized
qualified lemma names across its five candidates. The old model can
therefore learn menu-position preferences but no informative candidate
content discrimination in this cohort.

## Fresh test and paired representation ablation

**Source seed: \`2026112211\`.** The source generator remains
\`length_matched_near_miss_v2\`. Upstream Mathlib commit and exact Lean 4
toolchain remain \`809c3fb3b5c8f5d7dace56e200b426187516535a\`
and \`4.12.0\`.

The new candidate-feature module
\`hard02_semantic_features.py\` creates a 208-dimensional NumPy feature
vector independently for each source-visible tactic script. It contains
16 explicitly declared syntax features and three signed SHA-256 hash
banks (64 bins each) for lowercased unigrams, bigrams and trigrams of
full tactic text. Trailing whitespace is stripped before tokenization.
Features use the public goal, candidate syntax and stable hash buckets;
they never access gold actions, observed verifier labels, source split
identity, candidate list position or private evaluation data.
The vocabulary needs no fitting on held-out roots.

The source-only tests require all five new candidates to have distinct
content feature vectors in all 32 source roots. Permuting the candidate
order must permute feature rows without altering their values.

The new native campaign fits **two models on exactly the same new TRAIN
receipts**:

- **Content-aware learned policy:** logistic ranker with 208-dimensional
  full-script features and a CAL-only frozen stop threshold
- **Legacy-grounded learned policy:** the existing position-dependent
  features and an independently CAL-frozen threshold

Both are evaluated against the same five-candidate source-root budget
and three unchanged native controls: fixed full, random full, and
trivial-first. The ordinary and OOD development roots and theorem
statements are identical across arms, with actual Lean compiler
verification. Each attempted candidate is passed through the
\`LeanKernelChecker\`; the independently audited kernel proofs and
explicit allowed axioms alone determine successful closure. Both
learned-policy arms receive independent source replays.

This **within-cohort paired ablation** is required to isolate the effect
of the feature representation. The earlier length-matched source seed
is a separate development experiment and is not used to choose a
threshold or to fit the new models.

## Claims, cuts and acceptance

All outcome analysis must retain any learned failure against fixed full
and random full search, and count false STOP decisions separately for
each learned arm. Also report actual compiler checks and wall time
alongside kernel-certified coverage. The independent source/receipt
auditor must reject wrong SHA-256 inputs or missing matched-policy arms.
Preflight source tests and native kernel CI are separate gates.

A positive within-cohort model-value result would require the new
content-aware policy to improve certified solves or cost over the
legacy-grounded policy and meaningful nonlearned controls. Structural
differences in features alone establish identifiability, not useful
learning. These source theorem families are still generated from a
limited Mathlib operation palette and each candidate menu is fixed;
RLT-HARD-02C remains below issue #24's hard Mathlib/LeanGym and
target-blinded MCP acceptance gate. Sealed confirmation remains unopened.

## Reproduction

\`\`\`bash
python -m pytest -q tests/unit/test_rlt_hard02_semantic_features.py \
  tests/unit/test_rlt_hard02_semantic_policy.py
ruff check src/datagoblin/training_gym/rlt_gym/hard02_semantic_features.py
\`\`\`

Use \`.github/workflows/rlt-hard02-semantic.yml\` to fetch Mathlib
at the exact pinned commit, install Lean 4.12.0 and execute:

\`\`\`bash
python -m datagoblin.training_gym.rlt_gym.hard02_mathlib \
  --lake "$HOME/.elan/bin/lake" \
  --mathlib /tmp/dg-rlt-mathlib \
  --per-split 8 --source-seed 2026112211 \
  --matched-menus --semantic-features \
  --out runs/rlt-hard02-semantic
\`\`\`

It retains \`report.json\`, full source roots, kernel proof sources,
training weights and native receipts, and independently runs the
source and false-stop auditor. This is a development-only run.
