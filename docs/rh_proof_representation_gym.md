# RH-GYM-01: budgeted proof-representation repair

2026-11-27. Status: **design only; no new training or Lean runs**.

The public readable and compact RH/Dirichlet-GRH editions (https://github.com/lsernau/grh-lean/tree/72fead102f663d5c7a407964a13dc0cf9d977207) verify the same five theorem statements with different elaborated representations. Independent external checks (https://math.mit.edu/~drew/shortgrh/) report checker-specific cost. Their entire dependency neighborhoods are exposed reference/evaluation-only roots.

Extend the existing RLT external Lean checker interface in docs/rlt_external_lean_mcp_verifiers.md. The model proposes an authorized candidate ID and a budget decision; the environment owns candidate source, frozen root, exact expected type, allowed axioms and measured outcome. Do not accept self-reported proof cost, correctness or memory. An agent must not run arbitrary shell or generated code on the verifier host.

## Episode

Public observation: task ID, source and verifier revisions, authorized edit IDs, syntax/dependency summary, accepted previous receipts, remaining verifier-call and time caps. A candidate transforms one proof body or a bounded local definition cluster; freeze semantics and public theorem signatures. Examples: extract a lemma, share a subterm, change elaboration shape, select an existing equivalent lemma, or STOP. Definitional rewrites must pass statement comparison and dependency/axiom checks; an edit may change downstream declarations only inside an explicitly authorized closure with all relevant exported types compared.

Private environment receipt: root family and split, candidate hash, frozen challenge hash, Lean/Mathlib/checker hashes, exact theorem comparison and axiom closure, independent verdict, build/checker timestamps and peak RSS, memory cap, CPU cap, exit status, timeout/OOM flags. Every candidate is replayed in a fresh constrained environment without network. Fail closed on missing or mismatched receipts. Resource exhaustion is a separate status, not proof invalidity.

Acceptance gate: same exact pinned challenge; accepted by configured checker(s); axiom closure contained in propext, Classical.choice, Quot.sound. A successful receipt contributes a Pareto vector of source bytes, elaborated export bytes, checker time, peak RSS. Track costs of all attempts. For fixed cap B and horizon H, primary reward is a verified equivalent root solved under cap; compare total verifier calls and total elapsed work. Report per-checker accepted-set cost and censored cases separately. Use a lexicographic gate rather than letting low memory compensate for a false proof.

## Data and evaluation contract

- Stage 0: deterministic synthetic import graphs and finite DAG scheduling fixtures, source-byte lifetime proxy only. Compare lexical, greedy, random ready order and exact subset DP on tiny graphs. A proxy failure remains negative.
- Stage 1: new small Lean construction families with verified equivalent candidates and mutations (wrong theorem, hidden axiom, malformed receipt, OOM, oversize). Root and dependency-family split before generation, source hash, family hash, candidate hash and exact statement hash. Hold out full families; use public RH as reference-only, never a source of eligible training labels.
- Stage 2: compare TRM/RLT vs static ranker, matched depth/compute, heuristic and random policy, plus memory-only and cost-only ablations. Matched candidate menus and verifier caps; paired fresh roots, account for all attempted checks; train with only environment-owned fresh TRAIN roots.
- Stage 3: optional cross-checker generalization. Keep checker identity explicit; train on one and evaluate on another, report separately. A tool-specific success does not establish compactification across checkers.

Leak controls: private future receipts/gold actions unavailable at inference; accepted proof state and provenance frozen; remove roots that share exact subproblem/near-duplicate proof lineage across splits. No public RH proof-derived train/test uplift claim. Do not use generated evaluation runs or model weights in source commits.

Resource budget proposal: operator-controlled hard caps 512 MiB, 2 GiB, 6 GiB, 12 GiB for small fixtures; no automatic full RH build. Upstream compact build calls for 24 GB, Comparator 64 GB; readable lean4lean exceeded 332 GB. The complete proof checker campaign needs separate operator authorization.

Promotion requires independently checked fresh families and a prespecified improvement in verified solves at matched total budget with a paired uncertainty interval; no promotion on source-only proxies. The compactification analyzer and receipt live in MoralityLabAI/Compactification-Theory, branch research/rh-proof-compactification-20261127.
