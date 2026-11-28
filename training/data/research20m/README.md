# DG-Research20M corpus gym

CPU-only generation toward 20,000,000 candidate nonpadding positions:
4,221,281 in v1 plus a 15,778,719 new-position target. The gym does not
train a language model. Four distinct views per lineage: optimal decision,
invalid certificate repair, explicit control comparison and premise revision.
No repeated presentations. Both before/after world identities belong to
the same lineage and split. All views are accepted together or discarded
when one would exceed 2048 frozen-tokenizer positions.

Two profiles for each of six research families: parity synergy and
dependent tests; shared and bundled controls; scenario conflict and
resource bottlenecks; prerequisite unlocking and refutation barriers;
unequal repeats and confounded budgets; finite offline cache optimization
and deterministic probe alias. Native finite-model assessor independently
grades the decisions, failures and counterfactual revisions. In the
memory lane, compare independent forward/backward DP plus replay, or
explicit prefix enumeration versus a streaming repeat ledger.
Proof-obligation claims are conditional on successful audit jobs, not
actual Lean proofs; causal gates are descriptive synthetic decisions.

Replay v1's exact root inventories against its published member-file
SHA-256 hashes, excluding them and its known historical registry. Adaptive
worlds use binary canonical identity; other worlds use raw content
identity. Pilot and full seed ranges differ. Full runs must provide the
pilot member inventories (both splits); additional known exclusions can
be supplied by repeated flags. This is not a global isomorphism or
private holdout audit. No production admission or quota credit.

Full default additional quotas: 3M adaptive, 3M discrimination, 2.8M
portfolio, 2.5M proofs, 2.5M causal, 1,978,719 memory. Divide each by its
two profiles. Stop when each profile first reaches its quota at a
complete lineage boundary, or the bounded root limit; record overshoot
and shortfall without substituting or repeating data. The pilot uses
8 train and 2 development lineages per profile. Full development is
separate, 16 per profile by default. Reports actual token/loss counts and
per-profile file hashes. Full generation may be expensive in CPU and
disk; no guarantee of throughput or meeting quota within the root cap.
Use one worker initially, no GPU/VRAM needed, recommend 8 GB free RAM
and 2 GB disk as starting resource allowances, not measured limits.
The output is candidate JSONL, not an indexed production training bundle.

## Operator run

Use DataGoblin PR 113 checkout and clean Distill checkout pinned to
7fa19c90372b13d71676e6fef13819d2a18b9be3. Install editable DG with
tokenizers==0.22.2, and obtain the exact SmolLM2 tokenizer pinned by v1.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m datagoblin.research_v2_2.research20m --distill-checkout "$DISTILL" --tokenizer "$TOKENIZER" --v1-manifest training/data/research_pretrain_v1/manifest.json --out "$PILOT" --pilot

After pilot, run the full candidate build with the same pinned inputs:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m datagoblin.research_v2_2.research20m --distill-checkout "$DISTILL" --tokenizer "$TOKENIZER" --v1-manifest training/data/research_pretrain_v1/manifest.json --exclude-members "$PILOT/train/members.json" --exclude-members "$PILOT/development/members.json" --out "$FULL" --target-positions 15778719

Do not merge development JSONL into train. Review failures, member
inventories, exact grades and token masks. Preserve independent
stewardship, source rights/consent, deletion, global split exclusion,
structural deduplication, approved mixture and checkpoint gates.
Full candidate generation intentionally requires a separate invocation.
