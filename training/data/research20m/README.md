# DG-Research-20M: next bespoke Aleph corpus increment

**2026-11-28 — candidate preparation, not a training-ready release.** Target: 20,000,000 new, distinct content tokens under the exact pilot tokenizer, after provenance, split, rights and cross-corpus deduplication review. This is a reservoir for later [10B curriculum](../aleph_10b_mix.json) checkpoints, within the existing processed-position budget. No new quota is added to 10B. No paid teacher, Colab or QPU calls.

## Material in hand

| Candidate | Source rows | Custody |
| --- | ---: | --- |
| BPD/TRM Inky v2 preferences, repair, SFT | 18,211 | Manifest-reported TRAIN-only; gzip intake pending |
| BPD/TRM multi-task Inky v3 preferences, repair, SFT | 16,472 | Manifest-reported TRAIN-only; gzip intake pending |
| BPD SCCG-10 native MMS decisions/admissions | 35,912 | Manifest-reported fit/feedback; gzip intake pending |
| MTR-02G/H/I observed option outcomes | 576 | Actual pinned JSON read, grouped into 96 source roots; schema checks passed |
| New finite research oracle problems | 640 | Generated in V8, cross-checked by two algorithms per input; exact token count unavailable |

First three sources total **70,595 reported raw records**, and the pinned export roster totals **71,171** with the MTR arrays. These are not counts of distinct roots, accepted training rows or tokens. The v2/v3/SCCG candidate lineages may overlap. A derived view never increases unique root credit. The 96 grouped MTR candidates and 640 new finite problems still require rights, objective, prior-corpus and reserved-family review. No exact tokenizer census has run. Raw training payloads stay outside Git, except historical source blobs already tracked upstream.

## Proposed mixture

| Research lane | Target unique tokens |
| --- | ---: |
| Recurrent mechanisms, BPD and Aleph | 4M |
| Research decision-making, Distill, MTR | 4M |
| Skill factory, persona, self-models and adaptive stopping | 3M |
| Formal mathematics and proof repair | 3M |
| Experimental audit, uncertainty and negative results | 2M |
| Quantum information, observability and compilation | 2M |
| ZK and protocol verification | 2M |
| Total | 20M |

Prioritize actual train-only exports, then native generator replay on fresh root namespaces. Prefer complete bounded research episodes with measured observations, alternatives, failed actions and independently supported repairs. Avoid generic reasoning filler and superficial paraphrases. Preserve the result's evidential limits: an exact finite task does not certify the originating paper, and an exploratory model failure does not imply a universal result. The five original synthetic families are a pilot, not authorization to generate millions of numerical variants.

## Files and commands

- [sources.json](sources.json): 30 pinned program references, 11 exact export pointers, source manifest/blob and payload hashes, and target lane allocations.
- [stage_recent.mjs](stage_recent.mjs): local read-only Git blob census, compressed/decompressed SHA-256 and row validation, exact duplicate metadata and MTR grouping. The three MTR normalizer cores were run against the actual source rows; full Node CLI and gzip recovery remain unexecuted.
- [mtr_remote_audit.json](mtr_remote_audit.json): 576 rows, 96 roots, nine rejecting controls. This is a source-schema audit, not a rerun of the original game or an independent truth certificate.
- [fresh_oracle.mjs](fresh_oracle.mjs): new finite evidence selection, experiment-value, Horn proof, observation-fiber and paired audit tasks.
- [fresh_oracle_audit.json](fresh_oracle_audit.json): 640 distinct generated inputs, 640 two-algorithm comparisons, four rejecting cap controls. Same runtime/generator, not independent native validation.
- [count_reviewed.py](count_reviewed.py): exact-tokenizer unique-content census, requiring separately hash-pinned steward approval and prior/held-family custody records. Python execution and real tokenizer are still pending.

Use the [Neon handoff](NEON.md). Prepare a JSON map of repository full names to existing local checkout paths (at minimum MoralityLabAI/DataGoblin and MoralityLabAI/BPD for the first eleven export files). Fetch missing pinned commits with authorized Git credentials; review before materializing any raw payload. Run from the DataGoblin branch checkout, with a fresh output directory under an existing private parent **outside source checkouts**.

```sh
node training/data/research20m/stage_recent.mjs --repos /private/research20m/local_repos.json --out /private/research20m/staged-v1
node training/data/research20m/fresh_oracle.mjs --per-family 128 --out /private/research20m/fresh-v1
python training/data/research20m/count_reviewed.py --selftest
```

The staging command reports exact duplicate raw rows; it intentionally does not infer canonical root identity or near-duplicate status for the heterogeneous gzip exports. Review their schemas and native source manifests, then implement source-specific adapters. MTR root records preserve action-conditioned features and measured outcomes separately; the source feature order needs its method contract before rendering natural-language supervision. Preferences and failed outcomes are not positive SFT answers.

For the reviewed-token census, render one canonical text per root in private JSONL with keys `canonical_root`, `family`, `lane`, `canonical_text`. Provide a separate steward approval JSON (`schema: dg.research20m.steward-approval.v1`, `records` keyed by canonical root; each value binds `content_sha256`, `family`, `lane`, `decision: admit_candidate`, `split: train`, and nonempty `rights_evidence`, `lineage_evidence`, `near_duplicate_review`, `split_review`, `objective_review`, `source_identity`). The private prior custody JSON (`schema: dg.research20m.prior-custody.v1`) binds `prior_snapshot_digests`, `reserved_family_digest`, `roots`, `content_sha256` and `reserved_families`. Provide the two independently pinned SHA-256 values plus the exact tokenizer.json. The script measures unique content before packing; loss-bearing positions, separator/padding behavior, verifier identity and actual release acceptance require the native packer/replay gates.

```sh
python training/data/research20m/count_reviewed.py --records /private/research20m/reviewed.jsonl --approvals /private/research20m/approval.json --approvals-sha256 APPROVAL_SHA --prior /private/research20m/prior.json --prior-sha256 PRIOR_SHA --tokenizer /private/pilot/tokenizer.json --out /private/research20m/unique_census.json
```

Do not estimate the 20M quota from bytes or token counts under another tokenizer. Measure shortfalls by lane and stop on a supply or rights gap rather than replaying older data. The public report can contain aggregate counts, hashes and provenance, not held-family IDs or private data. The pilot tokenizer hash is pinned in the source plan. A release is not training-ready until the existing Aleph packer, root-custody, and continuation gates pass.
