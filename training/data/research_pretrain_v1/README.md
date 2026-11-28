# DG research pretraining candidate v1

Fresh synthetic exercises derived from RV22/RV23 research, not copied
held-out packets or raw Distill referee exports. All prior reference-only
custody remains unchanged. This candidate is not the frozen 10M release,
and no continuation or production training is authorized.

Default generation: 512 train roots and 64 development roots per family,
six families (adaptive evidence, robust discrimination, research
portfolio, proof obligations, paired causal gate, research memory).
Three candidate views per root: decision, certificate repair, comparison
against an explicit analytic control. No more than four presentations of
any root across all objectives. Separate development artifact; do not
train on it. Generator permits larger independent candidate runs after
root registry audit. Do not multiply views to meet a token quota.

The native targets use DG's checked DP, replayable cover proof, independent
subset enumeration and rational arithmetic; Distill's pinned assessor
independently grades optimality, feasibility and stated values. Faults
must fail that assessor. Analytic controls are graded as returned, including
unresolved or suboptimal proposals. Memory examples compare explicit cyclic
cache simulation with derived ledgers, and binomial marginal dimensions
with an independent Pascal recurrence. The scope is the stipulated finite
model. These are public derivations, not private model reasoning or
scientific truth beyond the model.

Record format: prompt, target, delimiter (newline Answer newline),
semantic_key, view, split, provenance and typed witness, with candidate
custody. Packed records optionally use the exact Aleph-A tokenizer
revision/hash and response-only offset mask from the 10M builder,
2048-position context cap, 49152 vocabulary. No special-token insertion;
same newline-terminated transcript as existing builder. Report accepted
nonpadding positions and loss-bearing positions separately from byte
counts, and reject overlength views instead of truncating. This is not
the production trainer's indexed binary bundle. EOS supervision for any
Baby Goblin-specific objective needs its own gate.

Reconstruct known native A/B/C/D and Z evaluation/train roots, excluding
all generated root IDs against that inventory and between new splits.
Adaptive worlds use full canonical keys for exclusion; other families
use raw content IDs. This is not a global historical or private registry
audit; rights/consent, deletion tombstones, cross-corpus structural
deduplication, independent stewardship and final mixture admission
remain required. Candidate records intentionally carry
training_admission=false until those gates. No old referee rows used as
targets. No model API calls, GPU training, or checkpoint alteration.

The hosted job publishes separate train and development archives plus
manifest with SHA-256 member hashes. Exact count results will be recorded
after the job completes. The builder is CPU-only, depends on the pinned
public Distill checkout, and can reproduce the candidate locally.

    python -m datagoblin.research_v2_2.research_pretrain --distill-checkout "$DISTILL_CHECKOUT" --out "$FRESH_OUT" --tokenizer "$PINNED_TOKENIZER"

Source pin DataGoblin 1a73ecc4db75857973f9472ca86bc435af48b433,
Distill 7fa19c90372b13d71676e6fef13819d2a18b9be3.
Suggested intake lane: custom_research. Repair and comparison views
are typed within that lane, so a root has exactly one accounting lane.


## Observed hosted materialization

Source generator commit 63c103383c1b22ff14907bba93648b5182bc89c5; workflow [46142396977](https://github.com/MoralityLabAI/DataGoblin/actions/runs/46142396977),
job 140971015645. Smoke tests and exact frozen tokenizer download passed.
The native build completed with no rejected or truncated overlength views.
Manifest ID a752b77a18bd6b1f62079dd888308d4e170b6426b68630082bc20eefc93e8910.

| Split | Roots | Examples | Nonpadding positions | Loss-bearing positions | Max context |
|---|---:|---:|---:|---:|---:|
| Train | 3072 | 9216 | 4221281 | 708670 | 1098 |
| Development (do not train) | 384 | 1152 | 527134 | 88985 | 1085 |

| Train family | Roots | Examples | Positions | Loss tokens |
|---|---:|---:|---:|---:|
| adaptive_evidence | 512 | 1536 | 729367 | 160540 |
| paired_causal_gate | 512 | 1536 | 1167351 | 179881 |
| proof_obligations | 512 | 1536 | 707398 | 110005 |
| research_memory | 512 | 1536 | 185743 | 94699 |
| research_portfolio | 512 | 1536 | 771629 | 83838 |
| robust_discrimination | 512 | 1536 | 659793 | 79707 |

Known root exclusion inventory: 800; rejected 63
overlapping/duplicate proposed train roots and 10 development roots.
Every accepted root contributes exactly three views; no split overlap
under the implemented semantic/raw identity. These are measured
candidate positions, not approved mixture quota credit.

- [Training archive](https://github.com/MoralityLabAI/DataGoblin/actions/runs/46142396977/artifacts/15262860262)
- [Separate development archive](https://github.com/MoralityLabAI/DataGoblin/actions/runs/46142396977/artifacts/15262410977)
- [Manifest](manifest.json)
- [Source-attributed research lessons](research_lessons.jsonl)

Training archive contains raw records, packed response-masked token
records, root inventory and common aggregate manifest. The per-packed
row receipt_sha256 binds the corresponding witness-bearing record,
not a standalone steward release receipt. The builder retains exact
reference grades but is not a production indexed-bundle exporter.
The separately curated twelve reference notes are not included in
the measured shard counts and are not verified action labels.
Artifacts have hosted retention until February 26, 2027; regenerate
from the source pin before expiry if necessary. This receipt commit
wraps four long source lines without intended semantic change.
No full cross-corpus, private heldout or source-rights admission
has been performed, and no model training has been launched.
