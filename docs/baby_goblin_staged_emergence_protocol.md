# Baby Goblin Staged Emergence Protocol: 20M / 50M / 100M / 200M Pretraining & Specialist Adaptation

**Base Lineage:** `AlephFunk/aleph-direct-a-10m-v2` (999,352,320 tied parameters, 14 unique blocks,
24 executed block visits `[0, 1..10, 1..10, 11, 12, 13]`, SmolLM2 tokenizer `9ca9acdd...`,
replacement v2 corpus index `53d27abf...`).  
**Machine-Readable Emergence Ledger:** [`configs/research_v2_1/baby_goblin_emergence_ledger.json`](../configs/research_v2_1/baby_goblin_emergence_ledger.json)  
**Executable Probe Runner:** [`scripts/research_v2_1/baby_goblin_emergence_probe.py`](../scripts/research_v2_1/baby_goblin_emergence_probe.py)

---

## 1. Empirical Lesson from the 10M Early-Specialization Audit

The 10M 4-arm matched-budget pilot (`receipt_sha256: 63a42e12...`) and its forensic exposure/lineage
audit (`receipts/baby_goblin_exposure_audit.json`) established four constraints that govern all
subsequent staged checkpoints:

1. **10M positions is pre-syntactic (mid-warmup, `lr = 1.25e-5`, loss `7.0351`):** Exit-shell SFT
   shifts probability mass into JSON punctuation (`"{\"{\"..."`) and lifts teacher-forced `p(EOS)`
   by >2,600× (`1.3e-5 -> 0.0346`), but fails to produce a single valid JSON object (`0/16`) or EOS
   stop (`0/16`) under greedy decoding.
2. **Base-then-specialist strictly dominates specialist-only at matched compute:** `arm3` (64 base +
   64 spec) beat `arm1` (128 spec) on target NLL (`3.9823` vs `4.1202`), TF token accuracy (`11.05%`
   vs `7.89%`), `p(EOS)` (`0.0346` vs `0.0336`), and general retention (`+0.72` vs `+3.60` nats).
3. **Match loss-bearing tokens AND preserve full 2,048-token context:** At `max_seq_len = 384`,
   prompt-masked SFT backpropagates through 6.76× fewer loss tokens than base LM per forward token
   and left-truncates long policy/compact-state prompts. Staged adaptation must train at full `2,048`
   context with explicit accounting of both forward positions and supervised loss tokens.
4. **Preserve optimizer lineage across pretraining milestones:** Continue full-model pretraining from
   durable training checkpoints (`group_039_7fbdb1f91cf8`: weights + FP32 AdamW moments + RNG +
   data cursor) rather than restarting from weights-only exports.

---

## 2. Staged Pretraining Milestones (20M / 50M / 100M / 200M Positions)

Each milestone resumes from the exact previous durable group checkpoint on the frozen replacement v2
corpus (`262,144` positions/group, BF16 compute + FP32 master weights + FP32 AdamW moments, full
999,352,320 tied parameters trainable, context length `2,048`, **zero held/prospective/Jinn-val/test
roots admitted**):

| Milestone | Cumulative Positions | Update Groups (262,144 pos/grp) | LR Schedule Position | Expected Pretraining Regime |
|---|---:|---:|---|---|
| **10M** *(observed)* | `10,000,000` | `39` (`group_039`) | Linear warmup end `1.25e-5` (halfway to 20M peak) | Unigram punctuation collapse; `loss ~7.04`; `0%` JSON / `0%` EOS |
| **20M** *(Stage 1)* | `20,000,000` | `77` (`group_077`) | Peak LR `2.50e-4` reached at 20M | End of warmup; first short-range syntax & EOS calibration check |
| **50M** *(Stage 2)* | `50,000,000` | `191` (`group_191`) | Cosine plateau `~2.45e-4` | Early compositional phrasing; candidate onset of valid JSON + EOS under SFT |
| **100M** *(Stage 3)* | `100,000,000` | `382` (`group_382`) | Cosine decay `~2.25e-4` | Multi-clause syntax & schema structure; candidate onset of action-verb selection & verifier checks |
| **200M** *(Stage 4)* | `200,000,000` | `763` (`group_763`) | Cosine decay `~1.80e-4` | Candidate onset of autonomous & controller-assisted learner fits (`primary_score > 0`) |

---

## 3. Matched Specialist-Adaptation Fork at Every Milestone

At each milestone $M \in \{20\text{M}, 50\text{M}, 100\text{M}, 200\text{M}\}$, export the native FP32
weights and run **three controlled forks** (strictly disjoint from all frozen held/prospective panels
and Jinn/Beast validation/test roots):

1. **`base_M` (0 specialist updates):** Raw checkpoint evaluated directly on the fixed emergence probe.
2. **`spec_replay_M` (full-model or LoRA+core SFT with 50% general replay):**
   - Full `2,048`-token sequence window (zero prompt truncation).
   - Balanced mixture: **50% train-only specialist pool** (all `211,906` supervised tokens across
     public Research v2 synthetic + private Jinn/Beast train-only, `>= 2` full epochs = `>= 423,812`
     supervised loss tokens!) + **50% disjoint general replay** to prevent catastrophic forgetting.
   - Peak LR `5e-5` (cosine decay to `5e-6`, 5% warmup, FP32 AdamW, grad clip `1.0`, explicit EOS
     supervision on every action turn).
3. **`continued_base_matched_M` (compute- & forward-token-matched base control):** Same additional
   forward tokens purely on the next replacement v2 base cursor positions (zero specialist SFT),
   verifying whether gains come from specialization vs raw pretraining compute.

---

## 4. Formal Emergence Thresholds & Recording Contract

Every milestone (`base_M`, `spec_replay_M`, `continued_base_matched_M`) is probed on the **same
frozen 16-case non-held emergence suite** (`eval_seed = 2026112799`, 2 cases/family, 0 overlap with
all training & held suites) plus the **same 56-chunk disjoint general retention set** and
`token_checks.json`, recording:

| Emergence Milestone | Quantitative Gate | Recorded Field in `baby_goblin_emergence_ledger.json` |
|---|---|---|
| **1. JSON Syntax Emergence** | `valid_json_rate >= 0.25` *(onset)* and `>= 0.80` *(reliable)* + `schema_valid_action_rate >= 0.25` | `emergence_milestones.first_json_onset`, `first_json_reliable` |
| **2. EOS Termination Emergence** | `eos_stop_rate >= 0.25` *(onset)* and `>= 0.80` *(reliable)* under greedy decoding (`max_new_tokens = 256`) | `emergence_milestones.first_eos_onset`, `first_eos_reliable` |
| **3. Action-Selection Emergence** | `action_verb_accuracy >= 0.50` across learner (`run_experiment`) vs symbolic (`check`/`submit`) families | `emergence_milestones.first_action_selection` |
| **4. Verified Research Progress** | `autonomous` or `rlt_state_correction` `ResearchSessionV21`: `mean_capability_score >= 0.25`, `>= 1` real optimizer fit OR verified check, and `mean_primary_score > 0.0` | `emergence_milestones.first_verified_capability`, `first_nonzero_primary_score` |
| **5. Retention Guardrail** | `general_val_nll_nats <= base_M_nll + 0.25` nats (no catastrophic forgetting) | `retention_guardrail_passed` |

Only checkpoints passing **all five gates** on the development emergence probe advance to a **single
registered pass** on the sealed prospective confirmation panel (`prospective-v2.1-20261127-final`).
