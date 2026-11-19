# Data Goblin Aleph: data landscape and 20B / 30B token plans

**As of 19 November 2026, Asia/Singapore.** This report audits source cards, repository trees and existing aggregate receipts. Public corpora were not downloaded or retokenized. External game shards remain unexamined. Proposed quotas count processed non-padding tokens and include repeated presentations.

## Recommended training target

Prepare a **20B-token run with two recurrent rounds**, retaining a **30B continuation** that depends on measured throughput and development results. Preserve a usable checkpoint at 10B. The public sources provide ample selection capacity, while the owned research material supplies tasks tied to Data Goblin's intended work.

The desired outcome is a compact model that can inspect a corpus, build a bounded gym, interpret an experiment receipt and choose a useful next action. Code, mathematical explanations and examples of responding to failed checks therefore receive substantial weight. This remains a training hypothesis whose value must be measured on held-out tasks.

The catalog contains **24 public pretraining/solution sources**, **16 formal-math/game sources or generator families**, and **eight owned corpus lanes**. Source size, accepted unique tokens and training exposures have separate fields. Every proposed source still needs an intake decision under a frozen tokenizer and source revision.

## What the owned work currently supports

| Corpus | Evidence recovered | Training preparation needed |
| --- | --- | --- |
| AICourt mechanics ensemble | 2,000 episodes; reported mean 18,822.8245 `cl100k_base` tokens per serialized episode. Derived volume: **about 37.65M tokens**. | Recover the raw shards, project actor-visible decision records, deduplicate and retokenize. Policies were seeded deterministic heuristics. |
| Earlier Court validation | 200 episodes, 3,753,256 `cl100k_base` tokens. | Join episode IDs and environment versions with the larger ensemble before calculating any combined total. |
| Sol/Terra/Luna campaign | A historical pilot records three games and 18 seats. The campaign document targets 8.0–8.5M `o200k_base` tokens. | Production completion and external artifact counts remain unresolved. |
| Control-Harness Mesh Blue | 40,029 accepted records and canonical-transition clusters; 4,999,622 whitespace units across five sealed shards. | Obtain the shards and count the approved training projection with Aleph's tokenizer. |
| TRMStoryworld export | 2,089 rows, including 2,064 base and 25 adapter rows; JSONL size 2,311,212 bytes. | Replay available transitions and count retained examples. |
| Control Observability | 6,400 records forming 3,200 counterfactual pairs. | Preserve paired roots and existing split custody. These labels describe a finite synthetic task. |
| Owned Lean source | Current four-repository metadata: 679 `.lean` files, 80,613,571 bytes. | Extract declarations and proof states; preserve held-out material; compile under pinned toolchains. |
| Research receipts | Concrete negative results and small checked curricula exist across LoopedTransformers and DataGoblin. | Join proposal, prediction, execution, verdict and subsequent revision into complete episodes. |

Primary receipts: [Court ensemble][court], [Court token method][court-method], [earlier Court validation][court-earlier], [model-agent pilot][court-pilot], [model campaign][campaign], [Mesh tranche][mesh], [TRM export][trm], [observability pilot][observability]. These receipts establish historical counts under their named methods. A total of accepted Aleph tokens remains unmeasured.

### Lean size and proof content

| Repository, pinned revision | Lean files | Source bytes |
| --- | ---: | ---: |
| [OntologicalProofs, 1c209654][ontology] | 399 | 2,176,267 |
| [MathCan, 9b2e9f5a][mathcan] | 122 | 560,412 |
| [EvalMath, bed334f0][evalmath] | 34 | 802,097 |
| [Semaev, 3c630e17][semaev] | 124 | 77,074,795 |
| **Total** | **679** | **80,613,571** |

Seven Semaev instance files account for 74,515,825 bytes, or about 92.4% of this source volume. Excluding those files leaves **672 files and 6,097,746 bytes**. File size is a diagnostic for this audit; the retained set still includes fixtures and evaluation material.

The earlier 10 November conversation reported 978,885 non-giant Lean tokens and 22,796,896 numeric-certificate tokens. Its original audit artifact was not recovered, so those quantities remain historical claims with an unresolved tokenizer. The current tree counts above are independently reproducible from the pinned Git trees.

For certificates, keep the theorem, verification algorithm, representative instance and checker outcome. Full numeric payloads can remain addressable artifacts. A compact explanation of how a certificate establishes a result can become a useful language example once checked against that result.

Several small corpora have particularly useful supervision:

- [LR-01][lr01] records 1,280 generated theorems, with 768 in training and 2,333 training proof states. Validation, standard test and OOD partitions retain their original custody.
- [BPD-LG-02][bpd-lean] has 384 proof/refutation examples, including 256 training records. Its independent Lean 4.19.0 audit records empty axiom inventories for every case. These examples can teach the model to find a counterexample to an unsupported universal claim.
- [RAM-17][ram17] records 326 candidate compilations from 63 roots and 20 accepted actions, including 18 training examples. Induction yielded zero accepted actions, which stopped training admission. The failed collection and its diagnosis form a useful research episode.

### Gameplay volume and useful decisions

The 37.65M Court estimate includes serialized bookkeeping. Projection will change its size. Recover one player's available observation, the action, subsequent feedback and a bounded continuation. Keep hidden state in the verifier's workspace. Shared full-history prompts should have explicit accounting so their repeated presentation remains visible.

The [Mesh capacity plan][mesh-plan] names 232,960 canonical transitions and estimates 29,104,302 cluster-adjusted whitespace units for one copy of them. Its 1B target has `materialized: false`. That transition registry gives us a concrete route to new compositions. Surface paraphrases can be sampled deliberately while their common ancestry remains recorded.

The observed policy action is suitable for behavior modeling. Training an optimal-action target requires a solver or other independent evidence about action quality. In social games, outcomes also depend on the opponent, seat, role and available information. Preserve those variables when selecting successful play.

## Public sources to prioritize

All sizes below are publisher counts under their own tokenization or record definitions. The machine catalogs retain additional alternatives and intake notes.

| Source | Published scale | Proposed use |
| --- | --- | --- |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Advertises 1.3T tokens | Broad explanatory English; sample across domains. |
| [FinePDFs-Edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu) | 350B+ tokens, 69 languages | Select English textbooks and instructional documents, checking extraction quality. |
| [Common Pile / Comma](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset) | Main-stage table: 463.6B source tokens | Named source subsets with retained provenance; useful papers, textbooks and code. |
| [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) | 125B advertised code tokens | Code and tests after source-byte access and per-file license checks. The release carries Software Heritage identifiers. |
| [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath/raw/main/README.md) | 9.6B tokens in 4+; 34B in 3+ | Start with the higher-score explanations. The 4+ subset is contained in 3+. |
| [Nemotron-CC-Math](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) | 52B in 4+; 133B in 3+ | Supplement math after access and overlap checks. |
| [peS2o](https://huggingface.co/datasets/allenai/peS2o) | 42.01B whitespace units in v2 | Select CS, statistics, mathematics and experimental methods; recount with Aleph's tokenizer. |
| [SYNTH](https://huggingface.co/datasets/SYNTH-Initiative/SYNTH) | About 75B Pleias-tokenizer tokens | A bounded quota of task-formatted synthetic material, grouped by original seed. |
| [Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | About 39.1M samples | Synthetic educational prose; keep generated factual claims labelled. |
| [FinePhrase](https://huggingface.co/datasets/HuggingFaceFW/finephrase/blob/main/README.md) | About 486.4B completion tokens | Selected compact tutorials and table tasks; generated output is in `rollout_results[0].text`. |
| [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning/raw/main/README.md) | 306K solved problem roots; CoT, tool and selection records | Choose short usable units and retain the answer-verification method. |
| [OpenCodeReasoning-2](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2/raw/main/README.md) | 2.57M Python/C++ rows over 34,799 questions | Critique and repair tasks; join omitted questions and use execution evidence. |
| [OpenThoughts3](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) | 1.2M rows | Selected complete reasoning episodes compatible with the context budget. |

FinePDFs-Edu's authors report that graduate technical content correlated poorly with their tracked benchmarks. Data Goblin needs a protected research-reference quota selected for its own tasks. Preserve methods sections, definitions and API examples even when a generic education score gives them little weight. The recommendation follows from the difference between that filter's measured objective and Aleph's intended workload.

For code, prioritize Python, practical numerical computing, tests, shell, configuration formats and the languages used by the gyms. Keep repository-linked examples and documentation together. Stack-Edu's upstream access requirements need checking before assigning it delivered capacity; its license metadata includes unresolved entries. [Common Pile's source collections](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset) offer an alternative route whose admitted code volume must also be measured.

For synthetic solutions, verification attaches to a specific claim. OpenMathReasoning distinguishes extracted reference answers from answers chosen by model voting. OpenCodeReasoning-2 has a measured `pass_rate`, with `-1` denoting unavailable validation. Its generated critique remains model-authored even when the code passes tests. Store these fields independently.

The [OpenThoughts paper](https://arxiv.org/pdf/2506.04178v1), Figure 4, describes approximately 75K sampled questions and 16 generations per question. This gives a reason to test multiple solutions per root. Its experiments concern posttraining a larger pretrained model; Aleph needs its own measurements of repetition and trace length.

## Formal and game sources

| Source | Available unit | Intake decision |
| --- | --- | --- |
| [mathlib4](https://github.com/leanprover-community/mathlib4) with [LeanDojo-v2](https://github.com/lean-dojo/LeanDojo-v2) | Source repository and extraction tooling | Pin the toolchain, extract proof states, split by theorem ancestry. |
| [Goedel-Prover-V2 SFT](https://huggingface.co/datasets/Goedel-LM/SFT_dataset_v2) | 1,745,010 chat samples | Extract bounded proof and repair episodes, then replay. |
| [STP_Lean](https://huggingface.co/datasets/kfdong/STP_Lean) | 1,738,754 prompt/target records | Deduplicate mathlib and LeanWorkbook ancestry; protect reserved benchmarks. |
| [NuminaMath-LEAN](https://huggingface.co/datasets/project-numina/NuminaMath-LEAN) | 104,155 formalization records | Filter proof status and check statement fidelity separately. |
| [MGC2025](https://huggingface.co/datasets/mindgameschallenge/MGC2025) | 94,132 player-perspective rollouts over four games | Group by game ID, preserve visibility and label invalid actions. |
| [Lichess](https://database.lichess.org/) | 6,193,530 puzzles in the checked snapshot | Select a small puzzle subset with difficulty metadata; join evaluated positions when search-depth metadata is needed. |
| [Reasoning Gym](https://github.com/open-thought/reasoning-gym) | Procedural generator covering 100+ task types | Generate bounded tasks with independent grader checks and structural holdouts. |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel) and [TextArena](https://github.com/TextArena/TextArena) | Game environments and algorithms | Start with tractable worlds; meter generation and solver cost. |

The formal catalogs contain overlapping theorem families. STP's [paper](https://arxiv.org/html/2502.00212v3) explicitly describes using miniF2F-valid and ProofNet-valid in later training. Existing evaluation commitments must be checked across those families before accepting records.

NuminaMath's `complete`, `with_sorry` and `statement` statuses support different uses. A task statement can seed proof generation. A successful proof target needs a kernel check under the recorded imports and axiom policy. Faithful translation of the original natural-language problem requires a separate check.

The generators provide capacity for future work. Their accepted tokens will only exist after generation, filtering and replay. Begin with a 10,000-example measurement across 12–20 families to estimate yield, root diversity and verification cost. This pilot is proposed here and has not run.

## Proposed 20B and 30B mixtures

| Accounting lane | 20B plan | 30B plan |
| --- | ---: | ---: |
| Educational and reference prose | 6.0B | 8.0B |
| Code, repository-linked docs and tests | 5.5B | 9.0B |
| Informal mathematics and algorithms | 4.0B | 6.0B |
| Selected synthetic explanations and solution episodes | 2.5B | 4.0B |
| Formal Lean and proof repair | 0.8B | 1.2B |
| Gameplay and procedural reasoning | 1.0B | 1.5B |
| Research and mechanistic-interpretability episodes | 0.2B | 0.3B |
| **Processed non-padding tokens** | **20.0B** | **30.0B** |

These are desired allocations. Accepted source capacity remains unresolved. Within reference prose, reserve 1B at the 20B endpoint and 1.5B at 30B for primary research and technical documentation. These amounts are already included in the totals. Give each document or problem root one accounting lane, even when it supports several capabilities.

Initially assign at most half of the synthetic lane to explanatory rewrites with unverified factual content. Use the remainder for task examples carrying suitable answer, execution or verifier evidence. Changing that balance is an experimental choice to record.

For static specialist material, start with at most **four presentations per canonical root across all views**, admitting up to four distinct verified views within that total. The cap is a planning hypothesis. Held-out learning curves can support a later change. Different names or serialization styles retain their root identity.

At that cap, 30M accepted specialist tokens could contribute 120M processed tokens, or 0.6% of a 20B run. Four presentations of the same serialized Court logs would amount to about 150.6M `cl100k_base` token exposures. The admitted Aleph volume depends on projection, filtering and retokenization. The 1B gameplay allocation therefore depends on public material and newly verified tasks as well as the owned pool.

The research allocation is a ceiling. Its delivered volume comes from admissible episodes. Route any shortfall equally to protected research references and executable code/tests. An unfilled formal allocation goes equally to informal math and independently checked solution data. Gameplay shortfalls go equally to code/tests and checked procedural or solution data, with educational/reference material filling unavailable checked supply. Recompute the achieved mixture after these substitutions.

### Curriculum

Expose every admitted lane early, then increase the share of checked actions and research episodes near the endpoint. Cumulative quotas remain fixed. For example, scheduling 70% of the 200M research allocation in the final 2B tokens makes it 7% of that segment and 1% of the complete run.

Choose the 20B versus 30B endpoint before terminal learning-rate decay. Save optimizer, scheduler, RNG and sampler state at the decision checkpoint. Count any repeated training from a fork against the compute and exposure budget.

## How the small owned corpora become useful

### Proof attempts and repairs

A proof record should preserve the theorem, available premises, current goals, proposed tactic, resulting goals and final checker result. Include a brief explanation when its concrete claims can be checked. A rejected tactic can be placed in the input context for a repair target. A separate outcome label records the rejection.

Expand by changing dependency structure, needed lemmas, counterexample construction or proof strategy. Root identity should group equivalent tasks and their generated siblings. The generator's train/dev/test split must survive extraction into individual proof steps.

### Decisions in games

Derive several tasks from a replayable transition: predict a consequence, choose an action under a cost limit, detect an illegal move, update a belief, or compare plans under a changed rule. These views share the parent transition and split assignment. Exact solvers can supply action-quality targets in tractable worlds. Other games supply observed outcomes whose uncertainty remains explicit.

Keep a budget for a bounded explanation, action or prediction. Repeated lore and protocol boilerplate receive their own count. This allows us to see whether the model is learning useful decisions as training volume grows.

### Research and mechanistic interventions

Join a research question to its data assumptions, predicted outcome, registered comparison, actual receipt, independent assessment and revised next choice. The 12 prior experiments can contribute failures as well as positive results once their artifacts are recovered. Their token yield is currently unknown.

For VPD/BPD work, store the exact intervention and readout definition. A training example can ask which experiment distinguishes two explanations, predict an intervention's outcome, or interpret a null result. The measured result supplies its label. An attribution score and a teacher's explanation each retain their own evidential status.

Start new synthetic research episodes with small systems whose generating mechanism is available to the independent verifier. Include faulty split manifests, changed normalizers, invalid controls, mistaken signs, and results that disappear under matched perturbations. Preserve enough variation that a classifier cannot answer from a family name or receipt format alone.

Any transfer from those exercises to research judgment on neural models needs measurement. Evaluate on new mechanisms, new codebases and changed failure types, including the decision to seek more evidence.

## Training objectives and accounting

Use a declared causal language-modeling objective for admitted reference text and code. Checked task episodes can use response-only supervision with the appropriate prefix mask. A future pretraining adapter must represent these objective types explicitly; the current data-etiquette implementation governs verified and discovery claims and has not been changed by this report.

Acceptance for language-model training can certify provenance, source permission, integrity and split checks while the document's factual claims retain their source-level status. Outcome supervision requires the corresponding environment-owned or independent receipt. The existing distinction between model proposals and authoritative labels remains in force.

Maintain the following accounting fields for every released shard:

| Field | Purpose |
| --- | --- |
| Source revision, original ID and parent root | Reconstruct ancestry and remove duplicated descendants from held-out sets. |
| License and allowed-use record | Preserve source terms through extraction. |
| Objective and loss mask | Identify which tokens train which prediction. |
| Unique retained tokens | Measure accepted source content after projection. |
| Processed input and target tokens | Charge repeated presentations against the run. |
| Tokens receiving loss | Report the actual supervised target volume. |
| Canonical roots and structural families | Describe diversity beyond token count. |
| Verification claim, method and receipt | Scope a successful check to the fact it establishes. |
| Generation and verification cost | Charge new data against its actual resources. |

A 20B run with an average 60% target fraction contains 12B target tokens. Its model still processes all 20B input-plus-target tokens. This is an arithmetic illustration; the real target fraction needs measuring after packing. Document packing must isolate independent examples and preserve prefix/response boundaries. Long solutions need complete bounded units or a separately costed context policy.

## Difference from HRM-Text's data recipe

Sapient [reports](https://arxiv.org/html/2605.20613v1) 40B unique sampled tokens and 60B processed tokens, using task-formatted public datasets and a response-only PrefixLM objective. Its [model card](https://huggingface.co/sapientinc/HRM-Text-1B) says the released checkpoint had no code-dataset training.

Aleph's proposed allocation adds substantial code, tool outcomes, formal repair and owned research episodes. Sharing a public source remains compatible with a distinct curriculum. Claims about data efficiency would require matching objectives, token accounting and compute; the current comparison is a design choice.

The dyadic companion could review difficult mathematical steps or suggest critiques. Route its proposals through the task verifier before using them as targets. Preserve which model proposed a solution, record the paid call cost and reserve examples where the companion's advice makes the result worse. Complementarity is measured through joint task performance and error overlap.

## 20B and 30B compute sensitivity

These figures come from [`aleph_compute.py`](../estimates/aleph_compute.py), evaluated for the existing approximately 1.007B-parameter design. Assumptions: 4,096-token context; prefix fraction 0.5; H100 SXM dense BF16 peak 989.4 TFLOP/s; full finite-unroll backward; 25% runtime allowance outside active training. Model FLOP utilization is hypothetical.

| Target | Recurrent rounds | GPU-hours at 20% MFU | GPU-hours at 30% MFU | GPU-hours at 45% MFU | 8-GPU wall hours at 30% |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20B | 2 | 478.2 | **318.8** | 212.6 | 39.9 |
| 30B | 2 | 717.4 | **478.2** | 318.8 | 59.8 |
| 20B | 3 | 691.5 | **461.0** | 307.3 | 57.6 |
| 30B | 3 | 1,037.3 | **691.5** | 461.0 | 86.4 |

At 10% active MFU, two-round training requires about 956.5 GPU-hours for 20B and 1,434.7 for 30B. Small operators, communication, attention masks, recomputation and host I/O can materially affect realized throughput.

With the earlier 400-hour main-training allocation, two-round 20B needs about 23.9% MFU under these assumptions. Two-round 30B needs about 35.9%, or a different allocation of campaign hours. The 700-hour scenario also covers probes, preflight, evaluation and recovery. A second full 20B control would use another 318.8 hours at the central assumption, leaving little capacity for the remaining work.

The $1,560 balance and 700 H100-hours remain separate user-reported planning quantities. Their ratio implies about $2.23 per GPU-hour before ancillary charges. At an illustrative $4 per hour, $1,560 would purchase 390 GPU-hours. Establish the provider conversion from an actual receipt before treating either token endpoint as funded.

For a 400-hour training allocation, the measured whole-job requirement is 50M tokens per GPU-hour for 20B, or 75M for 30B. Across eight GPUs those are about 111,111 and 166,667 tokens per second. Whole-job timing already contains its observed stalls and checkpoint costs, so applying another 25% allowance to that measured total would double-count them.

Generation, proof checking and mechanistic probes need separate meters. Prefer released suitable data for bulk coverage and direct new generation toward missing verified task families. Store accepted unique tokens per CPU-hour and per teacher GPU-hour, together with rejection reasons.

## Selection and evaluation plan

Create a small frozen development suite around Data Goblin's work: repairing a corpus manifest, implementing a gym with an independent grader, interpreting a failed run, comparing mechanistic claims, and choosing a bounded next experiment. Track success, invalid releases, executable correctness, calibration, token use and the cost of frontier escalation. Add conventional language, code and math evaluations for regressions.

Reserve source repositories, theorem ancestry, game episodes and generator families before expansion. Keep original documents and their synthetic rewrites in the same split. Preserve legacy evaluation partitions unless their evaluation role is explicitly retired in a separate decision. A new name or seed cannot establish an unseen mechanism.

At scheduled development checks, compare small matched data branches from the same checkpoint. Hold the architecture, token budget and loss objective fixed while changing one data choice. A repeated fixed evaluation suite belongs to development; final sealed tests retain independent custody. Short branch results support a local choice and carry their measured scale in the report.

A mechanistic signal can suggest where to seek more data. For example, an independently repeated intervention failure might nominate a new family of state-update tasks. Confirm its usefulness through held-out improvement per added token, with a matched selection control. Save the selection decision and its evidence so later gains can be attributed to the actual training history.

## Concrete next preparation

1. Resolve the external Court and Mesh shards and join their receipts to hashes. Use the committed TRM export to establish the projection format.
2. Freeze Aleph's tokenizer after checking compression on prose, code, Lean and structured game state. Count a stratified sample before assigning final byte/download budgets.
3. Pin the public sources and choose one initial supplier per major lane. Match derivatives to their shared roots and preserve permitted source metadata.
4. Extract train-only owned proof states and replay a sample. Build the 10,000-example procedural yield pilot under declared local caps.
5. Release a content-addressed corpus manifest with achieved quotas, repetitions, target fraction and exclusions. Keep raw content and held-out labels in their intended workspaces.
6. Measure the real distributed training path, including restart parity and billed occupancy. Use the measured throughput to choose the token endpoint before final decay.

Validation reproduced all 24 compute cases and both exact quota sums. The repository suite passed all 162 tests with `PYTHONPATH=src python -m pytest`; Ruff 0.17.5 reported all checks passed. Pytest emitted one deprecation warning from its installed AnyIO plugin. Corpus intake and measured GPU throughput remain pending.

[court]: https://github.com/MoralityLabAI/AICourt/blob/dad297ef6c103f57bf0bbcd90614c4c110b4c504/results/court-balance-ensemble-2000.json
[court-method]: https://github.com/MoralityLabAI/AICourt/blob/dad297ef6c103f57bf0bbcd90614c4c110b4c504/BALANCE_REPORT.md
[court-earlier]: https://github.com/MoralityLabAI/AICourt/blob/dad297ef6c103f57bf0bbcd90614c4c110b4c504/results/court-corpus-validation.json
[court-pilot]: https://github.com/MoralityLabAI/AICourt/blob/dad297ef6c103f57bf0bbcd90614c4c110b4c504/MODEL_AGENT_PILOT.md
[campaign]: https://github.com/MoralityLabAI/AICourt/blob/dad297ef6c103f57bf0bbcd90614c4c110b4c504/CAMPAIGN.md
[mesh]: https://github.com/MoralityLabAI/Control-Harness/blob/5d9556097e5d0df9e61b5fd6b258d1c27bbfd382/results/mesh_escape_pit/corpus_v1_progress.json
[mesh-plan]: https://github.com/MoralityLabAI/Control-Harness/blob/5d9556097e5d0df9e61b5fd6b258d1c27bbfd382/results/mesh_escape_pit/corpus_plans_v1/plan-1000000000.json
[trm]: https://github.com/MoralityLabAI/TRMStoryworld/blob/a1a4141adb6401a21d322479a96cddfe858c8961/zip_this/qblot_export/trm_rollout_corpus_existing_manifest.json
[observability]: https://github.com/MoralityLabAI/Control-Harness/blob/5d9556097e5d0df9e61b5fd6b258d1c27bbfd382/control_observability_pilot/README.md
[ontology]: https://github.com/patrickdugan/OntologicalProofs/tree/1c209654bf03b6eef62324edc10a9393987c6b33
[mathcan]: https://github.com/patrickdugan/MathCan/tree/9b2e9f5a258b2a0bfa31489333e1e5b9b86b7566
[evalmath]: https://github.com/MoralityLabAI/EvalMath/tree/bed334f032d31d777bad0e5152e732f57252092b
[semaev]: https://github.com/patrickdugan/semaev-index-calculus/tree/3c630e179ae837dad839949dcb03cf4fdb41815e
[lr01]: https://github.com/MoralityLabAI/LoopedTransformers/blob/faddf809f9a85b64c9b36869963df1c480053675/experiments/lean_reasoner/README.md
[bpd-lean]: https://github.com/MoralityLabAI/LoopedTransformers/blob/faddf809f9a85b64c9b36869963df1c480053675/experiments/bpd_lean_curriculum_v2/KERNEL_AUDIT.md
[ram17]: https://github.com/MoralityLabAI/LoopedTransformers/blob/faddf809f9a85b64c9b36869963df1c480053675/experiments/lean_ram17/RAM17_RESULT_20261116.json
