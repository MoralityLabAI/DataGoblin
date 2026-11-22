# MeTTa-AIRIS-02: fresh-root curation repair students

**Research status:** small original NumPy models trained on independently replayed synthetic-world oracle labels in an isolated, bounded CPU research fit. The public MeTTa corpus and data admission remain unchanged. This is a direct continuation of [AIRIS-01](AIRIS_RESULTS_20261122.md).

## Root custody

The earlier 3,072-root MeTTa curriculum used seeds 0–511 from six families. AIRIS-01 also used a small internal confirmation slice of those roots. AIRIS-02 registers a new root namespace, metta-airis02-v1, with **disjoint seed bands** for five familiar families:

| Source seeds per family | Intended role | Label exposure |
| --- | --- | --- |
| 512–703 | Synthetic research training | Bounded Python oracle, after full source replay |
| 704–767 | Public development diagnostics | Exact environment labels for post-fit scoring |
| 768–1023 | Registered public test reserve | **Root identities only; no worlds or labels opened** |
| 1024–1151 | Registered steward reserve | **Root identities only; no worlds or labels opened** |

No conflict-family OOD roots are included. The root registrations are deterministic and publicly regenerable; this is **not a cryptographically sealed evaluation**. A genuine independent steward pool should use separate private source data and a keeper-controlled verification process.

Each source root contributes at most **four presentations across all selection policies**, with one fresh intervened world for each of uniform, motif, budget-first, and MOSES-gated selectors. Every arm sees the same root IDs and source family distribution. The original MeTTa-SFT training corpus receives zero additional presentations. No prefabricated reference-sample or external licensed source is trained.

## Data interventions

The source world comes from the independent synthetic curation generator. Every arm receives the same initial evidence selection and the bounded candidate intervention set from AIRIS-01. A strict single-fact check regenerates the public MeTTa program before the exact finite subset oracle supplies the next canonical add/remove/STOP action. All four arms are evaluated on the same number of source roots.

The **MOSES-gated control** uses the frozen small program from the earlier train-only result: if selected set size exceeds two or revision keys collide, choose the budget-first mechanism; otherwise prefer lineage, then eligibility. It is a hand-specified mapping from a frozen observational classifier to a control intervention. It does not implement native MOSES action evolution or causal reasoning. Motif ranking was selected on the preceding seed-0–511 train roots; no new development labels are used in selecting that policy.

The v1 student pilot exposed an action-validity and stopping failure: all 60 public development episodes in each arm attempted an inapplicable edit, and none completed a solve with STOP. The [archived pilot](AIRIS02_PILOT_V1.md) preserves that result. The v2 corrective study uses a *public structural membership mask*: ADD targets must be absent candidate IDs and REMOVE targets must already be selected; STOP remains legal. This removes malformed membership moves from the decoder without granting the verifier's constraint or optimality decisions to the model.

A deterministic root-hash schedule designates about one fifth of fresh roots for fixed-point teaching. On those roots, the independent oracle's optimal set replaces the initial selected set **within the existing one presentation per arm**, giving genuine positive STOP targets without extra root exposures. Its origin is recorded in the row and rechecked by the source-root auditor. Intervention selection still uses the independently sampled source state. All four arms use the same fixed-point-root schedule, while their optimal sets can differ after their distinct one-fact interventions.

The student receives a fixed-length, explicitly structured public-feature vector. It contains candidate presence, selected status, eligibility, costs, quality, revision epoch, lineage collisions, prerequisite counts, topic coverage and coarse global resource quantities. Oracle outputs, optimal sets and target actions are excluded from these input features.

## Independent fit gate

The dataset is stored under ignored runs/ and remains training_ready=false. Before any research fit, airis02_audit.py replays every synthetic source world and teacher label against the original bounded Python oracle. It checks immutable root identities, seed bands, hash-bound artifacts, exactly four arm presentations per root, no OOD source participation, the frozen motif and rule identities, and absence of world contents or teacher labels in reserved registries.

The gate returns synthetic_research_fit_eligible=true for this isolated toy experiment. That is a narrow experiment permission and never turns a row into reusable verified training material. The DataGoblin EtiquettePolicy and ResearchMembrane keep authority for real-source training admission.

## Initial student experiment

The code trains two tiny randomly initialized models entirely on CPU, with a legal-action-masked cross-entropy objective:

- A direct linear 21-class repair-action predictor.
- A weight-tied recurrent tanh model with one shared hidden transition trained by exact manual backpropagation through four iterations. The same frozen weights are evaluated at depths 2, 4, and 8.

For each intervention arm, models receive exactly the same architecture, initialization seed, optimizer steps and root count. **All trained models are scored on the same set of uniform-intervened public development worlds**, which controls the evaluation-distribution confound. Within-arm diagnostics are retained separately. The direct and recurrent architectures have different parameter counts, which are recorded; their relative performance is descriptive. This is a **real small CPU training run** on synthetic worlds, with no pretrained weights, GPU lease, hidden reasoning traces, or native Hyperon execution. STOP is present in the repair-action classifier, and v3 adds a **separate binary learned STOP head** on the same public features. Its threshold is frozen from negative TRAIN examples, with a finite-sample 5% false-stop cap on those training examples. Both the original action-class STOP and the conservative learned gate are scored on the identical public development worlds.

The binary head makes an *action-level* stopping proposal. It does not implement adaptive computation-time halting: recurrent depth stays fixed and a separate cost is recorded for every STOP-head invocation. [The v1 negative pilot](AIRIS02_PILOT_V1.md) and [v2 premature-stop result](AIRIS02_PILOT_V2.md) remain separate, immutable development receipts.

The evaluator obtains the exact oracle independently of the model's predictions. Masking checks only public candidate membership, leaving the oracle's eligibility, dependency, budget, and optimality constraints as genuinely independent grading criteria. At up to eight proposed repairs it records first-edit correctness, completed solutions with explicit STOP, premature stopping, inapplicable edits, unsafe additions, final valid sets, goal coverage, remaining oracle edit distance and actual internal inference loops. Incomplete and invalid rollouts count against the model.

Equal examples, fixed vector width and optimizer updates do not establish equal source-token budgets. The original public MeTTa program strings vary in length, and no native tokenizer parity is claimed.

## Reproduction

Run from DataGoblin's repository root with the normal Python developer dependencies:

    python -m MeTTa.airis02_fresh --train-roots 32 --dev-roots 12 --out runs/metta/airis02_fresh
    python -m MeTTa.airis02_audit --bundle runs/metta/airis02_fresh --out runs/metta/airis02_fresh/audit.json
    python -m MeTTa.airis02_student --bundle runs/metta/airis02_fresh --steps 48 --hidden 18 --out runs/metta/airis02_fresh/student_development.json

The dedicated GitHub workflow runs these CPU steps and uploads **aggregate manifests and model score receipts only**. Source-world teacher datasets and public development labels remain ephemeral job files and are excluded from published artifacts.

## Next research gates

The development results should be treated as *public engineering diagnostics*. Before a transfer or recurrence claim, freeze the architecture and policy choices; have an independent steward evaluate on new, separately protected source roots; compare exactly matched model parameter counts, training-token totals and compute, and include a learned halting head with calibrated continuation targets. A positive gain must clear verified completed tasks and false-admission safety, including errors. Training on third-party MeTTa source text remains subject to source-specific licensing, provenance and ResearchMembrane decisions.

Related issue: [DataGoblin #31](https://github.com/MoralityLabAI/DataGoblin/issues/31). 
