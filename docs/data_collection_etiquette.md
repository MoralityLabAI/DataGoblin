# Data collection etiquette and LDT sorting

This subsystem turns private data intake into a receipted corpus decision without making the sorter an authority. The
initial executable is a deterministic metadata-only control. It validates the boundary, dataset splits, labels, metrics,
and larger-model handoff; it does not train an LDT or load model weights.

## The boundary

```text
private raw intake
       |
       v
public envelope: content hash + provenance + consent/license/privacy metadata
       |
       +--> deterministic policy --------------------------+
       |                                                    |
       +--> LDT recommendation + confidence + flags        v
                                               independent steward
                                                        |
                       +--------------------------------+------------------+
                       v                                v                  v
             verified supervision              soft discovery       evaluation custody
                  weight 1.0                     max weight 0.25          weight 0.0
                       |                                |                  |
                       +-------------------- larger-model trainer         +--> sealed evaluator
```

Raw content remains in an operator-owned private workspace. Public research state contains only typed metadata,
content hashes, and provenance references. It must not contain raw bytes, filesystem paths, secrets, hidden reasoning,
or model weights. A deletion request rejects the item and requires deletion of private content while retaining only a
tombstone and hash receipt.

Each envelope also carries a purpose ID, lawful basis, allowed uses, data-residency region, sensitivity class, finite
retention, and deletion state. A new purpose is a new adjudication: an item collected for evaluation cannot silently be
reused for training, and cross-region materialization requires explicit operator approval.

## Dispositions

| Disposition | Meaning | Training use |
|---|---|---|
| `REJECT` | prohibited or deletion-requested material | none |
| `QUARANTINE` | provenance, consent, license, redaction, or dedup work remains | none |
| `DISCOVERY_ONLY` | model, simulation, experience, or human-review claim | proposals and critiques only; max weight 0.25 |
| `VERIFIED_TRAIN` | independently or environment-verified clean material | authoritative supervision, weight 1.0 |
| `EVALUATION_ONLY` | ordinary or OOD evaluator material | never training |
| `SEALED_EVALUATION` | evaluator material whose labels are withheld until predictions exist | never training |

Verification receipts name their custody and are bound to the exact content SHA-256. Proposer-owned receipts remain
claims and cannot release training data. The LDT may recommend any disposition, flag risks, estimate confidence, or
abstain. It cannot release an item, create an
authoritative label, or convert proposer evidence into verified evidence. Only an environment-owned verifier or an
independent harness can hard-promote an item. Human review can resolve clerical questions but is not itself independent
verification.

## LDT training contract

The proposed LDT consumes public envelopes only and produces disposition logits, multi-label violation logits, and an
abstention score. The exact LDT implementation or local checkpoint is intentionally not selected in this repository.
That operator choice must be recorded before a real run.

The fixed splits are train, calibration, ordinary test, OOD test, and sealed poison. Selection metrics are critical false
release rate, violation-flag micro F1, and Brier score. Accuracy, OOD accuracy, and sealed critical false releases are
evidence metrics, not tuning signals. A candidate cannot promote unless split contamination is zero, sealed critical false
release is zero, resource receipts are complete, and the independent steward signs off. These are initial conservative
gates, not evidence that a model is generally safe.

Every real training process must use the hard-cap plan in `configs/data_etiquette.yaml`: 2,048 MB RAM, 50% CPU, 50 MB/s
I/O, 900 seconds, zero swap, batches no larger than 32 envelopes, checkpoints every 100 steps or 300 seconds, and the WSL
cgroup wrapper. Abort artifacts are valid results. No checkpoint is auto-downloaded.

## Larger-model enterprise handoff

The trainer consumes only a steward-released, content-addressed handoff. The verified and discovery lanes remain
separate. Evaluation examples are represented only by commitments and never appear in a training file. A larger-model
run additionally requires a local checkpoint, a separate approved resource plan, split and dedup receipts, deletion
tombstone checks, the LDT disagreement report, and sealed evaluation owned outside the trainer workspace.

Use the deterministic control offline:

```bash
datagoblin etiquette-demo --seed 7 --out runs/etiquette-demo
```

Build a fit dataset whose outer ML partitions are independent of each record's enterprise-use role:

```bash
datagoblin etiquette-build-training-dataset --seed 17 --out runs/ldt-fit/dataset
```

The trainable candidate is a learned public-envelope proposal head followed by the explicit LDT policy membrane. It is
not the unlicensed upstream implementation, and it must never be called directly. Run it only through
`scripts/run_etiquette_ldt_capped_wsl.ps1`, which stages public data on WSL's block-backed filesystem and invokes the
systemd cgroup wrapper. The trainer excludes unique identifiers, hashes, provenance values, synthetic category names,
sealed labels, raw content, and cross-partition duplicate groups from fitting.

After the capped run, the environment-owned evaluator and auditor are separate commands:

```bash
datagoblin etiquette-evaluate --dataset-manifest runs/ldt-fit/dataset/dataset_manifest.json \
  --predictions runs/ldt-fit/training/capped_run/predictions.jsonl \
  --sealed-labels runs/ldt-fit/private_steward/sealed_labels.jsonl --out runs/ldt-fit/evaluation
datagoblin etiquette-audit-candidate --evaluation-report runs/ldt-fit/evaluation/trained_sorter_evaluation.json \
  --run-receipt runs/ldt-fit/training/run_receipt.json --out runs/ldt-fit/promotion_decision.json
```

Critical false release counts only non-abstained `VERIFIED_TRAIN` recommendations. Abstentions remain in the adjudication
queue and reduce coverage. Thresholds chosen after seeing results are diagnostic only and cannot promote a candidate;
promotion requires a preregistered held-out confirmation and independent steward signoff.

## Evidence receipt registry

A collection record cannot authorize itself by copying `verified_by`, custody, and a receipt-looking hash into its public
envelope. Hard evidence is accepted only when the exact receipt is present in a separately supplied steward store and
validates against the frozen `configs/evidence_verifiers.yaml` registry. The v2 receipt commits to the content subject,
evidence type, verifier, custody, proposer, verdict, challenge, observation, replay specification, and replay artifacts;
its canonical hash must equal the hash carried by the collection record. Registry drift, unknown methods,
self-verification, non-passing verdicts, missing artifacts, or cross-subject receipt replay fail closed to `QUARANTINE`.
Receipts also declare claim types, and a policy requests the exact claims it needs. A receipt for content integrity,
for example, cannot silently stand in for policy eligibility or environment outcome validity.

The registry/store boundary is local operator authority, not a distributed signature system. A filesystem administrator
can replace both, so release receipts should retain the frozen registry hash and private steward custody. Public corpora
receive hashes and decisions only, never private replay artifacts or local paths.

Key outputs are `public_dataset/dataset_manifest.json`, `sorter_predictions.jsonl`, `sorter_evaluation.json`,
`adjudication_queue.jsonl`, and `larger_model_handoff/corpus_handoff_manifest.json`. The `private_steward` directory is a
demonstration of the custody boundary and must be outside the trainer workspace in a real run. Generated runs, hidden
seeds, private content, and model weights must never be committed.
