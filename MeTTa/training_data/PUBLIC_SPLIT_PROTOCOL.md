# MeTTa-04: public development root splits

This audit partitions the existing native-agreed MeTTa query **candidates** so a
development learner can be measured on whole unseen source files. It does not
approve any packet for training and does not manufacture a sealed benchmark.

## Scope and current inventory

- 187 source-conditioned query/expected-expression packets, observed on
  Hyperon 0.2.10 and still marked `training_ready=false`.
- 5 exceptional native observations, including one source root with no passing
  SFT candidate.
- All candidate output expressions are committed in the public source tree.
  Consequently, `dev_public` and `test_public` are **open diagnostics**, not
  uncontaminated evaluation labels.
- The existing source import, byte hashes, source catalogs and SFT packets stay
  byte-identical. The new plan contains only root identities, aggregate counts,
  source manifest digests, and release-gate declarations.

## Generate the plan

```sh
python MeTTa/training_data/scripts/public_root_splits.py \
  --out runs/metta/public_root_split/plan.json
python -m unittest discover \
  -s MeTTa/training_data/tests -p 'test_public_root_splits.py' -v
```

The auditor independently reruns the source-annotation/Git-blob provenance
validator. It checks each candidate ID, original source identity, Hyperon runtime
version, source-context hash, answer hash, byte budget metadata, unassigned
split, and quarantine flags. Exception IDs must be distinct from passing
observations, with failed or unobserved assertions kept out of SFT packets.

Source roots are joined when exact query text, query/answer pairs, or exact
context hashes are shared. Joined roots receive one partition. The deterministic
weighted allocation aims for 70/15/15 candidate counts while keeping each
root in a single public partition. An exception-only root stays in
`exception_quarantine`. The plan keeps these distinctions explicit:

- `train_candidate`: unapproved candidates for future training admission.
- `dev_public`: source-disjoint development diagnostics.
- `test_public`: source-disjoint public regression diagnostics.
- `exception_quarantine`: roots with only failed, unobserved, or unsupported
  candidate assertions.

The manifest includes SHA-256 digests of both input JSONL files and the seed
for deterministic regeneration. Changes to either upstream dataset invalidate
the old partition receipt. Exact deduplication can miss equivalent queries with
rewritten symbols, changing comments, or semantically identical programs.

## Next independent-verification gate

Before a training run, verify a fresh portion of MeTTa tasks with a separately
specified semantic oracle and a fixed runtime environment. Record successful,
failed, and halted repair steps with source commitments and verifier receipts.
The next *sealed* evaluation needs novel source roots and targets whose answers
were never published or presented to the learner.

Recommended model comparison: matched 8B or smaller recurrent and nonrecurrent
baselines at fixed query context and inference budgets; log per-root query
accuracy, oracle calls, correction steps, and voluntary stopping. Preserve
DataGoblin's source-rights and policy controls for any promoted training rows.
