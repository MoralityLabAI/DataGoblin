# Recover Aleph training inputs from recent research

The original 10M snapshot at 500a572 remains missing. Its 54,713,583-byte ZIP and SHA-256
8042336235ba39ae1786416d55d7fdf3f641bd2ed43c609b2ef11e59c3ea25b1 are recorded,
but the final raw data were not preserved in the reviewed PR tree.

**The earlier mixed pilot is in Git.** Evidence branch
[evidence/1b-pretrain-v1-20261124](https://github.com/MoralityLabAI/DataGoblin/tree/evidence/1b-pretrain-v1-20261124)
contains the [5,622,851-byte binary ZIP](https://github.com/MoralityLabAI/DataGoblin/blob/48c99a7d9c2d37c82fe127369dfa0124b88bc8ab/assets/evidence/DataGoblin-1B-Pretrain-Evidence-v1.zip).
The exact Git blob is c8b859ba326bb34fd48ef30ded6588ed0c2d4341. This archive includes
historical evaluation roles. Extract it in a custody workspace, not the trainer's
data directory. See ../../aleph_rlt_placement/RESUME.md.

## Materialized or reproducible inputs

| Source | Recorded scope | Recovery status |
| --- | --- | --- |
| DataGoblin mixed pretrain evidence | Native text, Lean candidate receipts, graph revision | Actual pinned archive; source for the original pilot |
| LoopedTransformers loop schedule architecture discovery | 7,507 train records, 64-token task encoding | Actual split-specific train.jsonl; rights and label semantics need new admission |
| LoopedTransformers RAM11F | 76 complete scripts, 230 positive prefix-action labels, 22 theorem roots | Actual train curriculum; compile/trace evidence, prefixes are not fresh proof states |
| LoopedTransformers LRM-REV-02 | 8,192 training episodes, four runs | Actual gzip train shards; original independent audit regenerated identities and Bayes targets |
| LoopedTransformers LBF18 | 64 train reference certificates, 24 counterfactual certificates | Actual train Lean files with archived compiler receipts; recheck in target intake |
| BPD run 44912385166, artifact 14709763227 | 96 STOP/CONTINUE rows, 96 outcome rows, 45 repair preferences | Live train-only artifact at audit; import using DataGoblin docs/rlt_pretrained_trm_exit_ingest.md |
| DataGoblin COL-DG-01 (PR 73) | 3,153 train, 1,082 dev records in completed run | Reproducible generator and checker; regenerate train-only, do not count dev |
| DataGoblin RLT rollout gym PR 2 | 16,052 views, 1,230 train roots | Local corpus reported; code and receipt tracked, full ZIP not found |
| DataGoblin hypergame PR 6 | 12,452 train presentations across six objectives | Local corpus reported; private replay key needed for original dataset |
| LoopedTransformers LC-GRT v1 | 24,000 planned train theorems, per-shard certified receipts | Manifest and generator found; original raw train shards absent from main tree |
| DataGoblin MeTTa native query pairs | Raw candidate JSONL and licensed source/receipt tree | Actual files; require role, source rights and overlap admission before adding |

Do not count repeated views as independent theorem roots. Use the strict pinned
SmolLM2 tokenizer to count positions and valid labels. Keep historical held
roles, sealed LBF proofs, calibration/evaluation trajectory JSON, and
benchmark outcome logs outside training. Re-run source verifiers and check
cross-source family, canonical root, and semantic duplicates. MeTTa assertions
and benchmark data are not automatically verified target labels.

## Recover the accessible train sources

From this PR's checkout, run:

    python -m pip install -e .
    PYTHONPATH=src python training/aleph_next/recovery/recover_sources.py --out runs/aleph-recovery-sources

The same bounded, zero-accelerator-credit public CI job materializes a candidate
source packet using the exact original evidence blob and exact LoopedTransformers
commit b14a51015996f0dee81ef033e2e198d9f84951b2. The script verifies Git object
identity, sizes, source manifests, historical mixed loader and expected record
counts. It exports train-role files only. All source inputs are permanently
referenced by Git commit; the Actions ZIP is a convenience copy with expiration.
The packet earns zero production quota until the new intake passes.

## Next snapshot

The original 10M reconstruction also depends on the original CONFIRM14 text
panel and source cache, tournament allocation logs, and private custody inputs.
A public reconstruction may recover much of the material, but the missing
original commitments cannot be silently replaced. Since the original direct A
ran zero optimizer updates, a separately versioned new initial 10M snapshot is
an appropriate option. Preserve 25/35/20/15/5 position shares and the production
recipe. Register new frozen batch digests and checkpoint identity; never replace
the old MILESTONE10M.json or weaken its hash gates. Generate new prospective
evaluation roots under separate custody. Publish the final train-only corpus as
a durable Git release or an operator-approved persistent artifact and verify
fresh extraction before authorizing a real run.
