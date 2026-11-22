# MeTTa-SFT-02: curation curriculum for looped transformers

**Status, 2026-11-22:** synthetic CPU exact-oracle training candidates. Model training, native Hyperon execution and held-out learned gains remain open.

This extends [DataGoblin MeTTa-01](README.md) beyond MCP or code retrieval. The learner must curate evidence and decide how many recurrent correction steps a selection needs. The MeTTa program describes a small research corpus. DataGoblin provides an independent bounded collection oracle, while the model learns a next-edit policy and when to stop.

## Generated task families

| Family | MeTTa relation | Curation behavior |
| --- | --- | --- |
| `lineage` | `Lineage`, `Eligible` | Deduplicate related evidence and avoid counting replicas as independent |
| `coverage` | `Provides`, `Goal` | Assemble a concise set covering required research capabilities |
| `dependency` | `Requires` | Repair missing prerequisites before admitting a dependent record |
| `revision` | `Key`, `Epoch` | Prefer the latest admissible revision and reject stale versions |
| `budget` | `Tokens`, `Budget` | Optimize goal coverage, quality and context use under hard limits |
| `conflict` | `Conflicts` | Detect mutually incompatible evidence in the fully held-out family |

Every world contains seven to nine synthetic evidence records, a MeTTa fact store and a native-shaped `match &self` query. Records carry source IDs, lineage, quality, token cost, goals and occasionally dependencies, revisions or conflicts. The exact reference maximizes the lexicographic tuple **(goals covered, summed quality, negative token cost)** subject to eligibility, lineage uniqueness, dependency closure, revision ordering, conflicts, maximum records and token budget. The Python reference exhaustively enumerates all permitted subsets. No language model supplies labels or authorization.

The generated `.metta` strings are syntactic teaching scaffolds. The separate Python collection oracle enforces the rules. It does not prove that Hyperon executed the program or that a model understands MeTTa semantics. A later native MeTTa replay must test the `match` results and record interpreter version, exit status and actual bindings.

## Canonical outputs

The generator writes one canonical source of supervision with no more than **four distinct presentations per root**:

- `train_candidates.jsonl`: learner-visible MeTTa facts and current selected records, a next `add/remove/stop` target, a chosen/rejected edit pair, exact-oracle receipts and recurrent-step targets. Every example has `training_ready=false`, `evidence_type=SIMULATED`, a stable root/presentation identity and a declared oracle scope.
- `reserved_public.jsonl`: unlabeled development, test and OOD prompts. These are public heldouts for development engineering, rather than cryptographically sealed independent evaluations.
- `manifest.json`: generator/oracle SHA-256, split counts, exact artifact hashes, stop/continue counts, depth summaries and training-custody gates.

A secondary `metta_query_supervision` field expresses a conjunctive, native-shaped `match &self` query for eligible source records that provide a requested skill. Its expected record IDs come from structured source facts. It is a second objective **on the same presentation**, rather than additional exposure of a root.

The loop targets are *synthetic edit distances and edit transitions*. They are labels for an observable repair controller. They must not be described as actual internal TRM thinking steps, a learned ACT schedule, or evidence of inference-time recurrence benefit.

## Build, replay and export

Run from the DataGoblin root after a development install:

```bash
python -m MeTTa.looped_sft build --roots-per-family 512 --out runs/metta/looped_sft
python -m MeTTa.looped_sft audit --out runs/metta/looped_sft
python -m MeTTa.export_looped_frames \
  --bundle runs/metta/looped_sft \
  --out runs/metta/alternate/looped_frames.jsonl
```

The default covers **3,072 canonical roots** from six families. Train/dev/test are hashed at root level across five familiar families. All 512 `conflict` roots are OOD. Only training-candidate roots receive oracle labels, with a four-presentation limit. Development/test sources never contribute corrections or preferences to the training export.

The alternate frame adapter follows [LoopedTransformers' documented SFT schema](https://github.com/MoralityLabAI/LoopedTransformers/blob/main/docs/sft_export.md), with `instruction`, `input`, `output` and `metadata`. It is an **alternate serialization of the same presentations**, rather than a second data augmentation stream. A trainer should consume one view at a time, keep root-grouped batches, and count repeated presentations across all objectives.

## Suggested training comparison

Train a fixed-capacity model on one approved source manifest. The initial experiment can compare direct one-pass action prediction, fixed two/four/eight recurrence passes, and a separate learned stop head, under the same optimizer, root splits, tokens and wall-clock budgets. A deterministic policy and a task-score oracle remain controls.

At each observed visit, measure exact next-edit accuracy, valid selected sets, final goal coverage, edit-distance reduction, unnecessary continuation after a correct fixed point, premature stopping, and average loops per successful solution. A strong next-edit score alone will not establish better curation if it increases false admission or fails on held-out source conflicts. Report actual provider/accelerator cost and use a frozen tokenizer for accepted-token counts.

For DataGoblin's RLT gym, keep failed proposals, independent verifier feedback and minimal repairs under the same root ID. Train an optional query-generation objective on `metta_query_supervision` only after native interpreter result checks. Hold all OOD roots out of SFT and preference training. Once a genuinely fresh steward-held pool is available, evaluate transfer to real MeTTa files, formal Lean data, math proofs and storyworld/agent trace curation.

## Authority and reproducibility

Existing `EtiquettePolicy` and `ResearchMembrane` control actual data admission. These generated worlds have an exact **synthetic** oracle; they do not qualify real external content for `VERIFIED_TRAIN`. Existing third-party license review, ancestry checks, deletion propagation and independently checked labels apply unchanged to any promoted output.

A separate GitHub Actions workflow runs CPU tests, generates a complete bundle, replays every training target and reserved prompt, checks four-presentation custody, and uploads the two alternate serialization views with their manifests. The repository retains only generator code and documentation; generated training material belongs in ignored `runs/` or the CI artifact.
