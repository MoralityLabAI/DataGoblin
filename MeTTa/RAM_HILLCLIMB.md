# MeTTa-10: RAM-only TRM / typed-LDT training hill climb

This gym consumes only CPU memory and uses the existing AIRIS-02 NumPy
learners for real, bounded gradient updates. It produces checked decision
traces for DataGoblin's RLT and LoopedTransformers-compatible SFT consumers.

**Scope of neural claims:** The TRM arm is an actual, small, weight-tied
recurrent tanh student with manual BPTT. The LDT arm trains a separate linear
proposal network and projects additions through the existing environment-owned
typed-constraint checker. It is an **LDT-style lattice refinement proxy**, not
a trained Lattice Deduction Transformer or a native transformer architecture.
The arms have unequal parameter counts. The first experiment measures repair
learning and intervention selection on synthetic curation worlds.

## RAM and compute envelope

- User machine: about 16 GB available RAM, CPU only, no GPU requirement.
- Default process budget: **8192 MiB**; upper configurable bound **12288 MiB**.
- Linux/WSL enforces an OS `RLIMIT_AS` virtual-address-space cap. Native Windows
  uses `psutil` for measured working-set accounting; run under WSL if you
  require the hard process address-space limit.
- NumPy, one BLAS/OMP thread; no torch, CUDA, checkpoint downloads or weights.
- At most 24 training roots and eight public development roots per family.
  The default uses eight training and three development roots per family.
- Trainer limits: 12–32 optimizer steps, recurrent hidden width 8–24,
  train loops 2/4/6, inference loops 2/4/6/8. A train-calibrated binary
  STOP head is a separate, optional learned control; its extra gradient
  updates and parameters are recorded. The proposal search permits at
  most eight candidate edits per run.
- The CPU models are deliberately tiny. The runtime receipt includes peak
  process resident memory, parameter counts, actual gradient updates,
  checkpoint hashes, wall time and verifier calls.

## Evaluation families and root custody

The registered public synthetic evaluation families are:

| Family | MeTTa facts and evaluation objective |
| --- | --- |
| Lineage | Avoid overcounting repeated or related records |
| Coverage | Choose records that cover requested evidence goals |
| Dependency | Resolve prerequisite constraints |
| Revision | Maintain latest viable source versions |
| Budget | Use limited context and record selection budgets |

The `conflict` family stays a separate OOD reserve. This is five variants
of a common exact **synthetic curation** task, not five native benchmark
implementations. Other requested evals (Lean theorem proving, boolean
worlds, Sudoku, MCP tool routing, novel LDT long-horizon tasks) require
task-specific adapters and independent evidence receipts.

TRAIN roots use the new `metta-ram10-v1` namespace with source seeds
20000 onward. Public DEVELOPMENT roots start at seed 30000. A registered
public test reserve starts at seed 40000; only its root **names** are
constructed. Existing AIRIS-02, MeTTa-SFT-02, native Hyperon query and
steward-held datasets are left untouched. Source-root presentations are
limited to four across all exported training views, with no new roots
silently admitted into the general corpus.

The exact Python `sft_oracle` enumerates feasible evidence sets and
produces the next canonical add/remove/STOP action. Model proposals see
only the public facts, selection and structural membership mask. The
LDT proxy's host checker vetoes ADD operations that introduce a new public
constraint violation, counts the extra verifier cost and never reveals
the canonical optimum to the model.

## Running the real CPU fits

From DataGoblin repository root:

```sh
python -m pip install -e ".[dev]"
# Native Windows: install psutil or use WSL2 for a hard virtual-memory cap
python -m MeTTa.ram_hillclimb \
  --train-roots 8 --dev-roots 3 \
  --max-proposals 4 --ram-limit-mib 8192 \
  --out runs/metta/ram_hillclimb

python -m MeTTa.ram_hillclimb_export \
  --bundle runs/metta/ram_hillclimb

# Optional genuine MeTTa syntax/result replay of host-created proposals
python -m pip install 'hyperon==0.2.10'
python -m MeTTa.ram_hillclimb_native \
  --steps runs/metta/ram_hillclimb/hillclimb_steps.jsonl \
  --out runs/metta/ram_hillclimb/native_proposal_receipt.json
```

A run independently refits from the same initialization and frozen training
roots for every proposed configuration. Each proposal changes **one** bounded
hyperparameter field. Acceptance requires an improved lexicographic
public-development score, and rejects any increase in unsafe or
new-violation additions. The initial uncalibrated pilot yielded zero
solves and premature STOP in all 10 development tasks per arm. The
next search therefore prioritizes a separate train-only STOP classifier
with a conservative false-stop threshold. This action head is not
adaptive inference-time depth halting. The ordering considers verified solved-and-stopped,
constraint safety, first-edit correctness, premature stops, verification
cost, remaining edits and inference iterations. No candidate can tune on
the registered test root contents.

The `report.json` includes baseline/champion config and family-by-family
results for both architectures, plus the current `next_teacher_prompt` with
knob choices and aggregate public-development metrics.

## Using a GPT teacher in ChatGPT

The model is an offline **proposer**. It chooses training hyperparameter
changes or a MeTTa construction, while DataGoblin's local CPU training and
independent verifiers decide the reward. You can use GPT-6.2 Sol in a
ChatGPT conversation to produce proposed edits without a local LLM.

After a CPU run, paste the `next_teacher_prompt` object from `report.json`
to your chosen GPT model with this instruction:

> Choose the next single CPU training change for MeTTa-10. Optimize
> independently scored verified completion and stopping; account for
> unsafe additions and verifier checks. Return exactly one JSON object
> containing `arm`, `field`, `value`, `source` and `model_id`.
> Use only the allowed grammar. Explain hypotheses separately if needed.

Collect the JSON edits, one line per proposal, in a JSONL file:

```json
{"arm":"trm","field":"lr","value":0.08,"source":"chat_import","model_id":"user_reported_model"}
{"arm":"ldt_proxy","field":"steps","value":24,"source":"chat_import","model_id":"user_reported_model"}
```

Then run:

```sh
python -m MeTTa.ram_hillclimb \
  --proposals /path/to/teacher_proposals.jsonl \
  --max-proposals 2 --ram-limit-mib 8192 \
  --out runs/metta/ram_teacher01
```

The runner checks that each proposal is a one-field edit in its allowlist.
The model ID is self-reported provenance, not verified provider identity.
No GPT API calls are made by the code in this milestone. The bundled
deterministic controls exercise the end-to-end training/evaluation and
verification flow even when no model connection exists.

Each edit emits a small host-owned **MeTTa program** with the candidate
configuration as a ground fact. Native Hyperon replays a `match &self`
query to establish construction syntax. This replay checks only that
the selected proposal atom can be read and queried in MeTTa. The
independent Python curation oracle supplies the training/evaluation labels.
This keeps the MeTTa construction, teacher decision and actual student
performance in separate receipts.

## DataGoblin trace training and RLT / LDT outputs

A run writes:

- `hillclimb_steps.jsonl`: teacher proposal, candidate configuration,
  native-query source, previous and candidate metrics, rejection reason,
  independently computed reward, resource and verifier accounting.
- `train_repair_traces.jsonl`: per-state learned proposal, checked next edit,
  independent optimum target, safety verdict, verifier calls, reward and
  model-checkpoint identity, all from TRAIN roots.
- `train_sft_alternate.jsonl`: SFT `instruction/input/output/metadata`
  training view for LoopedTransformers.
- `train_rlt_alternate.jsonl`: checked RLT proposal/feedback/reward view
  over exactly the SAME identities.
- `training_views_manifest.json`: byte hashes, root exposure counts and
  the one-view-per-training-run constraint.
- `native_proposal_receipt.json`: independent Hyperon query observations
  for the host-owned control facts.

The SFT and RLT files represent the same source-root/presentation pairs.
A learner should consume one view per training run and count that exposure
once, never double the corpus through parallel serialization. These are
synthetic research-fit candidates with `training_ready=false` under the
general DataGoblin EtiquettePolicy. The host's exact synthetic grader is the
teacher for action labels; GPT proposals do not make verifier labels true.

## What to measure next

The present hill climb uses repeated public-development feedback. Its
winning configuration is a **development-selected** result. A confirmation
claim requires frozen weights and a new, independently protected set of
source roots and unseen labels. The CPU run is also an explicit baseline
against the older AIRIS-02 negative study.

A next stage can use this gym's JSONL control/action contract as an adapter
for the real `HybridTRMLDT/research_gym` typed membrane, LoopedTransformers'
boolean worlds and EvalMath / Lean verification tasks. Those adapters
must carry their own source hashes, exact reward functions, time budgets,
root custody, model parameter/optimizer budgets and independent evaluation
receipts. Improvements on this synthetic curation gym are not evidence of
transfer to them.
