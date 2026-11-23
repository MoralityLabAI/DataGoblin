# DataGoblin Agonist: taste-to-world research gym (AGN-01)

The human supplies **research taste**. DataGoblin turns it into a bounded experiment
program and instructs a local Inkling 2 Mini builder to create storyworlds, small
playable games, counterfactuals and training candidates. The environment owns the
truth conditions. DataGoblin chooses which verified worlds deserve more work.

The first milestone is an executable **paired-world contract** for deciding whether
to deploy a fictional research expedition. A shallow policy follows the unchanged
surface cue. The exact verifier follows the causal prerequisites. Every valid pair
changes one prerequisite and reverses the right action. The two variants share one
source root, so their outcomes are paired observations.

This subfolder is an isolated gym and skill. It does not alter the existing DataGoblin
trainer, main entry points or RLT collections. Its data exports are candidate records,
rather than an admitted pretraining corpus.

## Quick smoke: no model or network

From the DataGoblin repository root, with Python 3.10+:

    python -m Agonist.gym demo \
      --tastes Agonist/fixtures/tastes.jsonl \
      --out runs/agonist01/fixture
    python -m Agonist.gym replay \
      --archive runs/agonist01/fixture/archive

The fixture uses three original taste seeds, two train roots and one development
root, to construct deterministic world pairs. It calls zero models, allocates no
GPU and has no benchmark significance. Expected: three verified pairs, six verified
decisions, three cue-baseline correct decisions and four unapproved training views.

## Stage for a real local Inkling builder

Prepare a JSONL file of user-paraphrased ideas following fixtures/tastes.jsonl.
Every seed has a unique taste_id, family_id, idea, source_kind, source_ref, rights,
and split. Source families may only appear in one split. The staging queue
supports train and dev; sealed task families live in a separate private registry.

    python -m Agonist.gym stage \
      --tastes Agonist/fixtures/tastes.jsonl \
      --out runs/agonist01/queue \
      --max-context-tokens 12288

The queue contains teacher_requests.jsonl with JSON-output prompts and
manifest.json with SHA-256 identities. It is inert until a separately authorized
operator sends requests to the **already running** Inkling 2 Mini llama.cpp
endpoint. Keep the model revision, quantization, tokenizer, chat template, full
request token counts and sampler settings in the operator's provider receipts.
Run the server's exact template and tokenization preflight, including tool schemas.
Use 12,288 tokens by default; reject requests above 16,384 tokens. No paid fallback.

Collect one response object per request in a separate teacher_replies.jsonl file.
Each has request_id, teacher_model, world_a, world_b, intervention and design_note.
The worlds have setting, scenario, surface_cue and state. State contains integer
available/required and boolean safety_approved/source_fresh. An intervention
specifies the one causal field, its old value and its new value.

    python -m Agonist.gym audit \
      --requests runs/agonist01/queue/teacher_requests.jsonl \
      --replies /path/to/teacher_replies.jsonl \
      --out runs/agonist01/import
    python -m Agonist.gym replay --archive runs/agonist01/import

Audit produces:

| Artifact | Meaning |
| --- | --- |
| public_worlds.jsonl | Fictional worlds and revision lineage, without gold actions |
| verifier_receipts.jsonl | Exact independent actions and pair integrity receipts |
| train_candidates.jsonl | Two single-world SFT candidates per eligible train pair; training_ready=false |
| rejected.jsonl | Invalid teacher replies, error codes and digest of the original reply |
| manifest.json | Input and artifact SHA-256, counts and surface-cue control accuracy |

The verifier accepts **deploy** if and only if available is at least required,
safety_approved is true, and source_fresh is true. **hold** is the alternate action.
This version has a deliberately simple, exact first mechanic. The world builder
may vary fiction and causal prerequisites while the evaluator remains stable.

When adapting the gym to full storyworld simulators or videogame engines, version
the world schema and supply an independent executable verifier. Game aesthetics,
designer explanations and model-generated claims provide training context; they
are not sources of truth about outcomes.

## Research experiment and baselines

The first planned experiment has 50 independent research-taste roots. A learned
DataGoblin agonist, a fixed curator and a random curator receive matched teacher
tokens and verifier calls. Each selects what world family or revision to generate
next. A tiny student learns from the selected train worlds; independent test-family
performance is the principal endpoint.

The current gym implements the queue, exact pair validation, a trivial cue-policy
control, train-only candidate preparation, SHA-256 manifest and replay checks.
There is no connected trained-agonist selector, live Inkling inference, or
measured downstream learner improvement in AGN-01.

Follow [the Agonist skill](SKILL.md) for the agent contract and EvalMath-inspired
promotion gates. The existing
[Research Taste Hill-Climbing skill](../training/rlt_gym/hypergame_taste/skills/research_hillclimb/SKILL.md)
covers the narrower K-depth RLT decision policies.

## Safety, rights and custody

- Tastes can come from the user's imagination, a model, a public reference or a
  user-written description of an idea encountered on social media. Keep source
  attribution and separately check permission for verbatim material.
- The generator and the simulator use fictional mechanics. Treat arbitrary
  model prose as untrusted proposals; no generated instructions may operate
  tools, spend money, read hidden roots, or promote their own results.
- Train/dev families remain disjoint. Full sealed confirmation cohorts are held
  outside the staging queue. No gold action enters an actor-visible prompt.
- Archive output goes to ignored runs/ paths. Reusing an existing output path
  fails closed so results and manifests cannot be silently overwritten.
- At most four presentations of a canonical root may enter the wider curriculum
  across all DataGoblin views. The current gym exports two candidate views and
  keeps training_ready=false pending rights and source-family review.

### Next milestones

Add an opt-in llama.cpp worker that implements exact tokenizer/template checks
against the resident service and retains actual provider receipts. Then introduce
multi-turn game adapters for costly information gathering, source revisions,
social play, and research selection. Connect the verified root exports to the
DataGoblin curator only after measuring the small-learner uplift against
matched-budget baselines.

## AGN-02 student-uplift pilot

[AGN-02](AGN02.md) adds a 50-family, three-decision offline experiment with a learned utility-selection surrogate, matched random and fixed curation, trained small-student comparisons, and independently recomputed receipts. Its [selection skill](skills/agonist02/SKILL.md) describes the downstream learner objective and promotion protocol. The current AGN-02 fixture uses no live Inky inference or sealed confirmation roots.
