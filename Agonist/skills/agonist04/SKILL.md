# AGN-04: Live teacher generation, mechanical verification and repair

Status: skill definition and verification-record schema only. No paid compute, live teacher
call, Lean kernel run or trained checkpoint is authorized by this document, consistent with
[AGN-03](../../../docs/storyworld_continual_learning.md). An operator must explicitly authorize
each live generation batch before this skill's pipeline is run against a real backend.

## Local serving path (operator-selected)

- **Inky Mini, fast local decode:** serve `thinkingmachines/Inkling-2-Mini` through the
  **squid5080** engine (`Documents\Codex\squid5090`, upstream `tensorfish/squid5090#5`) rather
  than a slower llama.cpp path. squid5080 is verified at 148–254 tok/s generate and ~1,900 tok/s
  prefill on the shared 5080; treat that as the realistic throughput budget when sizing a
  generation batch, not an assumed number. If "fast tps on Inky" names a different, newer loader,
  substitute it here — nothing else matching "Maya" exists on this machine as of this pass (see
  `docs/integration-log.md`).
- **Bonsai 2 27B:** served as in AGN-01–03, through the existing resident local server; do not
  restart it or displace another queued job (see `Agonist/SKILL.md`'s local teacher contract).
- **Bounded-context authoring surface ("Pocket Goblin GDK app inside Goose"):** run the
  encounter-by-encounter authoring loop (external memory + encounter index, Section 2's
  "bounded-context encounter-by-encounter generation") as a Goose session against `goblin-gdk`
  (`HybridTRMLDT/MCP/goose_gdk_app`), reusing its existing tool surface instead of a new harness.
  Resolve the actual model revision, quantization, tokenizer/chat template and context limit from
  whichever backend (squid5080 or the resident Bonsai server) a request targets; never guess an
  identifier.

## Authoring ticket (extends AGN-03's ticket to full encounter cards)

Data Goblin selects a source-attributed taste seed, family ID, mechanism family (provenance,
competing goals, finite resources, spatial reasoning, partial information, multi-hop logic, or
another registered family — Section 2's family list), target capability, a K difficulty vector
(Section 3), and a maximum teacher/verifier/storage budget. The teacher receives that ticket and
produces, one encounter at a time against an external world card + encounter index (never the
whole world in one context window):

- an SWMD-0-MIN encounter card and reaction formula, in the existing storyworld skill's format;
- a typed proposal against the GPTStoryworld SW-PROV whitelisted adapter (typed parameters only —
  never arbitrary executable code; unsupported mechanics are quarantined, per AGN-03);
- character-specific knowledge and an evidence ledger, with declared observation channels and
  source dependencies;
- a one-mechanism counterfactual world with matched exogenous assignment, budget and evidence
  visibility;
- an inductive hypothesis with a declared scope and an invitation for a counterexample;
- a proposed Lean statement (unverified draft; AGN-05 checks it, this skill never does).

Reject a reply that adds an inert variable, a hidden prerequisite with no inferable cue, a
redundant reaction, or an unreachable ending. Require a matched superficial-cue control for every
option that changes attainable utility or information.

## K as a difficulty vector (Section 3)

Record, per world, not a scalar: optimal opening/receding-horizon lookahead depth, branching
factor, epistemic ambiguity (uncertainty), source/provenance dependence depth, required memory
span, belief-revision depth, and opportunity-structure richness. See
`schemas/agn04_world_verification_record.schema.json#/properties/difficulty` for the exact typed
fields.

## Independent mechanical verification (Section 3)

Every generated world is checked by a solver independent of the teacher, before any candidate
view is emitted — reuse EvalMath's ADG-06 (`cad2a70`) exact latent-lineage/provenance oracle
where the mechanism matches, rather than re-deriving one. Use exact solvers where tractable;
report an explicit unresolved optimality gap for bounded-search worlds rather than claiming
optimality. Record, per world:

- realized policy utility and attainable-opportunity capture (`V* - V0` headroom, per AGN-03's
  shared mathematical object);
- informative decision accuracy;
- minimum optimal planning horizon;
- strength of the cheapest shortcut policy (the surface-cue / cue-baseline gap);
- counterfactual intervention effect, under the matched-budget contract;
- reaction-variable sensitivity;
- evidence provenance / belief-revision correctness;
- ending reachability.

`schemas/agn04_world_verification_record.schema.json` is the typed record for exactly these
fields plus the difficulty vector and family/root/source identity. A record with
`kernel_checked: true` or `training_ready: true` may never be emitted by this skill alone — both
require, respectively, AGN-05's Lean pass and an explicit operator promotion, per AGN-03's
training-custody rule.

## Repair cycle

On a verifier rejection, return the failure code and the independently-checked state (never a
hint derived from a sibling or counterfactual world) to the teacher and request a repair or a new
family, preserving the rejected attempt, its hash, and its lineage. Never retroactively edit a
previously emitted record. Stop on exhausted budget, repeated duplicate roots, unverifiable
mechanics, or a plateaued oracle/shortcut gap across a batch; report the reason and the budget
actually spent.

## Out of scope for this skill

Lean kernel checking (AGN-05), any training step (AGN-06), and curriculum selection (AGN-07) are
separate skills. This skill stops at an independently-verified, source-hashed, `training_ready:
false` candidate view.
