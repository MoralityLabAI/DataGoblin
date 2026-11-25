# AGN-03: Storyworld-driven continual learning

Status: development architecture and offline bridge. No paid compute, live teacher generation, new Lean kernel result or trained checkpoint is authorized by this document.

## Division of labor

| Component | Responsibility |
| --- | --- |
| Data Goblin / Agonist | Select research taste, candidate families, verify cost, estimate learner uplift, curate and stop |
| Bonsai 2 27B or Inky Mini | Draft original SWMD encounter cards, typed simulator proposals, candidate interventions and inductive hypotheses |
| GPTStoryworld | Render and validate narrative, evaluate bounded reaction semantics, execute trusted transitions with a private world |
| EvalMath | Opportunity capture, exact/approximate oracle, paired effects, resolution preflight, family-disjoint test custody |
| Lean adapter | Compile faithful formal statements, check candidate tactics and axiom policy on pinned frozen goals |
| Compact learners | Acquire reusable action, induction and proof skills; provide measured transfer and retention to the next round |

Start with the [SW-PROV-05 source](https://github.com/MoralityLabAI/GPTStoryworld/commit/f8ba60b66d719767848edc5cc72522738dc90f5f), [ADG-06 source](https://github.com/MoralityLabAI/EvalMath/commit/cad2a708aa5c66b1850746b1bc909b4a00e8afed), existing Agonist intake/uplift, and the RLT Lean/MCP checker. The new adapter is a reference to separately obtained upstream code, not a vendor copy.

## The shared mathematical object

A world is a typed partially observed causal decision process with exogenous assignment u, state x, actor-local evidence L_i, transition F, observation O_i, action a, and bounded reward R. Counterfactuals evaluate the same u and policy-visible prefix under a declared intervention, with matched budget; report the intervention, outcome distribution, and assumptions. A claim that trust enables registry access, for example, separates the structural statement (permission changes according to the explicit threshold) from the decision-policy effect (expected reward under optimal or fixed policy changes). These claims need different verifiers. Observational induction proposes a generalization, such as 'documented lineage can make an otherwise unavailable archive valuable'; a new generator family seeks a counterexample, and a proof target states exactly the assumptions needed. Proofs over finite models do not establish real historical causation.

Let V*(h,b) be optimal expected utility and V0 immediate terminal utility. Curation tracks opportunity mass V*-V0 and attained opportunity capture, alongside absolute utility and false assertion. Prefer new causal structure, not repeated high-gap roots. For train round t define learner improvement on designated development families as Delta_t and old-family retention change as r_t. A selection policy estimates expected Delta_t / full resource cost, subject to rights, replay cap, retention and independent confirmation requirements. This is a proposed objective; do not interpret oracle utility as student uplift.

## Initial lanes

| Lane | Input | Target and authority |
| --- | --- | --- |
| Action | Public packet at decision time | Verified legal optimal action set; independent environment/oracle |
| Dynamics | Public packet, executed action | Actually acquired public next packet; replay |
| Causal | Paired public histories with intervention contract | Matched outcome contrast, explicit assumptions; trusted simulator |
| Induction | Finite checked examples | Proposed general rule plus counterexample challenge; hypothesis until independently resolved |
| Lean | Faithful formal statement, frozen proof goal, prior proof state | Kernel-approved tactics/proof or verified failure; pinned Lean tool |

Keep hindsight outcomes separate from model inputs. An observed outcome may be a supervised dynamics target, never an advance hint to the acting policy. A model-authored explanation is a claim. All lanes retain family/root/source hashes and rights, and share the four-presentation cap. In a real training manifest the host must review and explicitly promote candidates. Development data is not a training release or a sealed benchmark.

## Staged experiments

**AGN-03 wiring:** replay the known North Quay finite law through its public session adapter. Verify legality, budget and paired potential outcomes for social vs blocked access. Generate a small collection of candidate views with an exact action target and a formal draft; record proof status as unverified. No new model run.

**AGN-04 author quality:** operator-controlled teacher queues for at least several independent mechanism families (provenance, competing goals, finite resources, spatial reasoning, partial information, multi-hop logic). Begin with small batches and compare teacher-generated candidates to scripted generators; log valid worlds per teacher token, unique causal families, repair rate, native-validator failures and oracle/shortcut gap. Fix context cap and local resource wrapper before any live generation; use a separate reviewed import and independently executable simulator. An authored draft is not a verified generator. Establish a fresh family holdout outside authoring. Do not use this document to authorize paid calls.

**AGN-05 induction/Lean:** on checked worlds, construct matched interventions, finite truth tables and faithful Lean goals. Train or evaluate the ability to infer preconditions and produce valid proof steps. Generate near-miss statements and counterexamples. Run kernel checks from original goal states, preserve failed paths, and verify equivalence between formal transition definitions and executed simulator on an independently audited domain. Do not turn natural-language causal hypotheses into axioms. Start with symbolic finite reachability and bounded arithmetic; advance to general invariants only as proofs warrant.

**AGN-06 continual student:** CPU pilot with small parameter-matched transformer/looped variants and explicit memory. Compare sequential-only, fixed rehearsal and active curation (random, heuristic, learned) with matched processed tokens and measured compute. Use a fixed anchor suite for retention and a distinct newly generated family suite for plasticity. Measure decision utility, informative accuracy, induction counterexample rejection and Lean closure separately. Three seeds are a development diagnostic, not sufficient confirmation. Preflight on independent development families; preregister meaningful utility and forgetting margins and hold out entire source and generator families. Use a checkpoint DAG and immutable manifests; roll back on failed gate. Scale to larger students only after independent transfer and retention gates pass.

**Long-range:** connect admitted corpus lanes to Aleph/RLT curriculum, promote only after independent improvement. Keep an exploration tranche for novel mechanisms and an anchor/replay tranche for retention. No self-scoring loop may hard-promote its own output.

## Offline replay

Run against a local copy of the SW-PROV-05 Python scripts, obtained separately and reviewed. The bridge requires a content digest of the four transitively imported files; `--print-source-digest` reports it. Compare the digest against a trusted checkout before passing `--expected-source-digest`. It treats the external module as trusted executable source, so do not run against untrusted model-generated code.

    python -m Agonist.storyworld_continual --source /path/to/sw_prov05 --print-source-digest
    python -m Agonist.storyworld_continual --source /path/to/sw_prov05 --expected-source-digest <verified digest> --out runs/agn03/north_quay_dev

The output is an immutable development archive with private paired replay traces, public candidate views (at most four across the shared root), and a digest manifest. The fixture belongs to one already exposed family; the oracle and transition model share implementation and no native engine parity or novel student uplift is measured. The Lean draft remains unverified until a pinned kernel and statement crosswalk are run. No generated archive belongs in Git.
