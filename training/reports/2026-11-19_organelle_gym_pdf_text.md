# DataGoblin Organelle Gym Research Design

> Archived, searchable extraction of the PDF created 2026-11-19. Preserve the PDF as the layout-authoritative report.
>
> Original SHA-256: `e921b5be770e8f6171925f571ae64ce35f62c443c48cf54a90700e8bc1a0bcf1`.
> Original PDF: https://chatgpt.com/library?file_id=libfile_fde3_f-6a204ebcd75c8191b05833f33fb2b384_c26eca361130819193bd5d8e21b255a7
> Parsed text retains page sequence; the tables may be visually flattened.



---

### PARSED TEXT FOR PAGE: 1 / 13

DATAGOBLIN / ORGANELLE GYM                                                                         RESEARCH DESIGN




 RESEARCH DESIGN / 19 NOVEMBER 2026



 DataGoblin
 Organelle Gym
 A 1B-class model that learns which data and recurrent computation help
 it improve

 Build the gym around the existing HRM-Text checkpoint. Compare
 small, typed organelles and the placement of steering before
 selecting a larger training recipe. Reserve the $1,560 campaign for
 experiments whose task outcomes can establish a useful effect.


   Text interface and research controller
   Encode tasks; propose curricula; forecast outcomes and cost




   Recurrent workspace                                                          Typed LDT boundary
   HRM clocks with a gated TRM or shared-loop specialist                        Check declared state updates




   Output and independent evaluation
   Score task outcomes; retain receipts for the next curation round




 The intended research claim
 A recurrent specialist can learn through a stable interface, improve unfamiliar task families, and
 retain useful behavior after replacement or changes in loop schedule. Mechanistic measurements
 should help select training data or update the shared operator in ways that survive independent
 evaluation.

 Where the work starts
 DataGoblin already contains an architecture gym, HRM-Text instrumentation, and experiment
 records. Its broad gym currently executes stage-zero mock evaluations; later task families remain
 required for architecture selection. This brief extends that foundation. [1]

 Status: proposed design informed by a read-only source audit. The current work created this brief. Model
 experiments and spending remain future stages. Source pins and references appear at the end.




19 November 2026 | Proposed experiments                                                                          1


---

### PARSED TEXT FOR PAGE: 2 / 13

DATAGOBLIN / ORGANELLE GYM                                                                        RESEARCH DESIGN




 01 / TARGET MODEL


 Architecture and the 1B budget
 Use the public sapientinc/HRM-Text-1B release for the first checkpoint-scale gym. Its configuration
 specifies width 1536, separate 16-layer H/L stacks, H=2 and L=3, and context 4096. The
 source-derived model size is about 1.18B parameters. A warm-start system with added organelles
 therefore belongs to the 1B class. [3,19]

 A strict 1.0B model needs a new allocation or compression step. Preserve the warm-start baseline
 while the gym selects the interfaces and schedules worth carrying into that student.

   Component                           Illustrative              Function
                                       strict-1B allocation

   Tied embeddings and readout         100M                      Shared text vocabulary

   Input/output stacks executed        120M                      Encode and decode stable representations
   once

   HRM or shared recurrent core        620M                      Task computation and slow/fast state

   TRM refinement organelle            100M                      Update candidate answer and latent state

   Routing and learned bridges         40M                       Translate between module interfaces

   Steering adapters and readouts      20M                       Small trained interventions

   Total unique neural weights         1,000M                    Illustrative envelope; count instantiated
                                                                 parameters


 The learned LDT proposer and its codecs count inside this envelope. The symbolic checker contributes runtime and
 engineering cost. Report those costs even when its neural parameter count is zero. The eventual layer counts
 depend on the gym results.


 An organelle contract
 Each module declares its input and output state, evidence epoch, reset behavior, visit index, and
 cache ownership. Its record also includes unique weights, trainable weights, execution count,
 gradient access, and elapsed cost. A 256-dimensional interface may reduce expense, but its width is
 an experimental choice.

 TRM can expose a candidate answer and latent refinement state. LDT receives an explicit candidate
 structure plus assumptions. The current HybridTRMLDT strict entry point, certify_and_apply_verified,
 requires environment-owned verification before a hard state refinement. [8]

 Interchangeability has an observable endpoint
 Measure a cold swap between independently trained modules, then allow a fixed bridge-only
 adaptation budget. Compare against an identity bridge and a bridge-only learner. Keep the host
 frozen during this test. Report immediate task loss and the adaptation data, trainable parameters,
 and updates needed to recover.




19 November 2026 | Proposed experiments                                                                             2


---

### PARSED TEXT FOR PAGE: 3 / 13

DATAGOBLIN / ORGANELLE GYM                                                                   RESEARCH DESIGN




 02 / TOPOLOGY


 Onion arrangements to compare
 Fixed describes several different choices. A layer can have frozen weights, occupy a fixed location,
 or execute a fixed number of times. Record these independently, including the visits through which
 gradients flow.

   Candidate                 Arrangement                            Question

   Native HRM                Original H/L schedule                  Reference behavior and cost

   Boundary steering         Small gated intervention at H/L        Can a stable interface improve task
                             boundaries                             outcomes?

   Pass-conditioned          The same adapter receives visit        Does phase-aware control transfer to new
   steering                  metadata; selected inner visits        depths?

   Fixed shell / recurrent   Untied input/output stacks surround    Does a stable shell help module
   center                    shared computation                     replacement?

   Nested HRM / TRM          Slow HRM state invokes a bounded TRM   Does specialization justify extra executed
                             loop; LDT checks typed proposals       work?


 The shell/recurrent-center pattern has prior art in Huginn. The contribution here would come from
 learned module interfaces, controlled replacement, and measured benefit from curation or steering.
 [9]

 Start with the modest changes
 The first checkpoint-scale comparison should retain native HRM and add a small residual side
 module. Compare H-boundary intervention with a limited inner-visit schedule. Test the larger nested
 arrangement after a side module has demonstrated useful task behavior.

 Give both location arms the same visit metadata. Match their trainable capacity, total intervention
 dose, and gradient access. Normalize dose relative to the local residual norm and record the
 cumulative perturbation across the full unroll.

 Backward access is part of the architecture
 The reference HRM training warms the backward horizon from two to five recurrent module visits.
 An intervention inside an early no-gradient region can change the output while receiving no direct
 training signal there. Log the actual autograd path and placement relative to normalization. [2,19]

 A learned halting module follows the fixed-loop comparisons. Measure how often extra loops repair
 an error and how often they damage a completed solution. Price the inner module by the number of
 calls it receives; nested loops can multiply those calls.




19 November 2026 | Proposed experiments                                                                          3


---

### PARSED TEXT FOR PAGE: 4 / 13

DATAGOBLIN / ORGANELLE GYM                                                                    RESEARCH DESIGN




 03 / EVALUATION


 What the gym must measure
   Family                 Example task                               Primary evidence

   Text competence        Reading, mathematical problems, code       Accuracy or verified solutions; retention
                          completion

   Repair and             Repair a typed program or invalid          Compiler/test success; minimal change
   compilation            research proposal

   Mechanism              Matched worlds differing in one hidden     Correct causal choice on unseen worlds
   discrimination         cause

   Research planning      Choose a data recipe and predict its       Realized outcome and cost forecast
                          training effect

   Evidence revision      Replace a stale premise and revise a       Revision accuracy; stale-state errors
                          decision

   Persistent state and   Long episodes with delayed information     Task success, excess loops, contamination
   halt

   Module replacement     Import a specialist trained                Transfer and bridge adaptation cost
                          independently

   Curriculum             Compose familiar skills in an unfamiliar   Whole-task success on held-out
   composition            order                                      compositions


 The existing DataGoblin plan already calls for repair, mechanism discrimination, experiment
 forecasting, and long-horizon state/halting families. Extend those contracts and keep deterministic
 checkers wherever task structure permits. [1]

 A fair comparison
 Select task difficulty using development data so each intervention has recoverable error to work on.
 Preserve the complete set of training seeds in the main comparison. Report whole-task and
 whole-family uncertainty separately from variation between training runs.

 Track learning curves against processed tokens and total billed cost. Architecture quality and
 sample efficiency answer different questions. A slower organelle may see fewer examples under a
 fixed cost cap, so both curves are useful.

 State and changing evidence
 Run clean-reset and declared-persistence episodes. Persistent records carry an evidence epoch and
 source hash. A lattice refinement remains monotone within a fixed set of premises; a changed
 premise starts a new state or triggers an explicit truth-maintenance operation. This allows evidence
 revision while preserving an audit trail.

 LDT endpoints
 Measure accepted-proposal precision together with coverage, task success after rejection, and
 fallback cost. Report the learner alone and the full system with its checker so the contribution from
 deterministic checking remains visible.




19 November 2026 | Proposed experiments                                                                          4


---

### PARSED TEXT FOR PAGE: 5 / 13

DATAGOBLIN / ORGANELLE GYM                                                                          RESEARCH DESIGN




 04 / FIRST ENGINEERING GATE


 Repair the HRM measurement path
 Code inspection at DataGoblin commit 02e9918d identified the following issues. These are source
 findings; this audit did not execute a real checkpoint. [5]

   Finding                     Required repair and evidence

   Verdict probability         Current code selects decision-word tokens from logits after the entire answer. Score
   position                    full candidate labels at the actual decision position, or validate a dedicated head.

   PrefixLM mask               Generation sets prefix token types; the later scoring forward omits them. Preserve
                               the prompt/answer boundary and verify against the released model.

   Trace semantics             Hooks are installed for a post-generation scoring pass. Label replay traces explicitly
                               and compare with generation-time states where causal claims require them.

   Trace overhead              H/L hooks copy full states to CPU and the accumulator converts them to float64.
                               Reduce on GPU and transfer bounded summaries.

   Fixed loop schedule         The adapter assumes L1,L2,L3,H1,L4,L5,L6,H2. Generalize traces to module ID, visit
                               index, state type, and gradient mode.



 Compare execution builds after semantics pass
 Begin with a faithful eager reference. Validate an optimized attention path and a compiled path on
 fixed inputs. Check output probabilities, task decisions, selected gradients, cache behavior, and a
 short optimizer trajectory. Set numeric tolerances in advance and report any decision flips.

 The current Transformers HRM implementation supports PrefixLM through SDPA or FlexAttention.
 The original Sapient training code has a specialized FA3 path. A backend choice must preserve
 bidirectional prefix attention and causal response attention. [4,19]

 Record compiler warmup, graph breaks, kernel choices, peak memory, and whole-job throughput.
 Fixed loop buckets help make compiled candidates comparable. Compilation remains an execution
 field alongside task composition in the candidate manifest.

 Resume is a training test
 Save and restore the optimizer, model, optional EMA, RNG, sampler cursor, learning-rate state, and
 backward-horizon schedule. Compare interrupted and continuous updates through a short fixed run.
 The same checkpoint must continue the intended optimization trajectory.




19 November 2026 | Proposed experiments                                                                                 5


---

### PARSED TEXT FOR PAGE: 6 / 13

DATAGOBLIN / ORGANELLE GYM                                                               RESEARCH DESIGN




 05 / STEERING


 Turn mechanistic ideas into experiments
 Recent local evidence points toward a carefully controlled study. BPD DT04 retained 16 of 59 roots
 under its strict eligibility gates; static state-space steering showed essentially zero advantage over
 its signed-Jacobian shams. Its sparse shared-operator fits showed strong depth extrapolation in a
 descriptive competent subset. Deuterium records DEVELOPMENT_NO_REGISTERED_ADVANTAGE for
 its native follow-up. [6,7]

 Signals that guide training
 Use mechanistic measurements to rank data, train small internal interventions, or update a shared
 recurrent operator. Each route has its own ablation. A favorable outcome from one route cannot
 establish the others.

   Pilot arm      Data selection                              Shaping

   A              Fixed admitted mixture                      Task loss

   B              Mechanistic curator                         Task loss

   C              Fixed admitted mixture                      Task loss plus proposed shaping

   D              Mechanistic curator                         Task loss plus proposed shaping


 Fit these arms at reduced scale, then retain the useful comparisons for a short HRM pilot. Add
 bounded controls using ordinary uncertainty selection and a random intervention matched for rank,
 location, frequency, and output norm. Reuse saved curricula across shaping conditions for a small
 replay study of their interaction.

 Behavior decides the outcome
 Use task success and retention as primary outcomes. Probe activation, reconstruction loss,
 subspace stability, and monitor accuracy remain explanatory measurements. Evaluate the learned
 model after removing steering to measure persistence; also report its performance with the
 intervention enabled.

 Goodfire reports that shaping can relocate or re-encode a concept. In one reward-hacking setting,
 the original monitor fell from 0.98 to 0.57 balanced accuracy while retrained monitors reached
 0.97-0.98. Keep an original probe, periodically refit on separate data, and test held-out behavioral
 outcomes. [11]

 Feature candidates
 Neuron communities, VPD/BPD factors, and small SAEs can identify candidate interventions.
 Community discovery establishes behavior within the original network context; portable organelles
 require replacement tests. [12] Begin with a few taps and simple probes before paying for a broad
 dictionary.




19 November 2026 | Proposed experiments                                                                   6


---

### PARSED TEXT FOR PAGE: 7 / 13

DATAGOBLIN / ORGANELLE GYM                                                               RESEARCH DESIGN




 06 / DATA GOBLIN LOOP


 How the model learns to curate
 At each declared checkpoint, a frozen snapshot proposes batches or generated tasks from an
 admitted candidate pool. A separate checker validates structured outputs. The learner updates for a
 bounded interval, after which the ledger records the observed learning result and charged cost.

 Learn from the marginal training effect
 A selected batch gets a causal reward through a sampled paired experiment: clone the same model
 and optimizer state; train one branch on the proposed batch and the other on a reference batch;
 evaluate their difference on a curator-reward split. Periodically extend the horizon to detect choices
 that help briefly and then stall.

 Train the curator on measured gain with a cost penalty and an explicit spend constraint. A raw
 gain/cost reward can amplify noise from tiny batches. Report cumulative downstream improvement
 against cumulative debit. Initially, fit an outcome predictor from the recorded experiments; adaptive
 selection can follow after that predictor is useful.

 Keep the 1B role coherent
 The same model family can act as learner and research controller through role adapters and
 checkpoint snapshots. Freeze the controller within each curation round. A deployment can load
 these roles sequentially; simultaneous snapshots add memory and execution cost even when they
 share an architecture.

 Related evidence
 IGDS selects fine-tuning examples using features screened by interventions. Mechanistic Data
 Attribution tests data interventions that alter interpretable heads in Pythia. These studies support
 targeted pilots for the proposed curation loop. Their reported results leave HRM transfer and
 autonomous repeated improvement as open experiments. [13,14]

 Split ownership

   Partition                          Allowed use

   Training pool                      Model updates and data recipes

   Mechanism calibration              Feature discovery and shaping design

   Curator reward                     Bounded paired-update feedback

   Development selection              Choose one topology and fixed endpoint

   Final test and transfer families   Independent reporting after the recipe is frozen


 Split by repository, theorem family, generator program, and causal-world lineage before creating
 variants. Hash exact examples and track near duplicates. Training batches retain their source and
 transformation history.




19 November 2026 | Proposed experiments                                                                   7


---

### PARSED TEXT FOR PAGE: 8 / 13

DATAGOBLIN / ORGANELLE GYM                                                                      RESEARCH DESIGN




 07 / CORPUS


 Build a corpus the objective can use
 The HRM reference objective trains on instruction-response pairs with response-only loss. Organize
 the custom corpus into tasks that exercise the intended competence. Raw document length and
 supervised target length should have separate counters. [2,4]

   Source lane                Useful transformations                         Acceptance evidence

   Custom research and        Explain a result, repair a proof, predict an   Source pin plus verifier or
   code                       ablation, test a patch                         documented evidence scope

   Broad educational          Reading, extraction, structured                Quality screen and held-out source
   material                   explanation                                    groups

   Mathematics and            Verified problems, execution traces,           Solver, compiler, Lean kernel, or unit
   programming                test-driven repairs                            tests

   Procedural task families   New worlds, counterfactual pairs, delayed      Independent generator and checker
                              causes

   Research episodes          Forecast, propose data, interpret failure,     Observed run receipts and retained
                              choose a follow-up                             failed predictions


 FineWeb-Edu, FineMath, and Stack-Edu provide candidate public material for the wider pool. Their
 dataset cards describe content and provenance. Evaluate a sample before committing a training
 fraction, and retain per-source metadata. [16-18]

 Inventory before mixture weights
 Count unique tokens under the selected tokenizer, deduplicate, and estimate the response-token
 fraction. Record how much custom data can be repeated before the mixture becomes dominated by
 copies. Treat synthetic expansions as a separate source with their own verification receipts.

 Use the audited custom corpus as a source of task families and exact generators. Training examples
 produced from independent instances can scale its useful coverage. Rephrased copies remain part
 of the same lineage for evaluation splitting.

 The 10B definition
 Use 10B as an upper bound on processed, non-padding token exposures in a selected arm. Also
 record unique tokens, repeated exposures, supervised response tokens, and padding fraction. A 10B
 unique-token corpus used for several epochs incurs a larger training bill.

 Curation should demonstrate value early
 Start from a fixed mixture and compare the mechanistic selector with uncertainty sampling on the
 same candidate pool. Charge candidate scoring and verifier work to the curator. Select the
 larger-run mixture from those results, with a broad retention set to expose domain
 over-specialization.




19 November 2026 | Proposed experiments                                                                           8


---

### PARSED TEXT FOR PAGE: 9 / 13

DATAGOBLIN / ORGANELLE GYM                                                                         RESEARCH DESIGN




 08 / COMPUTE BUDGET


 What $1,560 can support
 Sapient reports 16 H100s for 46 hours: 736 GPU-hours for 60B processed exposures drawn from 40B
 unique tokens. Its quoted $1,472 uses an assumed $2 per H100-hour. Linear scaling gives 122.7
 GPU-hours for 10B exposures under that reference throughput. These are reference calculations. [2]

   Assumed effective             10B, reference            10B, 1.5x time             Two 10B arms, 1.5x time
   rate                          speed

   $2 / GPU-hour                 $245                      $368                       $736

   $4 / GPU-hour                 $491                      $736                       $1,472

   $6 / GPU-hour                 $736                      $1,104                     $2,208


 The table prices one seed per arm. Additional seeds multiply exposures. The extra-time multiplier includes all added
 allocated time. Use measured whole-job throughput once available. Silico charges while a job holds a GPU; a public
 RU conversion was unavailable. Measure the account debit in a timed job. [15]


   Campaign stage                            Proposed           Deliverable
                                             cap

   Fidelity, profile, resume                 $120               Trusted model path and observed debit

   Small-model and HRM pilots                $240               One selected topology and comparison

   Main matched training                     $900               Two paired arms at a frozen endpoint

   Independent evaluation                    $180               Retention, transfer and mechanism checks

   Failure and restart reserve               $120               Preserved capacity for an interrupted run

   Total                                     $1,560             Campaign ceiling



 Recommended main endpoint
 Freeze two conditions at 2B tokens per run, with two training seeds per condition: four runs and 8B
 total exposures. At the illustrative $4/hour rate with 1.5x reference time, this costs $588.80. A
 single-seed pair would cost $294.40. Measured billing determines feasibility. Record any later
 extension as a new endpoint.

 For a $900 main allocation, two 10B arms require at least 24,691 useful tokens per GPU-second at
 an effective $4/hour. One 10B arm requires 12,346. Compute this from the whole-job debit and
 throughput, including compiler startup and checkpoint time.




19 November 2026 | Proposed experiments                                                                                 9


---

### PARSED TEXT FOR PAGE: 10 / 13

DATAGOBLIN / ORGANELLE GYM                                                                       RESEARCH DESIGN




 09 / RUNTIME


 Memory, traces and the execution split
 Count every visit
 For planning, separate forward and backward visits. If module i has P_i parameters and receives f_i
 forward visits and b_i gradient-carrying visits per exposure, the main matrix-multiply estimate is T x
 [2 sum(f_i P_i) + 4 sum(b_i P_i)]. Add attention, readout, tracing, and checker costs. This is a
 planning approximation; use measured runtime for billing.

 Two architectures with the same stored weights can execute very different work. A small inner
 organelle invoked repeatedly may dominate cost. The manifest must retain the complete module
 schedule and backward horizon.

 Local GPU role
 The user's 16 GB RTX 5080 suits inference, adapter work, probe fitting, and reduced-scale models.
 Under a common 16-byte full-Adam state accounting, 1.18B parameters require about 18.9 GB
 before activations. Full-model training on that GPU needs a different measured memory strategy.

 The native HRM source keeps caches per recurrent visit. At width 1536, BF16 MHA and 4096 tokens,
 storing K/V across 128 visited blocks takes about 3 GiB per sequence. This is a derived native-layout
 estimate; confirm the chosen backend's cache layout. [3,19]

 Trace a sample
 One BF16 hidden-state tap over 10B tokens at width 1536 occupies 30.72 TB. A 100,000-token
 calibration sample across eight taps occupies about 2.46 GB. Stream summaries for normal training
 and retain richer traces only on the calibration sample.

 Set a chosen instrumentation target, such as under 15% extra whole-job time, then measure it. Full
 Jacobians and per-token SAE training across the full corpus would need their own cost justification.

 Where each stage runs

   Place                       Work

   Local workstation           Dataset preparation, small-model pilots, HRM inference and adapters, trace
                               analysis

   Silico                      Verified checkpoint-scale training and the selected mechanistic comparisons

   Repository checks           Contract validation, deterministic fixtures, result arithmetic and replay
                               bookkeeping


 Use the actual plan runtime limit. If jobs must be chunked, preserve complete training state
 between chunks. Silico documents isolated on-demand filesystems and an artifacts directory for
 outputs passed to dependent work. [15]




19 November 2026 | Proposed experiments                                                                       10


---

### PARSED TEXT FOR PAGE: 11 / 13

DATAGOBLIN / ORGANELLE GYM                                                                       RESEARCH DESIGN




 10 / BUILD SEQUENCE


 Implementation order and promotion
   Step                Concrete work                                   Evidence before the next step

   1. Measurement      Fix masks, label scores, trace semantics,       Native output and gradient parity
   repair              bounded reductions

   2. Gym task         Add repair, mechanism, forecasting, state       Independent checkers plus lineage splits
   families            and halt tasks

   3. Organelle        Typed state, evidence epochs, routing and       Identity path and reset conformance
   interface           visit records

   4.                  Topology and curation/shaping pilots            Recoverable headroom and repeatable task
   Reduced-scale                                                       gain
   selection

   5. Native HRM       Short matched branches and charged profile      Retention and viable paired-run budget
   pilot

   6. Main             Two conditions, two seeds each, 2B tokens       Independent test with seed and family
   comparison          per run                                         uncertainty

   7. Optional         Increase paired exposure or train a strict-1B   Recorded decision based on earlier evidence
   extension           student



 Promotion evidence
 Before the main allocation, require a useful held-out task effect against the selected controls,
 acceptable retention, and complete runtime accounting. Freeze the minimum effect and retention
 tolerance from development needs before opening final labels. Include failed seeds and interrupted
 runs in the record.

 A module earns a portability claim only after the independent-swap test. A curator earns a
 learning-efficiency claim only through a paired update comparison. A persistent-training claim needs
 performance after steering removal. The full system also reports the contribution and cost of its
 deterministic checker.

 Repository layout
 Use DataGoblin as the orchestrator and candidate registry. Extend src/datagoblin/training_gym/ and
 src/datagoblin/hrm_text/. Place source-pinned organelle bridges beside these interfaces.
 HybridTRMLDT supplies the typed projection contract; BPD and LoopedTransformers provide prior
 protocols and retained evidence. [1,5-8]

 A candidate record should include the source/checkpoint hashes, topology graph, unique and active
 parameters, visit schedule, gradient horizon, intervention dose, corpus manifest, all token counters,
 backend and precision, charged debit, maximum memory, and resume receipt.




19 November 2026 | Proposed experiments                                                                           11


---

### PARSED TEXT FOR PAGE: 12 / 13

DATAGOBLIN / ORGANELLE GYM                                                                       RESEARCH DESIGN




 REFERENCES


 Sources and evidence
 Public sources were checked on 19 November 2026. GitHub links for the user's work are pinned to the inspected
 commit. Budget multipliers, parameter allocations and proposed thresholds in this brief are design choices or
 derived calculations.

 [1] DataGoblin architecture gym
 Existing stage-zero gym and required task families.

 [2] HRM-Text: Efficient Pretraining Beyond Scaling
 Training objective, processed/unique token counts, compute and gradient horizon.

 [3] Official HRM-Text checkpoint configuration
 Architecture dimensions and recurrence. Viewed 19 November 2026.

 [4] Transformers HRM-Text documentation
 PrefixLM token types and supported attention paths.

 [5] DataGoblin HRM instrumentation source
 Source inspection underlying the first repair gate.

 [6] BPD DT04 results, 19 November 2026
 Registered eligibility result and descriptive depth analyses.

 [7] Deuterium evidence and outcome
 Native steering result and retained precision issue.

 [8] HybridTRMLDT lattice implementation
 Candidate lattice and strict verification entry point.

 [9] Scaling by Thinking in Continuous Space
 Huginn prelude, recurrent core and coda prior art.

 [10] Less is More: Recursive Reasoning with Tiny Networks
 TRM refinement architecture.




19 November 2026 | Proposed experiments                                                                          12


---

### PARSED TEXT FOR PAGE: 13 / 13

DATAGOBLIN / ORGANELLE GYM                                                                       RESEARCH DESIGN




 REFERENCES


 Sources and evidence / continued
 Public sources were checked on 19 November 2026. GitHub links for the user's work are pinned to the inspected
 commit. Budget multipliers, parameter allocations and proposed thresholds in this brief are design choices or
 derived calculations.

 [11] Goodfire: Shaping and Monitoring Models
 5 November 2026; shaping and monitor drift.

 [12] Goodfire: Discovering Neuron Communities
 17 November 2026; causal studies within existing layers.

 [13] Interpretability-Guided Data Selection
 Intervention-screened features for fine-tuning data selection.

 [14] Mechanistic Data Attribution
 Data interventions and emergence of interpretable heads.

 [15] Silico compute documentation
 Allocated compute and artifact handling; no public RU conversion found.

 [16] FineWeb-Edu dataset card
 Candidate educational source.

 [17] FineMath dataset card
 Candidate mathematics source.

 [18] Stack-Edu dataset card
 Candidate programming source.

 [19] Sapient native HRM implementation
 Source-pinned recurrence, gradient gating and cache behavior.




19 November 2026 | Proposed experiments                                                                          13
