# Data Goblin Aleph 1B Research Dossier

> Archived, searchable extraction of the PDF created 2026-11-19. Preserve the PDF as the layout-authoritative report.
>
> Original SHA-256: `5aa18ac872ef3d856cb5c898b6b6d49c6f8c4c5b799e0d4da505afb25db9ce91`.
> Original PDF: https://chatgpt.com/library?file_id=libfile_fde3_f-d241b573d780819190cc95e761a1d221_c26eca361130819193bd5d8e21b255a7
> Parsed text retains page sequence; the tables may be visually flattened.



---

### PARSED TEXT FOR PAGE: 1 / 32

 RESEARCH DOSSIER / 19 NOVEMBER 2026



 Data Goblin
 Aleph 1B
 A learned recurrent graph, trained on 10B tokens with causal feedback
 about its own computation

 The target is a working 1B-class model and an experiment that can
 establish whether mechanistic feedback improves how its modules
 communicate. The full 10B-token run remains the main deliverable.

   Target                                  Proposed value

   Distinct neural parameters              About 1.007B in the explicit design

   Training exposure                       10B processed, non-padding tokens

   Compute layout                          Eight H100 SXM GPUs; benchmark DDP and operator sharding

   Work window                             November 20-30, following the stated eleven-day window



 What this dossier contains
 An HRM evidence audit, a concrete alternative to its fixed H/L schedule, a cost model, and a research
 plan. The annotated catalog covers shared-depth transformers and closely related methods through
 November 19, 2026. Most catalog entries were checked against primary abstracts; selected
 mechanisms and implementation details received full-text or source-code review.

 Status: research and design. The cost calculator has run and its parameter assertion passed. Model training, dataset
 preparation and GPU throughput tests have not run in this review. The $1,560 balance is user supplied; the conversion
 from Silico RUs to occupied H100 time remains to be measured.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                     1


---

### PARSED TEXT FOR PAGE: 2 / 32

 01 / REPORTED RESULTS


 HRM-Text has external evidence
 Each column below comes from a different reported run or evaluation. DROP uses F1. The DFM column
 preserves its historical protocol. Empty cells indicate that the cited source did not report the result.
 [S1-S3]

   Benchmark                    Sapient paper          External dense retrain   DFM checkpoint eval

   GSM8K                        84.53                  83.93                    84.80

   MATH                         56.16                  54.96                    56.00

   DROP F1                      82.21                  83.06                    78.10

   MMLU                         60.73                  61.38                    58.70

   ARC Challenge                81.91                  83.02                    82.20

   HellaSwag                    63.42                  61.96                    60.40

   WinoGrande                   72.38                  71.98                    70.40

   BoolQ                        86.18                  87.25                    87.50

   HumanEval                    Unreported             Unreported               0.00



 How to read the corroboration
 Xiaoye08 reports a newly trained dense HRM-Text XL baseline beside a larger MoE model. DFM
 evaluates Sapient's released checkpoint. Their math scores support useful task competence. Shared
 evaluation code can carry a shared measurement problem into an external result.

 DFM reports 0.0 on HumanEval under that protocol. Sapient's card says the checkpoint was trained
 without code datasets. Aleph needs a code-containing curriculum and execution-based evaluation
 before it can support Data Goblin's programming work. [S3,S4]

 A result worth keeping in the gym
 Use the released checkpoint as a reference for task completion, prefix masking and recurrent traces.
 Compare Aleph against it with a common generation budget and the original native prompt mode.
 Preserve raw generations so every score can be recomputed.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                        2


---

### PARSED TEXT FOR PAGE: 3 / 32

 02 / EVIDENCE AUDIT


 Where the HRM claims need work
 Fractional credit for invalid choices
 The pinned Sapient evaluator adds 1/k to the correct counter when a generated answer is outside the
 valid choice set. Standard MCQ tasks and MMLU use this behavior. DFM's historical fork inherited it; its
 August 31 change removes the relevant lines for StandardMCQ and MMLU. The historical result table
 has no verified strict rescore. [S5,S6]

 For a micro-averaged four-choice task: strict accuracy = reported score - 0.25 x invalid-output
 fraction. For two choices the coefficient is 0.5. Category-level averages require the same
 category weights as the original metric.

 A 10% invalid rate adds 2.5 percentage points on a four-choice task. The actual inflation depends on
 the saved generations. A fair rerun should report strict generated accuracy, invalid rate and a separate
 option-likelihood result.

 Architecture and training recipe
 Sapient includes approximate-compute controls and objective ablations. These support an architectural
 contribution within its pipeline, while also showing a large benefit from response-only PrefixLM training.
 The broad comparison against models trained on trillions of tokens combines several training choices.
 [S1]

 Aleph should therefore hold the data order, tokenizer, objective, sequence lengths and output budget
 fixed when isolating routing. A separate comparison at equal billed cost answers whether the added
 machinery earns its runtime.

 Original puzzle HRM
 ARC Prize reproduced 32% on ARC-AGI-1 semi-private and 2% on ARC-AGI-2 semi-private. Its ablations
 emphasized outer refinement and augmentation. TRM subsequently reported better public puzzle
 scores with a smaller shared network. Those findings motivate alternatives to the prescribed H/L
 hierarchy. Textual HRM uses a later training recipe. [S7,S8]

 Accounting and contamination
 Store unique tokens, repeated presentations, response-loss tokens and source hashes separately.
 Exact n-gram checks should be supplemented with document and task-family exclusions. An
 unresolved pipeline report about HRM token counts remains a reproducibility question, with conflicting
 reports recorded in the audit JSON. [S9]




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                          3


---

### PARSED TEXT FOR PAGE: 4 / 32

 03 / PUBLIC RESEARCH AND PRODUCT CHOICES


 What Sapient has done recently
   Date in 2026       Observed activity                            What it establishes

   May 17-21          HRM-Text announcement, paper and native      A public language-model research release
                      Transformers checkpoint support

   August 28          PRAXIST announcement                         A product centered on autonomous research

   September 4        Original HRM-Text repository banner update   Documentation maintenance

   September          New sampled dataset and HRM-Text-1B-v3.2     Continued HRM experimentation; gated model
   23-24              listing                                      content

   October 15         PRAXIST report revision                      Further documentation of the research-agent
                                                                   system


 The current PRAXIST page identifies DeepSeek-v4-pro as the model used in its MLE-bench system.
 Sapient also keeps HRM visible in its model listings. These records show parallel research and product
 efforts. The gated HRM listing supplies no verified capability gain in this review. [S10-S12]

 What this says about confidence
 Public activity gives us choices the company has made. It cannot identify the founders' private
 probability that HRM will scale. A research group can continue architecture work while shipping a
 product with an available frontier model. Aleph's design should be justified by its own measurements.

 The literature has moved past a single topology
 October studies explicitly compare recurrent placement, input injection and recurrent state choices.
 Part I favors a Huginn-style shell at the tested scales; How to Loop Your Transformer finds middle-block
 looping attractive when training FLOPs constrain the design. These results make the shell and the
 recurrent graph sensible experimental axes. [P36,P39]

 The causal question is more specific: can measured effects of component and edge interventions teach
 useful communication that transfers to unseen compositions? This is the hypothesis proposed here.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                             4


---

### PARSED TEXT FOR PAGE: 5 / 32

 04 / READING PRIORITIES


 The papers that shape Aleph
   Paper or group                                            Decision it informs

   Universal Transformers; ALBERT; Deep Equilibrium          Define weight sharing, recurrence and the assumptions
   Models                                                    behind implicit gradients.

   Huginn; AlgoFormer; RINS; Part I; How to Loop Your        Choose the untied shell, input injection and physical layer
   Transformer                                               allocation.

   HRM-Text; TRM; Closing the Loop                           Specify the task objective, finite gradient horizon and
                                                             reference training recipe.

   Mixture-of-Depths; Mixture-of-Recursions; PonderNet       Separate selection of tokens, loop count and halt policy.

   Shared Weights, Selected Computations; Cross-Loop         Design causal tests of what a pass computes and what its
   Query Handoff                                             messages carry.

   Relaxed Recursive Transformers; Think-at-Hard;            Choose small pass-specific controls and evaluate depth
   Elastic Latent Recurrence                                 transfer.

   Parallel Loop Transformer; Dual-Slot; parallel            Understand which dependency is changed when execution
   nonlinear equation solving                                is parallelized.

   LoopSpec; LoopSkim; CHASE; Part II                        Evaluate caching, actual skipped work and decoding after
                                                             the training recipe works.

 The catalog in the appendix provides dates, primary links and a short assessment for each paper. The
 machine-readable file also records routing, gradient handling and the scope of source review. Results in the papers
 remain author-reported.


 A current novelty boundary
 Recurrent modules, learned routing, pass-specific adapters and parallel latent slots already have
 substantial prior art. Aleph's proposed contribution is a training method: use held-out causal
 measurements to supervise communication, then test whether the resulting model composes skills and
 tolerates module replacement. A claim of novelty needs a comparison against these close precedents.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                       5


---

### PARSED TEXT FOR PAGE: 6 / 32

 05 / ALTERNATIVE TOPOLOGY


 A router can learn the graph
   Router action              Meaning                                      Role in Aleph

   Choose messages            Weight edges between recurrent module        Main experiment
                              states

   Control a state update     Choose whether a module updates or           Later clock experiment
                              retains its state

   Change input               Apply a small pass-dependent adapter         Steering placement experiment
   coordinates                before shared computation

   Choose a depth             Allocate more recurrent visits to selected   Later efficiency experiment
                              tokens or examples

   Call a typed organelle     Invoke a specialist through an explicit      TRM and LDT extension
                              state contract


 The first model executes all four neural operators. Its router controls communication. This preserves
 dense tensor shapes and keeps the measured intervention clear. Sparse dispatch and variable clocks
 can be added once the fixed-round graph has a usable result.

 Shared state contract
 Each module receives causally valid token states, a prompt/response mask, the round index and a
 reset identifier. It returns states of the same shape and declared diagnostics. Cache ownership and
 versioned interface metadata belong in the contract. A donor swap or module replacement must
 preserve the mask and readout position.

 Organelle exchange
 A TRM-style module can carry a candidate answer and a separate refinement state. An LDT bridge can
 accept typed proposals and return checked updates. Learned bridges count toward the neural budget;
 the symbolic checker's runtime is recorded separately.

 Test a cold swap between independently trained modules, followed by a fixed bridge-only adaptation
 budget. Keep the host frozen. Report the task loss immediately after replacement and the samples and
 updates required to recover. This gives interchangeability an observable endpoint.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                       6


---

### PARSED TEXT FOR PAGE: 7 / 32

 06 / PROPOSED ARCHITECTURE


 An explicit 1B design
                                                       Input shell
                                            Embeddings and two untied blocks




                                        States from the previous round
                                    Causal 4 x 4 routing weights mix module messages




        Operator 1                    Operator 2                     Operator 3                      Operator 4

     Seven shared blocks            Seven shared blocks            Seven shared blocks             Seven shared blocks




                                      Next-round states or output shell

                            Repeat the bank; then mix states and apply two untied output blocks



   Component                              Specification

   Token and feedforward widths           1536 and 4096; 12 attention heads

   Vocabulary                             65,536; tied input and output embeddings

   Fixed shell                            Two input blocks and two output blocks

   Recurrent bank                         Four distinct operators, seven transformer blocks each

   Default unroll                         Two synchronous rounds; three-round alternative in the pilot

   Control system                         Small graph router, state mixer, reserved rank-16 steering channels

   Parameter ledger                       1,007,141,013 under the calculator's declared design


 There are 32 physical transformer blocks. Two rounds execute 60 block calls in aggregate; a route
 through the graph traverses 18 sequential blocks including the shell. Three rounds execute 88 calls
 and allow a 25-block path. Modules retain distinct parameters across rounds.

 The count is an analytic design check. The implemented model must assert its own count, including any new adapters
 or learned normalization. The calculator specifies every counted control head.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                     7


---

### PARSED TEXT FOR PAGE: 8 / 32

 07 / PROPOSED LEARNING METHOD


 Train communication from interventions
 Let z[j,r,t] be module j's state at round r and token position t. A row-normalized matrix A[m,j,r,t] mixes
 messages for each recipient. All modules in a round read the previous-round snapshot. Their outputs
 become the next snapshot together.

 message[m,r,t] = sum_j A[m,j,r,t] * z[j,r,t]
 next_state[m] = gated_update(previous_state[m], operator[m](normalized_input))

 The first round uses identity communication. Learned routing starts in round two, after the distinct
 operators have produced different content. The router projects each module to 16 coordinates and
 concatenates them in module order before its 64-to-16 head. This preserves module identity and the
 parameter count.

 Mix the recipient state and routed message with an initial communication gate of 0.1, normalize at the
 boundary, and inject the fixed input embedding. Compute the routing softmax in FP32. Declare
 whether an operator returns a state or an update so its residual is applied once.

 The feedback loop
 At declared checkpoints, freeze a measurement copy. Use a small calibration set to compare intact
 execution with prespecified edge ablations or matched donor substitutions. Record the exact changed
 tensor, pass and next readout. An effect target is the increase in task loss when the edge is perturbed.

 A critic predicts those effects from information available at the routing decision. Cross-fit it on held-out
 items and train it on detached records. The initial router imitates the best measured candidate from a
 small bounded set of row distributions. Retain the current routing when a candidate fails to clear the
 declared noise threshold.

 L = L_task + lambda * KL(stopgrad(A_teacher) || A_router)
 + beta * KL(A_router || A_previous)


 What the critic must learn
 A useful edge can be necessary for the current policy while a stronger version of that edge makes the
 policy worse. Marginal effects also interact. Compare several normalized alternative routing rows under
 the same checkpoint before turning measured effects into a routing target. Reserve joint-edge
 perturbations for validation.

 Keep ordinary language-loss gradients through every selected round. This gives the model a direct
 training signal while the causal auxiliary objective is evaluated. Probe fitting and data generation
 consume a separately recorded budget.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                            8


---

### PARSED TEXT FOR PAGE: 9 / 32

 08 / MECHANISTIC STEERING


 VPD and BPD as measurement tools
 Goodfire's VPD means adVersarial Parameter Decomposition. Its public language-model study uses a
 67M model and decomposes parameter matrices into rank-one subcomponents, with causal-use
 estimates tested by adversarial perturbations. That scale supplies a method to adapt and measure.
 [S13]

 Your BPD extends decomposition to tied recurrent weights using shared physical factors and
 pass-indexed importance. The November 15 amendment adopts max pooling for temporal minimality
 and emphasizes matched-pass item-contrast donor swaps. It also preserves the earlier failures and
 requires explicit observation semantics. [S14]

 A bounded first use

   Phase               Proposed operation                                Evidence to save

   Warmup              Train the neural graph with the fixed objective   Loss, state norms and routing behavior

   Measurement         Decompose selected output/MLP matrices on         Factor identities, pass labels, adversarial
                       calibration batches                               tests

   Counterfactuals     Intervene on selected factors or message          Matched donor, dose, next readout and
                       edges                                             outcome

   Steering interval   Fit detached targets; update the router under     Critic calibration and realized task change
                       a trust region

   Refresh             Recheck targets at a later frozen checkpoint      Drift, failure cases and held-out transfer


 Begin with a few matrices and a fixed allowance for probing. Record summaries and selected
 examples. A full activation dump across 10B tokens would create tens of terabytes per tap at this
 width.

 Controls that make the signal assessable
 Use shuffled effect labels, matched-rank and matched-norm directions, and donor swaps at the same
 recurrent pass. Report calibration on a task family that did not fit the critic. Compare state-only
 controls with state-plus-clock controls so the result cannot be explained solely by a pass lookup.

 A component's causal importance describes the current computation. Choosing a training sample
 requires a further estimate of the update's benefit. Data Goblin should measure that benefit using
 paired short training branches before using it as a curation reward.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                   9


---

### PARSED TEXT FOR PAGE: 10 / 32

 09 / PROPOSED DATA PLAN


 A 10B-token curriculum
 The main run starts from new weights. Ten billion tokens means processed non-padding input
 exposures; repeated examples consume that count. Maintain separate counters for distinct content
 and the positions that receive loss. Freeze the tokenizer before the training comparison.

   Source class                         Initial exposure     Purpose
                                        budget

   Custom corpus                        2.5B                 Project knowledge and the user's preferred domains

   Curated general text                 3.5B                 Language coverage and broader knowledge

   Verified task-completion data        2.0B                 Mathematics, reading and executable code

   Procedural compositions              1.5B                 New combinations, state updates and controlled
                                                             mechanisms

   Failure-directed examples            0.5B                 A bounded pool for adaptive curation

   Total                                10.0B                One main Aleph run


 These amounts are a starting proposal. The size and composition of the custom corpus have not been
 measured here. Reallocate its unused share to the general pool and cap repeated exposure by source.
 Document sources, licenses, deduplication and generation provenance before freezing a manifest.

 Loss and masks
 Use response-only PrefixLM for paired tasks. General text can use causal language modeling or a fixed
 prefix-completion transformation, with the objective recorded per source. Keep the source mixture and
 loss weighting identical across the main steering comparison.

 Curation with measured learning value
 For a candidate batch, take paired short updates from the same saved state and evaluate on
 untouched development families. Reward the gain per occupied GPU-hour. Maintain a random
 exploration fraction and a broad-data floor to prevent the curator from repeatedly selecting narrow
 easy-to-measure mechanisms.

 For the first causal-routing claim, replay the same chosen data to its control. A separate small-scale
 factorial experiment can isolate the effect of curation itself. The final test set remains outside the critic,
 curator and architecture-selection loop.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                          10


---

### PARSED TEXT FOR PAGE: 11 / 32

 10 / EVALUATION AND IMPLEMENTATION PARITY


 Build the gym around consequences
   Family                    What to score                                        Why it matters

   Text and mathematics      Strict MMLU/ARC, GSM8K/MATH, DROP, held-out          Capability and retention
                             perplexity

   Code and repair           Executable tests, compiler success and exact         Data Goblin's practical work
                             task completion

   Composition               Held-out graph rules, longer chains and new          Reuse beyond familiar examples
                             operation orderings

   Evidence revision         Correct state after a premise changes; stale-state   Controlled persistence
                             errors

   Causal intervention       Predicted and observed effects on new items and      Measurement validity
                             joint perturbations

   Module exchange           Cold-swap loss and recovery under bridge-only        The organelle contract
                             adaptation

   Compute scaling           Quality at each loop count and measured wall         Return from added recurrence
                             time



 Compilation is a separate axis
 Run eager and compiled execution on the same fixed batch. Compare logits and parameter gradients
 within dtype-appropriate tolerances, then compare one optimizer update. Repeat for the selected
 attention backend, prefix mask, checkpointing setting and distributed layout. A speed gain is
 meaningful after these semantics agree.

 Use a future-token perturbation test: changing a later answer token must preserve earlier answer
 logits. A whole-sequence router summary can violate this condition even when each attention layer has
 a correct causal mask.

 HRM reference execution
 Use its native condition tokens and prefix token types. Capture generation-time states or label replay
 traces explicitly. Save all valid and invalid answers. The source audit records a context-length
 discrepancy between current HRM and baseline configurations, which a common evaluation runner
 should eliminate. [S4,S5]

 Separate compilation cost, warmup time, active training steps, evaluation and time holding an
 allocated node. Save the package versions and the selected source revision with each run.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                               11


---

### PARSED TEXT FOR PAGE: 12 / 32

 11 / PROPOSED COMPARISONS


 The experiments that identify the effect
   Arm                      Graph policy                                Mechanistic signal

   A: fixed graph           Prespecified communication and round        Absent
                            count

   B: task-trained graph    Learned from ordinary task gradients        Absent

   C: causal-feedback       Same trainable graph as B                   Measured intervention targets
   graph

   D: sham feedback         Same training schedule and auxiliary cost   Shuffled or matched-control targets
                            as C


 Run the full set at reduced scale before committing the 1B model. A suggested pilot uses roughly
 100-200M parameters, a shared data slice and multiple seeds. Choose the final size and token count
 from the measured debit; reserve most of the allocation for Aleph's 10B run.

 Primary outcome
 Predeclare a macro-average of exact completion across held-out composition families at a fixed
 inference budget. Report each family, variation across seeds and a paired item-level interval. Set a
 practical improvement threshold from pilot variance and cost before the full-scale result is inspected.

 Main 1B comparison
 If the measured budget supports it, train C and B for 10B tokens each. This isolates the causal auxiliary
 objective while both learn their graph. A full-length fixed graph is useful when funds permit. When only
 the main 10B run fits, use declared checkpoint branches and limit the causal claim to the interval they
 test.

 Steering placement
 Compare boundary steering with pass-conditioned inner steering at small scale. Match trainable
 capacity, total intervention dose and gradient access. In a separate freeze experiment, keep the
 input/output shell weights fixed after warmup while the recurrent bank trains. Fixed location, fixed visit
 count and frozen weights are recorded independently.

 Failure remains informative
 A poor effect critic, collapse to the same graph, failure under module swaps, or gains matched by the
 sham each answers a different question. Save those results with the checkpoint. A successful model
 can still leave the mechanistic hypothesis unresolved.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                          12


---

### PARSED TEXT FOR PAGE: 13 / 32

 12 / COMPUTE LAYOUT


 What eight H100s can parallelize
   Layout                        Work distributed                           Engineering consequence

   Eight-way DDP                 Different examples through full replicas   Start here; exact joint gradients and a
                                                                            simple reference

   Four operators x two          Four branches of one recurrent round       Autograd-aware state exchange and
   replicas                                                                 matched-module gradient sync

   Pipeline parallelism          Different microbatches along               A sample still follows sequential
                                 successive layer groups                    dependencies

   Tensor or FSDP sharding       Matrix work or model state                 Communication can dominate at this
                                                                            model size

   Local or synthetic            Approximate targets for dependent          Additional learning hypothesis and
   gradients                     module updates                             calibration tests


 The proposed graph offers concurrency across its four operators within each round. Each seven-block
 operator retains internal dependencies, and the next round consumes completed previous-round
 states. Exact joint backpropagation applies to the resulting finite graph.

 Benchmark DDP against four-way operator sharding with two data replicas. Keep identical global
 batches and compare gradients before timing. The sharded variant needs a declared placement for the
 input/output shell and causal router. NVLink topology matters for repeated state exchange. [S15]

 Independent layer updates
 Synthetic gradients predict the downstream learning signal and can support asynchronous updates. A
 BPD factor map supplies a coordinate system for interventions; an accurate gradient predictor requires
 a separate objective and validation. Include occasional full-gradient anchors if this becomes a later
 experiment. [S16]

 Memory and recording
 The calculator budgets about 18-20 GB for a replicated mixed-precision training state with Adam and
 EMA, before activations and temporary tensors. Final BF16 weights are about 2 GB. Measure peak
 memory with the actual loss kernel, activation checkpointing and batch packing.

 At 10B exposures, uint16 token IDs require about 20 GB before metadata. Sparse calibration records
 keep mechanistic analysis within a manageable storage budget.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                  13


---

### PARSED TEXT FOR PAGE: 14 / 32

 13 / ANALYTIC ESTIMATE


 A cost model for the full 10B run
 The calculation assumes H100 SXM dense BF16 peak throughput of 989.4 TFLOP/s, full gradients
 through every chosen round, 4,096-token sequences and an average prompt fraction of one half. The
 central case uses 30% model FLOP utilization during active steps plus a chosen 25% allowance outside
 those steps. [S17]

   Rounds        Model FLOPs                   GPU-hours             8-GPU wall time              At $3 / GPU-hour

   2             1.363 x 10^20                 159                   19.9 h                       $478

   3             1.970 x 10^20                 231                   28.8 h                       $692

 All prices in this page are illustrative accounting rates. Silico's actual debit, availability and charges for other grant
 services remain unverified.

   Assumption                                                      Two rounds                       Three rounds

   20-45% active-step MFU, with allowance                          106-239 GPU-hours                154-346 GPU-hours

   At $2 per GPU-hour, central case                                $319                             $461

   At $4 per GPU-hour, central case                                $638                             $922

   At $6 per GPU-hour, central case                                $956                             $1,383

   10% MFU downside, with allowance                                478 GPU-hours                    692 GPU-hours



 Cost of a credible comparison
 At the central two-round estimate and an illustrative $3 rate, two 10B-token runs total about $956.
 Against a $1,560 envelope, that leaves about $604 for pilots, evaluation and recovery. At $4, the pair
 costs about $1,275. Whole-job throughput from a pilot should replace the MFU estimate before
 allocation.

 How to replace the assumptions
 Log processed non-padding tokens, GPU count, occupied wall time and RU debit. Project the remaining
 cost from tokens per billed GPU-hour. A one-hour eight-GPU allocation consumes eight GPU-hours.
 Include probe fitting and copies in the same ledger.

 The calculator includes 2K and 4K cases, parameter assertions and memory estimates. Within-step recomputation and
 communication are already represented when MFU is measured from step time; the outside-step allowance must avoid
 double counting them.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                          14


---

### PARSED TEXT FOR PAGE: 15 / 32

 14 / WORK SEQUENCE


 Use the eleven days to freeze a runnable
 experiment
   Window                Work                                              Completion evidence

   Nov 20-21             Freeze the data contract; repair strict scoring   Causal-mask test, gradient parity, restart
                         and PrefixLM; implement the graph                 test and real RU debit

   Nov 22-23             Run reduced-scale graph and steering              Loss stability, held-out effect prediction and
                         controls; select two or three rounds              a frozen comparison plan

   Nov 24-27             Train Aleph1B through 10B tokens; run the         Token ledger, resumable checkpoints and
                         full control if its debit fits                    probe records

   Nov 28-29             Score transfer, code repair and swaps;            Raw generations, verified outcomes and
                         execute declared checkpoint branches              uncertainty estimates

   Nov 30                Package the trained result and the research       Checkpoint, manifests, runner, evaluator
                         record                                            and candid report

 These dates use the eleven-day window stated on November 19. They do not assert an independently checked grant
 expiry or account limit. Queue delays and failed experiments belong in the recovery reserve.


 Freeze scope after the pilot
 The core bet is causal supervision of a learned recurrent communication graph. Give the 10B run a
 stable data order and finite gradient horizon. Record optional adapters and organelle experiments as
 separately named branches so they remain attributable.

 What to preserve from intensive ChatGPT work
 Produce a pinned environment, deterministic corpus manifest, executable training command and a
 small continuation note at every milestone. Generated code needs runtime checks. Save the evaluator
 and restart procedure early enough that the experiment can continue after the current access window.

 Release contract
 Publish the architecture and parameter ledger with weights when permitted by the data and grant
 terms. Include actual tokens, occupied GPU-hours, RU debit and all selected comparisons. Separate
 measured findings from proposed mechanisms. A release should say exactly which tasks the model
 can complete and which research claims remain open.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                    15


---

### PARSED TEXT FOR PAGE: 16 / 32

 15 / EVIDENCE BEHIND THE AUDIT


 Primary source register
 S1 HRM-Text paper, v1
 https://arxiv.org/html/2605.20613v1

 S2 External dense retraining in the HRM-MoE model card
 https://huggingface.co/Xiaoye08/HRM-MoE

 S3 DFM historical comparison table
 https://huggingface.co/danish-foundation-models/DFM-Mimir-v1.5/blob/main/original_comparisons.md

 S4 Sapient checkpoint card and usage contract
 https://huggingface.co/sapientinc/HRM-Text-1B

 S5 Pinned Sapient scorer, aaa948ea
 https://github.com/sapientinc/HRM-Text/blob/aaa948ea674fd84b7bc455c9cfb455ecfefdf914/evaluation/benchmarks.py

 S6 DFM scorer change, August 31
 https://github.com/schneiderkamplab/HRM-Text/commit/d6a83e39ca1d284e20b9773fd69f351f1c80f0ed

 S7 ARC Prize independent HRM analysis
 https://arcprize.org/blog/hrm-analysis

 S8 TRM primary paper
 https://arxiv.org/html/2510.04871v1

 S9 Unresolved token-accounting report
 https://github.com/sapientinc/HRM-Text/issues/27

 S10 Sapient news
 https://sapient.inc/en/news

 S11 Sapient public model and dataset listings
 https://huggingface.co/sapientinc

 S12 PRAXIST official system description
 https://praxist.sapient.inc/en/about

 S13 Goodfire VPD research
 https://www.goodfire.com/research/interpreting-lm-parameters

 S14 BPD amendments, November 15
 https://github.com/MoralityLabAI/BPD/blob/main/mechinterp/bpd/docs/amendments_2026-11-15_fable.md

 S15 NVIDIA eight-GPU reference architecture
 https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html

 S16 Synthetic gradients, ICML 2017
 https://proceedings.mlr.press/v70/jaderberg17a.html

 S17 NVIDIA H100 architecture whitepaper
 https://resources.nvidia.com/en-us-hopper-architecture




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                               16


---

### PARSED TEXT FOR PAGE: 17 / 32

 APPENDIX / SCOPE AND EVIDENCE


 Annotated literature catalog
 This catalog contains 48 primary papers: 41 core entries, five adjacent methods for module routing and
 parallel learning, and two additional scaling/decoding studies. Search cutoff: November 19, 2026. This
 scoped research map covers the main design decisions for Aleph.

 Dates indicate first submission. Later material revisions appear in the machine-readable record.
 Primary abstracts and metadata were checked for every core entry. Full training details were reviewed
 selectively; each record states that boundary. All model results remain author-reported.

 P01 Adaptive Computation Time for Recurrent Neural Networks
 2016-03-29 | Foundation: differentiable adaptive depth

 https://arxiv.org/abs/1603.08983

 Finding: Introduces learned recurrent computation time, with a differentiable halting construction and a cost for extra
 computation.

 Boundary: The evidence predates modern large decoder language models. The character-language gains were
 modest.

 Use for Aleph: Implement a simple ponder-cost controller as a reference and track quality against actual executed
 visits.



 P02 Universal Transformers
 2018-07-10 | Foundation: shared recurrent transformer depth

 https://arxiv.org/abs/1807.03819

 Finding: Combines a shared transformer transition over depth with parallel sequence processing and optional
 position-wise adaptive computation.

 Boundary: Theoretical universality depends on assumptions. It does not establish learnability under the proposed
 10B-token budget.

 Use for Aleph: Use as the architectural ancestor and implement fixed-depth sharing before adding learned routing.



 P03 ALBERT: A Lite BERT for Self-supervised Learning of Language
 Representations
 2019-09-26 | Foundation: fixed-depth parameter sharing

 https://arxiv.org/abs/1909.11942

 Finding: Combines cross-layer parameter sharing, factorized embeddings, and a sentence-order objective to reduce
 parameters in a BERT-style encoder.

 Boundary: Encoder results do not establish autoregressive recurrent reasoning. Fewer distinct parameters do not
 remove per-visit arithmetic.

 Use for Aleph: Include a plain tied-weight baseline to separate the effect of sharing from adaptive control.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                   17


---

### PARSED TEXT FOR PAGE: 18 / 32

 APPENDIX / PRIMARY PAPERS


 Literature catalog, continued
 P04 Deep Equilibrium Models
 2019-09-03 | Foundation: equilibrium recurrence and implicit gradients

 https://arxiv.org/abs/1909.01377

 Finding: Solves for the equilibrium of a tied network and differentiates implicitly, avoiding storage of every
 effective-depth activation.

 Boundary: Root convergence and Jacobian conditioning matter. Constant activation memory with depth does not
 imply negligible solver cost.

 Use for Aleph: Use as a conceptual and small-scale control for stability, and distinguish implicit gradients from
 last-step approximations.



 P05 PonderNet: Learning to Ponder
 2021-07-12 | Foundation: probabilistic halting

 https://arxiv.org/abs/2107.05407

 Finding: Learns a distribution over stopping times and trains expected prediction quality alongside a
 computation-related regularizer.

 Boundary: Its results do not establish behavior in a 1B-parameter decoder trained on 10B tokens.

 Use for Aleph: Compare probabilistic stopping with ACT-style and fixed-budget policies, calibrating all policies on the
 same development split.



 P06 Scalable Adaptive Computation for Iterative Generation
 2022-12-22 | Adjacent: Recurrent Interface Networks and latent interfaces

 https://arxiv.org/abs/2212.11972

 Finding: Recurrent Interface Networks use a small latent computation array and cross-attention to read and write a
 much larger set of interface tokens.

 Boundary: The measured efficiency comes from diffusion, so its numerical gains cannot be transferred directly to
 autoregressive text.

 Use for Aleph: Borrow a typed read/compute/write contract for small organelles that interact with the backbone
 through a narrow interface.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                       18


---

### PARSED TEXT FOR PAGE: 19 / 32

 APPENDIX / PRIMARY PAPERS


 Literature catalog, continued
 P07 Looped Transformers as Programmable Computers
 2023-01-30 | Theory: programmable recurrent computation

 https://arxiv.org/abs/2301.13196

 Finding: Constructs looped transformers that implement program counters, branches, memory operations, and useful
 numerical routines.

 Boundary: Representability leaves open how training discovers the program and how robustly it runs under finite
 precision.

 Use for Aleph: Build algorithmic evaluation families with known state transitions and separately test whether training
 learns them.



 P08 CoTFormer: A Chain-of-Thought Driven Architecture with
 Budget-Adaptive Computation Cost at Inference
 2023-10-16 | Architecture: repeated computation with loop-history attention

 https://arxiv.org/abs/2310.10845

 Finding: Enables repeated model computation and variable token-wise inference work while exposing information
 from earlier repetitions.

 Boundary: Attention over loop history creates memory and dependency costs that differ from a single recurrent state.

 Use for Aleph: Provide a comparator for designs where an organelle can retrieve prior loop states instead of
 overwriting one state.



 P09 Investigating Recurrent Transformers with Dynamic Halt
 2024-02-01 | Architecture: gated recurrence and global halting

 https://arxiv.org/abs/2402.00976

 Finding: Studies depth-wise Universal Transformers and chunk-wise Temporal Latent Bottleneck models, introducing
 gated updates and a transition-aware global stopping rule.

 Boundary: Diagnostic-task success needs separate validation on broad text modeling.

 Use for Aleph: Compare reversible update gating with irreversible halting and test a global controller when per-token
 routing complicates compilation.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                  19


---

### PARSED TEXT FOR PAGE: 20 / 32

 APPENDIX / PRIMARY PAPERS


 Literature catalog, continued
 P10 AlgoFormer: An Efficient Transformer Framework with Algorithmic
 Structures
 2024-02-21 | Architecture: fixed prelude, recurrent core, fixed output

 https://arxiv.org/abs/2402.13572

 Finding: Organizes a model into a pre-transformer, a shared looped transformer for iterative computation, and a
 post-transformer.

 Boundary: A useful algorithmic topology alone does not establish natural-language pretraining quality.

 Use for Aleph: Use this as a clear architectural ancestor for onion-like fixed shells around a recurrent center.



 P11 Mixture-of-Depths: Dynamically allocating compute in
 transformer-based language models
 2024-04-02 | Routing: fixed-capacity token selection

 https://arxiv.org/abs/2404.02258

 Finding: Allocates attention and feed-forward work to a fixed number of selected tokens at each layer, keeping tensor
 capacity static while token identities vary.

 Boundary: Causality, routing overhead, and agreement between training and inference need implementation-level
 checks.

 Use for Aleph: Use as a non-MoE routing baseline whose fixed capacities suit compiled kernels.



 P12 On Expressive Power of Looped Transformers: Theoretical Analysis
 and Enhancement via Timestep Encoding
 2024-10-02 | Theory and architecture: loop-index conditioning

 https://arxiv.org/abs/2410.01405

 Finding: Analyzes approximation power and introduces timestep-dependent scaling to strengthen a shared looped
 transformer.

 Boundary: Approximation capacity does not quantify sample efficiency or optimization reliability at the proposed
 scale.

 Use for Aleph: Ablate loop-index embeddings or scales before adding independent parameters to every recurrent
 pass.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                 20


---

### PARSED TEXT FOR PAGE: 21 / 32

 APPENDIX / PRIMARY PAPERS


 Literature catalog, continued
 P13 Relaxed Recursive Transformers: Effective Parameter Sharing with
 Layer-wise LoRA
 2024-10-28 | Architecture: shared core with loop-specific low-rank parameters

 https://arxiv.org/abs/2410.20672

 Finding: Converts a pretrained dense transformer into repeated shared computation and restores depth-specific
 flexibility with low-rank adaptations.

 Boundary: A converted model inherits the donor's pretraining. Theoretical throughput estimates need measurement
 on the intended runtime.

 Use for Aleph: Use depth-specific LoRA as a low-cost organelle specialization mechanism and test conversion as an
 alternative to training a new backbone.



 P14 Training Large Language Models to Reason in a Continuous Latent
 Space
 2024-12-09 | Adjacent: latent-token reasoning

 https://arxiv.org/abs/2412.06769

 Finding: Feeds the last hidden state back as the next input embedding to perform reasoning in continuous latent
 positions.

 Boundary: Latent positions extend the sequence and its cache, giving different costs and dependencies from recurring
 over depth.

 Use for Aleph: Keep a latent-token comparator so that extra useful computation can be distinguished from one
 specific recurrent architecture.



 P15 Scaling up Test-Time Compute with Latent Reasoning: A Recurrent
 Depth Approach
 2025-02-07 | Architecture and training: large recurrent-depth language model

 https://arxiv.org/abs/2502.05171

 Finding: Uses fixed input and output sections around a recurrent core, repeated input injection, and variable forward
 depth.

 Boundary: The training scale exceeds Aleph's budget. Its appendix documents an implementation-specific interaction
 between autocast and truncated backpropagation where only compiled execution had correct gradients.

 Use for Aleph: Use the sandwich layout and gradient horizon as major controls; verify eager/compiled loss and
 selected gradients in the actual implementation.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                     21


---

### PARSED TEXT FOR PAGE: 22 / 32

 APPENDIX / PRIMARY PAPERS


 Literature catalog, continued
 P16 Reasoning with Latent Thoughts: On the Power of Looped
 Transformers
 2025-02-24 | Theory and evaluation: recurrence versus untied depth

 https://arxiv.org/abs/2502.17416

 Finding: Analyzes the reasoning capacity of looped transformers, their relationship to chain-of-thought, and empirical
 differences between reasoning and memorization.

 Boundary: Theorems and controlled tasks leave architecture-specific optimization and broad knowledge retention
 open.

 Use for Aleph: Separate knowledge tests from computation tests and compare models under both effective-depth and
 FLOP budgets.



 P17 Hierarchical Reasoning Model
 2025-06-26 | Architecture and training: hierarchical recurrent state

 https://arxiv.org/abs/2506.21734

 Finding: Runs high-level and low-level recurrent modules at different frequencies and applies deep supervision across
 refinement rounds.

 Boundary: Puzzle performance does not establish language-model pretraining quality, and approximate credit
 assignment is a design choice requiring measurement.

 Use for Aleph: Retain H/L recurrence as a benchmarked candidate and expose its clocks and gradient horizon
 separately.



 P18 Mixture-of-Recursions: Learning Dynamic Recursive Depths for
 Adaptive Token-Level Computation
 2025-07-14 | Routing: adaptive depth with a shared transformer

 https://arxiv.org/abs/2507.10524

 Finding: Combines parameter sharing with token-dependent recursive depth, attention over active tokens, and
 selective KV caching.

 Boundary: Dynamic routing and cache policies create runtime overhead and require train/inference consistency
 checks.

 Use for Aleph: Use as a central non-MoE allocation baseline; measure routing cost and effective batch utilization.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                      22


---

### PARSED TEXT FOR PAGE: 23 / 32

 APPENDIX / PRIMARY PAPERS


 Literature catalog, continued
 P19 Less is More: Recursive Reasoning with Tiny Networks
 2025-10-06 | Architecture and training: tiny latent and answer refinement

 https://arxiv.org/abs/2510.04871

 Finding: A small shared network repeatedly updates a latent state and then an answer state, with deep supervision
 and learned correctness-based stopping.

 Boundary: Fixed-context puzzle learning leaves transfer to 1B-parameter text modeling untested.

 Use for Aleph: Represent a TRM-style solver as an interchangeable latent/answer-update organelle with an explicit
 gradient contract.



 P20 Parallel Loop Transformer for Efficient Test-Time Computation
 Scaling
 2025-10-28 | Execution and architecture: parallel recurrent-depth work

 https://arxiv.org/abs/2510.24824

 Finding: Staggers loop computation across tokens and combines first-loop KV sharing with gated sliding-window
 attention.

 Boundary: The dependency graph changes. A later matched study, How to Loop Your Transformer, finds that input
 and normalization changes are needed for robust quality in its setting.

 Use for Aleph: Keep as an execution experiment after a serial reference exists, checking task behavior and selected
 gradients before trusting speed.



 P21 Scaling Latent Reasoning via Looped Language Models
 2025-10-29 | Architecture and training: full-stack looped language models

 https://arxiv.org/abs/2510.25741

 Finding: Pretrains looped language models with a learned exit mechanism and entropy regularization.

 Boundary: Its data scale is far above 10B tokens. Parameter-efficient inference quality does not establish affordable
 from-scratch training.

 Use for Aleph: Use a full-stack loop as a leading control and evaluate its exit policy against fixed-depth and simple
 statistical criteria.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                     23


---

### PARSED TEXT FOR PAGE: 24 / 32

 APPENDIX / PRIMARY PAPERS


 Literature catalog, continued
 P22 Think-at-Hard: Dynamic Looped Transformers for Improved
 Reasoning
 2025-11-11 | Routing: selective token refinement and depth-aware LoRA

 https://arxiv.org/abs/2511.08577

 Finding: Uses a learned decider, depth-aware low-rank adaptation, and attention across token and iteration axes to
 allocate latent refinement.

 Boundary: Oracle headroom and a learned policy have different attainable performance. Savings require validation
 under actual sparse execution.

 Use for Aleph: Compare pass-conditioned low-rank steering and token selection with a controller that only decides
 halting.



 P23 Fast and Accurate Causal Parallel Decoding using Jacobi Forcing
 2025-12-16 | Adjacent execution: training for parallel token refinement

 https://arxiv.org/abs/2512.14681

 Finding: Trains on model-generated Jacobi decoding trajectories while preserving causal structure, with progressive
 distillation.

 Boundary: Benefits concern decoding and require the relevant trajectory distribution. They do not directly reduce
 recurrent-depth pretraining cost.

 Use for Aleph: Use as evidence that an execution change may require training on its own intermediate states.



 P24 Prioritize the Process, Not Just the Outcome: Rewarding Latent
 Thought Trajectories Improves Reasoning in Looped Language Models
 2026-02-11 | Training: credit assignment across latent trajectories

 https://arxiv.org/abs/2602.10520

 Finding: Distributes reward-derived supervision across latent reasoning trajectories instead of concentrating learning
 only on the final state.

 Boundary: Evidence concerns post-training. Reward construction and task validity still govern what the trajectory
 learns.

 Use for Aleph: Include trajectory reward as a training control beside a mechanistic auxiliary loss and test whether
 either improves held-out behavior.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                      24


---

### PARSED TEXT FOR PAGE: 25 / 32

 APPENDIX / PRIMARY PAPERS


 Literature catalog, continued
 P25 Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth
 Transformers
 2026-04-09 | Evaluation: compositional and depth generalization

 https://arxiv.org/abs/2604.07822

 Finding: Studies recurrent-depth models on compositional reasoning and depth extrapolation, including learning
 dynamics and degradation from excessive iteration.

 Boundary: Additional depth can reduce quality. Results on controlled tasks do not guarantee broad-language gains.

 Use for Aleph: Include held-out composition and depth tests, with performance curves across iteration counts.



 P26 HRM-Text: Efficient Pretraining Beyond Scaling
 2026-05-20 | Architecture and training: hierarchical recurrence for text

 https://arxiv.org/abs/2605.20613

 Finding: Adapts hierarchical recurrence to text with MagicNorm and a PrefixLM task-completion pretraining objective.

 Boundary: That reported cost does not quote a Silico run. The task-structured 40B-token recipe does not establish
 performance from an arbitrary 10B-token mixture.

 Use for Aleph: Use as the closest baseline for the proposed Gym and preserve PrefixLM scoring, recurrence, and
 gradient semantics in each backend.



 P27 Training-Free Looped Transformers
 2026-05-22 | Intervention: frozen-block recurrence with damping

 https://arxiv.org/abs/2605.23872

 Finding: Reuses contiguous middle layers of frozen models through damped substeps motivated by a
 numerical-integration interpretation.

 Boundary: Reported gains are modest and task-dependent; naive repeated application can hurt.

 Use for Aleph: Use a cheap frozen-model baseline to test whether damped recurrence helps before funding
 pretraining.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                    25


---

### PARSED TEXT FOR PAGE: 26 / 32

 APPENDIX / PRIMARY PAPERS


 Literature catalog, continued
 P28 CHASE: Cache-Hole-Adapted Skip Exit for Looped State-Space
 Language Models
 2026-07-11 | Architecture and execution: looped state-space models

 https://arxiv.org/abs/2607.10110

 Finding: Adapts looped Mamba and hybrid models to the missing recurrent-state updates produced by real skipped
 computation.

 Boundary: The latest version still reports stronger perplexity from deep untied controls at matched FLOPs. State
 semantics constrain speedups.

 Use for Aleph: Require each stateful organelle to define what happens to hidden state when its update is skipped.



 P29 Beyond the Training Horizon: Mechanisms and Limits of Length
 Generalization in Looped Transformers
 2026-09-27 | Mechanisms and evaluation: generalization beyond training depth

 https://arxiv.org/abs/2609.33144

 Finding: Studies algorithmic state representations and causally tests directions associated with processing status,
 such as live/consumed or fresh/aged.

 Boundary: Local errors can accumulate beyond the training horizon. Decoding an intermediate state does not by itself
 prove faithful algorithm execution.

 Use for Aleph: Probe both content and processing status at each pass, then test interventions on held-out lengths and
 depths.



 P30 Shared Weights, Selected Computations: How Looped
 Transformers Route What Each Loop Does
 2026-09-30 | Mechanisms and control: state-dependent computational routing

 https://arxiv.org/abs/2609.39892

 Finding: Shows that a learned linear transform of the state entering a frozen looped transformer can select among
 computations, with attention interventions testing the causal route.

 Boundary: Selecting computations already available in a backbone does not establish the creation of arbitrary new
 algorithms.

 Use for Aleph: Prioritize small pass-conditioned state controllers as an organelle design and use matched
 attention-path interventions.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                   26


---

### PARSED TEXT FOR PAGE: 27 / 32

 APPENDIX / PRIMARY PAPERS


 Literature catalog, continued
 P31 Closing the Loop: Practical Training Recipes for Looped Language
 Models
 2026-09-30 | Training: scratch and pretrained conversion recipes

 https://arxiv.org/abs/2610.00673

 Finding: Studies learning-rate warmup, exit regularization, curated midtraining, and conversion of a pretrained dense
 model using scalar input mixing and a smoothed exit loss.

 Boundary: The scratch recipe uses 31 times the proposed token budget. Conversion needs a matched
 continued-training dense control.

 Use for Aleph: Use as a recipe reference when selecting scratch training versus continued training from an existing
 checkpoint.



 P32 Towards Looped Models Done Right, Part II: Rethinking at Fixed
 Points
 2026-10-05 | Architecture and training: adaptive near-equilibrium recurrence

 https://arxiv.org/abs/2610.06833

 Finding: Develops prediction-feedback and entropy-based depth control, orthogonal input injection, terminal-state KV
 sharing, and methods for efficient adaptation and prefill.

 Boundary: Near-fixed-point behavior must be checked empirically, and distilled prefill introduces an
 accuracy/compute tradeoff.

 Use for Aleph: Ablate input channels and scheduler signals independently; log convergence and useful change rather
 than relying on an assumed equilibrium.



 P33 Recurrent Looped Transformer
 2026-10-06 | Architecture: temporal recurrent state with a causal encoder

 https://arxiv.org/abs/2610.07591

 Finding: Combines a parallel causal encoder with a recurrent decoder that carries the preceding token's final state.

 Boundary: Temporal recurrence introduces serial dependencies and differs from depth recurrence. Broad-language
 scaling remains unestablished here.

 Use for Aleph: Treat persistent inter-token state as a separate organelle contract and evaluate chunk boundaries
 explicitly.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                     27


---

### PARSED TEXT FOR PAGE: 28 / 32

 APPENDIX / PRIMARY PAPERS


 Literature catalog, continued
 P34 One Model to Loop Them All: Elastic Latent Recurrence for
 Language Models
 2026-10-08 | Training and control: elastic recurrence budgets

 https://arxiv.org/abs/2610.12582

 Finding: Uses gated step sizes, loop-index and budget conditioning, and mixed-depth training to support different
 inference budgets in one model.

 Boundary: Transfer to 1B parameters and 10B tokens is untested. Last-loop KV approximations require separate
 quality checks.

 Use for Aleph: Use late elastic-depth training as a curriculum candidate and track retention of useful one-loop
 behavior.



 P35 Monitoring latent reasoning in looped language models
 2026-10-13 | Observability: comparing latent reasoning trajectories

 https://arxiv.org/abs/2610.18127

 Finding: Uses several inspection perspectives to compare looped language-model trajectories, including the effects of
 carrying state and rereading the input.

 Boundary: A monitoring signal can describe a trajectory without demonstrating that training on the signal improves
 task performance.

 Use for Aleph: Organize instrumentation across passes and test whether each proposed metric predicts errors on
 held-out tasks.



 P36 Towards Looped Models Done Right, Part I: Topology, Input
 Injection, Recurrent-State Design
 2026-10-16 | Matched ablations: topology, input injection, and state design

 https://arxiv.org/abs/2610.23036

 Finding: Separates layout, input injection, and recurrent-state choices under matched parameter count, logical depth,
 and training tokens.

 Boundary: Results for H/L state and random initialization are mixed or negative in this setting. They do not establish a
 universal ranking across budgets and objectives.

 Use for Aleph: Use as a template for the Gym's ablation matrix and keep input injection separate from
 hierarchical-state claims.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                    28


---

### PARSED TEXT FOR PAGE: 29 / 32

 APPENDIX / PRIMARY PAPERS


 Literature catalog, continued
 P37 Dual-Slot: From Serial Loops to Parallel Slots in Language Models
 2026-10-18 | Architecture and training: parallel slots and Jacobi iteration

 https://arxiv.org/abs/2610.25086

 Finding: Separates latent and prediction slots so current predictions can access deeper states of prior tokens, using a
 small number of parallel Jacobi iterations during training.

 Boundary: Changed dependencies and finite iteration counts require convergence and causal checks. It cannot be
 assumed to preserve a serial HRM's gradients.

 Use for Aleph: Keep as a later candidate for reducing training serialization once a measured serial baseline exists.



 P38 Cross-Loop Query Handoff: How Later Loops Redirect Two-Hop
 Retrieval in the Ouro Looped Language Models
 2026-10-25 | Mechanistic interventions: attention-query routing

 https://arxiv.org/abs/2610.34639

 Finding: Transplants later-loop queries into earlier loops to test how a looped model redirects attention during two-hop
 retrieval.

 Boundary: The intervention uses a collection of query states. It does not locate every computation that creates the
 bridge representation.

 Use for Aleph: Add matched query-patching and shuffled-donor controls, then report synthetic and natural-language
 transfer separately.



 P39 How to Loop Your Transformer: Loop Scaling, Specialization, and
 Parallel Inference
 2026-10-29 | Matched scaling study: loop placement, LoRA, and parallel inference

 https://arxiv.org/abs/2610.40325

 Finding: Studies how loop count, loop placement, per-loop specialization, and parallel execution affect quality under
 different resource budgets.

 Boundary: Depth gains can saturate, reverse, or become unstable. Full-stack recurrence wins a fixed-parameter
 comparison while middle recurrence wins a training-FLOP comparison.

 Use for Aleph: Use as the central budget-aware architecture reference; include current-token injection and separate
 normalization in parallel-loop controls.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                       29


---

### PARSED TEXT FOR PAGE: 30 / 32

 APPENDIX / PRIMARY PAPERS


 Literature catalog, continued
 P40 When Does Adaptive Halting Win? Selection Rules Matter in
 Looped Language Models
 2026-11-02 | Evaluation: policy selection and halting validity

 https://arxiv.org/abs/2611.02333

 Finding: Shows that conclusions about learned halting depend on how its operating point is selected, and compares
 trained stability signals with native and untrained criteria.

 Boundary: Savings are estimated from full-depth traces rather than demonstrated as end-to-end runtime savings.
 Out-of-distribution budget failures remain.

 Use for Aleph: Predeclare a quality-loss/compute selection rule, retain an untrained KL baseline, and verify the chosen
 policy with true skipping.



 P41 LoopSkim: Lossless Multi-Depth Speculative Decoding for Looped
 Language Models
 2026-11-15 | Execution: speculative decoding with intermediate-loop readouts

 https://arxiv.org/abs/2611.19896

 Finding: Uses intermediate-depth readouts to screen draft tokens and reserves full-depth evaluation for commitment,
 aiming to preserve the target model's output.

 Boundary: The lossless statement assumes exact arithmetic and specified random draws. Decoding improvements do
 not establish cheaper pretraining.

 Use for Aleph: Consider after the model is trained, with finite-precision output parity and latency checked on the
 target runtime.



 P42 Mixture-of-Depths Attention
 2026-03-16 | Adjacent architecture: attention across sequence and depth

 https://arxiv.org/abs/2603.15619

 Finding: Lets each attention head access current sequence keys and values alongside depth keys and values from
 preceding layers; tested on1.5B models.

 Boundary: A cross-depth attention mechanism requires its own memory and kernel accounting. The paper does not
 validate Aleph's causal router.

 Use for Aleph: Consider depth-state access as a controlled alternative to the proposed module-message mixer.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                   30


---

### PARSED TEXT FOR PAGE: 31 / 32

 APPENDIX / PRIMARY PAPERS


 Literature catalog, continued
 P43 Recurrent Independent Mechanisms
 2019-09-24 | Adjacent: modular recurrent networks and selective communication

 https://arxiv.org/abs/1909.10893

 Finding: Divides recurrent computation into separate mechanisms with selective activation and communication.

 Boundary: Modular recurrence alone does not establish plug-compatible learned modules at 1B language-model scale.

 Use for Aleph: Inform an organelle interface where inactive modules retain state and active modules exchange a
 bounded message.



 P44 Dynamic Inference with Neural Interpreters
 2021-10-12 | Adjacent: reusable learned functions and modular computation

 https://arxiv.org/abs/2110.06399

 Finding: Routes inputs through learned functions in a self-attention architecture, composing computation along width
 and depth.

 Boundary: A modular model design does not establish interchangeability between independently trained modules
 without an interface and calibration contract.

 Use for Aleph: Inform typed function signatures and compatibility tests for LDT/TRM-style organelles.



 P45 Decoupled Neural Interfaces using Synthetic Gradients
 2016-08-18 | Adjacent training: synthetic gradients and decoupled credit assignment

 https://arxiv.org/abs/1608.05343

 Finding: Learns local gradient predictors so a module can update before downstream gradient computation has
 completed.

 Boundary: Approximate local gradients can drift from the end-to-end task objective and add another learned
 estimator.

 Use for Aleph: Consider only as a later organelle-decoupling experiment after full-gradient and truncated-gradient
 baselines are measured.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                    31


---

### PARSED TEXT FOR PAGE: 32 / 32

 APPENDIX / PRIMARY PAPERS


 Literature catalog, continued
 P46 Accelerating Feedforward Computation via Parallel Nonlinear
 Equation Solving
 2020-02-10 | Adjacent execution: parallel solution of sequential network equations

 https://arxiv.org/abs/2002.03629

 Finding: Recasts sequential network evaluation as a system of nonlinear equations that iterative solvers can update in
 parallel.

 Boundary: Finite solver iteration can approximate sequential computation, so residual, convergence, and gradient
 agreement require checks.

 Use for Aleph: Provide the numerical-method context for block-parallel and Jacobi experiments; measure residuals
 and gradient parity.



 P47 Recursive Inference Scaling: A Winning Path to Scalable Inference
 in Language and Multimodal Systems
 2025-02-11 | Architecture and training: recursive inference schedules

 https://arxiv.org/abs/2502.07503

 Finding: Studies a recursive-depth pattern under matched model size and training compute, with stochastic dropout
 and small linear adapters in the later conference version.

 Boundary: The useful schedule and adapter recipe must be tested in the chosen data regime; an improved paper
 baseline does not settle Aleph topology.

 Use for Aleph: Add RINS to the pilot reference set and preserve its recurrence schedule when comparing cost.



 P48 LoopSpec: Pipelined Self-Speculative Decoding for Looped
 Transformers
 2026-09-15 | Execution: pipelined self-speculative decoding

 https://arxiv.org/abs/2609.17184

 Finding: Uses early recurrent readouts to draft future tokens while full-depth verification proceeds, with a selective
 deeper proposal and a lossless decoding rule.

 Boundary: Reported speed depends on acceptance rate, batching and model kernels. This changes inference
 execution rather than the main pretraining objective.

 Use for Aleph: Measure as a serving baseline after Aleph has usable intermediate readouts; compare with LoopSkim.




DATA GOBLIN ALEPH 1B / Research design / 19 Nov 2026                                                                      32
