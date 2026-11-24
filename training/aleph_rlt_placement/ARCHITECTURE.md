# Aleph dense recurrent architecture and TPU decision

Keep one input shell, ten shared reasoning blocks, two answer blocks, one output shell, causal grouped-query attention, SwiGLU, RMS normalization and tied embeddings. Each reasoning visit reuses the same ten blocks in the same order. Both tournament arms must have the same pass count. Arm A carries each endpoint directly. The first whole-pass challenger worth testing is Arm B with a remaining-time write at the end of the ten-block group, measured against local constant damping. Its promotion depends on the registered data. The decision can remain open after the screen.

Inkling 2 Mini applies its two middle six-layer blocks three times. Its later passes can read the prelude and a learned mixture of prior block outputs. Thinking Machines reported that middle looping beat last-layer looping on 27 of 28 placement evaluations in its comparison. That supports concentrating repeated work in the middle. Aleph can specialize in proposing a proof repair, rule revision or counterexample under a fixed budget while Inkling builds diverse worlds. A strictly dense core and an external exact checker make that division of labor testable. Every trained Aleph parameter executes on every example. The checker receives public proposed actions and has no route into the residual stream. [Inkling design and measurements](https://thinkingmachines.ai/news/inkling-2-mini/)

Use a fixed dense residual stream for the placement tournament. A learned prelude reread or cross-visit attention reader would change the available information and the optimization path simultaneously with the placement intervention. After the comparison, an independent factorial could test a source reread and planned interpolation separately. When measuring retained information under an edit, add a diagnostic that prevents access to the old source through all prompt, residual and KV paths outside the registered message. Report ordinary full-context utility separately. [LSA-02 execution and sharing calculus](https://github.com/MoralityLabAI/LoopedTransformers/blob/31256abc921ae9aa500ad4fb85a167a990be0984/docs/lsa_typed_architecture_calculus_v0.md)

## What the placement changes

Let the ten-block reasoning pass be $C$ and its input at visit $j$ be $x_j$. The state update is $x_{j+1}=x_j+a_j(C(x_j)-x_j)$. Direct feedback uses $a_j=1$. RLT uses $a_j=1/(r-j)$ at zero-based visit $j$. Matched damping uses the average nonterminal coefficient and keeps the terminal coefficient at one. A terminal write preserves the native endpoint and its gradient exactly. At $r=2$, both interpolating schedules use $(1/2,1)$ for every nonlinear core.

At three passes RLT uses $(1/3,1/2,1)$ and matched damping uses $(5/12,5/12,1)$. Write $M=J-I$ for the stationary linear core $C(x)=Jx$. The exact difference between the two output operators is

$$
J(I+M/2)(I+M/3)-J(I+5M/12)^2=-JM^2/144.
$$

The linear terms cancel. A mode near the identity gives weak leverage for discriminating the schedules. For fixed linear visit maps $J_1,J_2,J_3$ that can differ across visits, put $M_i=J_i-I$. The schedule difference is exactly

$$
J_3\left[(M_2-M_1)/12-M_2M_1/144\right].
$$

The first term is sensitive to changing visit dynamics. These identities concern the same fixed maps under each schedule. In a nonlinear model the two paths typically visit different states, and each path's Jacobians must be measured at its own states. Evaluate both schedules on identical checkpoints and inputs. The reversed schedule $(1/2,1/3,1)$ is another cheap diagnostic. For a shared affine core it commutes exactly with the original ordering in augmented homogeneous coordinates. An output difference between forward and reversed schedules, after numerical error is accounted for, demonstrates order dependence on that input. This diagnostic can motivate a new experiment. It has no role in the registered promotion decision.

A differentiable core has tangent factor $(1-a_j)I+a_j DC(x_j)$ at its actual visit. Products of these factors control perturbation gain. Smaller writes can reduce early excursions while still allowing amplification by an expansive final pass. The realized RMS displacement from the pass anchor is $a_j\,\mathrm{RMS}(C(x_j)-x_j)$. The displacement from the input of the last physical block measures a different quantity. Compare both traces and task behavior. Improvement of both interpolating schedules over direct supports the utility of damping at this boundary. Reproducible gain over the matched constant provides evidence about the clock. [RLT paper](https://arxiv.org/abs/2611.16265)

For the physical-edge clock, each pass's first and last reasoning blocks blend against their immediate inputs. In the near-identity expansion of the pass, the eight interior increments still enter at full first-order strength. Whole-pass RLT scales the combined first-order contribution of all ten. The global first-layer write applies $1/K$ of the first block displacement while retaining the embedding residual path. A global penultimate write is always a half increment at the executed penultimate block. Each singleton has an exact constant-coefficient twin. The combined first/penultimate policy can separate from its constant control. The matched coefficient equals $13/48$ at $K=24$ and $9/34$ at $K=34$. Pair global-full RLT with global-full damping. A gain that appears only at the penultimate site supports readout-boundary damping.

## Dense parameter and work ledger

The original planning allowance set aside 47,185,920 parameters for bridges and routing. Reallocate that entire allowance to the ten repeated core FFNs in both arms. Increasing each SwiGLU intermediate width by 768 adds exactly $10\times3\times2048\times768=47{,}185{,}920$ weights. This is a deliberate revision shared by direct and RLT, with the strict recipe's unique count preserved. Earlier bridge-equipped planning figures for executed work must be recomputed. [Strict-1B planning recipe](../../docs/one_billion_pretraining_recipe.md)

| Component | Physical count | Parameters per component | Parameters |
|---|---:|---:|---:|
| Tied 49,152 by 2,048 embedding and readout | 1 | 100,663,296 | 100,663,296 |
| Input, output and answer blocks, intermediate width 8,192 | 4 | 60,821,504 | 243,286,016 |
| Reasoning blocks, intermediate width 8,960 | 10 | 65,540,096 | 655,400,960 |
| Final RMS norm | 1 | 2,048 | 2,048 |
| **Total** | | | **999,352,320** |

For model width $d$, KV width $v=512$ and intermediate width $f$, each bias-free block has $2d^2+2dv+3df+2d$ trainables. Stored weight memory in BF16 is 1,998,704,640 decimal bytes. A full 16-byte Adam inventory is 15,989,637,120 bytes before sharding and activations. `fullscale.py` instantiates the actual model on Torch's meta device, counts its unique tensors and checks the output shape of both policies.

| Schedule | Executed blocks | Block parameters visited per position | Approximate training GFLOP per position at length 2,048 |
|---|---:|---:|---:|
| Two reasoning passes and one answer pass | 24 | 1,554,087,936 | 10.532 |
| Three reasoning passes and one answer pass | 34 | 2,209,488,896 | 14.716 |

This arithmetic includes all dense linear forward and backward products, the tied vocabulary projection and ideal triangular causal-attention products. It assumes a full context and dense logits at every position. Padding, actual attention kernel shape, BF16 collectives, rematerialization, pointwise and optimizer work, I/O, evaluation and recovery change the billed cost. Record both padded and nonpadding block-positions. The actual TPU credit ledger needs end-to-end throughput and observed credits per hour for the frozen topology and runtime. A 10B leg in 55 hours requires 50,505 processed nonpadding positions per second, equivalent in these two estimates to 532 or 743 useful aggregate TFLOP/s. At 350 credits per leg, two legs cost 700 and leave 300 of the 1,000-credit balance. Three legs cost 1,050 and expose a 50-credit shortfall before contingencies. If the original 350-credit estimate was measured at two passes, the linear work projection at three is about 489 credits per leg. Profile both before budgeting.

## Freezing the full-scale inputs

The JSON config proposes a common 2,048 context, 262,144 processed positions per global update, BF16 matrix work, FP32 reductions, AdamW and a token-clock schedule. The first 10B uses warmup and stable learning rate. The winning checkpoint can continue on the same schedule with a late decay to 20B. Verify gradient clipping after global accumulation and normalization on the TPU backend. Final partial updates must charge only actual nonpadding positions. The 10M restart and 500M calibration checkpoints can be prefixes of each first leg when configuration, sampler and continuation have been frozen in advance. Freeze the actual tokenizer, shard digests, mixture, stability limits and optimizer values before launch. [Machine-readable proposal](fullscale.json)

The full corpus input has two levels. A shard manifest binds a source revision, rights/attribution, byte SHA-256, tokenizer digest, lineage and split policy, ordered file list and canonical-root repeat ledger. The packed example provides token IDs, segment boundaries, objective kind, loss mask or valid-action set, reference to a receipt where applicable, and nonpadding length. Pack after splitting whole documents, proof-template families, rule lineages, repair programs and counterexample-generator families. Assign all paraphrases, failed variants and repairs of one root to the same split. Enforce the static cap of four total views per root and keep unique admitted tokens, processed nonpadding exposures, padded slots and loss-bearing positions separate. A byte window from the public pilot cannot supply a 49,152-token shard identity.

For causal text, the label at position $i$ is token $i+1$ in the same segment. Block attention across packed documents and mask cross-segment targets. For checked actions, make the grader an offline join keyed by task, action and epoch. Unknown grades stay masked. Use normalized likelihood over verified acceptable choices on the fixed candidate interface. A move to multi-token actions must use a frozen action codec with an explicit terminator and prefix semantics so the union-of-actions likelihood counts each valid event once. The external verifier can issue replay metrics on both snapshots using identical budgets. Preserve the first-10B stream and batch order byte-for-byte across arms, then advance the winner from its own exact optimizer, scheduler, sampler, RNG and tokenizer state.

## Packed-stream reference interface

The streaming reader accepts pretokenized JSONL shards and a separately frozen SQLite custody index. The manifest schema is `aleph.packed-stream.v1`. It names the context, vocabulary size, padding ID, tokenizer and registry files with SHA-256 values, and an ordered list of shards. Each shard entry carries a relative path, digest, split and record count. Supply the externally recorded manifest digest on every open. The reader verifies each used shard before yielding an example and opens the registry read-only. [Packer](packing.py) · [Executable example and tests](test_packing.py)

The steward builds the index using `INDEX_SCHEMA` after deduplication and independent verification. Create one split per family, one family per canonical root and one admitted entry per presentation, with at most four presentations per root. Bind the admission to the exact JSONL record bytes (excluding the line ending), shard SHA-256, zero-based ordinal and checker receipt digest. The index must be finalized without a live SQLite WAL. The reader verifies these bindings and the complete tokenizer file identity. Semantic validity of a checker receipt is the steward's responsibility.

Every record supplies `root`, `family`, `split`, `presentation`, `lane`, `objective`, `token_ids` and `target_mask`. The mask is indexed by the target token, and its first element is false. For a checked response, also supply the index of the first response token and its independently bound receipt SHA-256. A row without a known checked outcome contributes zero supervised response targets. This reference path handles token-response CE. A set-valued candidate objective needs a separate choice/grade table and its own normalization. The mixed placement gym exercises that path.

Each returned pack includes source segment IDs, per-segment positions, nonpadding length and already-shifted `labels_at_query`. `token_loss_sum` accepts these labels directly and returns a loss sum plus its valid-target count. Aggregate both across replicas before normalizing. A rank containing only unknown grades contributes zero loss and gradient. Skip the optimizer update if the global valid count is zero. The segment's final query and every padding position have target `-100`. A new block can start partway through a long record and resets its local context. `reference_logits` executes each segment independently through the native Aleph model and supports gradient parity checks. Use `sdpa_mask` as the allowed-pair reference for a vectorized or XLA implementation. A boolean mask per position suffices for loss selection, while attention itself must enforce segment boundaries.

The stream cursor stores the shard, next byte position, record ordinal, within-record token offset, processed and loss-bearing counts, padding, pack count, lane counters and started presentations. Canonical-view positions count exposures from presentation 1. Unique admitted corpora and repeat ancestry come from the custody ledger. Commit this cursor together with the optimizer update so an interrupted worker can repeat an uncommitted batch. Freeze absolute token stops and all update boundaries in advance for identical pack tensors after restart. Inserting a new stop inside a pack alters its available left context. The exact final partial update charges its real positions, including masks and padding in separate columns. Tests compare both tensors and cumulative ledgers across an interrupted two-shard run.

To inspect a snapshot, run `python training/aleph_rlt_placement/packing.py MANIFEST.json --sha256 MANIFEST_DIGEST --split train --stop-at 10000000`. The command prints a checkpoint cursor and fails on an undersupplied stream. Use `--cursor CURSOR.json` with a later registered absolute stop to continue. No production snapshot or tokenizer is admitted by the fixture checks.

## Full-scale decision

Freeze the chosen B placement on the independent small-scale confirmation cohort, along with its strongest damping comparator. When a clock-specific claim rests on three passes, use three passes for both A and B and reprofile. When the evidence supports a simpler placement effect at two passes, train both at two and record the coefficient identity. Keep B pending if the competent controls and fresh confirmation cannot separate the candidates. [Development and confirmation protocol](PROTOCOL.md)

At 10B use a fresh adjudication cohort, common decoder, equal checker budget and matched hardware. Apply the preregistered text, verified-task and throughput guards. Test the primary language difference using paired document-level observations, report checked success by family and use simultaneous bounds for the registered comparisons. A prespecified near tie may favor a confirmed checked-task gain if all guards pass. Hold the decision if the intervals cannot resolve the agreed practical margin. Retain the two checkpoints and the entire cost history. Continue the favored checkpoint for 10B new processed exposures only with available credits and a frozen continuation stream. Reserve another untouched cohort for the final 20B checkpoint. Two matched first legs plus one continuation charge 30B positions. At most 20B positions can be distinct even when all continuation data are fresh.

To regenerate the count and cost ledger from the repository root, run

```sh
python training/aleph_rlt_placement/fullscale.py --r 3 --candidate pass_rlt
python training/aleph_rlt_placement/fullscale.py --r 2 --candidate pass_rlt
```

Supply measured `--tokens-per-second` and `--credits-per-hour` together to replace the operator's planning case.
