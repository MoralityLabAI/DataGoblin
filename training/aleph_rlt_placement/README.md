# DG-RLT-ARCH-01 — Aleph RLT placement tournament

[Resume the development matrix](RESUME.md) · [Frozen promotion protocol](PROTOCOL.md) · [Dense architecture and TPU costs](ARCHITECTURE.md) · [A distributed clock-order research gym](CLOCK_ORDER_GYM.md)

Status: development gym and prospective promotion protocol, 2026-11-24.
No full pretrain, accelerator allocation, efficacy result or sealed evaluation
is claimed. The sole internal architecture intervention is RLT; RLP, SGRT/HoloArc,
teacher/curator, verification and LDT are external typed tools. Keep the
November 23 corpus and engineering gates.

## Launch tonight

Python 3.10+ and PyTorch 2.4+ (installed locally; no provider calls).
Run from repository root. First run the self-test:

    python training/aleph_rlt_placement/gym.py --smoke

Then a bounded CPU plumbing run:

    python training/aleph_rlt_placement/gym.py --out runs/aleph-rlt-arch01-smoke --width 16 --steps 2 --batch 2 --eval-n 4 --seeds 137 --arms direct terminal first_last penultimate first_penultimate_last first_penultimate_damp core_edges pass_rlt global_full --seconds 180

The default 64-wide model is a small actual causal GQA/SwiGLU/RoPE Transformer
with tied embeddings and 14 distinct physical blocks. It is not the complete
strict-1B trainer. The bridge bank, native training corpus and optimizer sharding
are absent. Each example has 20 prompt positions and one answer-loss position;
the public synthetic generator exercises pointer-chase and modular accumulation,
with longer horizons for development extrapolation. These are wiring tasks,
not evidence of LM pretraining or research taste. The raw development target
rows are deliberately public and must never be described as sealed.

After hardware admission, choose a free GPU explicitly, or use a single CPU
worker. For example, with operator approval and an idle CUDA device:

    python training/aleph_rlt_placement/gym.py --device cuda:0 --out runs/aleph-rlt-arch01-dev --width 128 --r 2 --s 1 --steps 512 --batch 16 --eval-n 128 --seeds 137 163 191 --arms direct first_last penultimate first_penultimate_last first_penultimate_damp core_edges pass_rlt global_full --seconds 3600 --swaps

The example budget is a **soft process deadline**, checked between steps and
optional sweeps. One step, checkpoint, native eval or process teardown may
overrun it. Before GPU execution, use the existing machine resource guard or
supervisor, cap memory and wall time independently, and check other jobs.
Never launch this example automatically on a busy device. A paused matrix
can be resumed using exactly the same arguments plus --resume (a fresh time
allowance is allowed). Per-cell atomic checkpoint includes weights, optimizer,
step, config and torch RNG; batches derive from the seed and step. The code
records parameter count, processed and loss positions, block-token visits,
checkpoint digest, session elapsed time, and per-root dev scores. It does
not yet prove bitwise resume, cumulative cost reconciliation, data-family
de-duplication or hard runtime enforcement. Add those gates before promotion.

## Exact schedule and placement semantics

LSA execution word P C^r Y^s D where C has ten distinct tied-across-pass blocks,
Y has two distinct tied-across-answer-pass blocks, and P,D each execute once.
Default r=2,s=1 gives:

* 1=P; 2..11=C pass 1; 12..21=C pass 2; 22..23=Y; 24=D.
* Block 23 is the second physical answer block on this schedule. The
  `penultimate` policy is relative to execution depth, never a silently fixed
  absolute layer number when r/s changes.
* At an executed site i (zero based), global RLT uses alpha=1/(K-i).
  The proposal is the entire residual block endpoint F_i(x), blended once
  against its incoming state. Native inner attention/MLP residuals remain.
* Sparse placement is a new heuristic use of the RLT coefficient. It does not
  inherit the paper's full-schedule mathematical guarantees. Terminal alpha=1
  is exact direct replacement. First+last is exactly first-only.
* A repeated physical core-edge site can instead use its own pass clock
  alpha=1/(r-j). Whole-pass RLT blends the entire ten-block output with the
  **input of that pass**. It is distinct from damping the final block's residual.
* Matched damping preserves the mean of the nonterminal coefficients on the
  same selected sites and terminal replacement; it does not match actual
  displacement. Single nonterminal site damping equals RLT; at r=2 core/pass
  damping equals local RLT. Run r=3 or greater for a nondegenerate comparison.

| Arm | Executed boundary / anchor |
|---|---|
| direct | Native residuals |
| terminal | Last layer only, exact identity control |
| first_last | Global first and terminal |
| penultimate | Global penultimate |
| first_penultimate_last | Global first, penultimate and terminal |
| first_penultimate_damp | Same sites; constant 13/48 at first/penultimate for K=24 |
| global_full | Global RLT on every executed block |
| core_edges | First and last physical C blocks on each pass, pass clock |
| core_edges_damp | Same sites, matched pass-clock constant damping |
| pass_rlt | End of each C pass, whole-pass anchor |
| pass_damp | Same whole-pass anchor, matched constant damping |

No extra trainable weights; same module objects and gradient horizon across
arms. No learned clocks in this miniature trainer. If adding time conditioning
or all-visit auxiliary losses from the paper, cross each as a separate matched
factor. Keep objective, token masks, initializer, optimizer, data order and
parameter count fixed. Record per-site alpha, residual and realized update RMS,
gradient norm, coda loss and perturbation gain before promoting a mechanism.

## Tonight's stages (Astra)

1. Review source and run the CPU self-test; test checkpoint restart (model,
   optimizer, next batch and next loss), schedule identities, no future-token
   leakage, tie aliasing, and matching native/runtime rule. Audit exact prompt
   root overlap and score the uniform 16-label baseline. All-at-chance cells
   are construction failures, not architecture evidence.
2. Run bounded 1–5M screening with two or three matched seeds, then test
   nondegenerate r=3 controls. Measure actual cumulative wall/compute budget.
   Keep failed cells and incomplete matrices. Never promote a result based
   on a single convenient seed. Use matched seed/root paired rows.
3. Integrate the existing checked rule-revision, Lean rank/repair and
   licensed text lanes from `rlt_pretrain_*`, plus the pretrain recipe's
   constant mix; add task-specific verifiers outside the model. Test text
   held-out NLL with strict masks, exact revision, Lean kernel-checked
   rank/repair, and counterexample generation on separate families. Add
   training/runtime crossovers, K=1/2/3/4/6 with independent trajectories,
   activation/gain traces, direct and matched damp controls. Keep all
   evaluation labels outside the proposer process in the confirmation stage.
4. At 12–30M, then 100M if competent and budgeted, predeclare finite sample
   gate using EvalMath and disjoint dev/CAL/family-held-out roots. Include
   relative wall time and verifier budgets, assess depth transfer. Freeze
   candidate choice using development results and report a fresh confirmation
   once. Use root-clustered paired intervals and multiplicity correction
   over the prespecified comparisons. Hold if text or critical checked tasks
   regress beyond preregistered limits. Do not use sealed results to iterate.

## 30B processed-token campaign

The two independent 10B runs use the same initialization seed, tokenizer,
admitted document/root order, packing, loss masks, optimizer schedule,
maximum context and fixed recurrence depth. One arm is the direct strict-1B
baseline. The other is one nondegenerate RLT placement chosen on small-scale
development and confirmed before pretraining. If no RLT arm is competent,
keep the second run on HOLD rather than spending compute on a known failed
recipe. Record exact trainable parameters (strict 1B maximum), bridge weights,
measured throughput and resource use. RLT itself changes no weight count.

Before each run: execute the existing 10M-token engineering gate (Nov 27)
and 500M-token calibration only with authorized allocation. These can be
prefixes of the 10B run if the recipe is unchanged and the exact continuation
is preregistered. Use the same shared first 10B training stream in both arms.
Use genuinely new admitted data for the continuation where available. At
10B, freeze both checkpoints and use a separate adjudication split and
predeclared scoring/guardrails. Keep both weights. Continue the chosen model
from its own optimizer state and token cursor for another 10B processed tokens
with a frozen continuation LR schedule. Do not reset or mix checkpoint weights.
Reserve a final untouched evaluation for the 20B model; the adjudication set
is selection data thereafter. Report processed tokens vs accepted unique
tokens, loss-bearing positions, repeats, block-token work and actual costs.
Two first runs plus one continuation = 30B total processed tokens, with
20B unique exposure at most if first streams match and continuation is fresh.

Primary decision: lower held-out language NLL subject to checked-task and
throughput noninferiority; a predeclared composite of verified research-task
success can break a near tie if both arms meet guardrails. Calibrate thresholds
before candidate selection; proposed initial limits for discussion are <=1%
relative text NLL degradation, <=2 pp aggregate verified task regression,
and <=10% measured throughput penalty. These are planning suggestions,
not a frozen statistically powered gate. One seed per 1B arm does not support
a population-level architectural superiority claim; rely on replicated small
scale evidence and report checkpoint-specific full-scale results.

## Existing source references

* [Strict 1B recipe](../../docs/one_billion_pretraining_recipe.md)
* [RLT research track](../../docs/rlt_research_track.md)
* [Synthetic data value, rival theories and the held Lean acquisition experiment](AGONIST_VALUE.md)
* [Matched placement results and the promotion decision](FINDINGS.md)
* [RLT feedback coefficients](../../src/datagoblin/training_gym/rlt_feedback.py)
* [Pretrain cell](../../src/datagoblin/training_gym/rlt_pretrain_cell.py)
* [November 23 execution plan](../reports/2026-11-23_aleph_pretrain_execution_plan.md)
* [LSA-02 typed calculus](https://github.com/MoralityLabAI/LoopedTransformers/blob/main/docs/lsa_typed_architecture_calculus_v0.md)
* [RLT source paper](https://arxiv.org/html/2611.16265v2)

No paid runs, teacher fallback, weights download, full corpus ingestion or
new full-scale architecture is authorized by this development gym alone.
