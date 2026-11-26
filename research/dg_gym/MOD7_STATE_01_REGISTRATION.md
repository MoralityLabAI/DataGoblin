# MOD7-STATE-01 public-development registration

Question: Can a learned row-stochastic state transition preserve an exact modular
residue over recurrent depth, and does state supervision improve the soft dynamics?

Use F_7 with nonzero affine multipliers. A 42 x 7 x 7 trainable logit table
(2058 scalar parameters) gives one row-stochastic matrix per instruction.
Input and output use the canonical residue coordinates: this is an explicitly
structured inductive bias, not discovery of an unknown latent basis.
No exact transition values are initialized or supplied to the task-only arm.

Three seeds 11,23,37, matched initialization and sampled streams. 600 updates,
96 roots per update, Adam lr 0.05, length 1/2/4 curriculum with fractions 1/4,
1/4,1/2. Arms: task-only terminal negative log likelihood; task plus 0.5
times average intermediate-state negative log likelihood (exclude terminal);
task plus same auxiliary loss with rolled batch labels. No auxiliary on length 1.
One thread CPU; no paid compute. Fixed exposure, not fixed wall-clock.

Audit all 294 elementary transitions. Record row argmax correctness and worst
row TV distance to its exact delta. Evaluate 512 public generated roots at
lengths 6,32,128,512, with identical draws per arm and training seed. Record
soft argmax accuracy, mean true-state mass, hard row-argmax execution accuracy,
and worst-row TV telescoping bound and correct-path mass lower bound.
Hard execution is an explicitly separate derived policy; an exhaustive row
audit certifies arbitrary-length correctness only if every row argmax is exact.
FP64 evaluation for soft numerical stability; training FP32.

Competence gate for intervention: >= 461/512 soft length-6 correct and
all 294 row argmax correct. If passed, length-32 intervention replaces
state after 16 instructions with a different-residue donor from the same
evaluation set. Compare identity, full donor state, wrong-residue donor
and exact point-mass oracle on the same 512 roots. Native coordinate replacement
is an oracle-assisted diagnostic; no discovered neural concept claim.
Report fidelity and target mass. No causal audit for failed fits.

Fixed public development seeds 57006,57032,57128,57512; no sealed test or
family-transfer claim. Report all fits and costs. Peak per-fit memory and
GPU rates unavailable unless actually instrumented. No training/corpus admission.
