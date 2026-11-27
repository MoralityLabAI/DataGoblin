# ZKL-17: recurrent exploration gate

Pinned exact checker: SatsHedgeZero 84c1681a99ef5897f0f6b90140a0ac9e3293f5cb. ZKL-16 showed that uniform
random row selection recovered 46/48 certificates at 96 columns, versus
38/48 for indexed frontier and 37/48 for the trained feedback row ranker.
The indexed arm was cheaper in process CPU. This pilot gates exploration
within a single sparse elimination state. Exploit takes the indexed global
maximum; explore samples a row uniformly and selects its best unused column.
Either action updates the same residual, basis, frontier and postings.

The eight arms include the original indexed and row-random controls, fixed
periodic exploration, stagnation-triggered exploration, 50% random gate,
trained linear gate, trained recurrent gate, and reset-hidden-state ablation.
The latter two share exactly the same learned weights. The model is an
eight-hidden-unit tanh RNN with 12 observed features (177 parameters);
masked teacher-label backpropagation through time trains it. A 12-parameter
linear model uses the same labeled sequences. No GPU or LLM is needed.
This is an actual small recurrent neural controller, not a canonical
TRM/LDT/BPD implementation.

Training: 12 roots 17001–17012 on four ZKL-16 training families over F7/F11,
two trajectories per task. The teacher is one independently verified full
bounded-span certificate. At each step when exactly one proposed action is
in its support, that action supplies a binary label; ambiguous labels are
masked. Remaining actions use a seeded mixture. This is imitation with
nonunique, imperfect labels, not a proof that a particular action is required.
No evaluation-root data enters training. Train/reference work is separate.

Test: fresh public roots 17101–17108 and 17201–17208 on ZKL-16 test families
over F11/BN254. This is a new-root comparison on known topologies, not a
blinded topology holdout. 64 tasks, 8 arms, 3 protocols (1536 runs).
Same 24/96-column and cooperative 10 ms process CPU protocols as ZKL-16.
Late verified certificates receive no deadline credit. The CPU includes
setup, index maintenance, policy computation and verification; a 512 MiB
virtual-address-space cap covers each campaign. Whole-process peak RSS
is recorded, not per-arm RSS. All candidate certificates go through the
independent exact polynomial verifier. No generated per-case Lean replay
or production authorization.

This pilot tests whether recurrence contributes over simple exploration
controls and the identical reset model. If it does not, preserve the
strongest transparent control as the next baseline.
