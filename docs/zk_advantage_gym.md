# ZKL-19: signed outcome advantage and matched memory ablations

ZKL-18's best policy was the unweighted sign-trained RNN, 46/48 at 96
columns and at the 10 ms deadline. Its reset control used different,
class-balanced weights. ZKL-19 closes that gap and compares magnitude
regression against binary sign imitation. Both use the same exact teacher
rollouts, ZKL-18 training roots 18001–18008 and initialization. The optional
collect flag records a normalized signed difference (sum of explore minus
exploit utilities divided by the number of paired schedules times horizon).
Observed equal utility receives zero for regression; unobserved states
are masked. The binary teacher remains unchanged. Train a linear model
and the 177-parameter RNN with Huber regression on the tanh output;
inference uses the same zero-logit threshold as the sign-trained gate.
Report label counts, observed ties, mean magnitude, separate fitting CPU
and teacher work. No evaluation tuning or root leakage.

Each objective has a linear, recurrent and same-weights reset arm. Four
transparent controls (indexed, row random, periodic, random gate) remain.
The exact polynomial verifier at SatsHedgeZero 84c1681a99ef5897f0f6b90140a0ac9e3293f5cb owns acceptance.

Fresh roots 19101–19108 and 19201–19208, F11/BN254. Eight families per
block: four previous ZKL-16 test families and four new families:
inverse-product transport, idempotent transport, zero-pair transport,
and essential inverse transport. These new core identities are absent
from the training set; they share the existing connected decoy generator
and related algebraic motifs. Product-chain-zero is intentionally
shared with training at disjoint roots, so report the new-family subset
separately. All essential controls have concrete independently checked
violating witnesses. No topology-blind claim.

Total 128 evaluation tasks (96 certifiable, 32 essential controls), ten
arms, three protocols (3840 runs). Same 24/96-column and 10 ms CPU
protocols, with 512 MiB process address-space cap and whole-process peak
RSS. Per-case Lean replay and production authorization remain false.
This is a minimal RNN experiment, not a full TRM/LDT/BPD architecture.
