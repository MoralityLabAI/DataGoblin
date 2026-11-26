# MOD7-FB-01 preregistration: closed-loop causal steering

Public-development CPU pilot. Reuse the canonical seven-state simplex and
42 x 7 x 7 trainable transition logits (2058 parameters). No paid compute.
Seed 11/23/37, paired initialization and streams, 600 updates, 96 roots/update,
Adam lr .05. Length 1/2/4 curriculum: 150/150/300 updates. All arms use terminal
NLL. On length >1, construct a donor and a wrong donor with pairwise distinct
intermediate residues using deterministic cyclic search within the batch.
Patch at site j with dose alpha in {.5,1} via (1-alpha)*p + alpha*p_donor.
The exact terminal target distribution is the same mixture of the ordinary
one-hot and the exact donor-counterfactual one-hot after the recipient suffix.
Donor state gradients remain enabled. The ordinary and patched passes are
separate full rollouts, with the edit inserted at j; count their visits.

Every 50 updates, before any optimizer step of that window, evaluate all
candidate (site,dose) pairs on a fixed 96-root feedback-only stream for the
current length (seed 61000+length). These roots are never optimizer examples.
For each candidate measure mean target mass for correct donor, identity and
wrong donor on the same roots, and estimate specificity S = mean(correct
donor mass - max(identity mass,wrong donor mass)). In the exact state model
S = alpha. The feedback deficit is 1-S/alpha. All arms perform identical
feedback audits. Length 1 has no candidate. Select and freeze one action per
50-update window, with ties by candidate index.

Five arms:
* task: zero-weight counterfactual shadow loss; static cycling action.
* concept: 0.5 mean intermediate state NLL; zero-weight shadow CF loss;
  static cycling action.
* static: 0.5 counterfactual mixture cross entropy, action cycles by window.
* adaptive: same CF loss, choose maximum feedback deficit.
* shuffled_feedback: same CF loss, choose the maximum after permuting score
  assignment using a separate deterministic seed (70000+training seed+window).
No steering at length 1. All arms have identical base exposures and two
full model rollouts on multistep training batches. Backpropagation and audit
wall cost measured separately; no claim of matched FLOPs or wall time.
Record feedback scores, chosen action, exposure/visit counts, training/audit
time, checkpoint digest. No adaptive action may access final evaluation roots.

Primary descriptive endpoint: mean true-state mass without steering at
length 512, adaptive minus static, and adaptive minus shuffled_feedback.
Require both mean gains >= .01, positive sign on every paired seed, all
adaptive fits >=461/512 soft accuracy at length 6 and 294/294 row argmax.
Report accuracy/mass at 6,32,128,512 and worst row TV. Fixed public evaluation
seeds 62000+length; no sealed outcomes. Also report length-32 donor/identity/
wrong-donor specificity at site 16, alpha 1, separately from unsteered retention.
Screening only: 3 seeds and one explicit-state environment. No general
transformer, hardware transfer, or corpus admission claim. RAM/GPU peaks
unavailable unless measured. Preserve source-bound receipts for all arms.
