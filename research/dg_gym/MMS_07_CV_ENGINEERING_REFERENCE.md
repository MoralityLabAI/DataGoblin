# MMS-07-CV engineering reference and noise-budget gate

Status: synthetic fixed-state mathematics and engineering checks only. No gym training, new learning evidence, or registration freeze. This note supplements the Fable review and second-review addendum.

## Reference and receipt

Run `python research/dg_gym/mms07_cv_reference.py`. Local execution with Python 3.13.5 / NumPy 2.3.5 passed; the recorded receipt is `receipts/mms07_cv_reference_receipt.json`.

- Source SHA-256: `8b95516711355f76f30a5fbb335fbef5df84062424c3d43c71b007db94d80dd0`.
- Receipt SHA-256: `9b86ce013ded6f3be932aee78670457f8f157a4663c21a42362726e7a3b0064b`.
- Exhaustively enumerate all 64 logical draws under uniform and nonuniform sampling for capacities 16 and 64. Max absolute mean error 5.56e-17; running sum matches logical table after collision, duplicate writes and eviction. Overflow is rejected before mutation.
- Uniform variance ratios 0.784716 and 0.333084; nonuniform 0.935974 and 0.470207. The synthetic signal dictionary is built into the data-generating process. Its comparison with a random rank-four dictionary is a positive control, not evidence about BPD.
- The implementation is deliberately slow reference code: dense least-squares, per-call full probability validation, Python writes and modulo slots. The latter deliberately produces collisions. Do not use it as a production bank, treat measured runtime as representative, or adopt its oracle alpha in an online policy.

## Cost-aware variance reduction: a necessary pilot question

Fix parameters, an objective with uniform target over N examples, and an IID full-support sampling distribution p. Define X = g_i/(N p_i), Y = t_i/(N p_i), with table and codec fixed before sampling. The single-draw estimator X - alpha (Y - E[Y]) is unbiased. Let v_g = E||X-E[X]||^2, v_t = E||Y-E[Y]||^2, c = E< X-E[X], Y-E[Y] >. With positive variances, the optimal scalar alpha is c/v_t and the exact relative trace variance is

    R(alpha) = 1 - r^2 + (alpha-alpha_star)^2 v_t/v_g,
    r^2 = c^2/(v_g v_t).

The formula remains valid when the coefficients are stale, compressed or missing, using zero for misses and a population-consistent sum. It requires the coefficient to be fixed independently of the current draw. The optimum is an oracle diagnostic. Estimate alpha on earlier independent draws, report the error penalty and validate on separate draws. For the zero-variance table use alpha=0. Include zero-alpha and permuted-table arms.

For a fixed parameter, independent equal-size minibatches, and a compute budget long enough to ignore integer rounding, variance of the mean of the sampled gradient estimates is proportional to variance per batch times step cost. With step-cost ratio kappa, CV improves this fixed-state noise-per-compute objective iff

    kappa R(alpha) < 1.

At oracle alpha this needs r^2 > 1-1/kappa. At historical cost ratios 1.38 and 2.06 the illustrative thresholds are 0.2754 and 0.5146. Measure CV's own overhead; do not reuse replay cost ratios as CV costs. The inequality does not establish an improvement in optimization speed, AdamW behavior, held-out regret or learning. Temporal correlation, parameter drift, optimization bias and curvature require separate training evidence.

For a rank-r matrix dictionary D with columns vec(u_j v_j^T), compute analysis scores from the ordinary backward pass, then obtain synthesis coefficients with a frozen rank-tolerant Gram pseudoinverse. This avoids storing a full per-example weight gradient. The Gram entries are products of the read/write-vector inner products. Use fixed dictionaries in the first pilot. For a shared weight, consistently aggregate the chosen visits and compare covariance against the total gradient. All-key moments (including zeros for misses) are required for the gate.

Preflight recommendation: plot held-out R(alpha) against measured kappa by rank, logical capacity, age bin, visit and sampling policy. A point is promising for the fixed-state objective only when an uncertainty interval supports kappa R < 1. Calibrate alpha without current-draw information, include table metadata and codec workspace in RAM accounting, and keep ordinary optimizer state and objective semantics unchanged. These measurements remain engineering telemetry.
