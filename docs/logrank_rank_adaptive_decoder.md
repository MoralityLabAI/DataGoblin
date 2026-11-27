# Rank-adaptive verifier budget curriculum

Source: https://github.com/MoralityLabAI/LoopedTransformers/tree/6f4d83449f3378997a41db9c4f05b74aaca9bf23/experiments/logrank_bpd_11.

11 compares nested budgets 2r through 32r on known quotient bases with independent binary symmetric noise. At .35 noise and 32r, sparse quotient observation schedules gave exact posterior query accuracy 740/768 and belief propagation 709/768; dense schedules gave 750/768 exact and 538/768 BP. Larger sparse BP pilots: rank 24, 128/128 query decisions; rank 48, 111/128. These are 24 small and eight large development worlds, not a general neural result.

Use the supplied-basis sparse decoder as a classical control. Distinguish physical masks from coefficient sparsity. Record observed rank, degree, redundant equations, exact or approximate confidence, covered errors, cost, independent world, and any learned basis mapping. Approximate BP confidence must be empirically calibrated. Treat duplicate parity observations as a combined log-likelihood factor when the noise is independent.

A rank-expanding call supplies information about a new coset; in-span calls refine existing posterior beliefs. For a new independent uniform bit U, target U xor q·S and noisy probe U xor a·S, expected Brier gain is (1-2p)^2 mu_(q xor a)^2/4. Use this alongside within-span expected Brier gain in one costed acquisition policy. At high noise repeated expansion directions may need redundancy before confidence gating. Compare against random, rank-only and exact small-r joint posterior controls; hold out fresh worlds and report abstentions. No native DataGoblin integration in this note.
