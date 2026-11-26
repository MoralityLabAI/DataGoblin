# LOGRANK trace compilation and identifiability

Source: https://github.com/MoralityLabAI/LoopedTransformers/tree/60603b52faa2dfd0af1e9efc49707af658b2d69b/experiments/logrank_bpd_09.

09 development: staged component calibration improved neural retrieval at 35% noisy teacher labels; a classical compiler enforcing the known bijection performed better. Clean singleton task labels already available during training made exact signed-coordinate compilation possible in every root, including all unseen compositions. This is specific to the planted structured family. Do not infer general compiler superiority.

For a trajectory corpus, record public descriptor syntax, teacher observation provenance, corruption type/rate, clean verifier labels, generator root and semantic split, independently of model inputs. Detect and report exact signed column matches, matching constraints, and learned stop semantics. When the structural compiler is applicable, compare it before allocating neural training; preserve failed matches and ambiguity rather than silently supplying oracle mappings.

Next dataset: no singleton labels; known coordinate-map parity with training incidence rank controlled over F_2. Record row-span membership and ambiguity witness for held-out queries. A row-span solver supplies exact signs for identifiable queries; out-of-span signs are ambiguous under a uniform solution prior. Compare a learned hard router with exact and noise-aware solver baselines; allow explicit abstention and account for verification calls. Hold out complete query functions and generator roots. Descriptor metadata is public, ground truth and ambiguity witnesses are offline evaluation fields. This is a design note only; no native DataGoblin validator changes or protected corpus generation.
