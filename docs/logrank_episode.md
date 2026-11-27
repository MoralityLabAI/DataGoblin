# LOGRANK-BPD-01 corpus handoff

Canonical fixtures: https://github.com/MoralityLabAI/LoopedTransformers/pull/404.

The envelope schema is an additive offline contract, not a generated dataset or trained model. Use root IDs for full matrix/generator-world partitions. Validate each episode against the schema, then separately enforce root-disjoint splits across the entire corpus, public schedule length <= bit_cap, generator/source provenance, and independent exact verifier replay. Reject inconsistent labels or certificates. A certificate records its evidence class and digest, not a floating numerical rank masquerading as an exact rank.

The Alice projection contains only public metadata and alice_private; the Bob projection contains public metadata and bob_private. Offline labels, rank certificates, generator hashes, root IDs and split labels are not model inputs. Audit public family identifiers and metadata for leakage; forbid communication via timing, shared state, tensor shape or uncharged routing indexes. Sealed episodes remain offline until independently authorized.

Candidate curriculum: parity with random feature pairs and nuisance coordinates; index-k; modular predicates; then real Lean/Sudoku verifier matrices. Hold out generator roots rather than just cells. Exact classical protocol search applies only to small matrices; report rank and protocol certificates separately. Do not use a structural oracle as evidence of learned model capability. No protected data generated or opened. Schema has not been integrated into DataGoblin's native validators.
