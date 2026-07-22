# ADR 0002: Provenance authority

Status: accepted

A proposer-authored evidence label is a claim, not a certificate. By default only environment-owned or independent-harness evidence can hard-promote a finding. Human-reviewed evidence is retained with its stated authority; model-, experience-, and simulation-derived evidence remains soft and may guide experiments.

The membrane records claimed evidence, verified evidence, disagreement, and the verification-receipt hash. A hard
verification receipt binds the prior public state, exact proposal, verifier identity, evidence type, and artifact
references. Bare evidence labels and caller-provided authority strings are not verification. Unsupported authority
produces `AUDIT`, `ABSTAIN`, `REVISE`, or `REJECT`, never silent promotion.
