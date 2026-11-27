# Semaev-RedTeam development gym

**Status:** development tasks, openly labeled. This gym teaches DataGoblin to diagnose mathematical proof scope, failed attempts and missing obligations in the Semaev index-calculus research program. It does not execute cryptanalytic attacks, verify new Lean claims, or establish deployed ECDSA security.

The task source is the pinned patrickdugan/semaev-index-calculus commit 3c630e179ae837dad839949dcb03cf4fdb41815e. The task descriptions identify exact read-only source paths. A future model-based proposer should retrieve the cited material and submit a structured answer. The independent answer rubric is transparent here, so these tasks are useful for debugging the pipeline but permanently excluded from sealed evaluations.

## Development topics

| Task | Mathematical claim to examine | Main failure mode |
|---|---|---|
| GDL-01 | Generic DLog Lean lower bound | Transferring generic-group bounds to algorithms that inspect curve coordinates |
| RMP-02 | Ideal-side ramp theorem | Dropping the S-unit hypothesis, characteristic restriction or ideal-to-closure gap |
| FEX-03 | Free-extension theorem | Treating a conditional all-height implication as an independently verified base instance |
| SEC-04 | secp256k1 applicability | Conflating split, Dickson and rational-step/isogeny tower regimes |
| TNS-05 | Strategy IV tensor contraction | Extrapolating exact toy contractions or asymptotics to a finite key-recovery speedup |
| RANK-06 | Rank and online relation collection | Mistaking relation-space rank for affordable independently tagged acquisition |

## Proposer answer format

One JSONL object per task, with these fields:

    {
      "task_id": "GDL-01",
      "claim_status": "scope_limited",
      "missing_obligations": ["non_generic_algorithm_model", "coordinate_reading_gap"],
      "evidence_paths": ["lean/SemaevIC/GenericDLog.lean", "paper/proof_strategies.md"],
      "next_check_kind": "formal_scope_audit",
      "confidence": 0.8
    }

The gold labels are openly available in tasks_dev.json. The accepted status types are scope_limited, conditional, unresolved and pilot_only. The permitted next actions are formal_scope_audit, lean_proof, exact_certificate_check and model_cost_audit.

## Scoring

The deterministic grader allocates 25% to the correct scope verdict, 40% to missing-obligation recall and precision, 20% to source references and 15% to the appropriate next verification step. It records a Brier error for the submitted confidence in the scope verdict, penalizes invented obligations, and rejects duplicate task IDs.

    python training/gyms/semaev_redteam/grade_dev.py --responses /tmp/proposer.jsonl --out /tmp/semaev_grade.json

The returned score measures whether the proposer recognizes the registered limitations. No score here constitutes independent evidence of new mathematical discovery.

## Research trajectory for the next iteration

Turn each task into a collection of source-pinned proposal, attempted proof, checker invocation, outcome and correction episodes. Include failed proof attempts from the Semaev dialogue history, preserving their exact chronology and parent statements. A frozen verifier must own every label. Reserve whole conjecture and curve families for later independent evaluation. Semaev's enormous Lean certificate arrays are verifier-side context and should not become generative target strings. No synthetic expansions or training shards have been accepted in this initial version.
