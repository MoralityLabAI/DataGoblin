# Aleph proxy on pools that can miss

The harder comparison freezes fresh roots from all seven gym families after excluding the campaign and previous proxy roots. Three quarters of each split and family use pools with every passing candidate removed by the environment. The other quarter retains its native pool. This makes out-of-pool repair measurable. The remaining drafts come from the gym's synthetic near-miss generator and retain its task structure.

Each repair policy gets the same bounded edit frontier. At each planned visit, the RLT reservoir prioritizes a draft and exposes up to 12 local edits. Examples include extending a graph path, changing a proposed operation result, revising a citation, or replacing a plan action. The editor uses the public world and the current draft. Every commit goes through the independent verifier.

The edit ranker extends the verified-pack proxy with 64 operation features and refits on new checked candidates. A second head learns whether the highest-ranked candidate will pass from its score, visit, frontier growth, and prior check count. The ranker and gate use different fitting roots. Threshold selection uses calibration roots and prioritizes verified coverage, checker calls, then recurrent visits.

The fixed-depth policy checks its final native proposal. The heuristic checks untried edits in generator order. The learned ordering control gives the same ranker without check deferral. The adaptive arm can defer a low-scoring check until a later visit, with a mandatory final check if a candidate remains. The three repair policies have the same four-check limit.

Run the harder variant in a fresh output directory through the Windows CPU cap wrapper.

```powershell
python scripts/run_aleph_proxy_windows.py --campaign CAMPAIGN --previous-proxy PROXY_RUN --out NEW_RUN --ramcap-package LCGRT_ROOT
```

The receipt reports pool-construction, fitting, calibration, policy and replay checks separately. The held-out traces retain the accepted candidate, operation, visit, and earlier deferred decisions.

## Held-out results

The completed evaluation covered 224 frozen held-out roots, including 168 pools with no passing native candidate. The ranker and gate each used 56 fitting roots. The separate 56-root calibration sweep used 287 checker calls and selected a threshold of `-0.10`.

| Policy | Verified on 224 roots | Verified on 168 pool misses | Verifier calls | Recurrent visits |
| --- | ---: | ---: | ---: | ---: |
| Fixed depth 8 | 15 | 0 | 224 | 1,792 |
| Heuristic repair | 154 | 137 | 503 | 503 |
| Learned order, check each visit | 168 | 124 | 459 | 459 |
| Learned order with deferral | 138 | 103 | 227 | 1,063 |

Learned ordering added 14 verified roots overall and used 44 fewer checks than the heuristic. On known pool misses, the heuristic repaired 13 more roots. Deferral skipped 836 checks. Compared with checking at every visit under the same learned order, it spent 232 fewer checks, took 604 more recurrent visits, and lost 30 solves. Of its 138 solves, 119 came from generated proposals outside the native pool. All 103 solves on pool misses required such a proposal.

The process peaked at 108 MiB of job commit under the 768 MiB hard cap. Two host-memory pauses preserved the same fit and held-out roots. The final summary covers 1,413 replayed policy receipts. A separate pool audit used 721 checks to confirm the miss and intact conditions. The initial run checkpoint lacked the pool-construction and fitting check totals. The updated runner records those counters at fit completion.

Resume with `--resume-hard` in the same run directory. The evaluator replays existing checker receipts and scores the remaining policy rows on the frozen roots. Its resume admission uses the observed previous memory peak plus a margin, with the same hard cap and live host reserve. The completed summary SHA-256 is `69de70947ef4f29036827052d54f647e413203cd2c3a5c7b5ad4c31390a227c9`.
