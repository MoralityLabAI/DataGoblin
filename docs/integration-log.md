# AGN-04 integration log

Running log of assumptions and unresolved issues for the AGN-04–08 storyworld
continual-learning program. Append; do not rewrite earlier entries.

## 2026-11-25 — AGN-04 Phase 1 (repo map, teacher skill scaffold)

**Assumptions made:**

- "Maya loader" (task instruction: "use the recently ported Maya loader for fast tps on Inky")
  is read as the recently-ported **squid5080** engine (`tensorfish/squid5090` port at
  `Documents\Codex\squid5090`, upstream PR `tensorfish/squid5090#5`), since no file, repo or
  tool named "Maya" exists anywhere searched (`goose`, `Goose-Experiments`, `HybridTRMLDT`,
  `adapter-bits`, `Documents\Codex`). This is the only recently-ported fast local loader for Inky
  Mini (Inkling-2-Mini) on this machine. Unresolved: confirm with the operator whether "Maya"
  names something else not yet recorded here.
- "Pocket Goblin GDK app inside Goose" is read as `goblin-gdk`
  (`HybridTRMLDT/MCP/goose_gdk_app`), the existing Rust/Goose tool app with its own CI smoke
  test. No literal "Pocket Goblin" name exists; this is the closest and only Goose-hosted GDK app
  on this machine.
- GPTStoryworld (`MoralityLabAI/GPTStoryworld`) is not cloned locally and its PR #21 content was
  not independently re-read; the exact seam (SW-PROV-05, commit
  `f8ba60b66d719767848edc5cc72522738dc90f5f`, four files bound by `storyworld_continual.py`) is
  taken from AGN-03's own doc, which already names that commit. A reviewed local checkout must
  be obtained and source-digested before any live `Agonist.storyworld_continual` run.
- EvalMath's ADG-01–06 (task's PR #81 reference) are already merged to `main` on the local
  clone, not an open PR; `cad2a70` (ADG-06) matches the hash AGN-03 already cites, so no new
  EvalMath PR exists to track for this lane.

**Unresolved issues:**

- No live Bonsai 2 27B or Inky Mini generation has been run. Section 2's teacher pipeline exists
  only as `Agonist/skills/agonist04/SKILL.md` plus a verification-record schema; there is no
  worker that actually calls squid5080 or routes through goblin-gdk yet.
- AGN-05 (Lean), AGN-06 (continual student training), AGN-07 (learned curriculum selector) and
  AGN-08 (Aleph integration) have not been started. Each needs an explicit operator go-ahead for
  live model calls, a Lean kernel run, or GPU training budget, per AGN-03's existing gate.
- `schemas/agn04_world_verification_record.schema.json` is new and unused by any live code path;
  it is validated against one synthetic fixture in
  `tests/unit/test_agn04_world_verification_schema.py` only.
