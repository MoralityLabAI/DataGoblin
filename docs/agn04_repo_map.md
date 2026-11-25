# AGN-04 repo map

Phase-1 seam inventory for the AGN-04–08 storyworld continual-learning program, built on
[AGN-03](storyworld_continual_learning.md). This records exact modules, not intent; update it
when a seam moves.

## DataGoblin (this repo)

| Concern | Location | Status |
| --- | --- | --- |
| Research taste intake, teacher custody | `Agonist/SKILL.md`, `Agonist/gym` | stable (AGN-01/02) |
| Storyworld bridge, SW-PROV digest pinning | `Agonist/storyworld_continual.py` | stable, offline-only (AGN-03) |
| Decision-trace schema/dataset | `schemas/dcf16_decision_trace.schema.json`, `src/datagoblin/datasets/dcf16.py` | stable |
| Cross-repo result ledger | `src/datagoblin/model_atlas.py`, `training/model_atlas/sources.json` | stable, append-only |
| Lean-adjacent intake | `scripts/stage_solved_conjectures.py`, `training/data/formal_game_catalog.json` | pinned corpus staging, no live kernel wiring here yet |
| New in this pass | `Agonist/skills/agonist04/SKILL.md`, `schemas/agn04_world_verification_record.schema.json` | design + schema only, no live calls |

## GPTStoryworld (`MoralityLabAI/GPTStoryworld`, not cloned on this machine)

AGN-03 already names the exact seam: the SW-PROV-05 fixture at commit
`f8ba60b66d719767848edc5cc72522738dc90f5f`, consumed through `provenance_access.py`,
`provenance_characters.py`, `provenance_session.py`, `provenance_design_audit.py` (see
`Agonist.storyworld_continual.FILES`). `storyworld_continual.py` binds those four files by
content digest and never executes untrusted model-authored code — only this pinned,
operator-reviewed checkout. No new SWMD authoring surface (SW-PROV-06+) is referenced by this
repo yet; AGN-04's live authoring loop (below) proposes typed parameters into that same
whitelisted adapter rather than new executable code.

**Unresolved seam:** the PR #21 (SW-PROV-01–05) checkout is not present on this machine. The
offline bridge's unit tests use synthetic fixtures and do not require it; the live
`python -m Agonist.storyworld_continual --source <path>` CLI does. Obtain and digest-pin a
reviewed checkout before any live run.

## EvalMath (`MoralityLabAI/EvalMath`, local clone, `main`)

ADG-01 through ADG-06 are merged to `main` (not a separate open PR on this clone):

- ADG-01 — `9c5d8da`/`b758b98`/`fd2b5ee`/`7e69786`: archival abductive-discovery oracle,
  `evaluation_abductive_discovery.py`, `test_abductive_discovery.py`,
  `verify_abductive_discovery.py`.
- ADG-02 — `b9af2b0`/`c837bd4`/`32f2ef1`/`e641f09`: budgeted archival environment, provenance
  grading, exact pilot receipt/trajectory export.
- ADG-03 — `ea15633`: matched controller pilot, negative recurrence gate.
- ADG-04 — `ddc550c`: revision-aware memory, matched training tails, donor interventions.
- ADG-05 — `c67855a`: exact provenance ledger, revocation witness, access-aware planner census.
- ADG-06 — `cad2a70` (full `cad2a708aa5c66b1850746b1bc909b4a00e8afed`, matches the hash already
  cited in AGN-03's SKILL.md): exact latent-lineage mixture, decision-sensitive provenance. This
  is the exact oracle/verifier seam for AGN-04's provenance and causal-counterfactual lanes.

**Why this is the right seam:** these are the only EvalMath modules that already implement an
independent, non-model oracle for provenance/access decisions on finite worlds — the same shape
AGN-04 Section 3 asks for (realized utility, opportunity capture, counterfactual effects). Reuse
them rather than re-deriving an oracle inside DataGoblin.

## LoopedTransformers / Compactification-Theory (avoid repeating failed runs)

Per project memory, before spending GPU budget on AGN-06 compact-transformer arms, note what is
already settled:

- Direct block-looping an **untrained** mid-size model is a negative result (ARG-02 on
  Qwen3.5-0.8B: loop16 67 vs base 68/168); a **trained-from-scratch** tied-recurrent small model
  can pass a strict gate (Algernon-02B/02C/02D "D1" fixed-slot, per-loop-supervised arm hits
  1.000 at depths 1–16), but naive slot-noise correctors do not repair damage until the
  supervision loss reaches the answer token (H-A3f/g in `algernon-02b-lotus`).
  **Implication for AGN-06:** matched-budget "flat vs. direct-loop vs. TRM" comparisons must
  train all arms from scratch at the target size; do not reuse an off-the-shelf instruction
  model's loop behavior as a control.
- RMP/Lean-CGRT work (`LoopedTransformers` research dirs) found early-hop causal structure
  replicates across seeds at scale S, but a 400M no-go on H1 is final (`lean-cgrt-hrm400`
  Gate 0); AGN-06's CPU-pilot arms should start well below 400M and re-check the same gate before
  scaling.
- `N1` (Nanbeige4.2-3B native loop) is the one mixed-positive native-loop result so far
  (loop8 > loop1, open-loop H2a not met) — the closest existing analogue to AGN-04's "tied
  recurrent computation" question in Section 7; read it before designing new loop arms.

No GPU time is spent by this pass; this section exists so AGN-06 does not re-run these.

## Local serving / authoring infra (Section 2's "teacher receives... produces" loop)

Two pieces of recently-landed local infra satisfy AGN-04's "fast tps on Inky" and "bounded
local resource wrapper" requirements, named directly in this task's own instructions:

- **Inky Mini fast serving ("Maya loader"):** no repo or file on this machine is literally named
  "Maya". The packaged, ready-to-run server is **squid5080** at
  `C:\Users\patri\tools\squid5080\` (`serve.sh` / `stop.sh`, OpenAI-compatible on port `18090`,
  model `inkling-2-mini`) — the Zig+CUDA engine ported from `tensorfish/squid5090` (dev clone
  `Documents\Codex\squid5090`, upstream PR `tensorfish/squid5090#5`), measured at ~165-254 tok/s
  decode and a 45K-token prompt prefilling in ~15 s on the 5080. Confirmed mid-task by the
  operator pointing at that tool's README; see `Agonist/skills/agonist04/SKILL.md` for the exact
  serve/stop/queue commands.
- **"Pocket Goblin GDK app inside Goose":** matches `goblin-gdk` at
  `HybridTRMLDT/MCP/goose_gdk_app` — a Rust app on `goose-agent`/`goose-providers`
  `0.1.0-alpha.14`, built in WSL, with its own CI smoke test (`ci_smoke.py`,
  `goose-gdk-app.yml`). This is the existing Goose-hosted tool/session surface; AGN-04's
  bounded-context, encounter-by-encounter authoring loop (external memory + encounter index)
  should run as a Goose session against this app rather than a new harness.

Both are outside this repo and are referenced, not vendored, consistent with AGN-03's existing
"pinned, separately obtained source" pattern for GPTStoryworld.

## What remains stubbed after this pass

- No live teacher call, no Lean kernel run, no training step. AGN-03's gate — "no paid compute,
  live teacher generation, new Lean kernel result or trained checkpoint is authorized by this
  document" — applies unchanged to this pass; Section 2–3's generation pipeline exists here only
  as a skill definition and a verification-record schema, not an executable worker.
- AGN-05 (induction/Lean), AGN-06 (continual student), AGN-07 (learned curriculum selector) and
  AGN-08 (Aleph integration) are not started. Each needs an operator go-ahead for live model
  calls or GPU training before it can produce a real result, per this repo's existing convention.
