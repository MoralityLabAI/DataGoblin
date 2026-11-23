---
name: metta-storyworld-routes
description: Plan or replay a concrete Storyworld courier route with permit gates, cargo state, and native MeTTa transition checks.
---

# Plan Storyworld routes

1. Identify the public world, initial `CourierState`, goal, and action cap.
   Read the actual corridor, permit, pickup, and delivery facts. Use a fresh
   native space for each world.
2. Query `CourierSuccessors` for legal action and next-state pairs. Track
   the full state when searching, since returning to a site with a permit
   or cargo can change its successors. Keep tied shortest routes when asked
   for all optimal actions or plans.
3. Replay the complete action sequence with `ReplayCourierPlan`. Compare
   the final state with the delivery goal and check that each replayed
   transition uses the same world. Use a missing-permit or unreachable case
   to probe the corresponding negative path. Keep unsuccessful or
   unreachable verdicts tied to the supplied search bound.

In the Storyworld checkout, use `engine/courier.metta` and the
`NativeCourier` adapter in `bridge/courier_planner.py`:

```sh
python -m pytest -q tests/test_courier_planner.py
```

Return the world and goal, action sequence, replayed states, and terminal
verdict. Report shortest-path status only if checked.
