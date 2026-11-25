# ZKL-20: bulk-lift typed curriculum

ZKL-19 verified all six closed typed certificates with Nova/Spartan IPA: 185 rows, 25,945 raw constraints per step, 11,472-byte compressed proofs, 31 forged constraint witnesses rejected. Measured evidence https://github.com/patrickdugan/zk-rusto-no-bitcoino/actions/runs/45586627039/job/139109768491 . The syntax arena is public.

ZKL-20 adds an explicit nonnegative lift distance and raw context lookup. Variable typing requires WF, retrieval of the stored binder type and a certified lift of i+1; substitution under a binder still requires exactly one lift. The bounded 16-bit distance is constrained inside Nova. Native evidence: 4096 repeated-unit versus bulk shift comparisons; six fixed fixtures and 189 rows; all 32 local cases; 5153 wrong-premise replacements and 42 invalid distance mutations rejected. Zero and maximum distances checked. Native generated diagnostics only; actual new Nova measurements pending.

| family | reused roots old/new rows | fresh roots old/new rows |
|---|---:|---:|
| weakening | 4260 / 4231 | 4150 / 4145 |
| dependent application | 14829 / 11678 | 14305 / 11294 |
| conversion | 6217 / 6226 | 6992 / 6985 |

64 roots per family per cohort. Reused seeds 22000/23000/24000 + 0..63; fresh disjoint seeds 32000/33000/34000 + 0..63. All source certificates replay with both relations and new equations. Dependent application improves on 58/64 in each cohort; other families have negligible total benefit. Reused 64-slot coverage stays 62/192, 256-slot coverage improves 166->173, all fit 512 slots. No learned model or sealed confirmation. No claim of optimal certificate lengths.

Training implication: use the native bulk and unit policies as controls and treat typed context complexity as a stratification variable. Do not reward choosing a shorter certificate from an incompatible circuit profile without checking its cost and capacity. Hold out source families rather than individual rows, retain all failed/capped roots in the denominator, and independently replay candidates. Keep distance-specific lifting separate from proof reuse; the unit strategy sometimes saves work via memoization. Strong next baseline is a profile-aware choice based on actual measured cost, subject to pinned verifier parameters. A richer single relation allowing both lookup modes and proved equivalence is a future candidate. No corpus release, neural training or paid compute.

Implementation and reproducible scripts: https://github.com/patrickdugan/zk-rusto-no-bitcoino/tree/15091c9160b262d393b453fd92050e315726003d/integrations/zkl-nova-calibration . Two separate hosted profiles (depth 6 small fixtures and depth 9 stress example with 270 rows and 112 arena objects) are configured. No claim of privacy or arbitrary Lean theoremhood.
