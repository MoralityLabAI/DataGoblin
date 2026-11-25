# ALEPH-STATE-05 | algebraic identification versus neural state feedback

2026-11-25. Exploratory CPU-only follow-up to STATE-04. Keep A and both paid slots on hold. Source and summary are committed; full numerical receipt SHA256 2d422003e84967a73a0466d996377475fdc43019628d463c9a270ffa01666b4c.

## Hypothesis and design

Two registers x=(a,b) in (Z/8Z)^2, 64 states, 19 instructions: odd translations of either register, cross-register addition/subtraction, swap, coordinate reset, copy and doubling. All instructions are affine over Z/8Z. Fix a balanced public held-edge mask (seed 2026112507) withholding 16 of 64 input states per instruction; 912 allowed and 304 withheld state-instruction pairs. Every oracle transition in a contextual training path is restricted to allowed edges. The local arm also samples only allowed edges. This holds out input/output transition pairs, not entire instruction families.

Nine independent 63,504-parameter direct-architecture miniatures: width 16, 14 physical blocks, 24 visits, tied 512-symbol projection, four attention heads and one KV head. Local transcription of the native architecture, random initialization; not a continuation of the historical checkpoints. Seeds 241/251/257, 256 AdamW steps, lr .001, gradient clipping 1, one CPU thread per run. All arms see exactly 51,200 actor-token positions. Terminal/state arms share paths of lengths 4 and 8. Local samples one-step edges independently and matches actor-token positions; number of supervised pairs differs (5,120 terminal, 30,720 state, 10,240 local). State loss splits mass equally between final and interior CE; both registers decoded via separate eight-class slices of the tied output. Gold states absent from contextual inputs; local examples contain the supplied one-step starting state.

Public diagnostic panel seed 2026112506: 16 words per regime and length (12,24,48); permutation, exactly one erasure instruction, exactly two erasure instructions. For each word, vary A over all eight possibilities while holding B fixed. This is an eight-element slice, not a complete 64-state orbit. Both inference modes evaluated: direct contextual prediction, or one-step prediction table composed from the network's own argmax outputs. Exact symbolic oracle also included. Panel is public and exploratory, no sealed promotion. No shuffled-label arm in this comparison. Equal actor tokens and update count are not equal compute, equal label count, or equal sampling distribution.

## Executed results

Pooled over three seeds; local train denominator 2736 and held denominator 912. Each program evaluation arm has 432 word-orbits (144 per seed).

|Arm|Train local exact|Held local exact|Direct terminal orbits|Feedback terminal orbits|Complete traces, either mode|
|---|---:|---:|---:|---:|---:|
|Terminal|111|29|4|3|0|
|Verified state|308|57|1|3|0|
|Explicit local|1390|229|0|1|0|

On all 48-step regimes pooled (1152 terminal cases per arm), direct vs feedback correct answers: terminal 22/15; state 25/22; local 17/22. No neural arm solved a complete permutation orbit at any evaluated length. Explicit local supervision improves held local rule acquisition but falls far below the accuracy needed for reliable composition. At length 48 every trace traverses at least one held edge. These runs do not justify an architecture or curriculum promotion; width/dose and sampling constraints prevent a conclusion about larger models. Exact oracle solves all cases.

The nine fits consumed 167.134629159 summed training CPU process seconds (excluding evaluation), zero paid compute. Sources state05.py SHA256 bb7a1eb605d918d8f4410bc8cd4dbec103942a9431db432b03ba8f4912858af8, affine_control.py SHA256 ca7c606992b7adba3d59c4128bf56d514a7b50e5e1955abbe85d19bd9e269009. Per-seed [summary](STATE05_SUMMARY.csv). Reproduce:
    for s in 241 251 257; do
      for a in terminal state local; do
        python state05.py --seed "$s" --arm "$a" --out runs
      done
    done
    python affine_control.py > affine_results.json

## Algebraic compactification

Let R=Z/mZ, x in R^d, and assume F_u(x)=A_u x+b_u. Write augmented coordinates X=(x,1). An affine map is uniquely identifiable from d+1 input/output witnesses whenever the square witness matrix W (rows augmented input vectors) has determinant a unit of R. In that case W is invertible and each output column has unique coefficients W^{-1}y. This is a conditional certificate under the affine-family promise, not a proof that arbitrary black-box maps are affine. For m=8,d=2, det W must be odd. Fewer than three witnesses cannot identify an unrestricted affine map in this class: mod 2 the at-most-two by three witness matrix has a nonzero kernel vector, and multiplying its lift by 4 produces a nonzero ambiguity mod 8. A singular mod-2 three-witness matrix likewise admits ambiguity.

For each instruction, select the first unit-determinant triple using only allowed input coordinates; fit its six coefficients by finite enumeration, then check all other available training labels. The 19 rules used 57 witness label pairs and 855 additional train-only validation pairs. All 912 allowed and 304 withheld transitions were recovered exactly. Independent affine composition matched final states and orbit ranks for all 144 public test words. Replacing one instruction by A*=B (mod 8) gives a nonlinear negative control: the candidate fitted to three witnesses fails 36 of the 48 available training labels and is rejected before relying on held labels. Fitting uses no held targets; the evaluation reads them only for scoring.

For composed affine word F_w(x)=A_w x+b_w, a slice x=(a,b0) has outputs differing by a v where v=A_w e_1. The order of v in (Z/mZ)^2 is r=m/gcd(m,v_1,v_2), with equal-size fibers. Therefore a suffix-only blind predictor has oracle upper bound 1/r, and remaining uniform initial-state information equals log2(r) bits. Composition updates A' = A_u A and b'=A_u b+b_u. Rank in this formula is orbit cardinality rather than matrix rank over a field. For this restricted task the resulting algebraic representation uses six mod-8 coefficients per instruction (342 bits excluding metadata) vs 1216 six-bit outputs (7296 bits) for an explicit transition table.

## Next gate

Separate rule discovery from execution. Use a typed candidate hypothesis class (affine, piecewise affine, then nonlinear), propose informative witnesses, use a train-only verifier to reject invalid candidates, and measure held-rule accuracy, coverage, fallback calls, and composition. Keep a symbolic exact executor as control. Escalate from known modular instruction names to unknown rule IDs and withheld families; verify conjectured rules with independent proof or exhaustive finite checker before certifying them. The small neural model can propose compact rules rather than predict every state. Train on counterexamples and model abstention. Avoid treating the 57 fitted witness labels as a full certificate outside the promised class. Historical 14M checkpoints were unavailable, so no native continuation or actual 1B cost estimate was attempted.
