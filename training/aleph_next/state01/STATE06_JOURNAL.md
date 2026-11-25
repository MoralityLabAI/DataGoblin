# ALEPH-STATE-06 | hypothesis discovery and grammar-relative certificates

2026-11-25. CPU-only diagnostic. Direct A remains the reference; neither paid slot authorized. Follows STATE-05. The neural arm is a small classifier, not Aleph or a continuation of a historical checkpoint.

## Operator grammar and split

Domain (Z/8Z)^2, scalar output in Z/8Z. Enumerate all affine a*x+b*y+c, then affine plus d times an indicator (x>=4, y>=4, x>y), affine plus d times a quadratic feature (xy, x^2, y^2), and bitwise XOR/AND/OR of two one-coordinate affine functions. d ranges 1..7. Deduplicate by full 64-output truth table, assigning the earliest class/template in that enumeration. These are canonical grammar labels, not mathematically disjoint classes by definition and not a proved minimum-description-length ordering. Prefer seen templates over withheld templates within each class when semantics overlap. Unique maps: affine 512, piecewise 10,752, quadratic 6,656, bitwise 6,304, total 24,224. Withheld templates x>y (3,584 unique), y^2 (1,536), OR (2,896); semantics already present in a preceding template remain with that template.

Split seed 2026112508. Train candidate pools 416/7072/5024/3312, held seen-template IID 96 per class; held-template panel 96 per non-affine class. Neural train episodes draw functions from train pools only, balanced per class. Panel seed 2026112509 supplies nested uniformly random distinct witness positions (4,8,16) for each of 384 IID and 288 held-template operators. Exact symbolic control has the entire enumerated grammar, including withheld templates; it does not receive a test operator identity or unqueried outputs. Equal class prior, uniform over distinct maps within each class. This prior differs from the sampling distribution of the held-template panel. Its greater structural knowledge is explicit, not an equal-information neural comparison.

Neural MLP has 65,284 parameters, 64 position-specific nine-way inputs (missing or output 0..7), two 96-wide layers and four class logits. Seeds 263/269/271, 800 AdamW updates, batch 128, lr .002, 102,400 sampled episodes and 954,880 observed witness labels per seed. Budgets cycle 4/8/16. No claim of architecture-optimality or training saturation. Python CPU torch 2.10.0+cpu, NumPy 2.3.5. Classification only: a neural class guess does not specify a mathematical operator.

## Executed results

|Panel|Budget|Neural class hits (3 seeds)|Exact symbolic class hits|Exact unique function|Exact unique class|
|---|---:|---:|---:|---:|---:|
|IID|4|419/1152|230/384|5/384|34/384|
|IID|8|468/1152|369/384|251/384|307/384|
|IID|16|547/1152|383/384|377/384|381/384|
|Held template|4|134/864|155/288|2/288|29/288|
|Held template|8|177/864|275/288|203/288|254/288|
|Held template|16|234/864|288/288|285/288|288/288|

Class identification and exact function identification are different endpoints. Accuracy on the three-class held-template panel is for a four-way classifier, with the absent affine label still available to both methods. Neural selection is substantially weaker than exact grammar filtering; it has not earned a deployment or training-budget promotion. Per-seed counts/NLL in STATE06_SUMMARY.csv.

The first run completed numerically but failed serializing a NumPy scalar; a rerun with only the serialization fix reproduced all per-seed class counts. Successful run training CPU 5.775554 s and symbolic scoring 0.483057 s, excluding setup; first discarded run training 6.200825 s and symbolic 0.515277 s. Zero accelerator credits.

## Version-space mathematics

Given deterministic hypotheses h:X→Y, observed data D={(x_i,y_i)}, let V(D)={h in H: h(x_i)=y_i for all i}. An output at a new x is entailed relative to H iff all surviving hypotheses agree and V(D) is nonempty. If the true function belongs to H, it belongs to V(D), and unanimous predictions are correct. A singleton certifies the entire function under this coverage assumption. A unique class alone certifies neither a function nor its unqueried outputs. An empty V(D) rejects the grammar under exact, noiseless observations.

For a posterior over V(D), noiseless deterministic queries give I(H;Y_x|D)=H(Y_x|D). For class C=c(H), I(C;Y_x|D)=H(Y_x|D)-H(Y_x|C,D). These differ: a query can distinguish functions within a class without reducing class uncertainty. Greedy maximizing either objective is a one-step policy, not a proof of global optimality. With 24,224 maps and eight outputs per query, any universally identifying deterministic adaptive decision tree requires worst-case depth at least ceil(log_8(24224))=5; this counting bound is not necessarily attainable.

## Post-hoc active query diagnostic

Same public operators as the first experiment; query policies designed after initial results and not sealed. Exact posterior over full grammar; equal unit query costs. Choose unqueried x maximizing predictive output entropy (function) or mutual information with the canonical class (class). Deterministic first-argmax ties. Query reveals only the true output at x; recompute survivors.

|Panel|Budget|Random class hits|Function-entropy class hits|Class-MI class hits|Random unique maps|Function-entropy unique maps|Class-MI unique maps|
|---|---:|---:|---:|---:|---:|---:|---:|
|IID|4|230|241|319|5|20|0|
|IID|8|369|383|384|251|382|225|
|IID|16|383|384|384|377|384|352|
|Held template|4|155|185|194|2|0|0|
|Held template|8|275|288|287|203|286|137|
|Held template|16|288|288|287|285|288|251|

Denominators 384 and 288 respectively. Class-MI wins early classification in IID while function entropy recovers more complete maps at eight queries. For the class policy, when all survivors share one class, information gain about that class is zero; this can waste further queries. A class-first/function-second switch or joint objective is a proposed next experiment, not a tested optimum. Both policies incur full version-space acquisition compute, and the measurements do not match wall-time costs against the neural selector. The two metrics are separately meaningful for an LDT query allocation objective.

## Grammar misspecification stress test

Remove all operators uniquely assigned to the three withheld templates from the symbolic grammar. Retain all seen-template operators including IID functions; this is a grammar-coverage test, not a stricter neural training split. On the held-template panel, random distinct queries leave:

|Budget|Rejected (empty version space)|Still compatible|Wrong singleton|Cases with at least one false unanimous coordinate|
|---|---:|---:|---:|---:|
|4|7|281|14|172|
|8|248|40|32|39|
|16|288|0|0|0|

The wrong singleton cases are conditional certificates under an invalid assumption, not independently verified mathematical facts. At eight witnesses, 32 of 40 surviving wrong-grammar cases appeared uniquely identified. All 288 held-template operators were rejected at 16 on this panel; no guarantee on a new grammar. No false unanimous coordinates in the IID panel. Test labels used only to score agreement errors. Diagnostic CPU 8.553240 s.

## Next gate

STATE-07: preregister a fresh semantic-family panel and equal verification budgets, compare fixed random, exact class-MI, exact function entropy, class-first switch and learned witness policies. Separate open-world abstention from closed-world recovery. Use known-grammar certificates only with an explicit grammar-coverage promise or independent verifier; otherwise escalate to counterexample search, enlarge the hypothesis grammar, or abstain. Track full operator recovery, false acceptance, witness count, candidate-evaluation cost and wall-time. A compact learned query policy should beat a cost-matched symbolic baseline or approximate exact acquisition much more cheaply at the same query budget. Train proposals only on public training families; do not use the diagnostic panel for promotion. No historical 14M checkpoints were available.

## Reproduction

From this directory: python state06.py ; python query_diagnostic.py. Sources SHA256 state06.py 22defa466dcbecfe13c89590fbc67251ba0a9f1a17d97e1ead94740e21af07ef, query_diagnostic.py 8f5b5c431bd77753bd3a1cf0e71190f99a8894d3ee5c88f8c5b3d762c532122a. Local full numerical receipt hashes (including timings): STATE06_RECEIPTS.json e76e39220bb0239747ae8f7c7046e6cf92e70b66fe436be65d111956aea0cc01; STATE06_QUERY_RECEIPTS.json 22f6610edfa1fb1ea791fbd4b5857edbee3c1324450d87d09c38653f71c32f79. Total measured CPU across original/repeated training, symbolic scoring and diagnostic about 21.53 s excluding setup and other work. No paid compute.
