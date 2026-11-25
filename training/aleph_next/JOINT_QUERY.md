# A gym where two individually worthless receipts win

The eight-site ALEPH-04 benchmark asks about exactly one independent site at deployment. Under that assumption, ideal gains from choosing sites add. A compiler that ranks sites by individual gain can recover the optimal public allocation in that game. As an exploratory follow-up, change only what a late question can depend on.

Three workers see independent fair bits A, B and C. Each selected worker transmits one bit before the query. The receiver must forecast C with probability p and A XOR B otherwise. It sees the query and the selected uplinks. Squared probability loss is the score. With two transmission slots every local worker can just send its bit; exchanging its two message labels preserves what can be inferred.

For any public subset S the reduction in Brier risk is

```text
V(S) = [p · 1(C in S) + (1 − p) · 1(A and B in S)] / 4.
```

A missing fair input leaves the XOR posterior at one half. Observing both operands determines it. Thus the marginal value of A increases from zero to (1−p)/4 after B is present. With 0 < p < 1/2, C is the unique best individual receipt and buying it first makes the optimal two-receipt solution unreachable. The greedy allocator achieves only p/(1−p) of the optimal utility. This ratio approaches zero even with perfectly calibrated individual values. At p = 1/4, exhaustive integer-rational enumeration gives risk 3/16 for the greedy choice and 1/16 for A+B. The regret is exactly 1/8. At p = 1/2 the two choices tie. Above it C belongs in an optimum.

This is a training opportunity for a small participant. A worker can submit a pair, a bound on the pair's gain, and a finite counterexample to an additive allocation. The environment replays all eight worlds and compares its verified gain per bit against the greedy and exhaustive controls. To transfer this idea to research, let one model discover a candidate constraint and another discover the condition under which it is valid. Before rewarding a pair, price the selection or coordination channel whenever its identity depends on private observations. The toy result is an exact counterexample to independent ranking on joint queries. Prevalence or learnability of such pairs in real mathematical work remains unmeasured. [Exact verifier](joint_query.py) · [Original channel assumptions](NONLINEAR_PROTOCOL.md)

```sh
python training/aleph_next/joint_query.py --p 1/4
python training/aleph_next/joint_query.py --p 1/2
```
