# A clock-order playground for small research agents

A worker can contribute one reproducible local experiment, even if it cannot design a whole training run. Give workers a common checkpoint, prompt, pass-input state, linear readout and precision contract. Let them choose two positive relaxation coefficients and propose an order-sensitive effect. Other workers can compose and audit the resulting receipts.

## The experiment that composes

Fix a deterministic shared core C and write g(x) = C(x) - x and T_a(x) = x + a g(x). Read the second-pass state, before any further nonlinear layer. The vector clock-order edge and its exact potential representation are

    q(a,b) = [T_b(T_a(x)) - T_a(T_b(x))] / (ab) = s(a) - s(b)
    s(t)   = [g(x + t g(x)) - g(x)] / t.

Expand the two compositions and cancel x and the shared first-order terms. For every third positive clock c this gives q(a,c) = q(a,b) + q(b,c). Any common linear projection preserves the identity. It holds even when the core is nonlinear. Final logits after a nonlinear coda require direct measurement. Input replay, stochastic routing or state changes between the calls can break the shared-core contract.

For a twice differentiable core the directional derivative of s(t) is an integral of the directional curvature of g along the initial residual ray. The divided difference -2q(a,b)/(b-a) is its normalized finite-scale curvature measurement for a != b. For a quadratic residual map it is exactly D²g(x)[g(x),g(x)]; in general it is a weighted average of curvature along the ray. It can vanish at an interesting state if the chosen ray or linear readout misses the effect. In a stationary affine core every order edge vanishes.

For g(x)=x² at x=1 and clocks 1/3, 1/2 and 2/3, the two short edges each measure -1/6. Their sum predicts -1/3 for the long edge and the measured curvature is 2. The executable finite game verifies all four values exactly. This use of a divided difference and a graph potential is a proposed research instrument. The algebra alone establishes no learning or model-quality advantage.

## What to buy and what to reward

On a fixed root, a forest of measurements determines the scalar potentials up to one additive constant per connected component. To see this, set one vertex potential arbitrarily in each component and propagate along the edges. A bridge removes one undetermined offset. A closing edge supplies a falsifiable prediction of the existing receipts. Use both types of contribution. One worker can propose an untested root or intervention, another can complete a path, and a third can test the implied edge without seeing its predicted value. Record a numerical error envelope from same-state repeats, exact affine controls, and a higher-precision replay on sampled cases. A large cycle residual diagnoses an incompatible or noisy receipt; every compatible deterministic nonlinear core also has zero cycle sum.

Unlock the task label only after committing the predicted sign, magnitude interval, root family and verifier budget. Compare the proposed selection policy to random probing and a fixed spanning-forest/closure rule. The primary endpoint is an independently checked downstream forecast, such as loss of a formerly verified repair after scrambling a prior-pass message. Predefine dose, sham and verifier tolerance. On text, report loss separately. Count paid oracle calls, replays and unique roots. A forest completed on public fixtures is a plumbing success. Transfer requires a new generator and another student.

## A second game for federated identifiability

This binary game deliberately models an unknown sign convention for each lab. An edge reports y_uv = g_u g_v H^t, where g_u and g_v are unknown signs, H is the rival mechanism sign, and t = 0 or 1 marks the designated clock contrast. Under this declared noiseless model, H is identifiable exactly when a cycle has an odd number of marked edges. Multiply around the cycle: the lab signs cancel and H survives. Conversely, if all cycles are even, assign orientation corrections along a spanning forest. Changing H and applying those corrections leaves the entire public record identical. This is an application of standard signed-graph parity to designing a research gym. Physical clock differences obey the additive identity above; applying this binary sign model to real workers would require separate calibration.

The executable triangle census has four ambiguous public transcripts on a two-edge tree, four on an even triangle, and eight fully identified transcripts on an odd triangle. With independent sign errors of probabilities e_i, an odd cycle's expected signed reliability is the product of (1-2e_i); cycles sharing an edge share noise. Model the order of disclosure and selection when estimates guide the next query. The small model plays researcher by proposing the cheapest reliable missing comparison and giving a possible observation that would falsify its current choice.

## Physical replay at step 96

On the 80-wide direct checkpoint, seed 137, the FP64 last-prompt-token channel-zero edges for the public repair root were −0.953779 and −0.474473. The directly measured long edge was −1.428251. For the counterexample root the values were −4.086581, −2.380816 and −6.467397. Full-tensor relative RMS closure error was 1.26e−15 and 1.35e−15 respectively. The matched affine scalar order effects stayed below 5.4e−15. Deliberately substituting the second edge from the counterexample root into the repair-root forecast gives a scalar residual of −1.906343. This negative control shows the cost of combining incompatible roots. These two deterministic replays test the measurement interface. A task-performance forecast needs its own verifier. The receipt records the exact public roots, checkpoint SHA-256, code and readout.

Run `python training/aleph_rlt_placement/clock_order_gym.py` for both finite examples. A submitted signed graph can be scored with `--graph graph.json`. The input is an array of [lab name, lab name, binary clock flag, sign] records. An empty compatible set exposes inconsistent evidence.

[EvalMath experimental separation](https://github.com/MoralityLabAI/EvalMath/blob/497f87cd83fd1ca8d53f9e39ebd74474612f1dfe/paper/bpd_interventional_separation.md) · [Loop schedule algebra](https://github.com/MoralityLabAI/LoopedTransformers/blob/31256abc921ae9aa500ad4fb85a167a990be0984/docs/lsa_typed_architecture_calculus_v0.md) · [Dense placement protocol](PROTOCOL.md)
