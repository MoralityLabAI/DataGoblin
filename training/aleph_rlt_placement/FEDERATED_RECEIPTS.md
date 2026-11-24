# How much of a research receipt must a small lab publish?

**A compact summary must survive the experiments other researchers can still perform.** A locally complete answer to today's target can throw away the one relation another lab needs tomorrow. This observation connects EvalMath's timed predictive equivalence to the practical question of how a network of small challengers should exchange synthetic data.

## The two-lab failure

The public target is a bit t. One lab verifies s = 1 and the other verifies s ⊕ t = 0. Each lab alone must report that t could be either 0 or 1. Merging those two reports still gives no answer. Merging the two equations yields t = 1 exactly. The first lab can accrue value by measuring s even if it cannot solve a target and its contribution has zero standalone accuracy gain. Reversing either measured bit reverses the prediction.

The rival theory says each agent need publish only what it learned about the target. We tested it on every ordered pair of affine binary relations on two bits, including inconsistent records. It returned a different feasible target set from a centralized solver in **32 of 144 cases**. Eight errors persisted when the original records were jointly consistent. The remaining 24 concealed a contradiction. These are exhaustive counts over a constructed finite class, not estimated lab failure rates.

## The repair and its exact communication cost

Declare a public boundary B consisting of the target directions and every coordinate that multiple labs may constrain. Give lab i additional private variables P_i. Assume different P_i are disjoint and that all cross-lab constraints are expressed through B. Let R_i be a lab's feasible assignments on B and P_i. Publish the relation M_i on B that admits at least one local private completion. Then

    project_B(join_i R_i) = intersection_i M_i.

Every global witness projects to a witness for each M_i. Conversely, a boundary assignment in all M_i admits a completion at each lab. Their private coordinates are disjoint, so the local completions join to one global witness. The equality holds for arbitrary relations. Noiseless binary linear receipts specialize it to Gaussian elimination: place private columns first, row-reduce, publish the remaining boundary-only equations, then merge them by another row reduction. In the example the boundary must include s as well as t, or nodes must meet to eliminate s together before publishing a target-only receipt. Identical equations are idempotent on this interface.

**Tight finite bit count.** On b binary boundary coordinates, the attainable messages are all affine subspaces and the empty set. A k-dimensional subspace has 2^(b−k) affine translates; there are Gaussian-binomial [b choose k]_2 distinct k-dimensional subspaces. The number of messages is

    N(b) = 1 + sum_{k=0}^b [b choose k]_2 · 2^(b−k).

If future partners may pin an arbitrary complete boundary assignment and contradiction is observable, any two distinct boundary relations require different messages. Choose an assignment in their symmetric difference and have the partner pin exactly that assignment. One merge is consistent and the other is impossible. Conversely, indexing the N(b) relations suffices. The worst-case fixed-length message on this declared finite interface therefore needs exactly ceil(log2 N(b)) bits. For b = 0, 1, 2 and 3, the counts are 2, 4, 12 and 52, requiring 1, 2, 4 and 6 bits. The count excludes the shared schema, variable names, decoder, provenance and verifier evidence. Row reduction supplies a convenient canonical representation; it need not achieve the shortest encoding.

The executable checker compared independently projected global truth tables with the actual row-reduction merger for **all 2,704 pairs** of three-bit affine lab relations on a two-bit boundary and one private bit per lab. All agreed. The rule also returns inconsistent for contradictory receipts. A shared hidden variable invalidates the disjoint-extension proof. Noisy evidence generally requires likelihood information and provenance to avoid counting a shared observation twice. A new intervention outside the declared boundary can make an old compressed report inadequate. These are falsifiable conditions of the model.

## A playable researcher episode

Give two local models different receipts, the shared names, a future target family and an upload budget. Each can propose a distinguishing query, an equation, a boundary variable to retain, a contradiction or an abstention. Another node predicts a withheld comparison before a verifier opens it. Compare target-only messages, full transcripts and boundary equations at equal call cost and reported wire bytes. Score reconstruction, incompatible merges, downstream student risk and the value added to an existing ledger. Hold out partner types and rename coordinates to test generalization. A node that over-exports all coordinates pays communication cost. A node that erases the shared witness loses the future decision. This is a mathematical proposal for a gym; a trained model's acquisition advantage still needs to beat the exact non-neural merger and the simple acquisition policies.

Use `python training/aleph_rlt_placement/federated_receipts.py` for the census. With `--merge instance.json` it accepts JSON containing a boundary list and a list of labs, each with variable names and equations of the form `[["s","t"],0]`. The default receipt includes a complete playable example. [Compactification Theory intervention contract](https://github.com/MoralityLabAI/Compactification-Theory/blob/e70534fbfebc0d1124e4cdf23f55531694dd2870/note/intervention-preserving-compactification.md) · [EvalMath interface-relative equivalence](https://github.com/MoralityLabAI/EvalMath/blob/accf4d9040bee8daeabc6b07003bb484708996c0/paper/interface_relative_compactification.md) · [Data-value comparisons](AGONIST_VALUE.md)
