# Frozen sender, fresh receiver

Lock this protocol and the executable before the fresh containment draw. Use all nine v2 primary LDT senders from C, D and MLP at seeds 137, 163 and 191, checking their hashes and their coordinated parity parents against the v2 manifest. Cross each sender with fresh C, D and MLP receivers. Reinitialize receivers from seed 100000 plus the sender seed, so each receiver architecture has the same numerical starting parameters on different senders at a given seed. The sender is a separate frozen object. Only its thresholded binary upload crosses the interface.

Train each fresh receiver for 500 Adam updates at learning rate .003, clipping gradient norm to 1. At each update draw one of the 24 original v2 training frames in a seeded balanced rotation. Enumerate the eight worlds. For each positive-mass cell indexed by public query, verified side bit and upload, compute the exact frequency of a positive target. Minimize squared probability error to this frequency weighted by its cell and query probability. This equals expected Brier loss minus the receiver-independent conditional variance. Both networks have access to the public features defined by the v2 game. World identity, cell mass and target frequency are unavailable as receiver inputs. No held data or checkpoint selection enter fitting.

At updates 0, 100, 200, 300, 400 and 500 evaluate the frozen sender's logits on all 13 previously public v2 parity priors. Record the exact digest and maximum change against step 0. Compare the original coordinated parity sender, the v2 LDT-adapted sender and the original v1 outcome sender at the four published bracket priors. Check the frozen weight digest at both ends. These probes monitor the sender only and do not influence optimization.

After every trained receiver and source hash has been frozen and verified, draw one new private seed. Evaluate four fresh frames in each of the four v2 containment families. These are new frames of the same relation topology. State the family-balanced realized Brier for the intact v2 pair, randomly initialized and trained receivers on the same frozen upload, the ideal posterior using that upload, and the ideal side-only receiver. Include the per-frame best linear and unrestricted bit ceilings. The intact and refitted predictions use their actual hard bits and public receiver information. The oracle controls are calibrated separately to each evaluation frame.

The primary comparison is fresh C against fresh MLP on identical v2 C senders. Count a readout rescue if the fitted C receiver beats the intact receiver, the paired MLP and side-only on all three seed aggregates, and has no held family with mean Brier worse than side-only. The D receivers and the other sender families are secondary matched comparisons. Also report residual loss to the ideal frozen-bit receiver and progress from random initialization. This assay tests the decoder and leaves all architecture-tournament nominations unchanged.

From the repository root with the v1 and v2 checkpoint bundles in sibling directories, reproduce the fit and private fresh draw as follows.

```powershell
$d = 'training/aleph_rlt_placement'
python "$d/cpu_worker.py" --wall 900 receipt_readout.py train --v1 ../receipt-cd --v2 ../receipt-v2 --out ../receipt-readout
python "$d/cpu_worker.py" --wall 180 receipt_readout.py score --v1 ../receipt-cd --v2 ../receipt-v2 --out ../receipt-readout
```

The scorer creates its private seed in the local output directory only after validating the complete fit. Retain it privately for exact replay together with both source bundles and receiver weights. Publish the hashed receipts and authored aggregates.
