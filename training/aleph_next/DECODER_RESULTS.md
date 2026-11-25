# ALEPH-03 frozen-channel tournament

The recurrent C reader lost accuracy when it received the frozen C sender's bit. Its mean Brier loss was .20755 with side information alone and .22011 with the valid upload. The increase appeared on all three training seeds and all four held relation/status families. The paired gain was −.01255 with a seed/panel bootstrap interval of [−.02048, −.00448]. A logistic reader given the very same bits gained .03336 [+.02998, +.03659]. Receiver choice changed whether the transmitted information helped.

All 216 fits finished. Nine original v2 senders were frozen, with separate side, constant, within-side shuffled and valid fits for six reader families. Each seed saw the same 24 public training frames and update order. Four fresh private panel draws gave 64 frames across the four previously defined containment strata. Each frame exhaustively scored eight worlds and six queries, giving 3,072 paired world/query cases per fit. Every frame was disjoint from both earlier held panels by full-frame SHA-256. The intervals resample seed and panel, keeping the four tested relation/status families fixed. [Registered protocol](DECODER_PROTOCOL.md)

## Paired receiver utility

Each entry gives **valid Brier / learned gain**. Positive gain means the valid bit reduced loss relative to the same learned side-only reader. An asterisk marks a passed registered gate, which includes checks against constant and shuffled receipts. The primary comparison was C sender to C reader. All ten passing pairings were secondary.

| Reader | Frozen C sender | Frozen D sender | Frozen MLP sender |
|---|---:|---:|---:|
| Recurrent C | .22011 / −.01255 | .21735 / −.00980 | .22350 / −.01594 |
| Recurrent D | .21813 / +.01467* | .22488 / +.00791* | .22659 / +.00620 |
| MLP | .20961 / +.02956* | .22639 / +.01278 | .20969 / +.02949* |
| Logistic | .19313 / +.03336* | .21099 / +.01549 | .19060 / +.03588* |
| Empirical conditional | .20012 / +.02740* | .21340 / +.01412 | .20153 / +.02599* |
| Structured relation-aware | .14177 / +.02728* | .15545 / +.01360 | **.13924** / +.02980* |

On C uploads the recurrent C control losses were .20588 with a constant and .21309 with a shuffled bit. Its frame-level ten-bin calibration error rose from .13465 side-only to .18890 with the valid upload. Exact side and valid Bayes risks were .16905 and .11303. An oracle gain of .05602 was offset by a .06857 increase in the recurrent reader's excess risk. This gives a negative recovered fraction of −22.4% and a channel carrying .21156 conditional bits. The bits carried information that this reader used poorly.

The logistic reader recovered 59.5% of C's frame-calibrated value, and 64.1% of the MLP sender's value. The structured reader obtained the lowest absolute Brier with every sender. It starts with the documented public two-rival survival rule and enumerates all eight hypothetical worlds compatible with the side bit, fitting 16 upload-likelihood logits from authorized training frequencies. Its side predictor already attains .16905. On MLP uploads its valid loss is .13924, recovering 53.3% of oracle gain. The 15-parameter logistic reader starts at .22649, so it has more side error to correct. Relative gain and final predictive accuracy select different readers.

The D senders averaged .02806 of frame-conditioned oracle value and .10510 conditional bits; MLP senders averaged .05594 and .20950 bits. D seed 191 produced zero conditional information and zero oracle value. Its recurrent D reader tied its constant and shuffled controls exactly, while the neutral absent-symbol control scored .00081 worse. Aggregate D→D passed, and zero channel value at that seed precludes a recovered fraction there. The positive mean gains of the D→MLP, D→empirical, D→logistic and D→structured pairings missed the all-seed criterion.

The recurrent readers each use 6,880 unique parameters and two shared visits. The feedforward MLP uses 1,601 parameters. The empirical estimator stores 163–325 occupied table entries across these fits. The 216 fit records reference 198 distinct file digests because some empirical tables are byte-identical across conditions or senders. Absolute scores incorporate the structured model's known public geometry, which transferred to these related containment frames. The fixed topology, four independent panel draws and three trained sender seeds limit extrapolation to new verification rules.

An exploratory diagnostic pooled identical public query, side and message observations across all 64 frames before computing the ideal posterior. This retrospective oracle can see aggregate evaluation frequencies, and its predictions depend only on features available to a learned reader. Its utility was .05073 for C, .02739 for D and .05456 for MLP, with .18314, .10181 and .20258 conditional bits respectively. C's frame-identity premium was .00529. The remaining .05073 measures potential improvement using the authorized public inputs themselves.

## Reproduction and custody

The registered train and score commands write local checkpoints, an answer-free fit manifest, per-frame metrics and aggregate comparisons. The fitted-receiver manifest SHA-256 is `8475093e40981af6f6f9671ea842c00fbc8db07b18aed81f025221d626d61ebd`. Each original producer, coordinated parent and fitted receiver is independently checked. Local scoring verifies all receiver bytes before drawing or loading its private seed. Exact replay uses the existing local seed. A new output directory makes an independent replication.

```text
python training/aleph_next/bounded.py --wall 5400 --core 2 decoder_lab.py train --v2 ../receipt-v2 --out ../aleph-next/decoder
python training/aleph_next/bounded.py --wall 3600 --core 2 decoder_lab.py score --v2 ../receipt-v2 --prior-readout ../receipt-readout/results.json --out ../aleph-next/decoder
python training/aleph_next/bounded.py --wall 1800 --core 2 decoder_pooled.py --v2 ../receipt-v2 --prior-readout ../receipt-readout/results.json --out ../aleph-next/decoder
python training/aleph_next/decoder_release.py --out ../aleph-next/decoder --figure ../aleph-next/decoder/decoder-utility.svg
python -m pytest -q training/aleph_next/test_decoder_lab.py
```
