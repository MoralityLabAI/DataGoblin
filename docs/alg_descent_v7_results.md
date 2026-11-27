# ALG-DESCENT-07 / BPD-RISK-01 development results

Protocol registered before execution: 3da5bee751ef7e4d5e8b9d810418f9896ab37174. Source: src/datagoblin/research/alg_descent_v7/risk.py, commit 9e10106e409a3edbb4fffb5a969abbbb879755da, verified local/remote git blob a8ae29e841bfb414f7aad34f284d288ee8c84c88. Synthetic public-witness polynomial certificate routing. This is a BPD-inspired shared-factor pilot, not the full BPD engine, DLP solving, or a cryptographic security result.

## Design

96 training roots (62,221 candidate/history examples); 24 calibration roots (22,908 examples); 24 IID evaluation roots (21,717 examples); 32 OOD evaluation roots (232,363 examples). Training corruption families adjacent and separated; OOD second-difference and two-child. All seeds and public development splits follow protocol. All arms use common necessary-condition masks at field elements 2 and 3 after edits preserving evaluations at 0 and 1, and exact product verifier on each attempted action. OOD candidate population 152,384, invalid 39,382, masked 18,177, surviving invalid 21,205; second family 8,303 surviving invalid, two-child 12,902. IID 18,352 candidates, 2,305 invalid, 1,445 masked, 860 surviving invalid. Some evaluations coincide in small fields; the exact verifier remains decisive. No candidate validity oracle used in routing decisions. Offline training and prediction examples include simulated prior rejected proposals. Their distribution need not match on-policy histories.

Three model seeds (19,41,67). Logistic residual 42 parameters, MLP 517, recurrent factor model 553. The latter uses twelve shared rank-one factors with up to two previous rejection features and current candidate. Logistic and MLP see mean history features and history count. Similar but unequal MLP/factor parameter counts. No adversarial BPD minimality training. Calibration selected components 10,7,5 respectively before evaluation. Eight random rank-one shams per checkpoint match the targeted parameter Frobenius norm, not activation-space perturbations. All interventions leave the exact verifier unchanged.

## Results

| Predictor | OOD root-average log loss | Brier | mean routing work | warmed mean route ms |
|---|---:|---:|---:|---:|
| bucket | .3968 | .1183 | 1017.91 | 1.157 |
| logistic | .3546 | .1172 | 988.47 | 2.472 |
| MLP | .3330 | .1123 | 997.26 | 2.787 |
| factor | .3256 | .1096 | 989.52 | 3.399 |

All 2,240 routing rollouts closed within work/call caps. Root means aggregate model seeds. Prediction losses weight roots equally and include simulated histories. Factor vs MLP OOD paired loss -0.0073364 (2.2% relative improvement), 14 wins/18 losses; descriptive root-bootstrap 95% [-0.0170617,+0.0015067]. Second-difference family bucket .233224, MLP .285321, factor .272088; two-child bucket .560407, MLP .380605, factor .379165. Prediction gate fails (5% and both-family requirement). Factor vs logistic routing +1.05208 work, 10/14/8 wins/ties/losses, interval [-7.15625,+10.3125]. Factor vs bucket -28.3854 work, 16/5/11, interval [-67.3125,+0.47917]. Routing gate fails. Warmed timings are single randomized-order executions per root/arm, exclude bank construction, and show no neural speed advantage.

## Parameter intervention

Targeted deletion vs intact factor: pooled log loss +0.00153943, descriptive 95% interval [-0.00184208,+0.00477595]. Second family +0.00409429, two-child family -0.00101543. Median matched-sham damage -0.00008534; targeted damage minus sham median about +0.001625, below 0.01 threshold. Enhancement pooled change +0.00051611. Both targeted edits exactly tie intact routing work in every OOD root/seed comparison. Causal gate fails.

Architectural limitation: W multiplies the initially zero hidden state. Neither targeted deletion nor enhancement can change a first proposal's score. Only 6 of 96 intact OOD factor rollouts encounter a rejection (8 total); IID factor rejects none. This creates low causal exposure. A second limitation is factor nonidentifiability: W=UV=(UA)(A^-1 V) for invertible A, so the end-to-end loss does not select a unique atom basis. Factor labels lack the behavioral-minimality objective used by native BPD.

## Independent audit and replay

Local audit.py reconstructs the train-only prior, regenerates two fixed evaluation roots per family (8 total), computes predictions from saved state dicts with an independent tensor formula, and checks eligible certificate labels using coefficient convolution plus power-sum evaluations at 0..3. 10,907 eligible identities; 640 metric fields reproduce within 5.96e-8; 32,721 zero-history ablation comparisons exactly equal; all 2,240 route receipts checked for closure and caps. Sham Frobenius norm error <=2.39e-7. This is a sampled checkpoint-metric and identity replay, not independent full routing replay. Replay package includes local audit script, saved checkpoints, receipts and v1-v4 dependencies; raw artifacts not committed here.

SHA256:
risk.py 348245d02e094de67d46763660bd6056bbf7002694806a25dd43f4b8109796c0
audit.py 3afd066017b8f81fd292b795ec02935bce2a062a81a1435c9389dfac6dd9bf12
receipt.json 088d455457dcfaba05d79e972fccb1c7338008f5bc6aa2b4e40818a0b9bc7b5d
models.pt 1f1d814ce7578ded115f709d4bb2849dd1e64a30f257d66478238368f9ed3e09
audit.json 79c63d187b45a962c84c3ab2c78f5e2918cdf4d828023106395f6f1845871496

Decision: fail all three registered gates. Next development run should preregister minimum causal exposure, introduce an encoder-initialized recurrent state before first candidate selection, and separate forced-history stress from natural routing. Use matched reset and sham activation perturbation, strong family-specific symbolic calibration, and a frozen-model BPD decomposition with behavioral minimality before attributing a causal mechanism to atoms. Preserve existing ARG-03-N seals.
