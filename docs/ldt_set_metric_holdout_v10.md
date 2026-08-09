# V10 set-metric held-out verification

## Result

The unchanged v10 exact status-vector and exact selected-set metrics passed one
environment-owned, sequestered robustness pack. Evals-reviewer observed and
accepted all 20 preregistered metric/probe rows across 10 independent probe
families. Each metric cleared two held-out families for invariance,
sensitivity, monotonicity, anti-gaming, and clean controls.

The audit reports `supported_on_declared_holdout`, claim status `supported`,
and operational action `accept`. This accepts the two metrics for use in a
separately frozen v10 development tournament. It is not a task-model result
and does not authorize a checkpoint, corpus, deployment, larger model, or
sealed model-confirmation reveal.

## Custody and independence

The metric source was frozen at commit `8e27385` before concrete held-out
assignments existed. The held-out contract was then staged before a fresh
256-bit seed generated the private pack. Neither the seed nor the concrete
candidate/status assignments entered the repository.

The steward used a separately coded reference implementation for candidate-ID
validation, exact status equality, and selected-set equality. Every observed
candidate-metric value had to match that reference before it could enter the
evals-reviewer audit. The candidate author received neither the private pack
nor its generation seed before the scorer was frozen.

This independence is classified as `environment_owned_artifact_bound`: code,
specification, seed commitment, private pack, observations, and audit are
content-hash bound. It is not third-party identity-authenticated. The local
environment-owned authority satisfies this experiment's verifier boundary,
but the result should not be described as an externally replicated finding.

## Frozen probe surface

| Probe type | Held-out families | Requirement | Result |
|---|---|---|---|
| invariance | cyclic order; bijective ID remap | exact stability | 2/2 |
| sensitivity | selected-to-unsafe; unresolved-to-selected | 1.0 decrease | 2/2 |
| monotonicity | mixed membership; expanded selection swap | no improvement | 2/2 |
| anti-gaming | high-cardinality decoy; inventory substitution | detect | 2/2 |
| clean control | exact shuffle; exact remap | no violation | 2/2 |

The concrete array sizes, identifiers, statuses, transformations, and orderings
remain in steward custody. Probe names and expected behaviors are public; raw
assignments are not.

## Evidence layers

| Layer | Status | Interpretation |
|---|---|---|
| metric robustness | supported on declared holdout | both metrics passed every frozen cell |
| task result | not run | no v10 adapter has been trained or scored |
| measurement reliability | complete | coverage 1.0; no parse, mismatch, or missing rows |
| claim support | supported locally | limited to the declared metrics and held-out probe families |
| operational decision | accept metric gate | freeze the v10 development tournament next |

No mechanistic hook was proposed. An SAE or VPD result could not replace these
behavioral probes or upgrade a future failed task result.

## Bound receipts

- held-out specification:
  `21aaec2e05ac352984de5f929d20c458f297e40a77dbfa3eae87c81e9326fb32`
- staged probe plan:
  `33e6c47342a69ee55a2e78aa04c24e85471a26c5f465335a64b14a402d543372`
- staging report:
  `e08d27aa91c7165f2b000d766f5e015b20030daea0d626ca26a31c3b82049dbe`
- frozen metric implementation:
  `ca326a9e3144abb970b38c0a8675fca82d6ca4b947719cc9101aafa827447712`
- private seed commitment:
  `567ec65527f1d809c155b4ec2c0e3e79f7143cc1937c3e9620e079aa902bd6ff`
- private probe-pack custody hash:
  `b51fe89f1df55470c3b5bd23bc25d20d39bcf431a84a81ca24a01a49b651beee`
- held-out observations:
  `cc9b0d6fdd79864ebc4fd41230d711efac8996456d666ffb8f7ed4b1a55e15a2`
- custody receipt:
  `b89d846a12db60f76dff6bd0753fc23b326c44c602e43ab2e4fa77df7c715483`
- robustness audit JSON:
  `2ff88579e010953378c160ec10c25e94f5093585e697a1fabb726fec72899087`
- robustness audit Markdown:
  `bdf4f44be9007a4e6e96ef45c9b832bbeb2e5d81e65b37597951df35ca2beb90`

All generated observations, private assignments, seeds, and audit run files
remain outside the repository.

## Next gate

Freeze a fresh-lineage v10 development tournament comparing the v9 pair
reducer and recurrent listwise controls with an edge-difference message head
and an otherwise matched head trained with a genuinely set-coupled selected-set
surrogate. Preserve exact antisymmetry, independent input primitives,
lineage-heldout selection, mean and worst-fold whole-array gates, hard resource
caps, and the existing sealed model-confirmation firewall.
