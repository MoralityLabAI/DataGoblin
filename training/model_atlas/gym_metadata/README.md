# Recent gym metadata

This reviewed discovery catalog supplements the LBF model-hash catalog in PR #60.
It does not modify its source roster, assume that task counts are independent
roots, or promote any example to Aleph training. The files identify eight
gym/module studies and 16 previously reported checkpoint digests. Only three
are terminal fits in the new catalog's trained-gym lane; ROC-02B and NOR-03
point to existing models. A recipe, fixture, or aggregate report alone is not
a checkpoint receipt.

From the repository root, produce a separate dataset staging bundle:

```sh
python -m datagoblin.gym_metadata \
  --sources training/model_atlas/gym_metadata/sources.json \
  --out runs/model-atlas/recent-gym-metadata
```

This uses the existing Model Atlas deterministic writer. It reads the reviewed
catalog only, emits module rows, a digest-keyed checkpoint index, and an
inventory, and makes no network requests. Sources are pinned as provenance, not
inferred to be externally verified by the offline command. Checkpoint bytes,
raw examples, weights, hidden seeds, evaluation artifacts and full activations
are out of scope. All emitted records have `training_ready=false` and retain
their family grouping. Missing fields, such as ROC parameter count and
checkpoint digests for the Aleph RLT report, stay unknown.
