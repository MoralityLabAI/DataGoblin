# Checkpoint identities and gate receipts

The public release uses DataGoblin's Model Atlas discovery schema and Gym Metadata collator. `models.jsonl` contains a single discovery record for each distinct fitted checkpoint digest, with all scored roles retained. `fit_roles.jsonl` maps each study cell to that digest and distinguishes weight lineage from a sender used as an input to a fresh receiver. Shared ALEPH-01/02 weights appear under both questions. Some empirical conditional tables serialize identically for different frozen senders or controls. Treating those aliases as independent models would overstate the population.

`gym-sources.json` is the reviewed source list accepted by the existing gym collator. It pins protocol Git blobs and includes reused participants in the separate train-only population gym. The synthetic joint-query verifier has zero trained checkpoints. The discovery records carry upstream custody flags. Each new endpoint's bytes were read and SHA-256-verified locally before registration, recorded in `local_bytes_verified_sha256`; public remotes offer hash identities and code for reconstruction. `inventory.json` gives identity counts and canonical output digests. The local receipt bundle also includes the expanded gym and checkpoint indexes.

- [Gate outcomes](gates.json)
- [Receipt hashes](receipt-manifest.json)
- [Model records](models.jsonl)
- [Study and seed roles](fit_roles.jsonl)
- [Gym registration sources](gym-sources.json)
- [Reproduction and upstream inputs](../REPLAY.md)

To reproduce the native gym catalog, run `python -m datagoblin.gym_metadata --sources training/aleph_next/catalog/gym-sources.json --out ../aleph-gym-discovery` with DataGoblin installed. To regenerate the scientific join, provide the local retained fit directories to `training/aleph_next/atlas.py` and the source commit pinned in `inventory.json`. To generate usable supervised data, follow the separate [research corpus procedure](../CORPUS.md).
