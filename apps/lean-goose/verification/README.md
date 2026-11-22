# OxiLean export compatibility pilot

This experiment accompanies [DataGoblin issue #42](https://github.com/MoralityLabAI/DataGoblin/issues/42). It uses eight independent declarations written for Lean 4.32.0-rc1 and NDJSON 3.1.0. The Lean 4.12.0 rollout certificates remain the primary evidence for the Goose experiment.

On Windows, native Lean compiled all eight declarations and reported no axiom dependencies. OxiLean verified all eight declarations and ten reachable dependencies, with zero unsupported and zero rejected. Substituting a proposition for the proof term in `dgOxiUnit` and `dgOxiCompose` produced exactly two rejected declarations. Native Lean also refused both corresponding source controls. A secondary audit of the rollout certificates requires proving their statements under the version-matched toolchain.

Install Lean 4.32.0-rc1. Build [lean4export at `3de59f1`](https://github.com/leanprover/lean4export/tree/3de59f10bc4b4a0f2de698597aeb1246caa0df0a) and [OxiLean at `3adb983`](https://github.com/cool-japan/oxilean/tree/3adb983d2aefc6a596bdc8b63438f54feecd1f55) in separate checkouts. Build the exporter with `lake build lean4export`. Generate a Cargo lockfile in the temporary OxiLean checkout and run `cargo build --release --locked -p oxilean-verify`. The measured lockfile SHA-256 was `449ee0f31e91ad1e629f2b52e6655fe001a3a4718fa80e59aa4e9cb041f8477f`.

```bash
python scripts/check_oxilean_exports.py \
  --lean /path/to/lean-4.32.0-rc1/bin/lean \
  --exporter /path/to/lean4export/.lake/build/bin/lean4export \
  --checker /path/to/oxilean/target/release/oxilean-verify \
  --exporter-repo /path/to/lean4export --checker-repo /path/to/oxilean \
  --out runs/oxilean-version-pilot
```

The runner retains native compiler output, both NDJSON inputs, per-declaration secondary verdicts and artifact checksums. The substituted export uses ordinary in-range expression indices so that a rejection measures type checking. Export parser failures use a different exit code and cannot pass this check.
