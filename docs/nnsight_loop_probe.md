# NNsight pass-aware probes for Inkling

This is an instrumentation lane, separate from the existing Inkling depth-sweep
completion API. It targets repeated calls to one shared decoder layer during one
prefill forward pass, not successive generated tokens or student RLT visits.

NNsight's tracer.iter[:N] indexes repeated occurrences of the chosen module.
The layer-17 decoder output is a tensor with shape [batch, sequence, hidden].
The probe requests the last prompt position and saves only a norm, feature values,
width and next-token readout per architectural pass. It rejects incomplete
observations, records NNsight/Torch versions, the declared revision, local model
config pass count (if exposed), requested pass count, and SHA-256 of the prompt.
The observed count establishes at least N passes; an exact pass-count claim
requires the locally exposed loop_num_passes config to match. A remote runtime
may override the config and must be checked separately through server receipts.

## Setup

~~~shell
python -m pip install -e ".[dev,interpretability]"
# Set NDIF_API_KEY privately, outside shell history
python scripts/inkling_ndif_smoke.py --model thinkingmachines/TML-d24-varloop-research --expected-passes 3
~~~

NNsight is pinned to 0.8.3 as an optional dependency. Remote model availability
varies; inspect nnsight.status() when a deployment is unavailable. The smoke
script refuses a remote call without NDIF_API_KEY, avoids printing the key,
and has no model-weights download. The GPT-2 smoke path remains available.

## Causal interventions

The controlled suite runs a clean observation, same-site sham, a targeted lesion,
and a disjoint random-coordinate lesion with the same number of coordinates.
If a donor is specified, the suite reads donor coordinates at the same
architectural pass and performs a same-feature donor patch. Dose 1 entirely
replaces the chosen donor coordinates or zeros them for a lesion. Dose 0 is a
sham. All target traces use the same model, token position, pass count and prompt;
the donor observation uses a separate prompt with its own digest.

~~~shell
python scripts/inkling_ndif_interventions.py \
  --revision main --prompt 'The Eiffel Tower is in the city of' \
  --passes 3 --pass-index 1 --site model.language_model.layers.17 \
  --features 2027 --dose 0.5 --seed 7 \
  --out runs/inkling/ndif/layer17-coordinate2027.json
~~~

The optional --target-token-id adds a next-token log-probability contrast
against the sham. For a donor patch add --donor-prompt and the corresponding
--donor-task-digest. Use an immutable revision when the NDIF deployment serves
one; main is mutable. The output includes pass measurements, arm-to-sham
contrasts, feature selections, configuration checks and a SHA-256 receipt.
Output paths are create-only. The JSON contains no API key or full activation
vectors. The suite makes four remote traces, or six with a donor, each
explicitly requested by running the script.

## Interpretation and next experiments

The default site reads the shared decoder layer output. It does not directly
expose Inkling's block-sum messages or their learned mixing gates. To study
those quantities, identify and validate the real .source operation or
submodule in the installed Inkling implementation, then supply its path
through --site. The selected site must return a [B, S, H] tensor, and its
fire count must represent architectural passes.

Top-token flips and log-probability differences are mechanistic readouts.
Every receipt says verifier_status=unverified and student_training_eligible=false.
To make training examples, join the frozen task digest to the existing RLT
teacher importer and let the environment verifier check completed answers.
The probe exports no synthetic SFT targets.

Candidate features should be selected on train/calibration tasks; reserve
ordinary and OOD tests for confirmation. Include donor identity, sham/random
controls, prompt hashes and a depth-matched baseline in every causal claim.

The tiny PyTorch recurrent fixture exercises three calls to a shared module
inside one model.trace, plus same-prompt sham and ablation runs. Its pure
contract tests have no NDIF key, teacher weights or GPU requirement. The
NNsight integration test is enabled by the interpretability CI job.

References: [NNsight documentation](https://github.com/ndif-team/nnsight/tree/main/docs),
[Inkling research release](https://thinkingmachines.ai/news/inkling-2-mini/),
[NDIF Looped Lens Challenge](https://ndif.us/looped-lens-challenge/).
