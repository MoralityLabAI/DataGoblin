# MOLT-03: public memory evaluation panels

This additive development harness reuses MOLT-GOBLIN-02's exact episode host.
Every model is given the same immutable public candidate list, objective, capacity and
source identities. Initial receipt contents are excluded from the static task packet.
A deterministic view adapter decides what past observations are delivered at each turn.

| View | What the model sees |
| --- | --- |
| full | All observed events |
| recent2 | Two most recent observed events |
| current | Current observed event |
| bank1 | Current event plus one valid public receipt |
| bank2 | Current event plus two valid public receipts |

Bank writes come only from public observations. A source-change event invalidates a stored
receipt for that source; the replacement cannot enter the bank until the host returns it.
Eviction is deterministic oldest-write-first. The bank is an external symbolic retrieval
policy, not neural latent memory. It contributes a distinct scaffold arm and receives its
own byte/access accounting. The exact public control is stateless between calls and
reconstructs only visible receipts. A missing receipt causes abstention.

Use the same root seed, count, mode, wall and action budgets for paired checkpoint panels.
Cross-mode experiments deliberately change information access; label them scaffold
comparisons. Full-history byte counts differ from bank counts; report context failures as
failures. Raw provider token usage is only available through a compatible model transport.
Native model passes and server-side resource caps are outside this offline meter.

```sh
python -m pytest -q tests/test_molt_memory_panel.py
python -m datagoblin.training_gym.molt_memory_panel --mode full --roots 4 --out /tmp/molt-full.json
python -m datagoblin.training_gym.molt_memory_panel --mode bank2 --roots 4 --out /tmp/molt-bank2.json
```

The optional model route reuses the pinned Research Verifiers v2 transport from PR #81.
It requires that module to be present, a registered immutable model identity and an
operator-provisioned binding. It does not provision or download a checkpoint.
No fallback HTTP implementation exists. The transport performs tokenizer preflight
with a 2,048-token context, 512 generation tokens, bounded time and provider usage receipts.
The driver reserves two HTTP requests per action and caps against the binding.
The external runtime and hard resource caps remain operator-attested.

```sh
python -m datagoblin.training_gym.molt_memory_panel --mode bank2 --roots 4 \
  --model-id baby-goblin-10m-v2-native --registry configs/research_v2/models.json \
  --binding /private/operator-binding.json --out /private/molt-baby-bank2.json
```

The developer control panel accepts train/calibration roots only. All planned episodes stay
in the denominator, including exhausted-budget, malformed and transport-failed episodes.
Each report preserves per-root outcomes, observation receipts, prompt digests, action attempts,
token usage when known, memory bytes and wall time. Keep these files outside git.
No held evaluation is opened, no model is promoted, and no endpoint credentials or bindings
are exported. Confirmation needs a separately approved, immutable steward release and
EvalMath preflight. The current runner is not itself a native Verifiers environment.
