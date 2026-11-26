# Public-policy audit and chat export

molt_episode_policy.py is a public-observation reference that never accesses the host's hidden
World fields. It uses a separate capacity dynamic program for array selection. The environment
retains the combinations oracle and independent validity check.

Compare selective refresh, always refresh, stale reuse and receipt forgetting. The deterministic
64-root test requires 128/128 optimal for selective and always refresh, with 64 versus 128 charged
verifier requests; stale reuse must solve only the 64 unrelated variants and forgetting must fail
all 128. These are exact expected outcomes on the generated TRAIN fixture; read CI for actual checks.
They demonstrate task discrimination and cost accounting, not model competence.

```sh
python -m datagoblin.training_gym.molt_episode_policy --seed 71 --roots 64
python -m datagoblin.training_gym.molt_episode_policy --seed 71 --roots 128 --train-out /tmp/molt-chat.jsonl
```

Chat export alternates observed events and exact-policy JSON actions. It is TRAIN-only with
root and task IDs. The full transcript is visible in this SFT format; a restricted-history
memory benchmark must use a different registered observation budget. The policy runner
already sends one observation at a time; real model adapters and Verifiers wiring are pending.
No learned model has been evaluated here.
