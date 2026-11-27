# DataGoblin MCP server

`python -m datagoblin.mcp` serves DataGoblin's research orchestrator over MCP (stdio, newline-delimited
JSON-RPC 2.0, protocol `2025-06-18`). It uses only the standard library plus DataGoblin's own dependencies and
calls the Python functions behind the CLI subcommands directly. Tools that need weights, a GPU, HRM checkpoints
or the network are not exposed.

```bash
PYTHONPATH=src python -m datagoblin.mcp [--root DIR]
```

`--root` defaults to this checkout. Stdout carries protocol frames only.

## Path rules

- Inputs resolve under `examples/`, `runs/` or `configs/` of the root.
- Outputs resolve under `runs/` of the root; a path that resolves elsewhere (including via `..`) fails with
  `path_not_allowed` and nothing is written.
- Ledger queries open `ledger.sqlite` read-only.

## Tools

`readOnly` tools never write. The others write only under `runs/`.

| Tool | CLI equivalent | readOnly | Purpose |
| --- | --- | --- | --- |
| `validate_recipe` | `validate` | yes | Constitutional decision and triggered rule ids for a DataRecipe |
| `propose_plan` | `propose` | no | Hypotheses, recipe, RAT critique, repaired recipe, tickets from a learner snapshot; optional `out` |
| `emit_tickets` | `emit-tickets` | no | Write builder tickets for a plan into `out_dir` |
| `run_demo` | `demo` | no | Offline toy research cycle into `out` (default `runs/demo`) |
| `synth_episodes` | `synth-episodes` | no | Synthetic planning episodes as JSONL |
| `export_episodes` | `export-sft`, `export-preferences`, `export-verifier`, `export-outcomes` | no | Training rows from a research episode file; `kind` is `sft`, `preferences`, `verifier` or `outcomes` |
| `report` | `report` | no | Markdown report for an experiment in a run's ledger; optional `out` |
| `ledger_summary` | | yes | Row counts per ledger table and experiment ids |
| `ledger_list` | | yes | Record ids, hashes and timestamps of one table |
| `ledger_get` | | yes | One ledger record payload |
| `run_files` | | yes | Files and sizes under a run directory |
| `run_read_file` | | yes | One text or JSON file up to 1 MB |
| `upstream_status` | `upstream-status` | yes | Which upstream repositories exist locally (git state, no network) |
| `etiquette_classify` | | yes | Collection-etiquette disposition and violations for `DataCollectionRecord` objects |
| `etiquette_demo` | `etiquette-demo` | no | Offline etiquette sorter demo |
| `etiquette_evaluate` | `etiquette-evaluate` | no | Score sorter predictions against sealed labels |
| `gym_demo` | `gym-demo` | no | Offline training-gym demo with mock candidates |
| `gym_evaluate` | `gym-evaluate` | no | Evaluate one candidate under a frozen contract |
| `gym_compare` | `gym-compare` | no | Compare candidate reports against a baseline |

## Errors

A failed call returns `isError: true` with `{"error": <code>, "detail": <text>}`. Codes: `unknown_tool`,
`invalid_arguments`, `invalid_input`, `path_not_allowed`, `not_found`, `too_large`, `run_failed`, `io_error`.

## Client configuration

goose (`~/.config/goose/config.yaml`):

```yaml
extensions:
  datagoblin:
    type: stdio
    name: datagoblin
    cmd: python
    args: [-m, datagoblin.mcp, --root, /path/to/DataGoblin]
    envs: { PYTHONPATH: /path/to/DataGoblin/src }
    timeout: 300
```

Claude (`.mcp.json`) and other clients that read the `mcpServers` form:

```json
{
  "mcpServers": {
    "datagoblin": {
      "command": "python",
      "args": ["-m", "datagoblin.mcp", "--root", "/path/to/DataGoblin"],
      "env": { "PYTHONPATH": "/path/to/DataGoblin/src" }
    }
  }
}
```

Typical flow: `validate_recipe`, `propose_plan`, `emit_tickets`, `run_demo`, then `ledger_summary`,
`ledger_get` and `report` to read the run, and `export_episodes` for SFT or preference rows.
