#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 6 ]] || { echo "cgroup guard received incomplete arguments" >&2; exit 64; }
cgroup=$1
events=$2
python_bin=$3
ram_bytes=$4
cpu_quota=$5
io_rule=$6
shift 6
[[ ${1:-} == "--" ]] || { echo "cgroup guard requires -- before command" >&2; exit 64; }
shift
[[ $# -gt 0 ]] || { echo "cgroup guard requires a command" >&2; exit 64; }

json_event() {
  "$python_bin" - "$events" "$@" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, event, *fields = sys.argv[1:]
record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event}
for field in fields:
    key, value = field.split("=", 1)
    record[key] = value
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
PY
}

[[ "$(cat "$cgroup/memory.max")" == "$ram_bytes" ]] || {
  json_event abort reason=memory_cap_mismatch weights_loaded=false
  exit 70
}
[[ "$(cat "$cgroup/memory.swap.max")" == "0" ]] || {
  json_event abort reason=swap_cap_mismatch weights_loaded=false
  exit 70
}
[[ "$(cat "$cgroup/cpu.max")" == "$cpu_quota" ]] || {
  json_event abort reason=cpu_cap_mismatch weights_loaded=false
  exit 70
}
[[ "$(cat "$cgroup/io.max")" == *"$io_rule"* ]] || {
  json_event abort reason=io_cap_mismatch weights_loaded=false
  exit 70
}

json_event caps_verified ram_bytes="$ram_bytes" cpu_quota="$cpu_quota" io_rule="$io_rule" swap_bytes=0
json_event owned_process pid="$$"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_MODULE_LOADING=LAZY
export DATAGOBLIN_GYM_EVENTS="$events"
export DATAGOBLIN_HARD_CAPS_VERIFIED=1
json_event candidate_process_start weights_loaded=unknown
exec "$@"
