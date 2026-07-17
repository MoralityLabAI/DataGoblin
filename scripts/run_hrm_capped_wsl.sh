#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 RUN_ID PYTHON REPO_ROOT MODEL_ROOT DATASET_MANIFEST OUT_DIR CONFIG" >&2
  exit 64
}

[[ $# -eq 7 ]] || usage
[[ $EUID -eq 0 ]] || { echo "run this launcher through: wsl -d Ubuntu -u root --exec ..." >&2; exit 77; }
run_id=$1
python_bin=$2
repo_root=$3
model_root=$4
dataset_manifest=$5
out_dir=$6
config=$7

ram_bytes=$((4096 * 1024 * 1024))
cpu_quota="50000 100000"
io_limit=50000000
model_device_path=$(findmnt -T "$model_root/model.safetensors" -no SOURCE | tr -d '[:space:]')
[[ -b "$model_device_path" ]] || { echo "model must reside on a block-backed WSL filesystem" >&2; exit 79; }
device=$(lsblk -ndo MAJ:MIN "$model_device_path" | tr -d '[:space:]')
run_user=${DATAGOBLIN_RUN_USER:-$(stat -c %U "$model_root")}
run_group=$(id -gn "$run_user")
cgroup="/sys/fs/cgroup/system.slice/${run_id}.service"

for controller in cpu io; do
  if [[ " $(cat /sys/fs/cgroup/cgroup.subtree_control) " != *" $controller "* ]]; then
    printf "+%s\n" "$controller" > /sys/fs/cgroup/cgroup.subtree_control
  fi
done
for controller in cpu io memory; do
  [[ " $(cat /sys/fs/cgroup/system.slice/cgroup.controllers) " == *" $controller "* ]] || {
    echo "system.slice lacks the $controller controller" >&2
    exit 78
  }
done

install -d -o "$run_user" -g "$run_group" "$out_dir"
events="$out_dir/wrapper_events.jsonl"
summary="$out_dir/wrapper_summary.json"

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

guard="$out_dir/capped_child.sh"
cat >"$guard" <<EOF
#!/usr/bin/env bash
set -euo pipefail
events='$events'
summary='$summary'
python_bin='$python_bin'
repo_root='$repo_root'
model_root='$model_root'
dataset_manifest='$dataset_manifest'
out_dir='$out_dir'
config='$config'
cgroup='$cgroup'
ram_bytes='$ram_bytes'
cpu_quota='$cpu_quota'
io_limit='$io_limit'
device='$device'

json_event() {
  "\$python_bin" - "\$events" "\$@" <<'PY'
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

[[ "\$(cat "\$cgroup/memory.max")" == "\$ram_bytes" ]] || { json_event abort reason=memory_cap_mismatch weights_loaded=false; exit 70; }
[[ "\$(cat "\$cgroup/memory.swap.max")" == "0" ]] || { json_event abort reason=swap_cap_mismatch weights_loaded=false; exit 70; }
[[ "\$(cat "\$cgroup/cpu.max")" == "\$cpu_quota" ]] || { json_event abort reason=cpu_cap_mismatch weights_loaded=false; exit 70; }
io_max="\$(cat "\$cgroup/io.max")"
[[ "\$io_max" == *"\$device rbps=\$io_limit"* && "\$io_max" == *"wbps=\$io_limit"* ]] || {
  json_event abort reason=io_cap_mismatch weights_loaded=false
  exit 70
}
model_device="\$(findmnt -T "\$model_root/model.safetensors" -no MAJ:MIN | tr -d '[:space:]')"
[[ "\$model_device" == "\$device" ]] || {
  json_event abort reason=model_not_on_capped_device weights_loaded=false
  exit 70
}
json_event caps_verified ram_bytes=\$ram_bytes cpu_quota="\$cpu_quota" io_bytes_per_second=\$io_limit swap_bytes=0
json_event owned_process pid=\$\$

export PYTHONPATH="\$repo_root/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_MODULE_LOADING=LAZY
"\$python_bin" - "\$out_dir/runtime_environment.json" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

packages = {}
for name in ("torch", "transformers", "safetensors", "numpy"):
    distribution = importlib.metadata.distribution(name)
    direct_url = None
    for item in distribution.files or ():
        if item.name == "direct_url.json" and item.parent.name.endswith(".dist-info"):
            direct_url = json.loads(distribution.locate_file(item).read_text(encoding="utf-8"))
            break
    packages[name] = {"version": distribution.version, "direct_url": direct_url}
payload = {
    "schema_version": "1.0",
    "python": sys.version,
    "platform": platform.platform(),
    "executable": sys.executable,
    "packages": packages,
}
Path(sys.argv[1]).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
json_event model_process_start weights_loaded=unknown
set +e
"\$python_bin" -m datagoblin trace-hrm \
  --dataset-manifest "\$dataset_manifest" \
  --out "\$out_dir/trace" \
  --config "\$config" \
  --local-model-path "\$model_root" \
  --device cuda \
  --confirm-hard-cap-wrapper \
  --maximum-examples 1
status=\$?
set -e
if [[ \$status -eq 0 ]]; then
  json_event complete exit_code=0 steps_completed=1
else
  json_event abort reason=model_process_exit exit_code=\$status
fi
exit \$status
EOF
chmod 700 "$guard"
chown "$run_user:$run_group" "$guard"

unit_properties="$out_dir/unit_properties.txt"
unit_stdout="$out_dir/unit.stdout.log"
unit_stderr="$out_dir/unit.stderr.log"
json_event wrapper_start run_id="$run_id" ram_mb=4096 cpu_percent=50 io_mb_per_second=50 wall_clock_seconds=1800
chown "$run_user:$run_group" "$events"
set +e
systemd-run --unit="$run_id" --no-block \
  -p User="$run_user" \
  -p Group="$run_group" \
  -p MemoryMax=4096M \
  -p MemorySwapMax=0 \
  -p CPUQuota=50% \
  -p IOReadBandwidthMax="$model_device_path 50M" \
  -p IOWriteBandwidthMax="$model_device_path 50M" \
  -p RuntimeMaxSec=1800 \
  -p StandardOutput="append:$unit_stdout" \
  -p StandardError="append:$unit_stderr" \
  "$guard"
launch_status=$?
set -e

status=$launch_status
if [[ $launch_status -eq 0 ]]; then
  while true; do
    state=$(systemctl show "${run_id}.service" -p ActiveState --value)
    [[ "$state" == "active" || "$state" == "activating" || "$state" == "reloading" ]] || break
    if [[ -r "$cgroup/memory.current" ]]; then
      memory_current=$(cat "$cgroup/memory.current")
      io_line=$(grep "^$device " "$cgroup/io.stat" || true)
      io_read=$(sed -n 's/.*rbytes=\([0-9]*\).*/\1/p' <<<"$io_line")
      io_write=$(sed -n 's/.*wbytes=\([0-9]*\).*/\1/p' <<<"$io_line")
      cpu_usage=$(sed -n 's/^usage_usec \([0-9]*\)$/\1/p' "$cgroup/cpu.stat")
      json_event resource_sample \
        memory_current_bytes="${memory_current:-0}" \
        io_read_bytes="${io_read:-0}" \
        io_write_bytes="${io_write:-0}" \
        cpu_usage_usec="${cpu_usage:-0}"
    fi
    sleep 2
  done
  systemctl show "${run_id}.service" \
    -p ActiveState -p Result -p ExecMainStatus -p MemoryPeak -p CPUUsageNSec \
    -p IOReadBytes -p IOWriteBytes >"$unit_properties"
  service_result=$(systemctl show "${run_id}.service" -p Result --value)
  status=$(systemctl show "${run_id}.service" -p ExecMainStatus --value)
  [[ "$service_result" == "success" && "$status" -eq 0 ]] || status=1
  systemctl reset-failed "${run_id}.service" >/dev/null 2>&1 || true
fi

result=aborted
[[ $status -eq 0 ]] && result=completed
"$python_bin" - "$summary" "$events" "$unit_properties" "$run_id" "$result" "$status" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path, events_path, properties_path, run_id, status, exit_code = sys.argv[1:]
events = []
path = Path(events_path)
if path.exists():
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
samples = [item for item in events if item.get("event") == "resource_sample"]
memory = [int(item["memory_current_bytes"]) for item in samples]
rates = []
for previous, current in zip(samples, samples[1:]):
    elapsed = datetime.fromisoformat(current["ts"]).timestamp() - datetime.fromisoformat(previous["ts"]).timestamp()
    previous_bytes = int(previous["io_read_bytes"]) + int(previous["io_write_bytes"])
    current_bytes = int(current["io_read_bytes"]) + int(current["io_write_bytes"])
    if elapsed > 0:
        rates.append(max(0, current_bytes - previous_bytes) / elapsed / 1_000_000)
properties = {}
properties_file = Path(properties_path)
if properties_file.exists():
    for line in properties_file.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        properties[key] = value

def optional_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

cpu_samples = [int(item["cpu_usage_usec"]) for item in samples if "cpu_usage_usec" in item]
cpu_pct = None
if len(cpu_samples) > 1:
    elapsed = datetime.fromisoformat(samples[-1]["ts"]).timestamp() - datetime.fromisoformat(samples[0]["ts"]).timestamp()
    if elapsed > 0:
        cpu_pct = max(0.0, cpu_samples[-1] - cpu_samples[0]) / 1_000_000 / elapsed * 100
payload = {
    "schema_version": "1.0",
    "run_id": run_id,
    "ts_utc": datetime.now(timezone.utc).isoformat(),
    "status": status,
    "exit_code": int(exit_code),
    "caps": {"ram_mb": 4096, "cpu_percent": 50, "io_mb_per_second": 50, "wall_clock_seconds": 1800, "swap_bytes": 0},
    "caps_verified": any(item.get("event") == "caps_verified" for item in events),
    "steps_completed": 1 if any(item.get("event") == "complete" for item in events) else 0,
    "weights_loaded": True if any(item.get("event") == "complete" for item in events) else (
        "possible" if any(item.get("event") == "model_process_start" for item in events) else False
    ),
    "abort_reason": next(
        (item.get("reason") for item in reversed(events) if item.get("event") == "abort"),
        None if status == "completed" else "wrapper_or_unit_failure",
    ),
    "owned_linux_pids": [int(item["pid"]) for item in events if item.get("event") == "owned_process"],
    "peak_ram_mb": max(memory, default=0) / 1024 / 1024,
    "avg_ram_mb": (sum(memory) / len(memory) / 1024 / 1024) if memory else 0.0,
    "peak_io_mb_s": max(rates, default=0.0),
    "cpu_usage_nsec": optional_int(properties.get("CPUUsageNSec")),
    "cpu_pct": cpu_pct,
    "systemd_result": properties.get("Result"),
    "unit_properties_path": str(properties_file),
    "events_path": str(path),
    "claim_boundary": "Execution receipt only; this run does not establish correctness, calibration, or alignment.",
}
Path(summary_path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
exit "$status"
