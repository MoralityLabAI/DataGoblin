#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 RUN_ID PYTHON WORK_DIR IO_PATH OUT_DIR -- COMMAND [ARG ...]" >&2
  exit 64
}

[[ $# -ge 7 ]] || usage
[[ $EUID -eq 0 ]] || { echo "launch through WSL as root so systemd can create the capped unit" >&2; exit 77; }
run_id=$1
python_bin=$2
work_dir=$3
io_path=$4
out_dir=$5
shift 5
[[ ${1:-} == "--" ]] || usage
shift
[[ $# -gt 0 ]] || usage
command=("$@")
case ${DATAGOBLIN_RAM_WATCHDOG:-off} in
  off) ;;
  report|reap)
    watchdog="$(cd "$(dirname "$0")/../tools" && pwd)/gym_ram_watchdog.py"
    [[ -r "$watchdog" ]] || { echo "missing CPU watchdog: $watchdog" >&2; exit 78; }
    watch_flags=(--cpu-only)
    [[ $DATAGOBLIN_RAM_WATCHDOG != report ]] || watch_flags+=(--report-only)
    command=("$python_bin" "$watchdog" "${watch_flags[@]}" -- "${command[@]}")
    ;;
  *) echo "DATAGOBLIN_RAM_WATCHDOG must be off, report, or reap" >&2; exit 64 ;;
esac
[[ $run_id =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo "run ID contains unsupported characters" >&2; exit 64; }

ram_mb=2048
ram_bytes=$((ram_mb * 1024 * 1024))
cpu_percent=50
cpu_quota="50000 100000"
io_bytes_per_second=50000000
wall_clock_seconds=900
device_path=$(findmnt -T "$io_path" -no SOURCE | tr -d '[:space:]')
[[ -b $device_path ]] || { echo "IO path must be on a block-backed WSL filesystem" >&2; exit 79; }
device=$(lsblk -ndo MAJ:MIN "$device_path" | tr -d '[:space:]')
run_user=${DATAGOBLIN_RUN_USER:-$(stat -c %U "$work_dir")}
run_group=$(id -gn "$run_user")
install -d -o "$run_user" -g "$run_group" "$out_dir"
for path in "$work_dir" "$out_dir"; do
  path_device=$(findmnt -T "$path" -no MAJ:MIN | tr -d '[:space:]')
  [[ $path_device == "$device" ]] || { echo "$path is not on the capped IO device" >&2; exit 79; }
done

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

events="$out_dir/events.jsonl"
summary="$out_dir/summary.json"
properties="$out_dir/unit_properties.txt"
unit="${run_id}.service"
cgroup="/sys/fs/cgroup/system.slice/$unit"
guard="$(cd "$(dirname "$0")" && pwd)/training_gym_cgroup_guard.sh"
io_rule="$device rbps=$io_bytes_per_second wbps=$io_bytes_per_second"

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

touch "$events"
chown "$run_user:$run_group" "$events"
json_event start run_id="$run_id" ram_mb="$ram_mb" cpu_percent="$cpu_percent" \
  io_mb_per_second=50 wall_clock_seconds="$wall_clock_seconds"

set +e
systemd-run --unit="$run_id" --no-block \
  -p User="$run_user" \
  -p Group="$run_group" \
  -p WorkingDirectory="$work_dir" \
  -p MemoryMax="${ram_mb}M" \
  -p MemorySwapMax=0 \
  -p CPUQuota="${cpu_percent}%" \
  -p IOReadBandwidthMax="$device_path 50M" \
  -p IOWriteBandwidthMax="$device_path 50M" \
  -p RuntimeMaxSec="$wall_clock_seconds" \
  -p PrivateNetwork=yes \
  /usr/bin/bash "$guard" "$cgroup" "$events" "$python_bin" "$ram_bytes" "$cpu_quota" "$io_rule" -- "${command[@]}"
launch_status=$?
set -e

status=$launch_status
if [[ $launch_status -eq 0 ]]; then
  while true; do
    state=$(systemctl show "$unit" -p ActiveState --value)
    [[ $state == "active" || $state == "activating" || $state == "reloading" ]] || break
    if [[ -r "$cgroup/memory.current" ]]; then
      memory_current=$(cat "$cgroup/memory.current")
      io_line=$(grep "^$device " "$cgroup/io.stat" || true)
      io_read=$(sed -n 's/.*rbytes=\([0-9]*\).*/\1/p' <<<"$io_line")
      io_write=$(sed -n 's/.*wbytes=\([0-9]*\).*/\1/p' <<<"$io_line")
      cpu_usage=$(sed -n 's/^usage_usec \([0-9]*\)$/\1/p' "$cgroup/cpu.stat")
      json_event resource_sample memory_current_bytes="${memory_current:-0}" \
        io_read_bytes="${io_read:-0}" io_write_bytes="${io_write:-0}" cpu_usage_usec="${cpu_usage:-0}"
    fi
    sleep 2
  done
  systemctl show "$unit" -p Result -p ExecMainStatus -p MemoryPeak -p CPUUsageNSec \
    -p IOReadBytes -p IOWriteBytes >"$properties"
  service_result=$(systemctl show "$unit" -p Result --value)
  status=$(systemctl show "$unit" -p ExecMainStatus --value)
  [[ $service_result == "success" && $status -eq 0 ]] || status=1
fi

result=aborted
[[ $status -eq 0 ]] && result=completed
[[ $status -eq 0 ]] && json_event complete exit_code=0 || json_event abort reason=capped_process_exit exit_code="$status"
"$python_bin" - "$summary" "$events" "$run_id" "$result" "$status" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

summary_path, events_path, run_id, status, exit_code = sys.argv[1:]
events = [json.loads(line) for line in Path(events_path).read_text(encoding="utf-8").splitlines() if line]
samples = [event for event in events if event.get("event") == "resource_sample"]
memory = [int(event["memory_current_bytes"]) for event in samples]
io_rates = []
for previous, current in zip(samples, samples[1:]):
    elapsed = datetime.fromisoformat(current["ts"]).timestamp() - datetime.fromisoformat(previous["ts"]).timestamp()
    before = int(previous["io_read_bytes"]) + int(previous["io_write_bytes"])
    after = int(current["io_read_bytes"]) + int(current["io_write_bytes"])
    if elapsed > 0:
        io_rates.append(max(0, after - before) / elapsed / 1_000_000)
cpu_pct = 0.0
if len(samples) > 1:
    elapsed = datetime.fromisoformat(samples[-1]["ts"]).timestamp() - datetime.fromisoformat(samples[0]["ts"]).timestamp()
    cpu_delta = int(samples[-1]["cpu_usage_usec"]) - int(samples[0]["cpu_usage_usec"])
    if elapsed > 0:
        cpu_pct = max(0.0, cpu_delta / 1_000_000 / elapsed * 100)
payload = {
    "schema_version": "1.0",
    "run_id": run_id,
    "status": status,
    "exit_code": int(exit_code),
    "weights_loaded": "unknown_requires_candidate_cleanup_receipt",
    "steps_completed": 0,
    "peak_ram_mb": max(memory, default=0) / 1024 / 1024,
    "avg_ram_mb": sum(memory) / len(memory) / 1024 / 1024 if memory else 0.0,
    "peak_io_mb_s": max(io_rates, default=0.0),
    "cpu_pct": cpu_pct,
    "caps_enforced": any(event.get("event") == "caps_verified" for event in events),
    "cleanup_passed": False,
    "cleanup_status": "pending_windows_post_run_audit",
    "events_path": str(Path(events_path)),
    "claim_boundary": "Execution receipt only; candidate quality requires separate gym evaluation.",
}
Path(summary_path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
systemctl reset-failed "$unit" >/dev/null 2>&1 || true
exit "$status"
