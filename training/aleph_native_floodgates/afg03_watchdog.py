"""External resource watchdog for the AFG-03 local GPU run (Windows, WDDM).

Launches the runner as a child process and samples, once per interval:
process-tree working set and private bytes (psutil), system available
memory, per-process dedicated and shared GPU memory from the Windows
"GPU Process Memory" performance counters, and GPU memory, temperature,
power, enforced power limit and clock events from NVML (nvidia-smi
when NVML is unavailable). Any cap breach kills the process tree and is
recorded as a violation. Before launch it records every process holding
GPU memory and refuses to start when free VRAM is below the floor. A
child exit code of 75 (the runner's thermal pause) is passed through.

    python afg03_watchdog.py --log-dir D:/run-logs --wall-seconds 1800 -- python afg03_checkpoint.py ...
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import afg03_nvml

GIB = 2**30
PAUSE_EXIT = 75
NVSMI = ("memory.used", "memory.free", "memory.total", "temperature.gpu", "power.draw",
         "utilization.gpu", "clocks_event_reasons.active")
PDH_FMT_LARGE = 0x00000400
PDH_MORE_DATA = 0x800007D2


class _Value(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]


class _Item(ctypes.Structure):
    _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _Value)]


def gpu_process_memory():
    """{pid: {"dedicated": bytes, "shared": bytes}} from the WDDM performance counters."""
    pdh = ctypes.WinDLL("pdh")
    for fn in ("PdhOpenQueryW", "PdhAddEnglishCounterW", "PdhCollectQueryData",
               "PdhGetFormattedCounterArrayW", "PdhCloseQuery"):
        getattr(pdh, fn).restype = wintypes.DWORD
    query = wintypes.HANDLE()
    if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)):
        raise OSError("PdhOpenQueryW")
    out = {}
    try:
        counters = {}
        for kind, name in (("dedicated", "Dedicated Usage"), ("shared", "Shared Usage")):
            h = wintypes.HANDLE()
            if pdh.PdhAddEnglishCounterW(query, f"\\GPU Process Memory(*)\\{name}", 0, ctypes.byref(h)):
                raise OSError("PdhAddEnglishCounterW")
            counters[kind] = h
        if pdh.PdhCollectQueryData(query):
            return out
        for kind, h in counters.items():
            size, count = wintypes.DWORD(0), wintypes.DWORD(0)
            status = pdh.PdhGetFormattedCounterArrayW(h, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None)
            if status != PDH_MORE_DATA:
                continue
            buf = (ctypes.c_byte*size.value)()
            if pdh.PdhGetFormattedCounterArrayW(h, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), buf):
                continue
            items = ctypes.cast(buf, ctypes.POINTER(_Item))
            for i in range(count.value):
                name = items[i].szName or ""
                if name.startswith("pid_"):
                    pid = int(name.split("_")[1])
                    slot = out.setdefault(pid, {"dedicated": 0, "shared": 0})
                    slot[kind] += int(items[i].FmtValue.largeValue)
    finally:
        pdh.PdhCloseQuery(query)
    return out


def nvidia_smi():
    raw = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(NVSMI)}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20, check=True).stdout.strip().splitlines()[0]
    values = [v.strip() for v in raw.split(",")]
    out = {}
    for key, v in zip(NVSMI, values, strict=True):
        try:
            out[key] = float(v) if key != "clocks_event_reasons.active" else v
        except ValueError:
            out[key] = None
    return out


def gpu_holders():
    rows = []
    for pid, m in sorted(gpu_process_memory().items()):
        if m["dedicated"] >= 64*2**20:
            try:
                name = psutil.Process(pid).name()
            except psutil.Error:
                name = None
            rows.append({"pid": pid, "name": name, **m})
    return rows


def tree(proc):
    try:
        return [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return []


def kill_tree(proc):
    for p in reversed(tree(proc)):
        with contextlib.suppress(psutil.Error):
            p.kill()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--wall-seconds", type=float, default=1800)
    p.add_argument("--ram-gib", type=float, default=12)
    p.add_argument("--min-available-gib", type=float, default=1.5)
    p.add_argument("--min-free-vram-gib", type=float, default=10.5)
    p.add_argument("--vram-gib", type=float, default=11.5, help="per-process dedicated GPU memory cap")
    p.add_argument("--shared-gib", type=float, default=1.0, help="per-process WDDM shared GPU memory cap")
    p.add_argument("--temp-c", type=float, default=95)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        p.error("missing child command")
    a.log_dir.mkdir(parents=True, exist_ok=False)
    log = (a.log_dir/"watchdog.jsonl").open("x", encoding="utf-8")

    def record(obj):
        log.write(json.dumps(obj, sort_keys=True)+"\n")
        log.flush()

    caps = {"wall_seconds": a.wall_seconds, "ram_tree_bytes": int(a.ram_gib*GIB),
            "min_available_bytes": int(a.min_available_gib*GIB), "min_free_vram_bytes": int(a.min_free_vram_gib*GIB),
            "vram_dedicated_bytes": int(a.vram_gib*GIB), "vram_shared_bytes": int(a.shared_gib*GIB),
            "temperature_c": a.temp_c}
    nvml = afg03_nvml.open_nvml()
    smi = nvidia_smi()
    pre = {"event": "preflight", "time": time.time(), "caps": caps, "gpu": smi, "gpu_holders": gpu_holders(),
           "nvml": nvml.read() if nvml else None,
           "available_bytes": psutil.virtual_memory().available, "cmd": cmd}
    record(pre)
    if smi["memory.free"] is None or smi["memory.free"]*2**20 < caps["min_free_vram_bytes"]:
        summary = {"event": "summary", "status": "not_launched_insufficient_vram", "violations": [],
                   "free_vram_bytes": None if smi["memory.free"] is None else int(smi["memory.free"]*2**20)}
        record(summary)
        (a.log_dir/"watchdog_summary.json").write_text(json.dumps({**summary, "preflight": pre}, indent=2,
                                                                  sort_keys=True)+"\n", encoding="utf-8")
        print(json.dumps(summary, sort_keys=True))
        return 3
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    stdout = (a.log_dir/"child_stdout.txt").open("xb")
    stderr = (a.log_dir/"child_stderr.txt").open("xb")
    start = time.monotonic()
    child = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
    proc = psutil.Process(child.pid)
    peaks = {k: 0 for k in ("tree_rss_bytes", "tree_private_bytes", "gpu_dedicated_bytes", "gpu_shared_bytes",
                            "gpu_memory_used_mib", "temperature_c", "power_w")}
    min_available = psutil.virtual_memory().available
    violations, samples, last_smi, capped_samples = [], 0, 0., 0
    while child.poll() is None:
        now = time.monotonic()
        procs = tree(proc)
        rss = private = 0
        for q in procs:
            try:
                m = q.memory_info()
                rss += m.rss
                private += getattr(m, "private", m.vms)
            except psutil.Error:
                pass
        pids = {q.pid for q in procs}
        gpu = gpu_process_memory()
        dedicated = sum(v["dedicated"] for k, v in gpu.items() if k in pids)
        shared = sum(v["shared"] for k, v in gpu.items() if k in pids)
        available = psutil.virtual_memory().available
        min_available = min(min_available, available)
        sample = {"event": "sample", "t": round(now-start, 3), "tree_rss_bytes": rss, "tree_private_bytes": private,
                  "gpu_dedicated_bytes": dedicated, "gpu_shared_bytes": shared, "available_bytes": available}
        if nvml is not None:
            g = nvml.read()
            if g["power_w"] is not None and g["power_w"] > 1000:
                # One AFG-03 sample read 138,567 W; drop readings no laptop GPU can draw.
                sample["gpu_power_glitch_w"] = g["power_w"]
                g["power_w"] = None
            smi = {"memory.used": (g["memory_used_bytes"] or 0)/2**20, "temperature.gpu": g["temperature_c"],
                   "power.draw": g["power_w"], "utilization.gpu": g["utilization_gpu"],
                   "enforced.power.limit": g["enforced_limit_w"], "clocks_event_reasons.active": g["clock_events"]}
            sample["gpu"] = smi
            capped_samples += bool(g["power_capped"])
        elif now-last_smi >= 2:
            try:
                smi = nvidia_smi()
                sample["gpu"] = smi
                last_smi = now
            except (subprocess.SubprocessError, OSError, IndexError) as error:
                sample["gpu_error"] = str(error)[:200]
        for key, value in (("tree_rss_bytes", rss), ("tree_private_bytes", private),
                           ("gpu_dedicated_bytes", dedicated), ("gpu_shared_bytes", shared),
                           ("gpu_memory_used_mib", smi.get("memory.used") or 0),
                           ("temperature_c", smi.get("temperature.gpu") or 0), ("power_w", smi.get("power.draw") or 0)):
            peaks[key] = max(peaks[key], value)
        breach = []
        if now-start > a.wall_seconds:
            breach.append(("wall_seconds", now-start))
        if rss > caps["ram_tree_bytes"]:
            breach.append(("ram_tree_bytes", rss))
        if available < caps["min_available_bytes"]:
            breach.append(("min_available_bytes", available))
        if dedicated > caps["vram_dedicated_bytes"]:
            breach.append(("vram_dedicated_bytes", dedicated))
        if shared > caps["vram_shared_bytes"]:
            breach.append(("vram_shared_bytes", shared))
        if (smi.get("temperature.gpu") or 0) >= a.temp_c:
            breach.append(("temperature_c", smi.get("temperature.gpu")))
        samples += 1
        if samples % 5 == 1 or breach:
            record(sample)
        if breach:
            violations.extend({"cap": k, "observed": v, "limit": caps[k], "t": round(now-start, 3)} for k, v in breach)
            record({"event": "kill", "violations": violations})
            kill_tree(proc)
            break
        time.sleep(max(0., a.interval-(time.monotonic()-now)))
    code = child.wait()
    elapsed = time.monotonic()-start
    stdout.close()
    stderr.close()
    summary = {"event": "summary", "status": "killed" if violations else "exited", "exit_code": code,
               "elapsed_seconds": elapsed, "violations": violations, "peaks": peaks,
               "min_available_bytes": min_available, "samples": samples, "power_capped_samples": capped_samples,
               "caps": caps}
    record(summary)
    log.close()
    (a.log_dir/"watchdog_summary.json").write_text(json.dumps({**summary, "preflight": pre}, indent=2,
                                                              sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status", "exit_code", "elapsed_seconds", "violations", "peaks")},
                     sort_keys=True))
    if violations:
        return 1
    return code if code in (0, PAUSE_EXIT) else 1


if __name__ == "__main__":
    sys.exit(main())
