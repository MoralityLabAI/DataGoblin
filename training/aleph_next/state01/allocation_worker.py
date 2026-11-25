"""Bound each external allocation worker and its subprocesses to local CPU resources."""

import argparse
import json
import os
import runpy
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--core", type=int, default=6)
    p.add_argument("--wall", type=int, default=3600)
    p.add_argument("--memory-log", type=Path)
    p.add_argument("worker", choices=("allocation_replay.py", "allocation_routes.py"))
    args, argv = p.parse_known_args()
    if not 1 <= args.wall <= 7200 or args.core < 0:
        p.error("resource limit outside allowed range")
    import psutil

    process = psutil.Process()
    cpus = sorted(process.cpu_affinity())
    if args.core >= len(cpus):
        p.error("core outside allowed affinity")
    process.cpu_affinity([cpus[args.core]])
    cap = 3072 * 1024**2
    if os.name == "nt":
        sys.path.insert(0, str(HERE.parents[1] / "aleph_rlt_placement"))
        from cpu_worker import windows_limits

        windows_limits(cap)
    else:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        os.nice(10)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.path.insert(0, str(HERE.parents[2] / "src"))
    timer = threading.Timer(args.wall, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    sys.argv = [str(HERE / args.worker), *argv]
    stop = threading.Event()
    usage = {
        "sample_seconds": 1,
        "peak_worker_rss": 0,
        "peak_lean_working_set": 0,
        "peak_process_tree_rss": 0,
        "minimum_system_available": None,
        "samples": 0,
    }

    def sample():
        while not stop.is_set():
            main_rss = process.memory_info().rss
            total = main_rss
            for child in process.children(recursive=True):
                try:
                    mem = child.memory_info()
                    total += mem.rss
                    if child.name().lower() in ("lean", "lean.exe"):
                        usage["peak_lean_working_set"] = max(
                            usage["peak_lean_working_set"], getattr(mem, "peak_wset", mem.rss)
                        )
                except psutil.Error:
                    continue
            avail = psutil.virtual_memory().available
            usage["minimum_system_available"] = min(usage["minimum_system_available"] or avail, avail)
            usage["peak_worker_rss"] = max(usage["peak_worker_rss"], main_rss)
            usage["peak_process_tree_rss"] = max(usage["peak_process_tree_rss"], total)
            usage["samples"] += 1
            stop.wait(1)

    thread = None
    if args.memory_log:
        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        stop.set()
        if thread:
            thread.join(timeout=2)
            args.memory_log.write_text(json.dumps(usage, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
