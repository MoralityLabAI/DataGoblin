"""Run a named Aleph experiment on one allowed CPU under hard wall and memory caps."""

import argparse
import ctypes
import os
import runpy
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLACEMENT = HERE.parent / "aleph_rlt_placement"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wall", required=True, type=int)
    p.add_argument("--core", required=True, type=int)
    p.add_argument("--memory-mib", type=int, default=3072)
    p.add_argument(
        "worker",
        choices=(
            "architecture_lab.py",
            "arch_spectral.py",
            "arch_edit_depth.py",
            "arch_scoring.py",
            "decoder_lab.py",
            "decoder_pooled.py",
            "nonlinear_lab.py",
            "continual_lab.py",
            "scale_lab.py",
            "scale_shard.py",
            "study.py",
            "research_stream.py",
            "packed_probe.py",
            "anchor14_train.py",
            "confirm_eval.py",
            "confirm_speed.py",
        ),
    )
    args, rest = p.parse_known_args()
    if not 1 <= args.wall <= 21600 or not 256 <= args.memory_mib <= 4096 or args.core < 0:
        p.error("invalid resource limit")
    sys.path[:0] = [str(HERE), str(PLACEMENT)]
    if os.name == "nt":
        from ctypes import wintypes

        from cpu_worker import windows_limits

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        proc = wintypes.HANDLE(-1)
        allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(proc, ctypes.byref(allowed), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        cpus = [i for i in range(allowed.value.bit_length()) if allowed.value & (1 << i)]
        if args.core >= len(cpus):
            p.error("core outside allowed affinity set")
        if not kernel.SetProcessAffinityMask(proc, ctypes.c_size_t(1 << cpus[args.core])):
            raise ctypes.WinError(ctypes.get_last_error())
        windows_limits(args.memory_mib * 1024**2)
    else:
        import resource

        cpus = sorted(os.sched_getaffinity(0))
        if args.core >= len(cpus):
            p.error("core outside allowed affinity set")
        os.sched_setaffinity(0, {cpus[args.core]})
        os.nice(12)
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_mib * 1024**2,) * 2)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    script = (PLACEMENT if args.worker == "study.py" else HERE) / args.worker
    timer = threading.Timer(args.wall, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    sys.argv = [str(script), *rest]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
