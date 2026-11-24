"""Bound a single offline gym worker before importing Torch. Pass a local gym script and its args."""

import argparse
import ctypes
import os
import runpy
import sys
import threading
from pathlib import Path


def windows_limits(cap):
    from ctypes import wintypes as w

    class Basic(ctypes.Structure):
        _fields_ = (  # ctypes field declaration
            ("process_cpu", ctypes.c_longlong),
            ("job_cpu", ctypes.c_longlong),
            ("flags", w.DWORD),
            ("min_ws", ctypes.c_size_t),
            ("max_ws", ctypes.c_size_t),
            ("processes", w.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority", w.DWORD),
            ("scheduling", w.DWORD),
        )

    class Extended(ctypes.Structure):
        _fields_ = (
            ("basic", Basic),
            ("io", ctypes.c_ulonglong * 6),
            ("process_mem", ctypes.c_size_t),
            ("job_mem", ctypes.c_size_t),
            ("peak_process", ctypes.c_size_t),
            ("peak_job", ctypes.c_size_t),
        )

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateJobObjectW.restype = w.HANDLE
    k.GetCurrentProcess.restype = w.HANDLE
    job = k.CreateJobObjectW(None, None)
    proc = k.GetCurrentProcess()
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    job, proc = w.HANDLE(job), w.HANDLE(proc)
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not k.GetProcessAffinityMask(proc, ctypes.byref(allowed), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    limit = Extended()
    limit.basic.flags = 0x200 | 0x10  # JOB_MEMORY and AFFINITY
    limit.basic.affinity = 1 << (allowed.value.bit_length() - 1)
    limit.job_mem = cap
    if not k.SetInformationJobObject(job, 9, ctypes.byref(limit), ctypes.sizeof(limit)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not k.AssignProcessToJobObject(job, proc) or not k.SetPriorityClass(proc, 0x4000):
        raise ctypes.WinError(ctypes.get_last_error())
    return job  # intentionally retain the handle for the entire process lifetime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wall", type=int, default=3600)
    p.add_argument("--memory-mib", type=int, default=3072)
    p.add_argument(
        "worker",
        choices=(
            "gym.py",
            "audit.py",
            "study.py",
            "pass_diagnostic.py",
            "review.py",
            "fullscale.py",
            "repair_generation.py",
            "receipt_gym.py",
        ),
    )
    args, rest = p.parse_known_args()
    if not (1 <= args.wall <= 21600 and 256 <= args.memory_mib <= 4096):
        p.error("invalid bounds")
    cap = args.memory_mib * 1024**2
    if os.name == "nt":
        windows_limits(cap)
    else:
        import resource

        os.nice(12)
        os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    script = Path(__file__).resolve().with_name(args.worker)
    timer = threading.Timer(args.wall, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    sys.argv = [str(script), *rest]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
