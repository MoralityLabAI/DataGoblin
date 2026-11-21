"""Bounded NVIDIA telemetry for the Windows Inkling resource guard."""

from __future__ import annotations

import ctypes as c
import os
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Thread
from time import monotonic
from typing import TypedDict


class GpuSample(TypedDict):
    vram_free_mib: int
    gpu_util_percent: int
    compute_pids: list[int]


class _Memory(c.Structure):
    _fields_ = [("total", c.c_ulonglong), ("free", c.c_ulonglong), ("used", c.c_ulonglong)]


class _Utilization(c.Structure):
    _fields_ = [("gpu", c.c_uint), ("memory", c.c_uint)]


class _Process(c.Structure):
    _fields_ = [
        ("pid", c.c_uint),
        ("used_gpu_memory", c.c_ulonglong),
        ("gpu_instance_id", c.c_uint),
        ("compute_instance_id", c.c_uint),
    ]


def _check(name: str, status: int) -> None:
    if status:
        raise RuntimeError(f"{name} returned NVML error {status}")


class _Device:
    def __init__(self) -> None:
        dll = Path(os.environ["SYSTEMROOT"]) / "System32" / "nvml.dll"
        nvml = c.CDLL(str(dll))
        nvml.nvmlInit_v2.argtypes = []
        nvml.nvmlDeviceGetCount_v2.argtypes = [c.POINTER(c.c_uint)]
        nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [c.c_uint, c.POINTER(c.c_void_p)]
        nvml.nvmlDeviceGetMemoryInfo.argtypes = [c.c_void_p, c.POINTER(_Memory)]
        nvml.nvmlDeviceGetUtilizationRates.argtypes = [c.c_void_p, c.POINTER(_Utilization)]
        nvml.nvmlDeviceGetComputeRunningProcesses_v3.argtypes = [
            c.c_void_p, c.POINTER(c.c_uint), c.POINTER(_Process),
        ]
        _check("initialization", nvml.nvmlInit_v2())
        count = c.c_uint()
        _check("device count", nvml.nvmlDeviceGetCount_v2(c.byref(count)))
        if count.value != 1:
            raise RuntimeError("the Inkling guard requires exactly one NVIDIA device")
        self.handle = c.c_void_p()
        _check("device handle", nvml.nvmlDeviceGetHandleByIndex_v2(0, c.byref(self.handle)))
        self.nvml = nvml

    def sample(self) -> GpuSample:
        memory, utilization = _Memory(), _Utilization()
        _check("device memory", self.nvml.nvmlDeviceGetMemoryInfo(self.handle, c.byref(memory)))
        _check("device utilization", self.nvml.nvmlDeviceGetUtilizationRates(self.handle, c.byref(utilization)))
        processes = (_Process * 64)()
        count = c.c_uint(len(processes))
        _check(
            "compute processes",
            self.nvml.nvmlDeviceGetComputeRunningProcesses_v3(self.handle, c.byref(count), processes),
        )
        if count.value > len(processes) or memory.free > memory.total or utilization.gpu > 100:
            raise RuntimeError("invalid NVIDIA telemetry")
        return {
            "vram_free_mib": memory.free // (1024 * 1024),
            "gpu_util_percent": utilization.gpu,
            "compute_pids": sorted({process.pid for process in processes[:count.value]}),
        }


class WindowsNvmlMonitor:
    """Read a fresh sample on demand. A hung driver call cannot hold the guard open."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("the Inkling NVML monitor requires Windows")
        self._requests: Queue[tuple[float, Queue[GpuSample | Exception]]] = Queue(maxsize=1)
        Thread(target=self._collect, daemon=True).start()

    def _collect(self) -> None:
        device = None
        while True:
            deadline, reply = self._requests.get()
            if monotonic() >= deadline:
                continue
            try:
                if device is None:
                    device = _Device()
                reply.put_nowait(device.sample())
            except Exception as error:
                reply.put_nowait(error)

    def sample(self) -> GpuSample:
        deadline = monotonic() + 5.0
        reply: Queue[GpuSample | Exception] = Queue(maxsize=1)
        try:
            self._requests.put_nowait((deadline, reply))
            result = reply.get(timeout=max(0, deadline - monotonic()))
        except (Empty, Full) as error:
            raise RuntimeError("NVIDIA telemetry did not respond within five seconds") from error
        if monotonic() >= deadline:
            raise RuntimeError("NVIDIA telemetry arrived after the five-second deadline")
        if isinstance(result, Exception):
            raise RuntimeError(f"NVIDIA telemetry unavailable: {result}") from result
        return result
