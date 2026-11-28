"""Minimal ctypes NVML reader (GPU 0): temperature, power, limits, clock events, memory.

nvidia-smi takes about two seconds per query on the Windows laptop, so the
runner's thermal gate and the watchdog read NVML directly (under a
millisecond). Returns None where the driver library is unavailable.
"""
from __future__ import annotations

import ctypes
import sys

SW_POWER_CAP = 0x4
HW_SLOWDOWN = 0x8
SW_THERMAL = 0x20
HW_THERMAL = 0x40


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class NVML:
    def __init__(self, index=0):
        name = r"C:\Windows\System32\nvml.dll" if sys.platform == "win32" else "libnvidia-ml.so.1"
        self.lib = ctypes.CDLL(name)
        if self.lib.nvmlInit_v2():
            raise OSError("nvmlInit_v2")
        self.handle = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle)):
            raise OSError("nvmlDeviceGetHandleByIndex_v2")

    def _uint(self, fn, *args):
        v = ctypes.c_uint()
        return v.value if getattr(self.lib, fn)(self.handle, *args, ctypes.byref(v)) == 0 else None

    def read(self):
        events = ctypes.c_ulonglong()
        ok = self.lib.nvmlDeviceGetCurrentClocksEventReasons(self.handle, ctypes.byref(events)) == 0
        mem = _Memory()
        mem_ok = self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(mem)) == 0
        enforced = self._uint("nvmlDeviceGetEnforcedPowerLimit")
        default = self._uint("nvmlDeviceGetPowerManagementDefaultLimit")
        power = self._uint("nvmlDeviceGetPowerUsage")
        util = ctypes.c_uint*2
        u = util()
        util_ok = self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(u)) == 0
        return {
            "temperature_c": self._uint("nvmlDeviceGetTemperature", 0),
            "power_w": None if power is None else power/1000,
            "enforced_limit_w": None if enforced is None else enforced/1000,
            "default_limit_w": None if default is None else default/1000,
            "power_capped": None if enforced is None or default is None else enforced < 0.9*default,
            "clock_events": hex(events.value) if ok else None,
            "utilization_gpu": u[0] if util_ok else None,
            "memory_used_bytes": mem.used if mem_ok else None,
            "memory_free_bytes": mem.free if mem_ok else None,
        }


def open_nvml():
    try:
        return NVML()
    except (OSError, AttributeError):
        return None
