"""Native Google TorchTPU (device='tpu') backend and host-CPU FP32 AdamW offload for TPU v5e-1.

Allows the frozen 999,352,320-parameter Direct-A model (FP32 master weights, BF16 compute,
FP32 gradients, and FP32 AdamW moments) to train bit-identically inside TPU v5e-1's 15.75 GiB
HBM by keeping the 7.44 GiB FP32 AdamW first/second moments in the TPU VM's 49 GB cloud
host RAM and stepping FP32 AdamW on cloud CPU once per 262,144-token update group.
"""

from __future__ import annotations

import contextlib
import math
import os
import random
from typing import Any

import numpy as np
import torch


class TorchTPUBackend:
    """Backend adapter for Google Colab's native TorchTPU PrivateUse1 (`device='tpu'`) runtime."""

    def __init__(self) -> None:
        self.kind = "tpu"
        self.xla = None
        self.xm = None
        self.metrics = None
        if not (hasattr(torch, "accelerator") and torch.accelerator.is_available()):
            raise RuntimeError("TorchTPU accelerator is not available")
        if torch.accelerator.device_count() != 1:
            raise RuntimeError(f"require exactly one TPU device, got {torch.accelerator.device_count()}")
        self.device = torch.device("tpu")
        hw = "Google Cloud TPU"
        if hasattr(torch, "tpu") and hasattr(torch.tpu, "get_device_name"):
            with contextlib.suppress(Exception):
                hw = str(torch.tpu.get_device_name())
        mem_limit = 16_909_336_064  # v5e-1 default 15.75 GiB reservable HBM
        if hasattr(torch, "tpu") and hasattr(torch.tpu, "get_device_properties"):
            with contextlib.suppress(Exception):
                prop = torch.tpu.get_device_properties(0)
                if hasattr(prop, "total_memory") and int(prop.total_memory) > 0:
                    mem_limit = int(prop.total_memory)
        self.identity: dict[str, Any] = {
            "backend": "tpu",
            "hardware": hw,
            "world_size": 1,
            "memory_limit": mem_limit,
            "optimizer_offload": "host_cpu_fp32",
        }

    def sync(self) -> None:
        torch.accelerator.synchronize()

    def memory(self) -> dict[str, Any]:
        if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "memory_stats"):
            try:
                stats = torch.accelerator.memory_stats()
                if stats:
                    return {k: int(v) for k, v in stats.items() if isinstance(v, (int, float))}
            except Exception:
                pass
        return {"bytes_limit": int(self.identity["memory_limit"])}

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if hasattr(torch, "tpu") and hasattr(torch.tpu, "manual_seed_all"):
            with contextlib.suppress(Exception):
                torch.tpu.manual_seed_all(seed)

    def get_rng(self) -> dict[str, Any]:
        from restart import rng_state

        return {"host": rng_state(), "xla": None, "cuda": None}

    def set_rng(self, state: dict[str, Any]) -> None:
        from restart import restore_rng

        restore_rng(state["host"])


class HostOffloadFP32AdamW:
    """Bit-identical FP32 AdamW keeping m/v moments in cloud host CPU RAM (0 extra HBM).

    Exposes the exact same interface as ``colab10m_xla.FP32AdamW`` (``named``, ``moments``,
    ``step``, ``zero()``, ``update(rate)``) so checkpoint save/load and handoff are unchanged.
    """

    def __init__(self, net: torch.nn.Module, cfg: dict[str, Any], backend: Any) -> None:
        self.named = list(net.named_parameters())
        if len({id(p) for _, p in self.named}) != len(self.named):
            raise ValueError("duplicate optimizer identity")
        self.cfg = cfg["optimizer"]
        self.backend = backend
        self.moments: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.step = 0
        for name, p in self.named:
            if p.dtype != torch.float32:
                raise ValueError("master parameter is not FP32")
            self.moments[name] = (
                torch.zeros(p.shape, dtype=torch.float32, device="cpu"),
                torch.zeros(p.shape, dtype=torch.float32, device="cpu"),
            )
        backend.sync()

    def zero(self) -> None:
        for _, p in self.named:
            p.grad = None

    @torch.no_grad()
    def update(self, rate: float) -> float:
        ps = [p for _, p in self.named]
        if any(p.grad is None or p.grad.dtype != torch.float32 for p in ps):
            raise ValueError("missing or non-FP32 gradient")

        self.backend.sync()
        prev_threads = torch.get_num_threads()
        cpu_threads = max(1, min(16, os.cpu_count() or 4))
        torch.set_num_threads(cpu_threads)
        try:
            # Transfer summed FP32 gradients once per 262,144-token group
            grads_cpu: list[torch.Tensor] = [p.grad.detach().to("cpu", dtype=torch.float32) for p in ps]
            grad_norm = float(torch.stack([g.square().sum() for g in grads_cpu]).sum().sqrt())
            if not math.isfinite(grad_norm):
                raise FloatingPointError("nonfinite global gradient norm")

            b1 = float(self.cfg["beta1"])
            b2 = float(self.cfg["beta2"])
            eps = float(self.cfg["eps"])
            t = self.step + 1
            bc1 = 1.0 - b1**t
            bc2 = math.sqrt(1.0 - b2**t)
            clip = min(1.0, float(self.cfg["clip_global_norm"]) / (grad_norm + 1e-6))
            step_scale = -(float(rate) / bc1)

            for (name, p), g_cpu in zip(self.named, grads_cpu, strict=True):
                m, v = self.moments[name]
                g = g_cpu * clip
                m.lerp_(g, 1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)

                w_cpu = p.detach().to("cpu", dtype=torch.float32)
                decay = float(self.cfg["weight_decay_matrices"] if p.ndim >= 2 else self.cfg["weight_decay_norms"])
                w_cpu.mul_(1.0 - float(rate) * decay)
                w_cpu.add_(step_scale * m / (v.sqrt() / bc2 + eps))

                if not (
                    bool(torch.isfinite(w_cpu).all())
                    and bool(torch.isfinite(m).all())
                    and bool(torch.isfinite(v).all())
                ):
                    raise FloatingPointError("nonfinite update; no checkpoint published")
                p.copy_(w_cpu.to(p.device))
            self.backend.sync()
            self.step = t
            self.zero()
            return grad_norm
        finally:
            torch.set_num_threads(prev_threads)
