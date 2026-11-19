"""Read a limited PyTorch ZIP tensor layout without loading CUDA or pickle globals."""

from __future__ import annotations

import io
import math
import pickle
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile


@dataclass(frozen=True)
class Layout:
    dtype: str
    storage_key: str
    n_elements: int
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


def rebuild(storage, offset, shape, stride, requires_grad, hooks):
    if hooks or not isinstance(requires_grad, bool):
        raise ValueError("unsupported tensor metadata")
    return Layout(*storage, offset, shape, stride)


class LayoutReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return rebuild
        if module == "torch" and name in {"FloatStorage", "BFloat16Storage"}:
            return name
        raise pickle.UnpicklingError(f"unsupported checkpoint global {module}.{name}")

    def persistent_load(self, entry):
        kind, dtype, key, _, n = entry
        if kind != "storage" or dtype not in {"FloatStorage", "BFloat16Storage"} or not isinstance(key, str):
            raise pickle.UnpicklingError("unsupported checkpoint storage")
        if not key.isdecimal() or not isinstance(n, int) or n <= 0:
            raise pickle.UnpicklingError("invalid checkpoint storage")
        return dtype, key, n


def read_layout(path: Path):
    with ZipFile(path) as archive, path.open("rb") as raw:
        pickle_entry, = (name for name in archive.namelist() if name.endswith("/data.pkl"))
        prefix = pickle_entry.removesuffix("data.pkl")
        if archive.getinfo(pickle_entry).file_size > 2**20 or archive.read(prefix + "byteorder") != b"little":
            raise ValueError("unsupported checkpoint format")
        layouts = LayoutReader(io.BytesIO(archive.read(pickle_entry))).load()
        if not isinstance(layouts, dict) or not layouts or len(layouts) > 128:
            raise ValueError("expected a tensor-only model state")
        offsets = {}
        for name, layout in layouts.items():
            if not isinstance(name, str) or not isinstance(layout, Layout):
                raise ValueError("non-tensor checkpoint entry")
            if (len(layout.shape) != len(layout.stride) or not layout.shape
                    or any(not isinstance(n, int) or n < 1 for n in layout.shape)
                    or layout.offset != 0 or math.prod(layout.shape) != layout.n_elements):
                raise ValueError("unsupported tensor view")
            expected_stride = tuple(math.prod(layout.shape[i + 1 :]) for i in range(len(layout.shape)))
            if layout.stride != expected_stride:
                raise ValueError("unsupported non-contiguous tensor")
            info = archive.getinfo(prefix + "data/" + layout.storage_key)
            width = 4 if layout.dtype == "FloatStorage" else 2
            if info.compress_type != ZIP_STORED or info.file_size != layout.n_elements * width:
                raise ValueError("unsupported storage size or compression")
            raw.seek(info.header_offset)
            fields = struct.unpack("<IHHHHHIIIHH", raw.read(30))
            if fields[0] != 0x04034B50:
                raise ValueError("invalid local archive header")
            offset = info.header_offset + 30 + fields[-2] + fields[-1]
            if offset + info.file_size > path.stat().st_size:
                raise ValueError("tensor extends beyond the checkpoint")
            offsets[name] = (layout, offset)
    return offsets
