from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HRMAvailability:
    torch_available: bool
    transformers_available: bool
    transformers_version: str | None
    native_hrm_text_module: bool
    local_model_path: str | None
    local_model_exists: bool | None
    model_id: str | None
    weights_loaded: bool
    download_attempted: bool
    note: str

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "torch_available": self.torch_available,
            "transformers_available": self.transformers_available,
            "transformers_version": self.transformers_version,
            "native_hrm_text_module": self.native_hrm_text_module,
            "local_model_path": self.local_model_path,
            "local_model_exists": self.local_model_exists,
            "model_id": self.model_id,
            "weights_loaded": self.weights_loaded,
            "download_attempted": self.download_attempted,
            "note": self.note,
        }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def detect_hrm_availability(
    *, model_path: Path | None = None, model_id: str | None = None
) -> HRMAvailability:
    """Report optional support without importing frameworks or loading weights."""
    torch_available = importlib.util.find_spec("torch") is not None
    transformers_available = importlib.util.find_spec("transformers") is not None
    native_module = False
    if transformers_available:
        distribution = metadata.distribution("transformers")
        native_module = any("models/hrm_text" in str(path).replace("\\", "/") for path in distribution.files or ())
    exists = model_path.exists() if model_path is not None else None
    notes = []
    if not transformers_available or not torch_available:
        notes.append("install the optional hrm-text extra to load a real checkpoint")
    if transformers_available and not native_module:
        notes.append("installed Transformers does not expose a native hrm_text module")
    if model_path is not None and not exists:
        notes.append("configured local model path does not exist")
    if model_path is None and model_id is None:
        notes.append("no model reference configured; no network request was made")
    local_path = None
    if model_path is not None:
        local_path = str(model_path.resolve()) if exists else str(model_path)
    return HRMAvailability(
        torch_available=torch_available,
        transformers_available=transformers_available,
        transformers_version=_package_version("transformers"),
        native_hrm_text_module=native_module,
        local_model_path=local_path,
        local_model_exists=exists,
        model_id=model_id,
        weights_loaded=False,
        download_attempted=False,
        note="; ".join(notes) or "optional runtime appears available; weights were not loaded",
    )


def validate_model_reference(
    *, model_path: Path | None, model_id: str | None, allow_download: bool
) -> tuple[str, bool]:
    if (model_path is None) == (model_id is None):
        raise ValueError("choose exactly one of --model-path or --model-id")
    if model_path is not None:
        resolved = model_path.resolve()
        if not resolved.exists() or not resolved.is_dir():
            raise ValueError(f"local model directory does not exist: {resolved}")
        return str(resolved), True
    assert model_id is not None
    if not model_id.strip() or "/" not in model_id:
        raise ValueError("model ID must be an explicit owner/name reference")
    return model_id, not allow_download
