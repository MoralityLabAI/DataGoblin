from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_bytes
from datagoblin.serialization import canonical_json

_FORBIDDEN_SUFFIXES = {".ckpt", ".pt", ".pth", ".safetensors", ".env"}


@dataclass(frozen=True)
class ArtifactRef:
    sha256: str
    size_bytes: int
    media_type: str
    relative_path: str

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "relative_path": self.relative_path,
        }


class ArtifactStore:
    def __init__(self, root: Path, maximum_bytes: int = 64 * 1024 * 1024) -> None:
        self.root = root
        self.maximum_bytes = maximum_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, value: bytes, *, media_type: str, name_hint: str = "artifact") -> ArtifactRef:
        if len(value) > self.maximum_bytes:
            raise ValueError("artifact exceeds store size policy")
        if Path(name_hint).suffix.lower() in _FORBIDDEN_SUFFIXES:
            raise ValueError("weights, checkpoints, and secret-like files are not accepted")
        digest = sha256_bytes(value)
        relative = Path("sha256") / digest[:2] / digest
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != value:
            raise RuntimeError("content-address collision")
        if not path.exists():
            path.write_bytes(value)
        return ArtifactRef(digest, len(value), media_type, relative.as_posix())

    def put_json(self, value: Any, *, name_hint: str = "artifact.json") -> ArtifactRef:
        return self.put_bytes(
            (canonical_json(value) + "\n").encode("utf-8"),
            media_type="application/json",
            name_hint=name_hint,
        )

    def put_file(self, path: Path, *, media_type: str = "application/octet-stream") -> ArtifactRef:
        return self.put_bytes(path.read_bytes(), media_type=media_type, name_hint=path.name)

    def read(self, reference: ArtifactRef) -> bytes:
        value = (self.root / reference.relative_path).read_bytes()
        if sha256_bytes(value) != reference.sha256:
            raise RuntimeError("artifact hash mismatch")
        return value
