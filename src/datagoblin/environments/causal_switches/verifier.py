from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from datagoblin.hashing import sha256_json

from .generator import GeneratedDataset
from .solver import exact_label


@dataclass(frozen=True)
class VerificationReceipt:
    accepted: bool
    checked_rows: int
    mismatches: tuple[int, ...]
    verifier: str
    evidence_type: str
    receipt_hash: str

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "checked_rows": self.checked_rows,
            "mismatches": list(self.mismatches),
            "verifier": self.verifier,
            "evidence_type": self.evidence_type,
            "receipt_hash": self.receipt_hash,
        }


def verify_dataset(dataset: GeneratedDataset) -> VerificationReceipt:
    mismatches = tuple(
        index
        for index, (state, label) in enumerate(zip(dataset.mechanics, dataset.labels, strict=True))
        if exact_label(state) != int(label)
    )
    payload = {
        "checked_rows": len(dataset.labels),
        "mismatches": mismatches,
        "label_checksum": sha256_json(np.asarray(dataset.labels, dtype=int).tolist()),
        "verifier": "causal_switches_independent_verifier",
    }
    return VerificationReceipt(
        not mismatches,
        len(dataset.labels),
        mismatches,
        "causal_switches_independent_verifier",
        "INDEPENDENT_HARNESS",
        sha256_json(payload),
    )
