"""Comparison contract for a future component intervention adapter."""

from __future__ import annotations

import math
from typing import Protocol

from .runner import RecurrentAdapter

MATCH_FIELDS = ("task_digest", "checkpoint_sha256", "adapter_sha256", "readout_site", "operator",
                "component", "placement", "visit", "planned_depth", "evidence_epoch", "feedback_mode", "seed")


class ComponentInterventionAdapter(RecurrentAdapter, Protocol):
    """Measure a visit-level intervention without deciding its causal interpretation."""

    def prepare_component_intervention(self, comparison: dict, *, arm: str) -> dict:
        """Return the bound intervention accepted by the adapter's visits method."""
        ...


def validate_comparison(comparison):
    """Require a matched untreated, sham, random, and chosen intervention."""
    arms = comparison["arms"]
    if set(arms) != {"untreated", "sham", "matched_random", "component"}:
        raise ValueError("BPD comparison requires untreated, sham, matched random and component arms")
    component = arms["component"]
    if type(component["visit"]) is not int or not 0 <= component["visit"] < component["planned_depth"]:
        raise ValueError("intervention visit must fit the planned clock")
    for arm in arms.values():
        if any(field not in arm or arm[field] != component[field] for field in MATCH_FIELDS):
            raise ValueError("BPD arms differ in their frozen visit context")
        if not math.isfinite(arm["realized_absolute_dose"]) or arm["realized_absolute_dose"] < 0:
            raise ValueError("intervention dose must be finite and nonnegative")
    if arms["untreated"]["realized_absolute_dose"] != 0 or arms["sham"]["realized_absolute_dose"] != 0:
        raise ValueError("untreated and sham controls require zero realized dose")
    if not math.isclose(component["realized_absolute_dose"], arms["matched_random"]["realized_absolute_dose"],
                        rel_tol=1e-6, abs_tol=1e-9) or component["realized_absolute_dose"] <= 0:
        raise ValueError("component and random control require the same positive realized dose")
    if not comparison.get("selection_receipt_sha256") or not component.get("donor_receipt_sha256"):
        raise ValueError("component selection and donor provenance are required")
    if comparison.get("bpd_mechanism_verified") is not False:
        raise ValueError("a matched comparison does not establish a BPD mechanism")
