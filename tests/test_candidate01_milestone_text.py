"""Regression tests for intact spans and the exact-match floor of the historical text screen."""

import sys
from array import array
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/aleph_next/candidate01"))
import milestone_text as intake
from milestone_text_overlap import SPAN, STRIDE, Overlap


@pytest.mark.parametrize("phase", range(STRIDE))
def test_shared_phrase_screen_covers_every_anchor_phase(phase):
    source = array("I", range(200, 300))
    lookup = Overlap()
    lookup.add_prepared("first", frozenset({1}), source)
    start = phase + STRIDE
    for width in (SPAN - 1, SPAN):
        prefix = array("I", range(1000, 1000 + STRIDE - phase))
        candidate = prefix + source[start : start + width] + array("I", [3000])
        observed = lookup.check("second", frozenset({2}), candidate)
        assert observed == ("shared_24_words" if width == SPAN else None)


def test_single_passage_boundaries_retain_unmodified_source():
    def block(marker):
        return " ".join(f"{marker} " + " ".join(f"word{j}" for j in range(50)) + "." for _ in range(3))

    left, right = block("Before"), block("Après")
    forbidden = "Shared source text."
    original = "\n".join((left, forbidden, right))
    denied = {forbidden.encode()[: intake.MATCH]}
    spans = intake.passages(original, denied)
    assert len(spans) == 2
    assert {original[a:b] for a, b in spans} == {left, right}
    assert all(not intake.short_match(original[a:b], denied) for a, b in spans)


def test_rights_restriction_applies_before_excerpting():
    assert intake.MIXED_RIGHTS.search("Excerpt reprinted by permission from the original publisher.")
    assert intake.MIXED_RIGHTS.search("This image is released under CC BY-SA.")
    assert not intake.MIXED_RIGHTS.search("This chapter is released under CC BY 4.0.")
