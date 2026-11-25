"""Production data tests use fabricated public inputs."""

import sys
from contextlib import ExitStack
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "training/aleph_next/candidate01"
sys.path.insert(0, str(HERE))
from production import digest, read  # noqa: E402
from reserve import counts, freeze, verify  # noqa: E402


@pytest.mark.parametrize(
    ("family", "instance", "count"),
    [
        ("subset_sum", {"values": [1, 1, 2], "target": 2}, 2),
        ("perfect_matchings", {"n": 2, "allowed": [[1, 1], [1, 1]]}, 2),
        ("domino_tilings", {"width": 2, "height": 2, "blocked": []}, 2),
        ("proper_colorings", {"n": 3, "edges": [[0, 1], [1, 2], [0, 2]]}, 6),
        ("independent_sets", {"n": 3, "edges": [[0, 1], [1, 2]]}, 5),
        ("linear_extensions", {"n": 3, "edges": [[0, 2], [1, 2]]}, 2),
    ],
)
def test_independent_counters(family, instance, count):
    assert counts(family, instance) == (count, count)


def test_role_custody(tmp_path):
    base = tmp_path / "custody"
    freeze(base)
    manifest = verify(base)
    assert len({k for s in manifest["roles"].values() for k in s["families"]}) == 6
    assert manifest["training_rows"] == manifest["scored_models"] == 0
    with pytest.raises(FileExistsError):
        freeze(base)
    path = base / "final/targets.jsonl"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="held shard changed"):
        verify(base)


def test_mixture_stops_without_wrapping_or_advancing():
    pytest.importorskip("torch")
    from snapshot import MixedStream

    class Source:
        def next_pack(self, stop):
            raise AssertionError("exhaustion must be caught before advancing the cursor")

    stream = MixedStream.__new__(MixedStream)
    stream.totals = {"general": 0}
    stream.goals = {"general": 100}
    stream.sources = {"general": Source()}
    stream.index = {"lanes": {"general": {"positions": 12}}}
    assert stream.next_pack(0) is None
    with pytest.raises(ValueError, match="insufficient admitted positions in general"):
        stream.next_pack(20)
    assert stream.totals == {"general": 0}


def test_recipe_and_mixture_integrity(tmp_path):
    pytest.importorskip("torch")
    from readiness import milestone_supply
    from restart import boundaries, lr
    from snapshot import MixedStream

    cfg = read(HERE / "RECIPE.json")
    pin = read(HERE / "SUPPLY.json")
    demand = milestone_supply(cfg, pin)
    assert sum(lane["required"] for lane in demand.values()) == cfg["stream"]["checkpoints"][0]
    assert sum(lane["missing"] for lane in demand.values()) == 5_785_246
    assert demand["custom_research"]["missing"] == 2_092_917
    stops = list(boundaries(cfg))
    assert sum(cfg["stream"]["lane_positions"].values()) == stops[-1] == 10**10
    assert set(cfg["stream"]["checkpoints"]) <= set(stops)
    assert max(b - a for a, b in zip([0, *stops[:-1]], stops, strict=True)) == cfg["stream"]["batch_nonpadding"]
    assert lr(cfg, 0) == 0
    assert lr(cfg, stops[-1]) == cfg["optimizer"]["peak_lr"]
    assert lr(cfg, 20 * 10**9) == cfg["optimizer"]["ending_lr"]
    path = tmp_path / "index.json"
    path.write_bytes(b"{}")
    pin = digest(path)
    path.write_bytes(b"{}\n")
    with ExitStack() as stack, pytest.raises(ValueError, match="wrong production snapshot"):
        MixedStream(tmp_path, path, pin, stack)


def test_text_rights_component_split_and_signature():
    pytest.importorskip("torch")
    from text_intake import LICENSE, assign_groups, duplicate_signature, eligibility

    def document(author, book, section):
        url = f"https://math.libretexts.org/Bookshelves/Algebra/{book}/{section}"
        return {
            "id": url,
            "metadata": {
                "author": author,
                "book_url": f"https://commons.libretexts.org/book/{book}",
                "url": url,
                "license": LICENSE,
                "title": section,
                "provenance": "source locator",
            },
            "text": "a simple mathematics textbook passage " * 20,
        }

    rows = [document(f"author{i}", f"book{i}", f"section{i}") for i in range(6)]
    sibling = document("author0", "otherbook", "later")
    groups = assign_groups([*rows, sibling])
    assert groups[sibling["id"]] == groups[rows[0]["id"]]
    assert eligibility(rows[0]) is None
    rows[0]["metadata"]["license"] = "CC BY-NC"
    assert eligibility(rows[0]) == "license_outside_cc_by_4"
    assert duplicate_signature("One solution in case 172") == duplicate_signature("ONE solution in case 42")
