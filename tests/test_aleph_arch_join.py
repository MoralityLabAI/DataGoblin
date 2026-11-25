"""A joined architecture manifest must prove every seed and its token stream."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "aleph_next"))
from arch_join import join


def ledger(out, plan, members, arms=("direct3", "damp3")):
    out.mkdir()
    fits = {}
    for seed in members:
        for arm in arms:
            key = f"{arm}-{seed}"
            data = key.encode()
            (out / (key + ".pt")).write_bytes(data)
            fits[key] = {
                "checkpoint_sha256": hashlib.sha256(data).hexdigest(),
                "seed": seed,
                "arm": arm,
                "backbone_init": f"shared-{seed}",
                "stream_sha256": f"stream-{seed}",
                "positions": 12,
            }
    r = {
        "contract": {"sources": {"source": "x"}, "recipe": {"seeds": plan, "arms": list(arms), "width": 16}},
        "fits": fits,
    }
    (out / "training.json").write_text(json.dumps(r))


def test_complete_independent_shards_and_no_holdout_seed(tmp_path):
    main, extra = tmp_path / "main", tmp_path / "extra"
    ledger(main, [137, 191], [137])
    ledger(extra, [191], [191])
    (main / "private.seed").write_text("77")
    with pytest.raises(ValueError, match="before opening"):
        join(main, extra)
    (main / "private.seed").unlink()
    join(main, extra)
    record = json.loads((main / "training.json").read_text())
    assert set(record["fits"]) == {"direct3-137", "direct3-191", "damp3-137", "damp3-191"}
    assert (main / "training.pre-union.json").exists()
    assert json.loads((main / "shard-custody.json").read_text())["paired_stream_and_backbone_checks"]


def test_same_seed_disjoint_arms_must_share_the_stream(tmp_path):
    main, left, right = (tmp_path / name for name in ("main", "left", "right"))
    ledger(main, [137], [])
    ledger(left, [137], [137], ("direct3",))
    ledger(right, [137], [137], ("damp3",))
    path = right / "training.json"
    altered = json.loads(path.read_text())
    altered["fits"]["damp3-137"]["stream_sha256"] = "other"
    original = path.read_bytes()
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="stream mismatch"):
        join(main, left, right)
    path.write_bytes(original)
    join(main, left, right)
    assert len(json.loads((main / "training.json").read_text())["fits"]) == 2
