"""The deployment roster protects the matched streams and post-selection candidate check."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/aleph_next"))
import confirm_lock as gate


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def roster(tmp_path, monkeypatch):
    receipts, parents = {}, {}
    locked = []
    for arm in gate.ARMS:
        for seed in gate.SEEDS:
            key = f"{arm}-{seed}"
            folder = tmp_path / "cells" / key
            folder.mkdir(parents=True)
            ck = folder / "h0064.pt"
            ck.write_bytes(key.encode())
            roots = folder / "h0064-roots.json"
            write(roots, {"repair": [str(seed)]})
            data = dict(
                arm=arm,
                seed=seed,
                step=64,
                visits=24,
                params=14179584,
                recipe="test",
                prefix_sha=key,
                checkpoint_sha256=gate.digest(ck),
                roots_sha256=gate.digest(roots),
                tokens=50,
                padded=60,
                labels=10,
                stream=str(seed),
                prefix_stream=str(seed),
                prefix_tokens=40,
                prefix_padded=45,
            )
            dev = dict(
                text_nll=2.7,
                text_unigram_nll=3.1,
                repair=0.85,
                repair_cue=0.72,
                repair_chance=0.25,
                text_positions=200,
                repair_rows=80,
                competent=True,
            )
            parents["w256-" + key] = {"sha256": key}
            if arm == "direct":
                data["development"] = dev
            else:
                locked.append((folder, dict(horizon=64, seed=seed, checkpoint_sha256=gate.digest(ck), development=dev)))
            receipt = folder / "h0064.json"
            write(receipt, data)
            receipts[key] = gate.digest(receipt)
    selected = tmp_path / "selection.json"
    write(selected, dict(horizon=64, recipe="test", competent_direct_seeds=list(gate.SEEDS), receipts=receipts))
    for folder, r in locked:
        write(folder / "h0064-locked-dev.json", dict(r, certificate_sha256=gate.digest(selected)))
    monkeypatch.setattr(gate, "prefix_inventory", lambda unused: parents)


def test_fit_lock_byte_identity_and_no_replacing_frozen_roster(tmp_path, monkeypatch):
    roster(tmp_path, monkeypatch)
    out = tmp_path / "fits.json"
    first = gate.manifest(tmp_path, tmp_path, out)
    assert first == gate.manifest(tmp_path, tmp_path, out)
    assert len(json.loads(out.read_bytes())["fits"]) == 6
    file = tmp_path / "cells/global_full_damp-163/h0064.pt"
    file.write_bytes(b"different")
    with pytest.raises(ValueError, match="identity mismatch"):
        gate.manifest(tmp_path, tmp_path, out)


def test_requires_post_selection_candidate_screen(tmp_path, monkeypatch):
    roster(tmp_path, monkeypatch)
    file = tmp_path / "cells/global_full_damp-137/h0064-locked-dev.json"
    r = json.loads(file.read_bytes())
    r["certificate_sha256"] = "stale"
    write(file, r)
    with pytest.raises(ValueError, match="candidate development result is stale"):
        gate.manifest(tmp_path, tmp_path, tmp_path / "fits.json")
