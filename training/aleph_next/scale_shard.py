"""Run admitted, independent scale cells in a separate one-core worker."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from corpus import load_corpus
from gym import file_sha
from scale_lab import fit
from study import SEEDS, select_panel

HERE = Path(__file__).resolve().parent


def publish(source, destination):
    """Publish a complete file without overwriting any primary-worker result."""
    try:
        if os.name == "nt":
            # The managed Windows workspace permits ordinary copies but restricts hard links.
            # Windows rename fails if the destination exists. A pending copy is never visible to the trainer.
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".scale-stage-", delete=False) as saved:
                pending = Path(saved.name)
            try:
                shutil.copyfile(source, pending)
                os.rename(pending, destination)
            finally:
                pending.unlink(missing_ok=True)
        else:
            os.link(source, destination)
    except FileExistsError:
        if source.suffix == ".json":
            first = json.loads(source.read_bytes())
            second = json.loads(destination.read_bytes())
            fields = ("id", "stream", "checkpoint_sha256", "contract")
            if any(first[k] != second[k] for k in fields):
                raise ValueError("independent receipt collision") from None
        elif file_sha(source) != file_sha(destination):
            raise ValueError("independent checkpoint collision") from None


def run(directory, stage, pack):
    gate = json.loads((directory / "gates.json").read_bytes())["80"]
    if "prelude_reread" not in gate["survivors"]:
        raise ValueError("width-256 prelude was not admitted")
    if stage.resolve() == directory.resolve():
        raise ValueError("staging directory must be distinct from the primary worker")
    stage.mkdir(parents=True, exist_ok=True)
    train, dev, source = load_corpus(pack)
    train["lean_pos"] = [r for r in train["lean"] if r.good == (1,)]
    train["lean_neg"] = [r for r in train["lean"] if r.good == (0,)]
    panel = select_panel(dev, 8)
    contract = file_sha(HERE / "scale_lab.py") + "-" + file_sha(HERE / "SCALE_PROTOCOL.md")
    for seed in reversed(SEEDS):
        key = f"w256-prelude_reread-{seed}"
        dest_record = directory / f"{key}.json"
        dest_model = directory / f"{key}.pt"
        if dest_record.exists():
            row = json.loads(dest_record.read_bytes())
            if row["checkpoint_sha256"] != file_sha(dest_model) or row["contract"] != contract:
                raise ValueError("already published fit has an incompatible checkpoint")
            continue
        row = fit(256, "prelude_reread", seed, train, panel, stage, source, contract)
        model, record = (stage / f"{key}{ext}" for ext in (".pt", ".json"))
        if file_sha(model) != row["checkpoint_sha256"]:
            raise ValueError("staging checkpoint changed")
        publish(model, dest_model)
        publish(record, dest_record)
        print(f"published {key}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("out", "staging", "pack"):
        p.add_argument("--" + flag, type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    run(args.out, args.staging, args.pack)
