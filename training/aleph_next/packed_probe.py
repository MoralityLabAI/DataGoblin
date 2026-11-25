"""Exercise real Aleph gradients and exact cursor replay on the released train-only corpus."""

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "aleph_rlt_placement"))
from gym import Aleph  # noqa: E402
from packing import PackedStream, reference_logits, token_loss_sum  # noqa: E402


def run(path, sha256):
    torch.set_num_threads(1)
    torch.manual_seed(137)
    with PackedStream(path, sha256, "train") as stream:
        pack = stream.next_pack(4096)
        cursor = stream.checkpoint()
        next_pack = stream.next_pack(4096)
        vocab = stream.manifest["vocab_size"]
    with PackedStream(path, sha256, "train", cursor) as restored:
        if next_pack != restored.next_pack(4096):
            raise ValueError("packed stream cursor replay changed targets")
    m = Aleph(width=16, vocab=vocab)
    logits = reference_logits(m, pack, "direct", r=2)
    loss, count = token_loss_sum(logits, pack["labels_at_query"])
    if not count.item():
        raise ValueError("no supervised target in the first pack")
    (loss / count).backward()
    norm = torch.nn.utils.clip_grad_norm_(m.parameters(), float("inf"), error_if_nonfinite=True)
    if not norm > 0:
        raise ValueError("no research gradient")
    return dict(
        positions=pack["nonpadding"],
        targets=count.item(),
        mean_loss=(loss / count).item(),
        gradient_norm=norm.item(),
        exact_next_pack_replay=True,
        optimizer_updates=0,
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("manifest", type=Path)
    cli.add_argument("--sha256", required=True)
    args = cli.parse_args()
    print(json.dumps(run(args.manifest, args.sha256)))
