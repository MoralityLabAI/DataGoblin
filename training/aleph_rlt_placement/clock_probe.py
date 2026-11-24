"""Check the clock-order identity on a frozen public-development checkpoint."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path

CLOCKS = (1 / 3, 1 / 2, 2 / 3)


def measure(x, displacement):
    """Take both clock orders at the same x. l selects the last token's first channel."""
    def advance(z, a):
        return z + a * displacement(z)

    gaps = []
    pairs = []
    for i, j in ((0, 1), (1, 2), (0, 2)):
        a, b = CLOCKS[i], CLOCKS[j]
        forward = advance(advance(x, a), b)
        reverse = advance(advance(x, b), a)
        delta = forward - reverse
        gap = delta / (a * b)
        gaps.append(gap)
        pairs.append({
            "a": a, "b": b, "q": gap[0, -1, 0].item(),
            "forward_minus_reverse_rms": delta.square().mean().sqrt().item(),
            "forward_minus_reverse_max_abs": delta.abs().max().item(),
            "q_tensor_rms": gap.square().mean().sqrt().item(),
        })
    residual = gaps[0] + gaps[1] - gaps[2]
    scale = max(item["q_tensor_rms"] for item in pairs)
    rms = residual.square().mean().sqrt().item()
    return {"pairs": pairs, "probe_residual": residual[0, -1, 0].item(),
            "residual_max_abs": residual.abs().max().item(), "residual_rms": rms,
            "q_rms_scale": scale, "relative_rms": rms / scale if scale else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sha256", required=True, help="expected checkpoint SHA-256")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from cpu_worker import windows_limits
    if os.name == "nt":
        windows_limits(3 * 1024**3)
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})

    import torch
    from corpus import agonist_rows, generate_counter, generate_repair, identity
    from study import dense

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    raw = args.checkpoint.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != args.sha256:
        raise ValueError("checkpoint hash mismatch")
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    cfg = checkpoint["cfg"]
    if cfg["arm"] != "direct":
        raise ValueError("the clock-order diagnostic expects a direct reference")
    rows = []
    for generate in (generate_repair, generate_counter):
        _, dev = agonist_rows(generate, 512, 96)
        rows.append(min(dev, key=lambda row: identity(["panel", row.root])))
    receipt = {"scope": "two public development roots, deterministic architecture diagnostic",
               "checkpoint_sha256": actual, "step": checkpoint["step"], "seed": cfg["seed"],
               "training_arm": cfg["arm"], "width": cfg["width"], "training_reasoning_passes": cfg["r"],
               "linear_readout": "last prompt token, channel zero", "torch": str(torch.__version__),
               "source_hashes": {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                 for n in ("gym.py", "corpus.py", "study.py", "clock_probe.py")},
               "affine_control": "same base displacement and secant at t=1/2 along that displacement; rank-one slope",
               "roots": []}
    with torch.inference_mode():
        for row in rows:
            samples = {}
            for dtype in (torch.float32, torch.float64):
                model = dense(cfg["width"]).to(dtype)
                model.load_state_dict(checkpoint["model"])
                model.eval()
                x = model.blocks[0](model.embed(torch.tensor([row.public()])))

                def g(z, model=model):
                    y = z
                    for block in model.blocks[1:11]:
                        y = block(y)
                    return y - z

                g0 = g(x)
                chord = (g(x + .5 * g0) - g0) / .5
                norm2 = g0.square().sum()
                if norm2.item() == 0:
                    raise ValueError("zero core displacement on registered probe root")

                def affine(z, x=x, g0=g0, chord=chord, norm2=norm2):
                    return g0 + (((z - x) * g0).sum() / norm2) * chord

                samples[str(dtype)] = {
                    "input_rms": x.square().mean().sqrt().item(),
                    "base_displacement_rms": g0.square().mean().sqrt().item(),
                    "actual": measure(x, g), "affine": measure(x, affine),
                }
            receipt["roots"].append({"lane": row.lane, "root": row.root, "precisions": samples})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        if existing != receipt:
            raise ValueError("existing clock-order receipt differs")
    else:
        args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": actual, "step": receipt["step"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
