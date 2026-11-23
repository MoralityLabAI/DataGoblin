"""Allocate, fit-lock, and score a registered native courier experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

from .native import NativeCourier, canonical, digest, load_storyworld
from .split import HERE, allocate, load_manifest, plan, worlds, write_manifest

FITS = ("canonical", "hashed", "shortest", "budget", "soft", "obligation")
ARMS = {
    "A": ("canonical", "standard"),
    "B": ("hashed", "standard"),
    "C": ("shortest", "standard"),
    "D": ("canonical", "reserve"),
    "E": ("hashed", "reserve"),
    "F": ("shortest", "reserve"),
    "G": ("budget", "reserve"),
    "I": ("soft", "reserve"),
    "J": ("shortest", "early"),
    "K": ("shortest", "gate"),
    "L": ("obligation", "reserve"),
}
REPO = HERE.parents[1]


def json_out(path: Path, data) -> None:
    with path.open("xb") as handle:
        handle.write(canonical(data) + b"\n")


def jsonl_out(path: Path, rows) -> str:
    checksum = hashlib.sha256()
    with path.open("xb") as handle:
        for row in rows:
            line = canonical(row) + b"\n"
            handle.write(line)
            checksum.update(line)
    return checksum.hexdigest()


def file_sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def provenance(native: NativeCourier, source_hashes: dict, protocol: dict, manifest: dict) -> dict:
    import torch

    from ..looped.model import upstream_revision

    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(REPO), *args])

    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Commit source, split, and protocol before fitting or scoring")
    source_paths = str(HERE.relative_to(REPO).as_posix())
    tree = (
        git("ls-files", source_paths, "metta_research_bench/looped/model.py", "MeTTa/inky_skills.py", "MeTTa/skills")
        .decode()
        .splitlines()
    )
    return {
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "code_sha256": {path: hashlib.sha256(git("show", f"HEAD:{path}")).hexdigest() for path in tree},
        "looped_transformers": upstream_revision(),
        "storyworld_sources": source_hashes,
        "storyworld": protocol["storyworld"],
        "rules_sha256": native.rules_sha256,
        "protocol_sha256": digest(protocol),
        "manifest_sha256": digest(manifest),
        "hyperon": version("hyperon"),
        "torch": str(torch.__version__),
    }


def prepare(native, upstream, protocol, manifest, identity, output: Path) -> None:
    import torch

    from .learning import CourierPointer, fit, fit_stop_gate, training_data

    output.mkdir(parents=True, exist_ok=False)
    data = training_data(
        native, worlds(upstream, protocol, manifest, "train"), protocol["horizon"], protocol["verifier_ceiling"]
    )
    fits = []
    gates = {}
    for seed in protocol["model_seeds"]:
        torch.manual_seed(seed)
        initial = CourierPointer()
        for name in FITS:
            initial.obligation_aware = name == "obligation"
            objective = "shortest" if initial.obligation_aware else name
            model, fit_record = fit(initial, data, objective, seed, protocol["train_updates"], protocol["batch_size"])
            filename = f"{name}-{seed}.pt"
            torch.save(model.state_dict(), output / filename)
            print(f"fit {name}/{seed}: {fit_record['checkpoint_weights_sha256']}", flush=True)
            fits.append({**fit_record, "name": name, "file": filename, "file_sha256": file_sha(output / filename)})
            if name == "shortest":
                gates[str(seed)] = fit_stop_gate(model, data)
        if len({row["initial_sha256"] for row in fits if row["seed"] == seed}) != 1:
            raise RuntimeError("Fits diverged in matched initialization")
    frames_sha = jsonl_out(output / "frames.jsonl", data.frames)
    receipts_sha = jsonl_out(output / "train_receipts.jsonl", data.receipts)
    json_out(
        output / "fit_lock.json",
        {
            "identity": identity,
            "fits": fits,
            "gates": gates,
            "frames_sha256": frames_sha,
            "train_receipts_sha256": receipts_sha,
            "training_native_calls": native.calls,
            "native_frames": len(data.frames),
            "shortest_ties": sum(len(row["weights"]["shortest"]) > 1 for row in data.frames),
            "successful_nonshortest": sum(
                len(row["weights"]["budget"]) > len(row["weights"]["shortest"]) for row in data.frames
            ),
        },
    )


def _load_models(output: Path, protocol: dict, identity: dict):
    import torch

    from ..looped.model import weight_hash
    from .learning import CourierPointer

    lock = json.loads((output / "fit_lock.json").read_text(encoding="utf-8"))
    if lock["identity"] != identity:
        raise ValueError("Source, dependencies, or protocol differ from fit lock")
    if (
        file_sha(output / "frames.jsonl") != lock["frames_sha256"]
        or file_sha(output / "train_receipts.jsonl") != lock["train_receipts_sha256"]
    ):
        raise ValueError("Training data or proof traces changed after locking")
    models = {}
    for receipt in lock["fits"]:
        name, seed = receipt["name"], receipt["seed"]
        file = output / f"{name}-{seed}.pt"
        if (
            file.name != receipt["file"]
            or file_sha(file) != receipt["file_sha256"]
            or receipt["updates"] != protocol["train_updates"]
            or (name, seed) in models
        ):
            raise ValueError("A locked checkpoint differs from its registered fit")
        model = CourierPointer(name == "obligation").eval()
        model.load_state_dict(torch.load(file, map_location="cpu", weights_only=True))
        if weight_hash(model) != receipt["checkpoint_weights_sha256"]:
            raise ValueError("Checkpoint tensor digest differs from fit receipt")
        models[name, seed] = model
    if models.keys() != {(name, seed) for name in FITS for seed in protocol["model_seeds"]}:
        raise ValueError("Incomplete paired checkpoint lock")
    return lock, models


def score(native, upstream, protocol, manifest, identity: dict, output: Path) -> None:
    from .analysis import analyze
    from .controller import Proposal, Repair, run_episode
    from .native import STOP
    from .oracle import repair_routes
    from .probes import probe_world, summarize_probes

    lock, models = _load_models(output, protocol, identity)
    # Reserve the scoring destination before opening any held-out worlds.
    with (output / "score_started.json").open("xb") as marker:
        marker.write(canonical({"fit_lock_sha256": file_sha(output / "fit_lock.json")}) + b"\n")
    rows = []
    probes = []
    probe_native = NativeCourier(upstream) if protocol.get("native_state_probes") else None
    horizon, ceiling = protocol["horizon"], protocol["verifier_ceiling"]
    for root, world in worlds(upstream, protocol, manifest, "score"):
        exact = repair_routes(world, world.initial, horizon, 1)[0]
        base = {"root": root.world_sha256, "family": root.family, "shortest_moves": len(exact)}
        goal_actions = iter((*exact, STOP))
        oracle = run_episode(
            native.session(world, ceiling, horizon), lambda _obs, actions=goal_actions: Proposal(next(actions))
        )
        if not oracle["verified_completion"]:
            raise RuntimeError("Exact reference route failed native acceptance under the shared caps")
        rows.append({**base, "arm": "H", "seed": None, **oracle})
        for seed in protocol["model_seeds"]:
            intercept, slope = lock["gates"][str(seed)]

            def gate(proposal, offset=intercept, coefficient=slope):
                return offset + coefficient * proposal.stop_confidence < 0

            for arm, (name, mode) in ARMS.items():
                policy = Repair(
                    mode,
                    protocol["macro_horizon"],
                    protocol["macro_attempt_limit"],
                    gate=gate if mode == "gate" else None,
                )
                session = native.session(world, ceiling, horizon)
                rows.append(
                    {**base, "arm": arm, "seed": seed, **run_episode(session, models[name, seed].propose, policy)}
                )
        if probe_native is not None:
            probes.extend(probe_world(probe_native, world, models, root, protocol["model_seeds"], horizon, ceiling))
    if native.calls != sum(row["calls"] for row in rows):
        raise RuntimeError("Runtime queries differ from summed episode ledgers")
    metrics = analyze(rows, protocol["model_seeds"], protocol["bootstrap_draws"], protocol["bootstrap_seed"])
    report = {
        "identity": identity,
        "protocol": protocol,
        "fit_lock_sha256": file_sha(output / "fit_lock.json"),
        "receipts_sha256": jsonl_out(output / "score_receipts.jsonl", rows),
        "scoring_native_calls": native.calls,
        **metrics,
    }
    if probe_native is not None:
        report.update(
            probe_native_calls=probe_native.calls,
            probe_receipts_sha256=jsonl_out(output / "probe_receipts.jsonl", probes),
            probes=summarize_probes(probes),
        )
    json_out(output / "report.json", report)
    print(f"scored {len(rows)} attempts across {metrics['roots']} roots and {native.calls} native calls", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("allocate", "fit", "score"))
    parser.add_argument("--family", choices=("development", "confirmation"), required=True)
    parser.add_argument("--storyworld", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    options = parser.parse_args()
    protocol = plan(options.family)
    upstream, sources = load_storyworld(options.storyworld)
    manifest_file = HERE / f"{options.family}_split.json"
    if options.phase == "allocate":
        prior = (
            load_manifest(HERE / "development_split.json", plan("development"))["roots"]
            if (options.family == "confirmation")
            else None
        )
        write_manifest(manifest_file, allocate(upstream, protocol, prior))
        return
    if options.out is None:
        parser.error("fit and score require --out")

    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    native = NativeCourier(upstream)
    manifest = load_manifest(manifest_file, protocol)
    identity = provenance(native, sources, protocol, manifest)
    run = prepare if options.phase == "fit" else score
    run(native, upstream, protocol, manifest, identity, options.out)


if __name__ == "__main__":
    main()
