"""Matched training presentations with a native MeTTa curation policy."""
from __future__ import annotations

from importlib.metadata import version

from MeTTa.ram_hillclimb import FAMILIES, digest
from MeTTa.ram_hillclimb_native import PACKAGE_VERSION
from MeTTa.ram_trajectory import metta_transitions

QUERY = (
    "\n!(match &self (RepairTransition $step (remove $id) $after) $step)"
    "\n!(match &self (RepairTransition $step (add $id) $after) $step)\n"
)


def curate(trajectories: list[dict]) -> tuple[dict, list[dict]]:
    """Choose an early removal and late addition while preserving a two-edit quota."""
    from hyperon import Environment, MeTTa

    if version("hyperon") != PACKAGE_VERSION:
        raise RuntimeError(f"Curation requires hyperon=={PACKAGE_VERSION}")
    datasets: dict[str, list[dict]] = {"chronological": [], "metta_curated": []}
    receipts = []
    for row in trajectories:
        if row["split"] != "train_candidate" or row["family"] not in FAMILIES:
            raise ValueError("Native curation accepts only non-conflict TRAIN trajectories")
        frames = row["transitions"]
        source = metta_transitions(row) + QUERY
        results = MeTTa(env_builder=Environment.custom_env(create_config=False, is_test=True)).run(source)
        remove, add = (sorted(int(str(atom)) for atom in result) for result in results[-2:])
        expected = [[frame["step"] for frame in frames if frame["next_edit"]["op"] == op] for op in ("remove", "add")]
        if [remove, add] != expected:
            raise ValueError("Native curation query disagrees with the checked trajectory")
        quota = min(2, len(frames) - 1)
        priority = [*remove[:1], *add[-1:], *sorted((*remove, *add))]
        selected = {
            "chronological": list(range(quota)),
            "metta_curated": sorted(list(dict.fromkeys(priority))[:quota]),
        }
        for name, indices in selected.items():
            datasets[name].append({**row, "transitions": [*(frames[index] for index in indices), frames[-1]]})
        receipts.append({
            "root": row["root"], "source_sha256": row["source_sha256"], "split": row["split"],
            "presentation": row["presentation"], "native_program_sha256": digest(source.encode()),
            "native_runtime": f"hyperon=={PACKAGE_VERSION}", "native_remove_steps": remove, "native_add_steps": add,
            "selected_steps": selected, "stop_step": frames[-1]["step"],
        })
    return datasets, receipts
