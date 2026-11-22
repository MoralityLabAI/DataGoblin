"""AIRIS-02: fresh, disjoint synthetic curation roots and matched policy arms.

Root identity v2 is disjoint from the original v1 seeds 0..511 and the
AIRIS-01 confirmation cases. This is a CPU research dataset; every record is
training_ready=false until independently admitted. Steward roots have no
teacher labels and are never materialized by this module.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .airis_counterfactual import (
    MECHANISMS,
    apply_intervention,
    choose_guided,
    choose_uniform,
    digest,
    propose_interventions,
)
from .collect import ROOT, sha256_bytes
from .moses_control import Program
from .pattern_miner import generate_observations, mine_motifs, structural_features
from .sft_oracle import (
    FAMILIES, OOD_FAMILY, evaluate_subset, exact_optimum,
    first_state, make_task, next_edit,
)

SCHEMA = "metta-airis02-fresh-root-v1"
ARMS = ("uniform", "motif", "budget_first", "moses_gate")
TRAIN_RANGE = range(512, 704)
DEV_RANGE = range(704, 768)
PUBLIC_TEST_RANGE = range(768, 1024)
STEWARD_RANGE = range(1024, 1152)
SOURCE_FAMILIES = tuple(f for f in FAMILIES if f != OOD_FAMILY)
DEFAULT_OUT = ROOT / "runs" / "metta" / "airis02_fresh"

# Frozen from the original bounded MOSES control's documented historical run.
# This small Boolean program predicts initial invalidity. It is not a causal
# intervention selector. The mapping below is a pre-registered policy control.
FROZEN_MOSES_PROGRAM = Program("or", ("!selected_le_2", "pool_key_collisions"))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def root_id(family: str, seed: int) -> str:
    return f"metta-airis02-v1:{family}:{seed:06d}"


def partition(family: str, seed: int) -> str:
    if family not in SOURCE_FAMILIES or type(seed) is not int:
        raise ValueError("reserved family or invalid root seed")
    if seed in TRAIN_RANGE:
        return "train_candidate"
    if seed in DEV_RANGE:
        return "development"
    if seed in PUBLIC_TEST_RANGE:
        return "public_test_reserved"
    if seed in STEWARD_RANGE:
        return "steward_reserved"
    raise ValueError("seed outside registered AIRIS-02 source ranges")


def _frozen_motifs() -> tuple[tuple[str, ...], ...]:
    # This is the previously published v1 training partition, not a v2 root.
    prior_train = generate_observations(roots_per_family=512)
    selected = mine_motifs(prior_train, min_support=20, top_k=24)
    return tuple(tuple(row["features"]) for row in selected["candidates"])


def _select(
    *, arm: str, options: list, features: tuple[str, ...],
    motifs: tuple[tuple[str, ...], ...], root: str, seed: int,
):
    if arm == "uniform":
        return choose_uniform(options, root=root, seed=20261122 + seed)
    if arm == "motif":
        return choose_guided(options, features, list(motifs))
    if arm == "budget_first":
        return next((i for i in options if i.mechanism == "budget"), options[0])
    if arm == "moses_gate":
        if FROZEN_MOSES_PROGRAM.predict(features):
            return next((i for i in options if i.mechanism == "budget"), options[0])
        return next((i for i in options if i.mechanism == "lineage"),
                    next((i for i in options if i.mechanism == "eligibility"), options[0]))
    raise ValueError("unknown intervention arm")


def build_row(
    family: str, seed: int, arm: str, motifs: tuple[tuple[str, ...], ...],
) -> dict[str, Any]:
    split = partition(family, seed)
    if split in ("steward_reserved", "public_test_reserved"):
        raise ValueError("reserved roots cannot be materialized or labeled here")
    if arm not in ARMS:
        raise ValueError("unknown intervention arm")
    root = root_id(family, seed)
    task = make_task(family, seed)
    selected = first_state(task, seed)
    options = propose_interventions(task, selected)
    if not options:
        raise ValueError("fresh root lacks any structural intervention")
    public = structural_features(task, selected)
    proposal = _select(arm=arm, options=options, features=public, motifs=motifs,
                       root=root, seed=seed)
    changed = apply_intervention(task, proposal)
    if digest(task) == digest(changed):
        raise ValueError("one-fact edit leaves world unchanged")
    before = evaluate_subset(task, selected)
    after = evaluate_subset(changed, selected)
    verified = exact_optimum(changed)
    chosen = next_edit(selected, verified["selected"])
    if chosen["op"] not in ("add", "remove", "stop"):
        raise ValueError("invalid exact-oracle repair label")
    # Oracle labels and selected answers stay strictly inside train/dev rows.
    return {
        "schema_version": SCHEMA,
        "root": root,
        "family": family,
        "split": split,
        "arm": arm,
        "source_seed": seed,
        "source_world_sha256": digest(task),
        "intervened_world_sha256": digest(changed),
        "source_lineage_sha256": digest(root),
        "source_kind": "new_synthetic_curation_world",
        "public_world": changed,
        "selected": selected,
        "intervention": {"mechanism": proposal.mechanism, "field": proposal.field,
                         "item_id": proposal.item_id, "before": proposal.before,
                         "after": proposal.after},
        "source_violations": before["violations"],
        "changed_violations": after["violations"],
        "oracle_next_edit": chosen,
        "oracle_optimum_sha256": digest(verified["selected"]),
        "oracle_optimum": verified["selected"],
        "oracle_checked_subsets": verified["checked_subsets"],
        "single_fact_checked": True,
        "native_hyperon_verified": False,
        "training_ready": False,
        "presentation_number": ARMS.index(arm),
        "max_total_presentations_per_root": len(ARMS),
        "new_model_weights": False,
        "model_label_authority": "independent_bounded_synthetic_python_oracle",
    }


def make_rows(split: str, *, roots_per_family: int,
              motifs: tuple[tuple[str, ...], ...]) -> list[dict[str, Any]]:
    if split not in ("train_candidate", "development"):
        raise ValueError("only train and public dev material may be generated")
    horizon = TRAIN_RANGE if split == "train_candidate" else DEV_RANGE
    if not 1 <= roots_per_family <= len(horizon):
        raise ValueError("root budget exceeds registered partition")
    rows = [
        build_row(family, seed, arm, motifs)
        for family in SOURCE_FAMILIES
        for seed in range(horizon.start, horizon.start + roots_per_family)
        for arm in ARMS
    ]
    roots = Counter(x["root"] for x in rows)
    if any(n != len(ARMS) for n in roots.values()):
        raise ValueError("inconsistent four-arm root exposure")
    if any(x["split"] != split or x["training_ready"] is not False for x in rows):
        raise ValueError("train or development custody drift")
    if len(rows) != len(roots) * len(ARMS):
        raise ValueError("unequal data exposures")
    return rows


def steward_root_manifest(*, count_per_family: int = 64) -> dict[str, Any]:
    if not 1 <= count_per_family <= len(STEWARD_RANGE):
        raise ValueError("steward horizon exceeds registered reserve")
    # Names only: no programs, oracle answers, labels, or task contents.
    roots = [root_id(f, s) for f in SOURCE_FAMILIES
             for s in range(STEWARD_RANGE.start, STEWARD_RANGE.start + count_per_family)]
    return {
        "schema_version": "metta-airis02-steward-registration-v1",
        "root_count": len(roots),
        "family_count": len(SOURCE_FAMILIES),
        "seed_range_start": STEWARD_RANGE.start,
        "root_ids_sha256": sha256_bytes("\n".join(roots).encode("utf-8")),
        "public_only_identity_registration": True,
        "labels_generated": 0,
        "worlds_opened": 0,
        "not_cryptographically_sealed": True,
        "training_ready": False,
    }


def build_bundle(out: Path, *, train_roots_per_family: int = 32,
                 dev_roots_per_family: int = 12) -> dict[str, Any]:
    if out.resolve().is_relative_to((ROOT / "MeTTa").resolve()):
        raise ValueError("generated data must remain outside source tree")
    motifs = _frozen_motifs()
    if not motifs:
        raise ValueError("frozen development source produced no motifs")
    train = make_rows("train_candidate", roots_per_family=train_roots_per_family, motifs=motifs)
    dev = make_rows("development", roots_per_family=dev_roots_per_family, motifs=motifs)
    if {r["root"] for r in train} & {r["root"] for r in dev}:
        raise ValueError("source roots cross train/dev partitions")
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for filename, rows in (("train_candidate.jsonl", train), ("development_labels_research.jsonl", dev)):
        path = out / filename
        path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")
        files[filename] = sha256_bytes(path.read_bytes())
    steward = steward_root_manifest()
    (out / "steward_registry.json").write_text(
        json.dumps(steward, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    files["steward_registry.json"] = sha256_bytes((out / "steward_registry.json").read_bytes())
    public_reserved_roots = [root_id(f, seed) for f in SOURCE_FAMILIES for seed in PUBLIC_TEST_RANGE]
    public_register = {
        "schema_version": "metta-airis02-public-test-registration-v1",
        "root_count": len(public_reserved_roots),
        "root_ids_sha256": sha256_bytes("\n".join(public_reserved_roots).encode()),
        "worlds_opened": 0,
        "teacher_labels_written": 0,
        "not_cryptographically_sealed": True,
        "training_ready": False,
    }
    (out / "public_test_registry.json").write_text(
        json.dumps(public_register, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    files["public_test_registry.json"] = sha256_bytes(
        (out / "public_test_registry.json").read_bytes()
    )
    manifest = {
        "schema_version": "metta-airis02-bundle-v1",
        "root_exposure_policy": "one_per_arm_max_four_per_root",
        "train_roots": len(train) // len(ARMS),
        "development_roots": len(dev) // len(ARMS),
        "train_presentations_total": len(train),
        "development_diagnostic_presentations": len(dev),
        "arms": list(ARMS),
        "roots_per_family_train": train_roots_per_family,
        "roots_per_family_dev": dev_roots_per_family,
        "frozen_motifs_sha256": digest(motifs),
        "frozen_moses_key": FROZEN_MOSES_PROGRAM.key,
        "teacher_labels_in_steward_registry": 0,
        "teacher_labels_in_public_test_registry": 0,
        "old_sft_seed_overlap": 0,
        "new_training_presentations_original_v1": 0,
        "training_ready": False,
        "native_hyperon_verified": False,
        "files": files,
        "source_sha256": sha256_bytes(Path(__file__).read_bytes()),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-roots", type=int, default=32)
    parser.add_argument("--dev-roots", type=int, default=12)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    manifest = build_bundle(args.out, train_roots_per_family=args.train_roots,
                            dev_roots_per_family=args.dev_roots)
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items()
                      if k in ("train_roots", "development_roots", "train_presentations_total",
                               "development_diagnostic_presentations", "training_ready")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
