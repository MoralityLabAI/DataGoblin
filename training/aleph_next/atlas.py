"""Join frozen fit summaries to DataGoblin's discovery schema and offline gym catalog."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import jsonschema

from datagoblin.gym_metadata import CUSTODY, collate
from datagoblin.model_atlas import canonical, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://github.com/MoralityLabAI/DataGoblin/blob/"
# Two registrations share the arch weights, held frames and joint scoring pass.
SUITES = {
    "ALEPH-01": ("arch", "ARCH_PROTOCOL.md", "architecture_lab.py", "source-access and matched local control"),
    "ALEPH-02": ("arch", "ARCH_PROTOCOL.md", "architecture_lab.py", "correction magnitude and clock intervention"),
    "ALEPH-03": ("decoder", "DECODER_PROTOCOL.md", "decoder_lab.py", "matched learned side and receipt receivers"),
    "ALEPH-04": ("nonlinear", "NONLINEAR_PROTOCOL.md", "nonlinear_lab.py", "exhaustive distributed nonlinear game"),
    "ALEPH-05": ("continual", "CONTINUAL_PROTOCOL.md", "continual_lab.py", "continual parity and native LDT"),
    "ALEPH-06": ("scale", "SCALE_PROTOCOL.md", "scale_lab.py", "matched native mixed prefix frontier"),
    "ALEPH-mixed192": ("mixed192", "MIXED_REPLAY.md", "../aleph_rlt_placement/study.py", "native mixed 192 updates"),
}


def checked(path, base, expected):
    resolved = (base / path).resolve()
    if not resolved.is_relative_to(base.resolve()) or Path(path).is_absolute():
        raise ValueError("checkpoint outside study directory")
    with resolved.open("rb") as fp:
        if hashlib.file_digest(fp, "sha256").hexdigest() != expected:
            raise ValueError("local checkpoint digest mismatch")


def build(runs, output, commit):
    # Resolve and pin merge ancestry before turning any local paths into source links.
    rev = git("rev-parse", "--verify", commit + "^{commit}")
    schema = json.loads((ROOT / "schemas/model_atlas_record.schema.json").read_bytes())
    validator = jsonschema.Draft202012Validator(schema)
    output.mkdir(parents=True, exist_ok=False)
    models, modules, roles, checked_files = {}, [], [], set()
    for study, (directory, protocol, script, scope) in SUITES.items():
        path = "training/aleph_next/" + protocol
        impl = ((ROOT / "training/aleph_next" / script).resolve()).relative_to(ROOT).as_posix()
        blob = git("rev-parse", rev + ":" + path)
        git("rev-parse", rev + ":" + impl)
        src = dict(repository="MoralityLabAI/DataGoblin", commit=rev, path=path, git_blob_sha1=blob)
        uri = BASE + rev + "/" + path
        fits = json.loads((runs / directory / "atlas-input.json").read_bytes())
        references = {}
        for row in fits:
            h, key = row["sha256"], row["key"]
            local = (directory, row["checkpoint"], h)
            if local not in checked_files:
                checked(row["checkpoint"], runs / directory, h)
                checked_files.add(local)
            parent = row.get("parent")
            sender = row["mechanism"].get("sender_sha256")
            roles.append(
                dict(
                    study=study,
                    key=key,
                    sha256=h,
                    weight_parent=parent,
                    frozen_sender=sender,
                    training_seed=row["seed"],
                    steps=row["steps"],
                    local_relative_path=directory + "/" + row["checkpoint"],
                )
            )
            refs = references.setdefault(h, dict(sha256=h, training_seed=row["seed"], steps=row["steps"]))
            if refs.get("training_seed") != row["seed"]:
                refs.pop("training_seed", None)
            obs = dict(
                study_id=study,
                scope="receipt_observation",
                frozen_sender=sender,
                split="public-development" if directory in ("scale", "mixed192") else "registered-held",
                source_uri=uri,
                pointer=key,
                metrics=row["metrics"],
                denominators={"training_seeds_in_fit": 1},
                independent_verification=False,
                evidence="Locally measured and byte-checked study receipt",
            )
            if h in models:
                models[h]["phenotype"].append(obs)
                models[h]["provenance"]["study_roles"].append([study, key])
                continue
            models[h] = dict(
                schema_version="1.0",
                record_type="model",
                model_id="sha256:" + h,
                notation=key,
                family=directory,
                population_group="aleph-next",
                study_id=study,
                genotype=dict(
                    architecture=row["architecture"],
                    parameter_count=row["params"] or None,
                    passes=row["passes"] or None,
                    implementation_uri=BASE + rev + "/" + impl,
                ),
                training_recipe=dict(
                    declared={"protocol_uri": uri, "updates": row["steps"]},
                    training_seed=row["seed"],
                    reported_fit_receipt={"key": key},
                ),
                phenotype=[obs],
                mechanistic_profile=row["mechanism"],
                causal_evidence={"design": scope, "limitations": row["failures"]},
                formal_properties={"oracle_or_verifier": uri},
                lineage={"weight_parent": parent},
                checkpoint=dict(
                    reported_sha256=h,
                    verified_bytes=False,
                    archive_directory_uri=None,
                    local_bytes_verified_sha256=True,
                ),
                provenance=dict(study_roles=[[study, key]], source=src),
                **CUSTODY,
            )
        modules.append(
            dict(
                id=study,
                family="aleph-next",
                kind="trained_gym",
                scope=scope,
                sources=[src],
                architecture="see model identities",
                recipe=uri,
                metrics="see phenotype receipts",
                checkpoints=list(references.values()),
                limits="Finite registered study, discovery custody",
            )
        )
    stream = runs / "research-stream-v2"
    stream_inventory = json.loads((stream / "inventory.json").read_bytes())
    checked("audit-receipts.jsonl", stream, stream_inventory["receipts_sha256"])
    participants = set()
    with (stream / "audit-receipts.jsonl").open(encoding="utf-8") as receipts:
        for line in receipts:
            event = json.loads(line)
            if event["split"] != "train":
                raise ValueError("unexpected corpus split")
            participants.add(event["sender"])
            if "receiver" in event:
                participants.add(event["receiver"])
            participants.update(event.get("receivers", {}).values())
    source_path = "training/aleph_next/CORPUS.md"
    modules.append(
        dict(
            id="ALEPH-researcher-population",
            family="aleph-next",
            kind="checkpoint_reuse",
            scope="Train-only finite posterior verification and costed receiver choices",
            sources=[
                dict(
                    repository="MoralityLabAI/DataGoblin",
                    commit=rev,
                    path=source_path,
                    git_blob_sha1=git("rev-parse", rev + ":" + source_path),
                )
            ],
            architecture="Frozen C, D and MLP producers across heterogeneous calibrated readers",
            recipe=BASE + rev + "/" + source_path,
            metrics=stream_inventory,
            checkpoints=[dict(sha256=h) for h in sorted(participants)],
            limits="Generated from public training grammar. Byte tokenizer is pilot-only.",
        )
    )
    joint_source = "training/aleph_next/JOINT_QUERY.md"
    modules.append(
        dict(
            id="ALEPH-joint-query",
            family="aleph-next",
            kind="synthetic_gym",
            scope="Exact exploratory three-worker pair selection and finite counterexample",
            sources=[
                dict(
                    repository="MoralityLabAI/DataGoblin",
                    commit=rev,
                    path=joint_source,
                    git_blob_sha1=git("rev-parse", rev + ":" + joint_source),
                )
            ],
            architecture="Three decentralized one-bit workers",
            recipe="Enumerate eight uniform worlds",
            metrics="Conditional Bayes Brier risk and greedy regret per slate",
            checkpoints=[],
            limits="Analytical game, no training; selection fixed before observations.",
        )
    )
    records = [models[h] for h in sorted(models)]
    for model in records:
        validator.validate(model)
    sources = output / "gym-sources.json"
    sources.write_bytes(canonical(dict(schema_version="1.0", modules=modules)) + b"\n")
    gyms, index = collate(sources)
    inventory = dict(
        fit_roles=len(roles),
        byte_checked_local_files=len(checked_files),
        distinct_model_ids=len(records),
        gym_modules=len(gyms),
        source_commit=rev,
        training_examples_from_held_results=0,
        digests={
            name: write_jsonl(output / name, data)
            for name, data in [
                ("models.jsonl", records),
                ("gym_modules.jsonl", gyms),
                ("checkpoint_index.jsonl", index),
                ("fit_roles.jsonl", roles),
            ]
        },
    )
    (output / "inventory.json").write_bytes(canonical(inventory) + b"\n")
    return inventory


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("runs", "out"):
        cli.add_argument("--" + arg, type=Path, required=True)
    cli.add_argument("--source-commit", required=True)
    a = cli.parse_args()
    print(build(a.runs, a.out, a.source_commit))
