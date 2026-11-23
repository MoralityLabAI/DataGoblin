"""Independent manifest fingerprint and schema shape checks for CI."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def roots_for(manifest: dict) -> tuple[str, ...]:
    return tuple(
        f"DG12/{split}/{family}/{index:04d}"
        for split, counts in manifest["split_roots"].items()
        for family, count in counts.items()
        for index in range(count)
    )


def root_seed(manifest: dict, root_id: str) -> int:
    key = (manifest["seed_namespace"] + "|" + root_id).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def validate() -> dict:
    manifest = json.loads((BASE / "roots_manifest.json").read_text())
    protocol = json.loads((BASE / "protocol.json").read_text())
    schema = json.loads((BASE / "RESULT_SCHEMA.json").read_text())
    roots = roots_for(manifest)
    digest = hashlib.sha256(("\\n".join(roots) + "\\n").encode()).hexdigest()
    assert digest == manifest["root_ids_sha256"], "root-ID manifest changed"
    assert len(roots) == len(set(roots)) == manifest["total_roots"]
    assert len({root_seed(manifest, r) for r in roots}) == len(roots)
    assert protocol["experiment_id"] == schema["properties"]["experiment_id"]["const"]
    assert protocol["evaluation"]["five_call_cap_per_episode_for_C_and_D"] == 5
    if protocol["status"] == "wiring_only_until_native_kernel_pinned":
        assert protocol["source"]["native_storyworld_full_commit"] is None
    else:
        native = protocol["source"]["native_storyworld_full_commit"]
        assert isinstance(native, str) and len(native) == 40
    assert set(manifest["split_roots"]["train"]) == set(manifest["split_roots"]["development"])
    assert set(manifest["split_roots"]["train"]) == set(manifest["split_roots"]["frozen"])
    assert not set(manifest["split_roots"]["train"]) & set(manifest["split_roots"]["transfer"])
    return {"roots": len(roots), "root_ids_sha256": digest,
            "status": protocol["status"]}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
