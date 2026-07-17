from __future__ import annotations

import json
from pathlib import Path

from datagoblin.domain.experiments import ExperimentRecord
from datagoblin.domain.research_episode import ResearchEpisode
from datagoblin.research.demo import run_demo
from datagoblin.serialization import load_structured

REQUIRED = {
    "run_manifest.json",
    "learner_snapshot.json",
    "scout_findings.json",
    "hypotheses.json",
    "recipe.original.yaml",
    "rat_critique.json",
    "recipe.repaired.yaml",
    "constitutional_decision.json",
    "build_receipt.json",
    "train_split_manifest.json",
    "evaluation_split_manifest.public.json",
    "experiment_record.json",
    "report.md",
    "ledger.sqlite",
}


def test_demo_completes_offline_and_hides_evaluation_seed(tmp_path: Path, snapshot) -> None:
    output = tmp_path / "demo"
    episode = run_demo(snapshot, 7, output)
    assert {path.name for path in output.iterdir()} >= REQUIRED
    assert (output / "tickets").is_dir()
    assert (output / "artifacts" / "sha256").is_dir()
    public_manifest = load_structured(output / "evaluation_split_manifest.public.json")
    assert public_manifest["private_seed_values_disclosed"] is False
    public_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in output.iterdir()
        if path.is_file() and path.suffix in {".json", ".yaml", ".md"}
    )
    assert "900008" not in public_text
    assert "hidden_evaluation_seed" not in public_text
    record = ExperimentRecord.from_jsonable(load_structured(output / "experiment_record.json"))
    assert record.run_status.value == "EVALUATED"
    assert record.claim_boundary.startswith("Toy smoke evidence")
    assert ResearchEpisode.from_jsonable(episode.to_jsonable()) == episode
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["report.md"]["sha256"]
