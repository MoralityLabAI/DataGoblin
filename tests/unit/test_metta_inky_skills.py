import json
import sys
from pathlib import Path

import pytest

from MeTTa import inky_skills


@pytest.mark.parametrize("name", inky_skills.skill_names())
def test_request_loads_one_skill_and_keeps_public_task_separate(monkeypatch, capsys, tmp_path: Path, name: str):
    task = tmp_path / "public task.md"
    public_task = "Public fixture: reviewer Alice, epoch 3, cost ≤ 2.\n"
    task.write_text(public_task, encoding="utf-8-sig")
    monkeypatch.setattr(
        sys, "argv", ["inky_skills", "request", name, "--model", "local-inky", "--task", str(task)]
    )

    assert inky_skills.main() == 0

    request = json.loads(capsys.readouterr().out)
    assert request["model"] == "local-inky"
    system, user = request["messages"]
    assert system["role"] == "system"
    assert inky_skills.load_skill(name)[1] in system["content"]
    for other in set(inky_skills.skill_names()) - {name}:
        assert inky_skills.load_skill(other)[1] not in system["content"]
    assert user == {"role": "user", "content": public_task}


def test_unknown_skill_cannot_load_a_relative_path():
    with pytest.raises(ValueError, match="unknown MeTTa skill"):
        inky_skills.load_skill("../metta-verify-rules")
