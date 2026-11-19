from datagoblin.training_gym.rlt_lm_replication import select_windows


def test_heldout_selection_rejects_duplicate_documents_and_prior_windows() -> None:
    source = {
        "observed_repo_revision": "revision",
        "rows": [
            {"row_idx": 1000, "full_text_sha256": "old", "row": {"text": "abcdefgh"}},
            {"row_idx": 1001, "full_text_sha256": "new1", "row": {"text": "hi defg bye"}},
            {"row_idx": 1002, "full_text_sha256": "new2", "row": {"text": "xyzxyz"}},
        ],
    }
    seen_documents = {"old"}
    used_windows = {b"defg"}
    selected, rejected = select_windows(source, "demo", seen_documents, used_windows, count=1, width=4)
    assert [r["ordinal"] for r in selected] == [1002]
    assert rejected == {"document_hash": 1, "overlapping_window": 1}
    assert bytes(selected[0]["bytes"]) in used_windows
    assert seen_documents == {"old", "new2"}
