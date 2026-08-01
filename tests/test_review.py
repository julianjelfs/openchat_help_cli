import json

from ocqa.corpus import load_corpus
from ocqa.review import approve, load_decided, reject


def candidate(cid: str) -> dict:
    text = f"Question for {cid}\n\nAnswer for {cid}"
    return {
        "id": cid,
        "source_type": "help_channel",
        "status": "pending",
        "title": f"Question for {cid}",
        "text": text,
        "url": "https://oc.app/community/x/channel/1/2",
        "meta": {
            "question": f"Question for {cid}",
            "answer": f"Answer for {cid}",
            "confidence": 0.9,
            "stale": False,
            "answered_at": "2026-01-01T00:00:00+00:00",
        },
        "provenance": {"root_event_index": 1},
        "content_hash": "a" * 16,
    }


def test_approve_writes_approved_status_with_review_timestamp(tmp_path):
    out = tmp_path / "help.jsonl"
    approve(candidate("help:1"), out)
    row = json.loads(out.read_text())
    assert row["status"] == "approved"
    assert "reviewed_at" in row["provenance"]
    # Original provenance keys survive.
    assert row["provenance"]["root_event_index"] == 1


def test_decisions_are_remembered(tmp_path):
    out, rejects = tmp_path / "help.jsonl", tmp_path / "rejected.txt"
    approve(candidate("help:1"), out)
    reject("help:2", rejects)
    approved, rejected = load_decided(out, rejects)
    assert approved == {"help:1"}
    assert rejected == {"help:2"}


def test_approved_chunks_are_indexable_by_the_corpus_loader(tmp_path):
    out = tmp_path / "help.jsonl"
    approve(candidate("help:1"), out)
    # Pending sibling in the same directory must stay invisible.
    pending = candidate("help:2")
    (tmp_path / "candidates.jsonl").write_text(json.dumps(pending) + "\n")

    chunks = load_corpus(tmp_path)
    assert [chunk.id for chunk in chunks] == ["help:1"]
    assert chunks[0].status == "approved"
