import json
from collections import Counter

import pytest

from ocqa.corpus import CorpusError, load_corpus


def test_corpus_loads_expected_size(chunks):
    # 17 FAQ + 83 blog + however many help-channel chunks a human has
    # approved (14 at the time of writing; the count grows with review).
    by_source = Counter(chunk.source_type for chunk in chunks)
    assert by_source["faq"] == 17
    assert by_source["blog"] == 83
    assert by_source["help_channel"] >= 1


def test_ids_unique(chunks):
    ids = [chunk.id for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_every_chunk_has_url_and_hash(chunks):
    for chunk in chunks:
        assert chunk.url.startswith("https://")
        assert len(chunk.content_hash) == 16


def test_pending_help_chunks_are_never_loaded(tmp_path):
    pending = {
        "id": "help:1",
        "source_type": "help_channel",
        "status": "pending",
        "title": "q",
        "text": "q\n\na",
        "url": "https://oc.app/x",
        "meta": {},
        "provenance": {},
        "content_hash": "a" * 16,
    }
    approved = {**pending, "id": "help:2", "status": "approved"}
    path = tmp_path / "help.jsonl"
    path.write_text(json.dumps(pending) + "\n" + json.dumps(approved) + "\n")

    loaded = load_corpus(tmp_path)
    assert [chunk.id for chunk in loaded] == ["help:2"]


def test_duplicate_ids_are_a_hard_error(tmp_path):
    row = {
        "id": "faq:x",
        "source_type": "faq",
        "title": "q",
        "text": "t",
        "url": "https://oc.app/x",
        "meta": {},
        "provenance": {},
        "content_hash": "a" * 16,
    }
    path = tmp_path / "faq.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")

    with pytest.raises(CorpusError, match="duplicate"):
        load_corpus(tmp_path)
