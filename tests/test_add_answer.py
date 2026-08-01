import json

from ocqa.add_answer import build_chunk, existing_ids, slugify
from ocqa.corpus import load_corpus
from ocqa.embeddings import content_key


def test_slugify():
    assert slugify("Can I change my username?") == "can_i_change_my_username"


def test_built_chunk_is_loadable_and_indexable(tmp_path):
    chunk = build_chunk(
        "Can I change my username?",
        "Yes, in profile settings.",
        "https://oc.app/faq",
        None,
        "manual:username",
    )
    (tmp_path / "manual.jsonl").write_text(chunk.model_dump_json() + "\n")

    loaded = load_corpus(tmp_path)
    assert [c.id for c in loaded] == ["manual:username"]
    assert loaded[0].source_type == "manual"
    # No status field, so it indexes immediately — unlike mined help chunks.
    assert loaded[0].indexable


def test_text_carries_question_and_answer():
    chunk = build_chunk("Q?", "A.", "https://oc.app", None, "manual:x")
    assert chunk.text == "Q?\n\nA."
    assert chunk.title == "Q?"


def test_content_hash_matches_the_embedding_cache_key():
    """If these drift, a hand-edited entry reuses a stale embedding."""
    chunk = build_chunk("Q?", "A.", "https://oc.app", None, "manual:x")
    assert chunk.content_hash == content_key(chunk.text)


def test_stale_content_hash_cannot_poison_the_cache():
    """The cache key comes from the text, so an edit that forgets to update
    content_hash still embeds afresh."""
    chunk = build_chunk("Q?", "A.", "https://oc.app", None, "manual:x")
    edited = chunk.model_copy(update={"text": "Q?\n\nA different answer."})
    assert content_key(edited.text) != content_key(chunk.text)


def test_existing_ids_reads_the_file(tmp_path):
    path = tmp_path / "manual.jsonl"
    path.write_text(json.dumps({"id": "manual:a"}) + "\n" + json.dumps({"id": "manual:b"}) + "\n")
    assert existing_ids(path) == {"manual:a", "manual:b"}


def test_existing_ids_when_absent(tmp_path):
    assert existing_ids(tmp_path / "nope.jsonl") == set()
