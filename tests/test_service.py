"""Service tests. No LLM — answerers are faked; the stub is real."""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from ocqa.answering import StubRefusalAnswerer
from ocqa.models import Answer
from ocqa.service import ServiceState, create_app


class FixedAnswerer:
    name = "fixed"

    def __init__(self, answer: Answer):
        self._answer = answer

    def answer(self, question: str) -> Answer:
        return self._answer


def make_client(chunks, extra_answerers=None):
    answerers = {"stub": StubRefusalAnswerer(), **(extra_answerers or {})}
    state = ServiceState(answerers, chunks, index_build_ms=42)
    return TestClient(create_app(state), raise_server_exceptions=False)


def good_answer(chunks) -> Answer:
    return Answer(
        text="Buy CHAT on an exchange.",
        citations=[chunks[1].id],
        refused=False,
        confidence=0.9,
        strategy="fixed",
        retrieved=[chunks[1].id, chunks[0].id],
    )


def test_health(chunks):
    client = make_client(chunks)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["corpus_chunks"] == 100
    assert body["index_build_ms"] == 42
    assert "stub" in body["strategies"]


def test_ask_stub_refusal_contract(chunks):
    client = make_client(chunks)
    body = client.post("/ask", json={"question": "Anything?", "strategy": "stub"}).json()
    assert body["refused"] is True
    assert body["citations"] == []
    assert "help channel" in body["answer"]
    assert body["strategy"] == "stub"
    assert isinstance(body["latency_ms"], int)


def test_ask_resolves_citations(chunks):
    client = make_client(chunks, {"fixed": FixedAnswerer(good_answer(chunks))})
    body = client.post("/ask", json={"question": "How do I buy CHAT?", "strategy": "fixed"}).json()
    assert body["refused"] is False
    citation = body["citations"][0]
    assert citation["chunk_id"] == chunks[1].id
    assert citation["url"].startswith("https://")
    assert citation["title"]
    assert citation["source_type"] in {"faq", "blog", "help_channel"}
    assert "published" in citation


def test_fabricated_citation_is_a_500(chunks):
    fabricated = Answer(
        text="Made up.",
        citations=["faq:does_not_exist"],
        refused=False,
        confidence=0.9,
        strategy="fixed",
    )
    client = make_client(chunks, {"fixed": FixedAnswerer(fabricated)})
    response = client.post("/ask", json={"question": "q?", "strategy": "fixed"})
    assert response.status_code == 500
    assert "faq:does_not_exist" in response.json()["detail"]


def test_unknown_strategy_rejected(chunks):
    client = make_client(chunks)
    response = client.post("/ask", json={"question": "q?", "strategy": "nope"})
    assert response.status_code == 422


def test_empty_question_rejected(chunks):
    client = make_client(chunks)
    assert client.post("/ask", json={"question": "", "strategy": "stub"}).status_code == 422


def test_max_chunks_bounds(chunks):
    client = make_client(chunks)
    bad = client.post("/ask", json={"question": "q?", "strategy": "stub", "max_chunks": 0})
    assert bad.status_code == 422


def test_request_logged_as_json(chunks, caplog):
    client = make_client(chunks)
    with caplog.at_level(logging.INFO, logger="ocqa.service"):
        client.post("/ask", json={"question": "log me", "strategy": "stub"})
    record = json.loads(caplog.records[-1].message)
    assert record["question"] == "log me"
    assert record["strategy"] == "stub"
    assert record["refused"] is True
    assert record["retrieved"] == []
    assert "latency_ms" in record


@pytest.mark.parametrize("payload", [{}, {"strategy": "stub"}])
def test_question_required(chunks, payload):
    client = make_client(chunks)
    assert client.post("/ask", json=payload).status_code == 422
