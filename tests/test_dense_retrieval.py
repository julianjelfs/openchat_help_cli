"""Dense retrieval tests. No API calls — the embeddings client is faked."""

from types import SimpleNamespace

import numpy as np
import pytest

from ocqa.answering import ANSWER_RULES, RetrievalAnswerer
from ocqa.embeddings import OpenAIEmbedder, text_key
from ocqa.models import Chunk
from ocqa.retrieval import DenseRetriever


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        source_type="faq",
        title=chunk_id,
        text=text,
        url="https://oc.app/x",
        meta={},
        provenance={},
        content_hash=f"{abs(hash(text)):016x}"[:16],
    )


class FakeEmbeddings:
    """Deterministic embeddings: direction chosen per keyword."""

    def __init__(self):
        self.calls = 0

    def create(self, model, input):
        self.calls += 1
        data = []
        for text in input:
            lowered = text.lower()
            vector = [
                1.0 if "wallet" in lowered else 0.1,
                1.0 if "video" in lowered else 0.1,
                1.0 if "vote" in lowered else 0.1,
            ]
            data.append(SimpleNamespace(embedding=vector))
        return SimpleNamespace(data=data)


def make_embedder(tmp_path):
    client = SimpleNamespace(embeddings=FakeEmbeddings())
    return OpenAIEmbedder(client, model="fake-model", cache_dir=tmp_path), client


CHUNKS = [
    chunk("faq:wallet", "your wallet holds tokens"),
    chunk("blog:video", "video calls use daily rooms"),
    chunk("faq:voting", "vote on proposals with neurons"),
]


def test_dense_retrieval_ranks_by_similarity(tmp_path):
    embedder, _ = make_embedder(tmp_path)
    retriever = DenseRetriever(CHUNKS, embedder)
    top = retriever.retrieve("how does my wallet work", k=3)
    assert top[0][0].id == "faq:wallet"
    assert top[0][1] > top[1][1]


def test_embeddings_are_cached_across_instances(tmp_path):
    embedder, client = make_embedder(tmp_path)
    DenseRetriever(CHUNKS, embedder)
    assert client.embeddings.calls == 1

    # New embedder over the same cache dir: chunk vectors come from disk.
    embedder2, client2 = make_embedder(tmp_path)
    DenseRetriever(CHUNKS, embedder2)
    assert client2.embeddings.calls == 0


def test_query_embedding_cached_too(tmp_path):
    embedder, client = make_embedder(tmp_path)
    retriever = DenseRetriever(CHUNKS, embedder)
    retriever.retrieve("wallet?", k=1)
    retriever.retrieve("wallet?", k=1)
    assert client.embeddings.calls == 2  # 1 for chunks, 1 for the query


def test_only_missing_texts_are_embedded(tmp_path):
    embedder, _ = make_embedder(tmp_path)
    DenseRetriever(CHUNKS, embedder)

    extended = [*CHUNKS, chunk("faq:new", "a brand new chunk about wallets")]
    embedder2, client2 = make_embedder(tmp_path)
    DenseRetriever(extended, embedder2)
    assert client2.embeddings.calls == 1  # only the new chunk


def test_text_key_stable():
    assert text_key("abc") == text_key("abc")
    assert text_key("abc") != text_key("abd")


def test_retrieval_answerer_prompt_contains_only_top_k(tmp_path):
    embedder, _ = make_embedder(tmp_path)
    retriever = DenseRetriever(CHUNKS, embedder)

    captured = {}

    class FakeCompletions:
        def parse(self, **kwargs):
            captured.update(kwargs)
            draft = SimpleNamespace(
                parsed=SimpleNamespace(
                    response_type="answer",
                    answer="From your wallet.",
                    citations=["faq:wallet"],
                    confidence=0.9,
                ),
                refusal=None,
            )
            # model_dump-compatible shim not needed; RetrievalAnswerer reads attrs
            return SimpleNamespace(choices=[SimpleNamespace(message=draft)], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    answerer = RetrievalAnswerer(client, retriever, max_chunks=1)
    result = answerer.answer("how does my wallet work")

    assert result.strategy == "dense"
    messages = captured["messages"]
    assert messages[0]["content"] == ANSWER_RULES  # stable system prefix
    user = messages[1]["content"]
    assert 'id="faq:wallet"' in user
    assert 'id="blog:video"' not in user  # k=1: only the top chunk ships


def test_normalised_scores_bounded(tmp_path):
    embedder, _ = make_embedder(tmp_path)
    retriever = DenseRetriever(CHUNKS, embedder)
    for _, score in retriever.retrieve("voting", k=3):
        assert -1.0001 <= score <= 1.0001


def test_cache_file_written(tmp_path):
    embedder, _ = make_embedder(tmp_path)
    DenseRetriever(CHUNKS, embedder)
    assert (tmp_path / "embeddings-fake-model.npz").exists()
    archive = np.load(tmp_path / "embeddings-fake-model.npz")
    assert len(archive.files) == len(CHUNKS)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_k_respected(tmp_path, k):
    embedder, _ = make_embedder(tmp_path)
    retriever = DenseRetriever(CHUNKS, embedder)
    assert len(retriever.retrieve("anything", k=k)) == k


def test_citations_are_filtered_to_retrieved_chunks(tmp_path):
    """A hallucinated-but-plausible id must not reach the service, which
    treats an unresolvable citation as a hard failure."""
    embedder, _ = make_embedder(tmp_path)
    retriever = DenseRetriever(CHUNKS, embedder)

    class FakeCompletions:
        def parse(self, **kwargs):
            draft = SimpleNamespace(
                parsed=SimpleNamespace(
                    response_type="answer",
                    answer="From your wallet.",
                    # First is real and was retrieved; second never existed.
                    citations=["faq:wallet", "faq:wallet_does_not_exist"],
                    confidence=0.9,
                ),
                refusal=None,
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=draft)], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    answerer = RetrievalAnswerer(client, retriever, max_chunks=3)
    result = answerer.answer("how does my wallet work")

    assert result.citations == ["faq:wallet"]
    assert answerer.dropped_citations == 1


def test_bm25_finds_exact_jargon_dense_would_miss():
    """The case for hybrid: an identifier that appears verbatim in one chunk."""
    from ocqa.retrieval import BM25Retriever

    corpus = [
        chunk("faq:a", "The Pawt token index canister id is kbvhp-rqaaa-aaaak-aflnq-cai."),
        chunk("faq:b", "Tokens can be sent from your wallet to any address."),
        chunk("faq:c", "Diamond membership unlocks extra features."),
    ]
    top = BM25Retriever(corpus).retrieve("kbvhp-rqaaa-aaaak-aflnq-cai", k=3)
    assert top[0][0].id == "faq:a"
    assert top[0][1] > 0


def test_bm25_ignores_unknown_terms():
    from ocqa.retrieval import BM25Retriever

    corpus = [chunk("faq:a", "alpha beta"), chunk("faq:b", "gamma delta")]
    assert all(score == 0 for _, score in BM25Retriever(corpus).retrieve("zzz", k=2))


def test_rrf_promotes_chunks_both_retrievers_rank():
    """A chunk ranked 2nd by both should beat one ranked 1st by only one."""
    from ocqa.retrieval import HybridRetriever

    agreed, dense_only, lex_only = (
        chunk("c:agreed", "x"),
        chunk("c:dense", "y"),
        chunk("c:lex", "z"),
    )

    class Fake:
        def __init__(self, ordered):
            self.ordered = ordered

        def retrieve(self, question, k=10):
            return [(c, 1.0) for c in self.ordered][:k]

    hybrid = HybridRetriever(Fake([dense_only, agreed]), Fake([lex_only, agreed]))
    assert next(c.id for c, _ in hybrid.retrieve("q", k=3)) == "c:agreed"


def test_hybrid_dedupes_across_retrievers():
    from ocqa.retrieval import HybridRetriever

    shared = chunk("c:1", "x")

    class Fake:
        def retrieve(self, question, k=10):
            return [(shared, 1.0)]

    assert len(HybridRetriever(Fake(), Fake()).retrieve("q", k=5)) == 1
