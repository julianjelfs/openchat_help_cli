from ocqa.retrieval import StubLexicalRetriever, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("Buy CHAT tokens!") == {"buy", "chat", "tokens"}


def test_retrieval_is_deterministic(chunks):
    retriever = StubLexicalRetriever(chunks)
    first = [chunk.id for chunk, _ in retriever.retrieve("how do I buy chat tokens", k=10)]
    second = [chunk.id for chunk, _ in retriever.retrieve("how do I buy chat tokens", k=10)]
    assert first == second


def test_known_query_hits_expected_chunk(chunks):
    retriever = StubLexicalRetriever(chunks)
    top = [
        chunk.id
        for chunk, _ in retriever.retrieve(
            "why am I charged a transaction fee when sending tokens", k=5
        )
    ]
    assert "faq:send_tokens" in top


def test_k_limits_results(chunks):
    retriever = StubLexicalRetriever(chunks)
    assert len(retriever.retrieve("wallet", k=3)) == 3
