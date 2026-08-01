"""StuffedAnswerer tests. No API calls — the OpenAI client is faked."""

from types import SimpleNamespace

from ocqa.answering import DraftAnswer, StuffedAnswerer, build_stuffed_system


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.last_kwargs = None

    def parse(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        message = SimpleNamespace(parsed=outcome, refusal=None)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def fake_client(outcomes):
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(outcomes)))


def draft(**overrides) -> DraftAnswer:
    base = {
        "response_type": "answer",
        "answer": "Buy CHAT on an exchange.",
        "citations": ["faq:buychat"],
        "confidence": 0.9,
    }
    return DraftAnswer(**{**base, **overrides})


def test_system_prompt_contains_every_chunk(chunks):
    system = build_stuffed_system(chunks)
    for chunk in chunks:
        assert f'id="{chunk.id}"' in system


def test_successful_answer_maps_fields(chunks):
    answerer = StuffedAnswerer(fake_client([draft()]), chunks)
    result = answerer.answer("How do I buy CHAT?")
    assert result.text == "Buy CHAT on an exchange."
    assert result.citations == ["faq:buychat"]
    assert not result.refused
    assert result.strategy == "stuffed"
    assert answerer.input_tokens == 100
    assert answerer.output_tokens == 20


def test_refusal_clears_citations(chunks):
    # The model refused but still returned citations — the contract says a
    # refusal carries none.
    answerer = StuffedAnswerer(
        fake_client([draft(response_type="refuse", citations=["faq:wallet"])]), chunks
    )
    result = answerer.answer("Unknown thing?")
    assert result.refused
    assert result.citations == []


def test_clarify_is_not_a_refusal_and_carries_no_citations(chunks):
    answerer = StuffedAnswerer(
        fake_client([draft(response_type="clarify", citations=["faq:wallet"])]), chunks
    )
    result = answerer.answer("How much does it cost?")
    assert not result.refused
    assert result.citations == []


def test_confidence_is_clamped(chunks):
    answerer = StuffedAnswerer(fake_client([draft(confidence=1.7)]), chunks)
    assert answerer.answer("q").confidence == 1.0


def test_double_parse_failure_becomes_refusal(chunks):
    client = fake_client([ValueError("bad"), ValueError("bad")])
    answerer = StuffedAnswerer(client, chunks)
    result = answerer.answer("q")
    assert result.refused
    assert result.citations == []
    assert result.confidence == 0.0
    assert answerer.parse_failures == 1
    assert client.chat.completions.calls == 2


def test_retry_succeeds_after_one_failure(chunks):
    client = fake_client([ValueError("bad"), draft()])
    answerer = StuffedAnswerer(client, chunks)
    result = answerer.answer("q")
    assert not result.refused
    assert answerer.parse_failures == 0


def test_question_goes_in_user_message_not_system(chunks):
    client = fake_client([draft()])
    answerer = StuffedAnswerer(client, chunks)
    answerer.answer("How do I buy CHAT?")
    messages = client.chat.completions.last_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "How do I buy CHAT?" not in messages[0]["content"]
    assert "How do I buy CHAT?" in messages[1]["content"]
