"""Answer-eval harness tests. No API calls — the grader client is faked."""

from types import SimpleNamespace

from ocqa.answering import StubRefusalAnswerer
from ocqa.evals.answers import Grade, LLMGrader, deterministic_checks, run_eval
from ocqa.evals.golden import GoldenCase
from ocqa.models import Answer


def case(**overrides) -> GoldenCase:
    base = {
        "id": "g001",
        "category": "answerable",
        "question": "How do I buy CHAT?",
        "expected_chunk_ids": [],
        "must_mention": [],
        "must_not_mention": [],
    }
    return GoldenCase(**{**base, **overrides})


def answer(**overrides) -> Answer:
    base = {
        "text": "Buy CHAT on an exchange.",
        "citations": [],
        "refused": False,
        "confidence": 0.9,
        "strategy": "test",
    }
    return Answer(**{**base, **overrides})


GOOD_GRADE = Grade(
    grounded=True, correct=True, cited=True, appropriate_refusal=True, rationale="fine"
)


class FakeCompletions:
    """Mimics openai chat.completions.parse: Grade or Exception per call."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def parse(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        message = SimpleNamespace(parsed=outcome, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def fake_client(outcomes):
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(outcomes)))


def test_stub_answerer_always_refuses():
    result = StubRefusalAnswerer().answer("anything")
    assert result.refused
    assert result.citations == []
    assert "help channel" in result.text


def test_fabricated_citation_fails_deterministic_check():
    checks = deterministic_checks(case(), answer(citations=["faq:nope"]), {"faq:buychat"})
    assert not checks["citations_resolve"]
    assert checks["unresolved_citations"] == ["faq:nope"]


def test_must_mention_only_applies_to_non_refusals():
    the_case = case(must_mention=["exchange"])
    refused = answer(refused=True, text="I don't know.")
    assert deterministic_checks(the_case, refused, set())["must_mention_missing"] == []
    answered = answer(text="Use your wallet.")
    assert deterministic_checks(the_case, answered, set())["must_mention_missing"] == ["exchange"]


def test_must_not_mention_is_case_insensitive():
    the_case = case(must_not_mention=["HACKED"])
    checks = deterministic_checks(the_case, answer(text="you have been hacked"), set())
    assert checks["must_not_mention_hit"] == ["HACKED"]


def test_mention_check_ignores_spacing_and_hyphens():
    # 'homescreen' must match 'Add to Home Screen'; '30 day' must match '30-day'.
    the_case = case(must_mention=["homescreen", "30 day"])
    ok = answer(text="Use Add to Home Screen. Sessions last 30-day periods.")
    assert deterministic_checks(the_case, ok, set())["must_mention_missing"] == []


def test_grader_retries_once_then_gives_up():
    client = fake_client([ValueError("bad json"), ValueError("bad json")])
    grader = LLMGrader(client, model="test-model")
    assert grader.grade(case(), answer(), []) is None
    assert client.chat.completions.calls == 2


def test_grader_retry_succeeds_second_time():
    client = fake_client([ValueError("bad json"), GOOD_GRADE])
    grader = LLMGrader(client, model="test-model")
    grade = grader.grade(case(), answer(), [])
    assert grade == GOOD_GRADE
    assert client.chat.completions.calls == 2


def test_run_eval_aggregates_and_counts_failures(chunks):
    cases = [
        case(id="g001", category="answerable", must_mention=["exchange"]),
        case(id="g002", category="refusal", question="Delete my account?"),
    ]
    # One good grade per case.
    client = fake_client([GOOD_GRADE, GOOD_GRADE])
    report = run_eval(StubRefusalAnswerer(), LLMGrader(client, "test-model"), cases, chunks)

    assert report["overall"]["cases"] == 2
    assert report["overall"]["grading_failures"] == 0
    assert report["overall"]["citation_failures"] == 0
    # Stub refused everything, so must_mention was not applied.
    assert report["overall"]["must_mention_failures"] == 0
    assert set(report["by_category"]) == {"answerable", "refusal"}
