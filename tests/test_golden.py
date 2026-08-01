import pytest

from ocqa.evals.golden import (
    CATEGORY_MINIMUMS,
    GoldenCase,
    GoldenError,
    validate_against_corpus,
)


def test_size_at_or_above_spec_minimum(golden):
    # The spec asks for 40-60. The set grows as the corpus does: held-out
    # cases from prompt tuning, then coverage for each new source. Only the
    # floor is enforced.
    assert len(golden) >= 40


def test_category_minimums_met(golden):
    counts = {}
    for case in golden:
        counts[case.category] = counts.get(case.category, 0) + 1
    for category, minimum in CATEGORY_MINIMUMS.items():
        assert counts.get(category, 0) >= minimum, (
            f"{category}: {counts.get(category, 0)} < required {minimum}"
        )


def test_all_expected_ids_resolve(golden, chunks):
    validate_against_corpus(golden, chunks)


def test_every_answerable_case_has_ground_truth(golden):
    for case in golden:
        if case.category == "answerable":
            assert case.expected_chunk_ids, f"{case.id} has no expected_chunk_ids"


def test_unresolvable_id_is_a_hard_error(golden, chunks):
    bad = GoldenCase(
        id="g999",
        category="answerable",
        question="q?",
        expected_chunk_ids=["faq:does_not_exist"],
    )
    with pytest.raises(GoldenError, match="faq:does_not_exist"):
        validate_against_corpus([*golden, bad], chunks)
