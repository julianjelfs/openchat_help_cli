import pytest

from ocqa.evals.metrics import mean, recall_at_k, reciprocal_rank


def test_recall_at_k_full_hit():
    assert recall_at_k(["a", "b"], ["a", "b", "c"], 2) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k(["a", "b"], ["a", "c", "b"], 2) == 0.5


def test_recall_at_k_miss():
    assert recall_at_k(["a"], ["x", "y"], 2) == 0.0


def test_recall_at_k_respects_cutoff():
    assert recall_at_k(["a"], ["x", "y", "a"], 2) == 0.0
    assert recall_at_k(["a"], ["x", "y", "a"], 3) == 1.0


def test_recall_undefined_without_expected():
    with pytest.raises(ValueError):
        recall_at_k([], ["a"], 5)


def test_reciprocal_rank_first():
    assert reciprocal_rank(["a"], ["a", "b"]) == 1.0


def test_reciprocal_rank_third():
    assert reciprocal_rank(["a"], ["x", "y", "a"]) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_first_expected_found():
    assert reciprocal_rank(["a", "b"], ["b", "a"]) == 1.0


def test_reciprocal_rank_miss_is_zero():
    assert reciprocal_rank(["a"], ["x", "y"]) == 0.0


def test_mean_empty_is_zero():
    assert mean([]) == 0.0


def test_alternates_credit_any_single_match():
    # "either of these chunks answers it" — finding one is a full hit.
    assert recall_at_k([["a", "b"]], ["b", "x"], 5) == 1.0


def test_alternates_do_not_double_count():
    assert recall_at_k([["a", "b"]], ["a", "b"], 5) == 1.0


def test_alternates_mixed_with_required():
    # One required chunk plus one alternates group.
    assert recall_at_k(["a", ["b", "c"]], ["a", "c"], 5) == 1.0
    assert recall_at_k(["a", ["b", "c"]], ["a", "x"], 5) == 0.5


def test_alternates_miss():
    assert recall_at_k([["a", "b"]], ["x", "y"], 5) == 0.0


def test_reciprocal_rank_accepts_any_alternate():
    assert reciprocal_rank([["a", "b"]], ["x", "b"]) == 0.5
