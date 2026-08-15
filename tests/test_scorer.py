"""Unit tests for src/scorer.py."""
import numpy as np
import pytest

from src.scorer import compute_embedding_score, compute_match_score, compute_tfidf_score


# ---------------------------------------------------------------------------
# compute_tfidf_score
# ---------------------------------------------------------------------------


def test_tfidf_identical_texts_returns_1():
    """Identical non-empty texts produce a TF-IDF score of 1.0."""
    score = compute_tfidf_score("python developer", "python developer")
    assert score == 1.0


def test_tfidf_completely_different_texts_returns_0():
    """Texts with no shared vocabulary produce a TF-IDF score of 0.0."""
    score = compute_tfidf_score("python developer", "accountant finance spreadsheet")
    assert score == 0.0


def test_tfidf_empty_resume_returns_0():
    """Empty resume text returns 0.0 without error."""
    assert compute_tfidf_score("", "python developer") == 0.0


def test_tfidf_empty_jd_returns_0():
    """Empty JD text returns 0.0 without error."""
    assert compute_tfidf_score("python developer", "") == 0.0


def test_tfidf_both_empty_returns_0():
    """Both inputs empty returns 0.0."""
    assert compute_tfidf_score("", "") == 0.0


def test_tfidf_partial_overlap_between_0_and_1():
    """Texts with partial vocabulary overlap produce a score strictly between 0 and 1."""
    score = compute_tfidf_score("python java developer", "python ruby engineer")
    assert 0.0 < score < 1.0


def test_tfidf_score_rounded_to_4_decimal_places():
    """Score is rounded to exactly 4 decimal places."""
    score = compute_tfidf_score("python java developer", "python ruby engineer")
    assert score == round(score, 4)


def test_tfidf_score_in_valid_range():
    """Score is always in [0.0, 1.0]."""
    score = compute_tfidf_score("machine learning engineer", "data scientist python")
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compute_embedding_score
# ---------------------------------------------------------------------------


def test_embedding_score_empty_resume_returns_0(mock_model):
    """Empty resume text returns 0.0 without calling the model."""
    assert compute_embedding_score("", "python developer", mock_model) == 0.0
    mock_model.encode.assert_not_called()


def test_embedding_score_empty_jd_returns_0(mock_model):
    """Empty JD text returns 0.0 without calling the model."""
    assert compute_embedding_score("python developer", "", mock_model) == 0.0
    mock_model.encode.assert_not_called()


def test_embedding_score_uses_mock_model(mock_model):
    """
    With the mock model (fixed identical embeddings), cosine similarity = 1.0.
    Verifies that encode() is called and the result is rounded to 4 d.p.
    """
    score = compute_embedding_score("python developer", "java engineer", mock_model)
    assert mock_model.encode.called
    # Fixed (1,384) array of 0.5 vs itself → cosine similarity = 1.0
    assert score == 1.0


def test_embedding_score_rounded_to_4_decimal_places(mock_model):
    """Score returned by compute_embedding_score is rounded to 4 decimal places."""
    score = compute_embedding_score("some text", "other text", mock_model)
    assert score == round(score, 4)


def test_embedding_score_in_valid_range(mock_model):
    """Score is always in [0.0, 1.0]."""
    score = compute_embedding_score("resume text here", "job description here", mock_model)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compute_match_score — formula and rounding
# ---------------------------------------------------------------------------


def test_match_score_formula_zeros():
    """Both scores 0.0 → match score 0.0."""
    assert compute_match_score(0.0, 0.0) == 0.0


def test_match_score_formula_ones():
    """Both scores 1.0 → match score 100.0."""
    assert compute_match_score(1.0, 1.0) == 100.0


def test_match_score_formula_40_60_weighting():
    """Formula applies 40% TF-IDF and 60% embedding weight."""
    # tfidf=1.0, embedding=0.0 → (1.0 * 0.4 + 0.0 * 0.6) * 100 = 40.0
    assert compute_match_score(1.0, 0.0) == 40.0
    # tfidf=0.0, embedding=1.0 → (0.0 * 0.4 + 1.0 * 0.6) * 100 = 60.0
    assert compute_match_score(0.0, 1.0) == 60.0


def test_match_score_formula_midpoint():
    """Verify exact formula output for an intermediate case."""
    # (0.5 * 0.4 + 0.5 * 0.6) * 100 = (0.2 + 0.3) * 100 = 50.0
    assert compute_match_score(0.5, 0.5) == 50.0


def test_match_score_rounded_to_2_decimal_places():
    """Result is rounded to exactly 2 decimal places."""
    score = compute_match_score(0.333, 0.666)
    assert score == round(score, 2)


def test_match_score_result_in_valid_range():
    """Result is always in [0.0, 100.0]."""
    score = compute_match_score(0.7, 0.8)
    assert 0.0 <= score <= 100.0


# ---------------------------------------------------------------------------
# compute_match_score — invalid inputs
# ---------------------------------------------------------------------------


def test_match_score_negative_tfidf_raises():
    """tfidf_score below 0.0 raises ValueError."""
    with pytest.raises(ValueError, match="tfidf_score"):
        compute_match_score(-0.1, 0.5)


def test_match_score_tfidf_above_1_raises():
    """tfidf_score above 1.0 raises ValueError."""
    with pytest.raises(ValueError, match="tfidf_score"):
        compute_match_score(1.5, 0.5)


def test_match_score_negative_embedding_raises():
    """embedding_score below 0.0 raises ValueError."""
    with pytest.raises(ValueError, match="embedding_score"):
        compute_match_score(0.5, -0.1)


def test_match_score_embedding_above_1_raises():
    """embedding_score above 1.0 raises ValueError."""
    with pytest.raises(ValueError, match="embedding_score"):
        compute_match_score(0.5, 1.5)
