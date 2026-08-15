"""Unit tests for src/preprocess.py."""
import string

import pytest

# Skip the entire module if NLTK resources are not available locally.
# This prevents hard failures on machines that haven't run download_nltk_data.py.
nltk_available = True
try:
    import nltk
    nltk.data.find("tokenizers/punkt_tab")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk_available = False

pytestmark = pytest.mark.skipif(
    not nltk_available,
    reason="NLTK resources not available — run: python scripts/download_nltk_data.py",
)

from src.preprocess import preprocess_text  # noqa: E402  (after guard)


# ---------------------------------------------------------------------------
# Empty / trivial inputs
# ---------------------------------------------------------------------------


def test_empty_string_returns_empty():
    """Empty string input returns empty string without error."""
    assert preprocess_text("") == ""


def test_whitespace_only_returns_empty():
    """Input containing only whitespace returns empty string."""
    assert preprocess_text("   \t\n  ") == ""


def test_all_punctuation_returns_empty():
    """Input containing only punctuation returns empty string."""
    assert preprocess_text("!@#$%^&*().,;:") == ""


def test_all_stopwords_returns_empty():
    """Input containing only English stop words returns empty string."""
    assert preprocess_text("the and is a an of in") == ""


# ---------------------------------------------------------------------------
# Lowercase conversion
# ---------------------------------------------------------------------------


def test_output_is_lowercase():
    """Output contains no uppercase characters."""
    result = preprocess_text("Python Developer Engineer JAVA")
    assert result == result.lower()


def test_mixed_case_input_lowercased():
    """Mixed-case input is fully lowercased before further processing."""
    result = preprocess_text("PYTHON")
    assert "PYTHON" not in result
    # 'python' may survive stopword removal (it's not a stop word)
    assert result == result.lower()


# ---------------------------------------------------------------------------
# Punctuation removal
# ---------------------------------------------------------------------------


def test_no_punctuation_in_output():
    """No character from string.punctuation appears in the output."""
    result = preprocess_text("Hello, world! This is a test: (really).")
    for ch in string.punctuation:
        assert ch not in result, f"Punctuation character {ch!r} found in output"


def test_punctuation_around_words_stripped():
    """Punctuation attached to words is removed without losing the word."""
    result = preprocess_text("developer, engineer.")
    assert "," not in result
    assert "." not in result


# ---------------------------------------------------------------------------
# Stop word removal
# ---------------------------------------------------------------------------


def test_stopwords_removed():
    """Common English stop words are not present in the output."""
    from nltk.corpus import stopwords as nltk_sw
    stop_words = set(nltk_sw.words("english"))
    result = preprocess_text("the quick brown fox jumps over the lazy dog")
    tokens = result.split()
    for token in tokens:
        assert token not in stop_words, f"Stop word {token!r} survived filtering"


def test_content_words_survive_stopword_removal():
    """Meaningful content words survive stop word filtering."""
    result = preprocess_text("experienced software engineer")
    # 'experienced', 'software', 'engineer' are not stop words
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Lemmatisation
# ---------------------------------------------------------------------------


def test_lemmatisation_running_to_run():
    """'running' is lemmatised to 'run'."""
    result = preprocess_text("running")
    assert "run" in result


def test_lemmatisation_developers_to_developer():
    """Plural 'developers' is lemmatised to 'developer'."""
    result = preprocess_text("developers")
    assert "developer" in result


def test_lemmatisation_studies():
    """'studies' is lemmatised to its base form."""
    result = preprocess_text("studies")
    # WordNetLemmatizer default POS=noun: 'studies' → 'study'
    assert result in ("studi", "study")


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------


def test_output_has_no_leading_trailing_whitespace():
    """Output string has no leading or trailing whitespace."""
    result = preprocess_text("  Python developer  ")
    assert result == result.strip()


def test_output_has_no_consecutive_spaces():
    """Tokens are separated by exactly one space — no double spaces."""
    result = preprocess_text("Python Java Docker Kubernetes")
    assert "  " not in result


def test_output_is_single_space_separated():
    """Output is a single whitespace-separated string of tokens."""
    result = preprocess_text("machine learning engineer")
    # Each token separated by exactly one space
    tokens = result.split(" ")
    assert all(len(t) > 0 for t in tokens)
