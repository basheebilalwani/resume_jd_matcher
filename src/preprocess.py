"""
src/preprocess.py

Cleans and normalises raw text for NLP analysis.

Pipeline applied by preprocess_text():
  1. Lowercase all characters
  2. Remove all characters in string.punctuation
  3. Tokenise with nltk.word_tokenize
  4. Remove NLTK English stop words
  5. Lemmatise each token with WordNetLemmatizer
  6. Join tokens with a single space and strip whitespace

Run `python scripts/download_nltk_data.py` once before first use to
ensure the required NLTK corpora are available locally.
"""
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def _check_nltk_resource(resource_path: str, download_name: str) -> None:
    """
    Verify that an NLTK resource is available locally.

    Args:
        resource_path: The NLTK resource path used with nltk.data.find()
                       (e.g. 'tokenizers/punkt_tab').
        download_name: The name passed to nltk.download() if missing
                       (e.g. 'punkt_tab').

    Raises:
        LookupError: If the resource is not present, with a message that
                     names the missing resource and the download command.
    """
    try:
        nltk.data.find(resource_path)
    except LookupError:
        raise LookupError(
            f"Required NLTK resource '{download_name}' is not available locally. "
            f"Download it by running: python scripts/download_nltk_data.py  "
            f"(or: import nltk; nltk.download('{download_name}'))"
        )


def _verify_nltk_resources() -> None:
    """
    Check that all NLTK resources required by preprocess_text() are present.

    Raises:
        LookupError: If any required resource is missing, naming the first
                     missing resource and the command to fix it.
    """
    required = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]
    for resource_path, download_name in required:
        _check_nltk_resource(resource_path, download_name)


def _remove_punctuation(text: str) -> str:
    """
    Remove every character that appears in Python's string.punctuation.

    Args:
        text: Input string.

    Returns:
        String with all punctuation characters stripped out.
    """
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def _remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Filter out NLTK English stop words from a token list.

    Args:
        tokens: List of lowercase string tokens.

    Returns:
        New list containing only tokens that are not English stop words.
    """
    stop_words = set(stopwords.words("english"))
    return [token for token in tokens if token not in stop_words]


def _lemmatise(tokens: list[str]) -> list[str]:
    """
    Reduce each token to its dictionary base form using WordNetLemmatizer.

    Args:
        tokens: List of lowercase string tokens with stop words removed.

    Returns:
        New list where each token has been lemmatised (e.g. 'running' → 'run').
    """
    lemmatiser = WordNetLemmatizer()
    return [lemmatiser.lemmatize(token) for token in tokens]


def preprocess_text(text: str) -> str:
    """
    Normalise raw text for NLP analysis.

    Applies the following pipeline in order:
      1. Lowercase all characters.
      2. Remove all characters in Python's string.punctuation set.
      3. Tokenise into words using NLTK's word tokeniser.
      4. Remove English stop words using the NLTK stop-words corpus.
      5. Lemmatise each token using NLTK's WordNetLemmatizer.
      6. Join the resulting tokens with exactly one space.

    Args:
        text: Raw input string (resume text or job description text).

    Returns:
        Preprocessed string of space-joined tokens, with no leading or
        trailing whitespace. Returns "" if the input is empty, contains
        only punctuation, or all tokens are filtered out as stop words.

    Raises:
        LookupError: If a required NLTK resource (punkt_tab, stopwords,
                     wordnet) is not available locally. The error message
                     names the missing resource and the download command.

    Examples:
        >>> preprocess_text("Running tests for Python developers!")
        'run test python developer'
        >>> preprocess_text("")
        ''
        >>> preprocess_text("the and is")
        ''
    """
    if not text:
        return ""

    _verify_nltk_resources()

    # Step 1: lowercase
    lowercased = text.lower()

    # Step 2: remove punctuation
    no_punct = _remove_punctuation(lowercased)

    # Step 3: tokenise
    tokens = word_tokenize(no_punct)

    # Step 4: remove stop words
    tokens = _remove_stopwords(tokens)

    # Step 5: lemmatise
    tokens = _lemmatise(tokens)

    # Step 6: join — returns "" naturally if token list is empty
    return " ".join(tokens)
