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

Required NLTK resources are downloaded automatically on first use if they
are not already present, so the application works in fresh environments
such as Streamlit Community Cloud without any manual setup step.
"""
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def _check_nltk_resource(resource_path: str, download_name: str) -> None:
    """
    Ensure an NLTK resource is available locally, downloading it if needed.

    First checks whether the resource already exists via ``nltk.data.find()``.
    If it is missing, downloads it silently using ``nltk.download()``.  If the
    download also fails (e.g. no internet connection), raises ``LookupError``
    with an actionable message so the caller can surface a clear error.

    Args:
        resource_path: The NLTK resource path passed to ``nltk.data.find()``
                       (e.g. ``'tokenizers/punkt_tab'``).
        download_name: The package name passed to ``nltk.download()``
                       (e.g. ``'punkt_tab'``).

    Raises:
        LookupError: If the resource is not present and cannot be downloaded,
                     with a message naming the missing resource.
    """
    try:
        nltk.data.find(resource_path)
    except LookupError:
        # Resource is absent — attempt a silent automatic download.
        success = nltk.download(download_name, quiet=True)
        if not success:
            raise LookupError(
                f"Required NLTK resource '{download_name}' is not available "
                f"and could not be downloaded automatically. "
                f"Download it manually by running: "
                f"python scripts/download_nltk_data.py  "
                f"(or: import nltk; nltk.download('{download_name}'))"
            )


def _verify_nltk_resources() -> None:
    """
    Ensure all NLTK resources required by preprocess_text() are available.

    Each resource is checked via ``_check_nltk_resource``, which downloads
    it automatically if it is missing.  Raises ``LookupError`` (with an
    actionable message) only if a resource cannot be downloaded.
    """
    required = [
        ("tokenizers/punkt",                     "punkt"),
        ("tokenizers/punkt_tab",                 "punkt_tab"),
        ("corpora/stopwords",                    "stopwords"),
        ("corpora/wordnet",                      "wordnet"),
        ("taggers/averaged_perceptron_tagger",   "averaged_perceptron_tagger"),
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

    Required NLTK resources are downloaded automatically on first use if
    they are not already present locally.

    Args:
        text: Raw input string (resume text or job description text).

    Returns:
        Preprocessed string of space-joined tokens, with no leading or
        trailing whitespace. Returns "" if the input is empty, contains
        only punctuation, or all tokens are filtered out as stop words.

    Raises:
        LookupError: If a required NLTK resource is missing and cannot be
                     downloaded (e.g. no internet connection).

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
