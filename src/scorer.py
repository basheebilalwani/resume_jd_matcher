"""
src/scorer.py

Computes similarity scores between preprocessed resume text and job
description text. All functions are pure (no side effects) and expect
preprocessed strings produced by src/preprocess.py.

Functions:
    compute_tfidf_score   -- TF-IDF cosine similarity (lexical overlap)
    compute_embedding_score -- Sentence-embedding cosine similarity (semantic)
    compute_match_score   -- Weighted combination of both scores
"""
import logging

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def compute_tfidf_score(resume_text: str, jd_text: str) -> float:
    """
    Compute TF-IDF cosine similarity between two preprocessed texts.

    Fits a single TfidfVectorizer on both texts together so they share
    the same vocabulary matrix, then computes cosine similarity between
    the two resulting vectors.

    Args:
        resume_text: Preprocessed resume string.
        jd_text: Preprocessed job description string.

    Returns:
        Cosine similarity as a float in [0.0, 1.0], rounded to 4 decimal
        places. Returns 0.0 if either input is empty or vectorisation fails.
    """
    if not resume_text or not jd_text:
        return 0.0

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        return round(float(score), 4)
    except Exception as exc:
        logger.error("TF-IDF vectorisation failed: %s", exc)
        return 0.0


def compute_embedding_score(
    resume_text: str,
    jd_text: str,
    model: SentenceTransformer,
) -> float:
    """
    Compute sentence-embedding cosine similarity between two preprocessed texts.

    Encodes both strings into dense vectors using the provided
    SentenceTransformer model, then computes cosine similarity between
    the two embeddings.

    Args:
        resume_text: Preprocessed resume string.
        jd_text: Preprocessed job description string.
        model: A pre-loaded SentenceTransformer instance. Must already be
               available locally — no network calls are made here.

    Returns:
        Cosine similarity as a float in [0.0, 1.0], rounded to 4 decimal
        places. Returns 0.0 if either input is empty.
    """
    if not resume_text or not jd_text:
        return 0.0

    resume_embedding = model.encode([resume_text])
    jd_embedding = model.encode([jd_text])
    score = cosine_similarity(resume_embedding, jd_embedding)[0][0]
    return round(float(score), 4)


def compute_match_score(tfidf_score: float, embedding_score: float) -> float:
    """
    Combine TF-IDF and embedding scores into a single match percentage.

    Uses a weighted average that favours semantic similarity:
        match = (tfidf_score * 0.4 + embedding_score * 0.6) * 100

    Args:
        tfidf_score: TF-IDF cosine similarity, must be in [0.0, 1.0].
        embedding_score: Embedding cosine similarity, must be in [0.0, 1.0].

    Returns:
        Match percentage as a float in [0.0, 100.0], rounded to 2 decimal
        places.

    Raises:
        ValueError: If either score is outside [0.0, 1.0], with a message
                    identifying the invalid value and the valid range.
    """
    if not (0.0 <= tfidf_score <= 1.0):
        raise ValueError(
            f"tfidf_score must be in [0.0, 1.0], got {tfidf_score!r}."
        )
    if not (0.0 <= embedding_score <= 1.0):
        raise ValueError(
            f"embedding_score must be in [0.0, 1.0], got {embedding_score!r}."
        )

    match = (tfidf_score * 0.4 + embedding_score * 0.6) * 100
    return round(match, 2)
