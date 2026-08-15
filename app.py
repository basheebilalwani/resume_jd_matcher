"""
app.py

Streamlit entry point for the Resume-to-Job-Description Matcher.

Responsibilities:
  Task 8.1:
    - Load the SentenceTransformer model once per session via @st.cache_resource
    - Render the page title and description
    - Render the resume file upload widget (PDF/DOCX, max 10 MB)
    - Render the job description text area (max 10,000 characters)

  Task 8.2:
    - Validate inputs before running analysis
    - Orchestrate the full analysis pipeline
    - Display scores and keyword gap results

Run with: streamlit run app.py
"""

import streamlit as st
from sentence_transformers import SentenceTransformer

from src.keyword_gap import classify_skills, extract_skills_from_jd
from src.parser import parse_resume
from src.preprocess import preprocess_text
from src.scorer import compute_embedding_score, compute_match_score, compute_tfidf_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"
MAX_UPLOAD_MB = 10
MAX_JD_CHARS = 10_000


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model…")
def load_model() -> SentenceTransformer | None:
    """
    Load the SentenceTransformer model from the local cache.

    Uses @st.cache_resource so the model is loaded once per session and
    reused across Streamlit reruns without re-initialising.

    Returns:
        A SentenceTransformer instance, or None if the model is not found
        locally (OSError). The caller is responsible for checking the return
        value before use.
    """
    try:
        return SentenceTransformer(MODEL_NAME)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------


def configure_page() -> None:
    """Set Streamlit page metadata (title, icon, layout)."""
    st.set_page_config(
        page_title="Resume ↔ JD Matcher",
        page_icon="📄",
        layout="centered",
    )


# ---------------------------------------------------------------------------
# UI sections — Task 8.1
# ---------------------------------------------------------------------------


def render_header() -> None:
    """Render the page title and application description."""
    st.title("📄 Resume ↔ Job Description Matcher")
    st.markdown(
        "Upload your resume and paste a job description to see how well they "
        "align. The app calculates a **TF-IDF similarity score** (keyword "
        "overlap), a **semantic similarity score** (meaning and context), and "
        "an **overall match percentage**. It also identifies which technical "
        "skills from the job description are present or missing in your resume."
        "\n\n"
        "_All processing runs locally — no data is sent to any external service._"
    )
    st.divider()


def render_model_warning() -> None:
    """Display a clear instructional message when the embedding model is absent."""
    st.error(
        f"**Embedding model not found.**\n\n"
        f"The `{MODEL_NAME}` model could not be loaded from the local cache. "
        f"Download it once by running the following in your terminal:\n\n"
        f"```python\n"
        f"from sentence_transformers import SentenceTransformer\n"
        f"SentenceTransformer('{MODEL_NAME}')\n"
        f"```\n\n"
        f"Then restart the app with `streamlit run app.py`.",
        icon="🚫",
    )


def render_resume_uploader() -> object:
    """
    Render the resume file upload widget.

    Accepts PDF and DOCX files up to MAX_UPLOAD_MB in size. The size limit
    is enforced by Streamlit's server.maxUploadSize setting; this widget
    restricts accepted MIME types on the client side.

    Returns:
        The uploaded file object, or None if no file has been selected.
    """
    return st.file_uploader(
        label="Upload your resume",
        type=["pdf", "docx"],
        help=f"Accepted formats: PDF, DOCX. Maximum file size: {MAX_UPLOAD_MB} MB.",
        accept_multiple_files=False,
    )


def render_jd_input() -> str:
    """
    Render the job description text area.

    Returns:
        The job description text entered by the user (may be an empty string).
    """
    jd_text = st.text_area(
        label="Paste the job description",
        height=250,
        max_chars=MAX_JD_CHARS,
        placeholder="Paste the full job description here…",
        help=f"Maximum {MAX_JD_CHARS:,} characters.",
    )
    return jd_text


# ---------------------------------------------------------------------------
# Analysis pipeline — Task 8.2
# ---------------------------------------------------------------------------


def run_analysis(uploaded_file, jd_text: str, model: SentenceTransformer) -> dict:
    """
    Execute the full analysis pipeline and return all results.

    Pipeline order:
        parse_resume → preprocess_text (×2) → compute_tfidf_score
        → compute_embedding_score → compute_match_score
        → extract_skills_from_jd → classify_skills

    Args:
        uploaded_file: Streamlit UploadedFile object for the resume.
        jd_text: Raw job description string from the text area.
        model: Pre-loaded SentenceTransformer instance.

    Returns:
        Dictionary with keys:
            tfidf_score     (float)
            embedding_score (float)
            match_score     (float)
            present_skills  (list[str])
            missing_skills  (list[str])

    Raises:
        ValueError: Propagated from parse_resume for unsupported/corrupt files.
        RuntimeError: Propagated from parse_resume for image-only files.
        LookupError: Propagated from preprocess_text for missing NLTK resources.
    """
    # Step 1: extract raw text from the resume file
    raw_resume = parse_resume(uploaded_file)

    # Steps 2–3: preprocess both documents
    preprocessed_resume = preprocess_text(raw_resume)
    preprocessed_jd = preprocess_text(jd_text)

    # Steps 4–6: compute similarity scores
    tfidf_score = compute_tfidf_score(preprocessed_resume, preprocessed_jd)
    embedding_score = compute_embedding_score(preprocessed_resume, preprocessed_jd, model)
    match_score = compute_match_score(tfidf_score, embedding_score)

    # Steps 7–8: keyword gap analysis (uses raw JD, preprocessed resume)
    skills = extract_skills_from_jd(jd_text)
    present_skills, missing_skills = classify_skills(skills, preprocessed_resume)

    return {
        "tfidf_score": tfidf_score,
        "embedding_score": embedding_score,
        "match_score": match_score,
        "present_skills": present_skills,
        "missing_skills": missing_skills,
    }


def render_scores(tfidf_score: float, embedding_score: float, match_score: float) -> None:
    """
    Display the three similarity scores in a metric row.

    Args:
        tfidf_score: TF-IDF cosine similarity in [0.0, 1.0], 4 d.p.
        embedding_score: Embedding cosine similarity in [0.0, 1.0], 4 d.p.
        match_score: Combined match percentage in [0.0, 100.0], 2 d.p.
    """
    st.subheader("Match Scores")
    col1, col2, col3 = st.columns(3)
    col1.metric("TF-IDF Score", f"{tfidf_score:.4f}")
    col2.metric("Semantic Score", f"{embedding_score:.4f}")
    col3.metric("Overall Match", f"{match_score:.1f}%")


def render_skills(present_skills: list[str], missing_skills: list[str]) -> None:
    """
    Display the keyword gap results in two labelled sections.

    Args:
        present_skills: Skills found in the resume.
        missing_skills: Skills absent from the resume.
    """
    st.subheader("Keyword Gap Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Skills Found in Resume**")
        if present_skills:
            for skill in present_skills:
                st.markdown(f"- ✅ {skill}")
        else:
            st.caption("No matching skills detected.")

    with col2:
        st.markdown("**Skills Missing from Resume**")
        if missing_skills:
            for skill in missing_skills:
                st.markdown(f"- ❌ {skill}")
        else:
            st.caption("No skills gaps detected.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the full Streamlit UI."""
    configure_page()
    render_header()

    # Load model — halt with instructions if unavailable
    model = load_model()
    if model is None:
        render_model_warning()
        st.stop()

    # Input widgets (Task 8.1)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Resume")
        uploaded_file = render_resume_uploader()

    with col2:
        st.subheader("Job Description")
        jd_text = render_jd_input()

    # Analysis trigger (Task 8.2)
    st.divider()
    analyse_clicked = st.button("🔍 Analyse", use_container_width=True, type="primary")

    if analyse_clicked:
        # Input validation
        if not uploaded_file:
            st.warning("⚠️ Please upload a resume file before running the analysis.")
            st.stop()

        if not jd_text or not jd_text.strip():
            st.warning("⚠️ Please paste a job description before running the analysis.")
            st.stop()

        # Run pipeline with spinner
        try:
            with st.spinner("Analysing your resume…"):
                results = run_analysis(uploaded_file, jd_text, model)
        except ValueError as exc:
            st.error(f"**Could not read the resume file.** {exc}")
            st.stop()
        except RuntimeError as exc:
            st.info(f"**No text found in the resume.** {exc}")
            st.stop()
        except LookupError as exc:
            st.error(f"**Missing NLTK resource.** {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"**An unexpected error occurred.** {exc}")
            st.stop()

        # Display results
        st.divider()
        render_scores(
            results["tfidf_score"],
            results["embedding_score"],
            results["match_score"],
        )
        st.divider()
        render_skills(results["present_skills"], results["missing_skills"])


if __name__ == "__main__":
    main()
