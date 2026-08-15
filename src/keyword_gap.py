"""
src/keyword_gap.py

Extracts technical skills from a job description and classifies each
skill as present or missing based on the preprocessed resume text.

Module-level constant:
    SKILLS_LIST -- predefined list of technical skills for matching

Functions:
    extract_skills_from_jd -- find SKILLS_LIST entries in a JD string
    classify_skills        -- split skills into present / missing
"""

# ---------------------------------------------------------------------------
# Predefined skills list (source of truth for extraction and classification)
# ---------------------------------------------------------------------------

SKILLS_LIST: list[str] = [
    # Programming languages
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
    # Databases
    "SQL", "NoSQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
    # ML / Data Science
    "Machine Learning", "Deep Learning", "Natural Language Processing",
    "Computer Vision", "Data Science", "Statistics",
    "TensorFlow", "PyTorch", "scikit-learn", "Keras", "Pandas", "NumPy",
    # Web / backend frameworks
    "React", "Vue", "Angular", "Node.js", "Django", "Flask", "FastAPI",
    # DevOps / Cloud
    "Docker", "Kubernetes", "AWS", "GCP", "Azure", "Terraform",
    # Tooling / practices
    "Git", "CI/CD", "REST API", "GraphQL", "Microservices",
    "Linux", "Bash", "Agile", "Scrum",
]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def extract_skills_from_jd(jd_text: str) -> list[str]:
    """
    Identify skills from SKILLS_LIST that appear in the job description.

    Uses case-insensitive substring matching against the predefined list.
    Each matched skill is returned with the exact casing from SKILLS_LIST.
    Results are deduplicated; order follows SKILLS_LIST order.

    Args:
        jd_text: Raw job description text (not preprocessed).

    Returns:
        Deduplicated list of matched skill strings using SKILLS_LIST casing.
        Returns [] if jd_text is empty, whitespace-only, or no skills match.
    """
    if not jd_text or not jd_text.strip():
        return []

    jd_lower = jd_text.lower()
    seen: set[str] = set()
    matched: list[str] = []

    for skill in SKILLS_LIST:
        if skill.lower() in jd_lower and skill not in seen:
            matched.append(skill)
            seen.add(skill)

    return matched


def classify_skills(
    skills: list[str],
    preprocessed_resume: str,
) -> tuple[list[str], list[str]]:
    """
    Split a list of skills into present and missing based on resume content.

    A skill is classified as present when its lowercase form appears as a
    substring in the preprocessed resume text (also lowercased). Every input
    skill appears in exactly one output list — no skill is omitted or
    duplicated across both lists.

    Args:
        skills: List of skill strings, typically from extract_skills_from_jd.
                Casing is preserved in both output lists.
        preprocessed_resume: Preprocessed resume text from preprocess_text().

    Returns:
        A tuple (present_skills, missing_skills) where:
          - present_skills: skills whose lowercase form is found in the resume.
          - missing_skills: skills whose lowercase form is not found.
        Returns ([], []) if skills is empty.
        Returns ([], list(skills)) if preprocessed_resume is empty.
    """
    if not skills:
        return [], []

    if not preprocessed_resume:
        return [], list(skills)

    resume_lower = preprocessed_resume.lower()
    present: list[str] = []
    missing: list[str] = []

    for skill in skills:
        if skill.lower() in resume_lower:
            present.append(skill)
        else:
            missing.append(skill)

    return present, missing
