"""Unit tests for src/keyword_gap.py."""
import pytest

from src.keyword_gap import SKILLS_LIST, classify_skills, extract_skills_from_jd


# ---------------------------------------------------------------------------
# extract_skills_from_jd — empty / whitespace inputs
# ---------------------------------------------------------------------------


def test_extract_empty_string_returns_empty_list():
    """Empty JD string returns an empty list."""
    assert extract_skills_from_jd("") == []


def test_extract_whitespace_only_returns_empty_list():
    """Whitespace-only JD returns an empty list."""
    assert extract_skills_from_jd("   \t\n  ") == []


def test_extract_no_matching_skills_returns_empty_list():
    """JD with no recognisable skills returns an empty list."""
    assert extract_skills_from_jd("excellent communication and teamwork required") == []


# ---------------------------------------------------------------------------
# extract_skills_from_jd — case-insensitive matching
# ---------------------------------------------------------------------------


def test_extract_lowercase_jd_matches_skill():
    """Skills written in lowercase in the JD are matched case-insensitively."""
    result = extract_skills_from_jd("we need a python developer")
    assert "Python" in result


def test_extract_uppercase_jd_matches_skill():
    """Skills written in uppercase in the JD are matched case-insensitively."""
    result = extract_skills_from_jd("experience with DOCKER is required")
    assert "Docker" in result


def test_extract_mixed_case_jd_matches_skill():
    """Skills in mixed case in the JD are matched."""
    result = extract_skills_from_jd("knowledge of Machine LEARNING preferred")
    assert "Machine Learning" in result


def test_extract_preserves_skills_list_casing():
    """Returned skill strings use the casing from SKILLS_LIST, not the JD."""
    result = extract_skills_from_jd("experience with PYTHON and docker")
    # Regardless of JD casing, returned strings must match SKILLS_LIST exactly
    assert "Python" in result       # SKILLS_LIST casing
    assert "Docker" in result       # SKILLS_LIST casing
    assert "PYTHON" not in result   # JD casing must not bleed through
    assert "docker" not in result   # JD casing must not bleed through


# ---------------------------------------------------------------------------
# extract_skills_from_jd — deduplication
# ---------------------------------------------------------------------------


def test_extract_duplicate_mentions_deduplicated():
    """A skill mentioned multiple times in the JD appears only once in the result."""
    result = extract_skills_from_jd("Python experience required. Python is essential. Python preferred.")
    assert result.count("Python") == 1


def test_extract_result_has_no_duplicates():
    """The result list never contains duplicate entries."""
    jd = "Python Docker Python AWS Docker Git Python"
    result = extract_skills_from_jd(jd)
    assert len(result) == len(set(result))


# ---------------------------------------------------------------------------
# extract_skills_from_jd — multiple skills
# ---------------------------------------------------------------------------


def test_extract_multiple_skills_from_jd():
    """Multiple skills mentioned in a JD are all returned."""
    jd = "We require Python, Docker, and AWS experience."
    result = extract_skills_from_jd(jd)
    assert "Python" in result
    assert "Docker" in result
    assert "AWS" in result


def test_extract_returns_subset_of_skills_list():
    """All returned skills are members of SKILLS_LIST."""
    jd = "Looking for Python, React, and Kubernetes engineers."
    result = extract_skills_from_jd(jd)
    for skill in result:
        assert skill in SKILLS_LIST


# ---------------------------------------------------------------------------
# classify_skills — empty inputs
# ---------------------------------------------------------------------------


def test_classify_empty_skills_list_returns_two_empty_lists():
    """Empty skills list returns ([], [])."""
    present, missing = classify_skills([], "python developer")
    assert present == []
    assert missing == []


def test_classify_empty_resume_all_skills_missing():
    """Empty resume text causes all skills to be classified as missing."""
    skills = ["Python", "Docker"]
    present, missing = classify_skills(skills, "")
    assert present == []
    assert set(missing) == {"Python", "Docker"}


def test_classify_both_empty_returns_two_empty_lists():
    """Both empty inputs return ([], [])."""
    present, missing = classify_skills([], "")
    assert present == []
    assert missing == []


# ---------------------------------------------------------------------------
# classify_skills — present / missing classification
# ---------------------------------------------------------------------------


def test_classify_skill_present_in_resume():
    """A skill whose lowercase form appears in the resume is classified as present."""
    present, missing = classify_skills(["Python"], "experienced python developer")
    assert "Python" in present
    assert "Python" not in missing


def test_classify_skill_absent_from_resume():
    """A skill not in the resume is classified as missing."""
    present, missing = classify_skills(["Docker"], "python developer javascript")
    assert "Docker" in missing
    assert "Docker" not in present


def test_classify_mixed_present_and_missing():
    """Some skills present, some missing — each in exactly the right list."""
    skills = ["Python", "Docker", "AWS"]
    resume = "experienced python developer with aws cloud knowledge"
    present, missing = classify_skills(skills, resume)
    assert "Python" in present
    assert "AWS" in present
    assert "Docker" in missing


# ---------------------------------------------------------------------------
# classify_skills — partition invariant
# ---------------------------------------------------------------------------


def test_classify_partition_no_skill_in_both_lists():
    """No skill appears in both present and missing."""
    skills = ["Python", "Docker", "AWS", "Git"]
    resume = "python aws developer"
    present, missing = classify_skills(skills, resume)
    assert set(present).isdisjoint(set(missing))


def test_classify_partition_every_skill_accounted_for():
    """The union of present and missing equals the full input skills list."""
    skills = ["Python", "Docker", "AWS", "Git", "React"]
    resume = "python git developer"
    present, missing = classify_skills(skills, resume)
    assert set(present) | set(missing) == set(skills)


def test_classify_preserves_input_casing():
    """Skills in output lists preserve the casing of the input skills argument."""
    skills = ["Python", "Machine Learning"]
    resume = "python machine learning engineer"
    present, missing = classify_skills(skills, resume)
    # Should be "Python" not "python", "Machine Learning" not "machine learning"
    assert "Python" in present
    assert "Machine Learning" in present


def test_classify_case_insensitive_matching():
    """Matching is case-insensitive: uppercase skill found in lowercase resume."""
    present, missing = classify_skills(["Docker"], "docker container experience")
    assert "Docker" in present
    assert "Docker" not in missing
