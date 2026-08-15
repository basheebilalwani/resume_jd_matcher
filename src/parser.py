"""
src/parser.py

Extracts raw text from uploaded resume files (PDF and DOCX).
All functions accept raw bytes and return a single string.
"""
import io

import pypdf
from docx import Document


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF file.

    Reads each page using pypdf.PdfReader and joins page text
    with a single newline character between pages.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.

    Returns:
        A single string containing all extracted text, pages joined by '\\n'.

    Raises:
        ValueError: If the bytes cannot be parsed as a valid PDF.
        RuntimeError: If the PDF contains no extractable text (e.g. image-only).
    """
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Could not read PDF file: {exc}") from exc

    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    full_text = "\n".join(pages).strip()

    if not full_text:
        raise RuntimeError(
            "No text could be extracted from the PDF. "
            "The file may contain only images or scanned content."
        )

    return full_text


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract text from a DOCX file.

    Reads each paragraph using python-docx and joins paragraph text
    with a single newline character between paragraphs.

    Args:
        file_bytes: Raw bytes of the uploaded DOCX file.

    Returns:
        A single string containing all extracted text, paragraphs joined by '\\n'.

    Raises:
        ValueError: If the bytes cannot be parsed as a valid DOCX file.
        RuntimeError: If the DOCX contains no extractable text.
    """
    try:
        doc = Document(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Could not read DOCX file: {exc}") from exc

    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]

    full_text = "\n".join(paragraphs).strip()

    if not full_text:
        raise RuntimeError(
            "No text could be extracted from the DOCX file. "
            "The document appears to be empty."
        )

    return full_text


def parse_resume(uploaded_file) -> str:
    """
    Dispatch to the correct text extractor based on the file extension.

    Accepts a Streamlit UploadedFile object (or any object with .name and
    .read() attributes). Reads the file bytes once and forwards them to
    either extract_text_from_pdf or extract_text_from_docx.

    Args:
        uploaded_file: An object with a .name attribute (str) and a
                       .read() method returning bytes.

    Returns:
        Extracted resume text as a single string of at least 1 character.

    Raises:
        ValueError: If the file extension is not .pdf or .docx.
        ValueError: If the file is corrupted or cannot be parsed.
        RuntimeError: If the file contains no extractable text.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        file_bytes = uploaded_file.read()
        return extract_text_from_pdf(file_bytes)
    elif filename.endswith(".docx"):
        file_bytes = uploaded_file.read()
        return extract_text_from_docx(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file format: '{uploaded_file.name}'. "
            "Only PDF (.pdf) and DOCX (.docx) files are accepted."
        )
