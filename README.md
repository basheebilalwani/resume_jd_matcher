# Resume-to-Job-Description Matcher

A locally-run Streamlit application that quantifies how well a resume aligns with a job description. Upload a resume (PDF or DOCX), paste a job description, and the app produces two complementary similarity scores — a TF-IDF lexical score and a sentence-embedding semantic score — combines them into an overall match percentage, then identifies which technical skills from the job description are present or missing in the resume.

All processing runs on your local machine. No API keys, no cloud services, no data leaves your computer.

---

## Features

- **PDF and DOCX resume upload** — accepts both common resume formats up to 10 MB
- **Job description text input** — paste any job description directly into the app
- **Resume text extraction** — extracts text from every page (PDF) or paragraph (DOCX) using `pypdf` and `python-docx`
- **NLTK text preprocessing** — lowercasing, punctuation removal, stop-word filtering, and lemmatisation before comparison
- **TF-IDF cosine similarity** — measures vocabulary overlap between the preprocessed resume and job description
- **Semantic similarity** — encodes both documents with the `all-MiniLM-L6-v2` sentence-transformer model and computes cosine similarity to capture meaning beyond exact keyword matches
- **Combined match score** — weighted average of both scores (40% TF-IDF, 60% semantic) expressed as a percentage
- **Skill extraction** — identifies technical skills mentioned in the job description from a curated list of 49 technologies and tools
- **Skill gap analysis** — classifies each extracted skill as present in or missing from the resume
- **Input validation and error handling** — clear messages for missing inputs, unsupported file formats, corrupt files, and missing NLTK data
- **Fully local processing** — no external API calls at runtime; the embedding model runs from the local cache

---

## How It Works

1. **Resume parsing** — the uploaded file is read as bytes and passed to `src/parser.py`, which extracts plain text using `pypdf` (PDF) or `python-docx` (DOCX)
2. **Text preprocessing** — `src/preprocess.py` applies a five-step NLTK pipeline to both the resume and the job description: lowercase → remove punctuation → tokenise → remove stop words → lemmatise
3. **TF-IDF similarity** — `src/scorer.py` fits a single `TfidfVectorizer` on both preprocessed texts and computes their cosine similarity (0.0 – 1.0)
4. **Semantic embedding similarity** — both texts are encoded into 384-dimensional vectors using `all-MiniLM-L6-v2` via `sentence-transformers`; cosine similarity is computed between the two embeddings (0.0 – 1.0)
5. **Combined match score** — the two scores are merged as `(TF-IDF × 0.4 + Semantic × 0.6) × 100`, giving a percentage that weights semantic understanding more heavily than keyword overlap
6. **Skill extraction and gap analysis** — `src/keyword_gap.py` scans the raw job description for known technical skills using case-insensitive substring matching, then checks each skill against the preprocessed resume to produce a present/missing split
7. **Results display** — all scores and skill lists are rendered in the Streamlit interface with clear labels and section headings

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| [Python](https://www.python.org/) | 3.13+ | Runtime |
| [Streamlit](https://streamlit.io/) | 1.41.1 | Web UI and file upload |
| [scikit-learn](https://scikit-learn.org/) | 1.6.1 | TF-IDF vectorisation and cosine similarity |
| [Sentence Transformers](https://www.sbert.net/) | 3.3.1 | Semantic embedding with `all-MiniLM-L6-v2` |
| [NLTK](https://www.nltk.org/) | 3.9.2 | Tokenisation, stop-word removal, lemmatisation |
| [pypdf](https://pypdf.readthedocs.io/) | 4.2.0 | PDF text extraction |
| [python-docx](https://python-docx.readthedocs.io/) | 1.1.2 | DOCX text extraction |
| [pytest](https://docs.pytest.org/) | 8.2.2 | Unit and integration test runner |

---

## Project Structure

```
resume-jd-matcher/
├── app.py                        # Streamlit entry point and pipeline orchestrator
├── src/
│   ├── parser.py                 # PDF and DOCX text extraction
│   ├── preprocess.py             # NLTK text normalisation pipeline
│   ├── scorer.py                 # TF-IDF, embedding, and combined scoring
│   └── keyword_gap.py            # Skill extraction and gap classification
├── scripts/
│   ├── __init__.py
│   └── download_nltk_data.py     # One-time NLTK data bootstrap script
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Shared fixtures (mock SentenceTransformer)
│   ├── test_parser.py
│   ├── test_preprocess.py
│   ├── test_scorer.py
│   └── test_keyword_gap.py
├── .streamlit/
│   └── config.toml               # Streamlit server config (10 MB upload limit)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

> **Requires Python 3.13 or later.**

### 1. Clone the repository

```powershell
git clone <repository-url>
cd resume-jd-matcher
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Download NLTK data

```powershell
python scripts/download_nltk_data.py
```

This downloads `punkt_tab`, `stopwords`, `wordnet`, and `averaged_perceptron_tagger` into the project's virtual environment so they are found automatically at runtime.

### 5. Pre-download the sentence-transformer model

The `all-MiniLM-L6-v2` model (~90 MB) is fetched from Hugging Face the first time it is used and cached locally. Run this once while you have an internet connection:

```python
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
```

After that, the app runs fully offline.

---

## Running the App

```powershell
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. Upload a resume, paste a job description, and click **Analyse**.

---

## Running the Tests

```powershell
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

> **Note:** `test_preprocess.py` requires the NLTK data downloaded in the installation step. Tests are automatically skipped with a clear message if the data is not present.

---

## Skill Coverage

The skill gap analysis matches against a curated list of 49 technical skills across six categories:

| Category | Examples |
|---|---|
| Programming languages | Python, JavaScript, TypeScript, Go, Rust |
| Databases | PostgreSQL, MongoDB, Redis, Elasticsearch |
| ML / Data Science | Machine Learning, TensorFlow, PyTorch, scikit-learn |
| Web / Backend frameworks | React, Django, FastAPI, Node.js |
| DevOps / Cloud | Docker, Kubernetes, AWS, GCP, Azure |
| Tooling / Practices | Git, CI/CD, REST API, Agile, Scrum |

The list is defined in `src/keyword_gap.py` and can be extended by adding entries to `SKILLS_LIST`.

---

## Limitations

- **Scanned PDFs** — image-only PDFs contain no extractable text and will return an error. Use a text-based PDF or DOCX instead.
- **Substring matching** — skill detection uses substring matching, which can produce false positives for short skill names (e.g. `Go` may match inside `good`).
- **Lemmatisation and skills** — the preprocessed resume text is used for skill classification, which means compound skills like `Machine Learning` depend on both words surviving preprocessing.
- **File size** — resumes over 10 MB are rejected by the server before processing.

---

## License

This project is released under the [MIT License](LICENSE).
