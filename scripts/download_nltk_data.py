"""
Helper script to download required NLTK data packages.

Run this once before using the application:
    python scripts/download_nltk_data.py

Downloads are placed into <project_root>/.venv/nltk_data/ so they are
immediately visible to the running Python environment without any manual
extraction or path configuration.
"""
import sys
from pathlib import Path

import nltk

# ---------------------------------------------------------------------------
# Resolve the download directory
# ---------------------------------------------------------------------------

# This file lives at <project_root>/scripts/download_nltk_data.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Target the venv's nltk_data directory — NLTK searches <sys.prefix>/nltk_data
# at runtime when the venv is active, so this is the right place.
VENV_NLTK_DATA = PROJECT_ROOT / ".venv" / "nltk_data"

if VENV_NLTK_DATA.parent.exists():
    # .venv exists — download into the venv so the app finds the data immediately
    DOWNLOAD_DIR = str(VENV_NLTK_DATA)
    VENV_NLTK_DATA.mkdir(parents=True, exist_ok=True)
else:
    # No .venv directory — fall back to NLTK's default user directory
    DOWNLOAD_DIR = None  # nltk.download() will choose the default

# ---------------------------------------------------------------------------
# Packages required by src/preprocess.py
# ---------------------------------------------------------------------------

REQUIRED_PACKAGES = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger",
]


def download_nltk_data() -> None:
    """
    Download all NLTK packages required by the application.

    Writes packages into the project's .venv/nltk_data/ directory so they
    are found automatically by the Python environment without manual
    extraction or NLTK_DATA environment variable changes.
    """
    if DOWNLOAD_DIR:
        print(f"Downloading NLTK data into: {DOWNLOAD_DIR}")
    else:
        print("Downloading NLTK data into default NLTK user directory.")

    all_ok = True
    for package in REQUIRED_PACKAGES:
        print(f"  Downloading: {package} ...", end=" ", flush=True)
        success = nltk.download(package, download_dir=DOWNLOAD_DIR, quiet=True)
        if success:
            print("OK")
        else:
            print("FAILED")
            all_ok = False

    if all_ok:
        print("\nAll NLTK packages downloaded successfully.")
    else:
        print("\nSome packages failed to download. Check your internet connection and retry.")
        sys.exit(1)


if __name__ == "__main__":
    download_nltk_data()
