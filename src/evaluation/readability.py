import re
from pypdf import PdfReader
import textstat
from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class ReadabilityProfile:
    flesch_kincaid_grade: float
    smog_index: float
    gunning_fog: float
    coleman_liau_index: float
    automated_readability_index: float


def evaluate_readability(text: str) -> ReadabilityProfile:

    # Compute a readability profile for the given text using standard readability indices
    # Returns a structured profile that can be logged, exported, or compared across versions

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Input text is empty.")

    profile = ReadabilityProfile(
        flesch_kincaid_grade=float(textstat.flesch_kincaid_grade(cleaned)),
        smog_index=float(textstat.smog_index(cleaned)),
        gunning_fog=float(textstat.gunning_fog(cleaned)),
        coleman_liau_index=float(textstat.coleman_liau_index(cleaned)),
        automated_readability_index=float(textstat.automated_readability_index(cleaned)),
    )
    return profile


def print_report(profile: ReadabilityProfile) -> None:
    print("--- Readability Evaluation Report ---")
    print(f"Flesch–Kincaid Grade Level: {profile.flesch_kincaid_grade:.2f}")
    print(f"SMOG Index:                {profile.smog_index:.2f}")
    print(f"Gunning Fog Index:         {profile.gunning_fog:.2f}")
    print(f"Coleman–Liau Index:        {profile.coleman_liau_index:.2f}")
    print(f"ARI Score:                 {profile.automated_readability_index:.2f}")
    print("-" * 37)


def extract_text_from_pdf(path: str, max_pages: int = 2) -> str:
    # Extract text from the first `max_pages` of a PDF
    reader = PdfReader(path)
    chunks = []
    for page in reader.pages[:max_pages]:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def basic_cleanup(text: str) -> str:
    # Remove common PDF artifacts that can distort readability scores
    text = re.sub(r"-\n", "", text)        # fix hyphenated line breaks
    text = re.sub(r"\n+", "\n", text)      # collapse many newlines
    text = re.sub(r"\s+", " ", text)       # normalize whitespace
    return text.strip()


if __name__ == "__main__":
    # Optional local test only. This block is not used by the main pipeline
    pdf_path = "../../data/raw_pdfs/sample.pdf"

    raw_text = extract_text_from_pdf(pdf_path, max_pages=2)
    cleaned_text = basic_cleanup(raw_text)

    # quick sanity check: print first 500 chars to ensure extraction worked
    print(cleaned_text[:500])
    print("\n" + "=" * 60 + "\n")

    result = evaluate_readability(cleaned_text)
    print_report(result)
