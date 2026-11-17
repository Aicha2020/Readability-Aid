from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ValidationResult:
    passed: bool
    checks: Dict[str, bool]
    warnings: List[str]
    missing_hard_numbers: List[str]
    missing_soft_numbers: List[str]
    missing_acronyms: List[str]


# Uppercase words that are usually headings or extraction artefacts,
# not real acronyms that should drive validation.
UPPERCASE_STOPWORDS = {
    "THE", "AND", "OF", "IN", "ON", "FOR", "TO", "FROM",
    "WITH", "BY", "AT", "AS", "IS", "ARE", "AN", "A",
    "DOCUMENTS", "DOCUMENT", "SAMPLE", "STATISTICS", "TABLE",
    "FIGURE", "RESULTS", "METHODS", "ABSTRACT", "ARTICLE",
    "INTRODUCTION", "CONCLUSION", "REFERENCES", "ACKNOWLEDGMENT",
    "ACKNOWLEDGEMENTS"
}


def extract_percentages(text: str) -> List[str]:

    # Extract percentages such as 92% or 10.3%.
    # These are treated as hard-preserve values because percentages often
    # represent important quantitative findings.

    pattern = r"\b\d+(?:\.\d+)?%\b"
    return sorted(set(re.findall(pattern, text)))


def extract_years(text: str) -> List[str]:
    # Extract likely years such as 2021 or 1998.

    # These are logged as soft warnings if missing. They should not fail validation automatically

    pattern = r"\b(?:19\d{2}|20\d{2}|2100)\b"
    return sorted(set(re.findall(pattern, text)))


def extract_small_counts(text: str) -> List[str]:
    # Extract modest numeric values that may appear
    # These are treated as soft warnings only
    pattern = r"\b\d+(?:\.\d+)?\b"
    raw = re.findall(pattern, text)

    filtered = []

    for num in raw:
        # Percentages are handled separately as hard numbers
        if num.endswith("%"):
            continue

        # Years are handled separately.
        if re.fullmatch(r"(?:19\d{2}|20\d{2}|2100)", num):
            continue

        try:
            value = float(num)
        except ValueError:
            continue

        # Keep only modest counts
        # Larger values are more likely to be table or extraction noise
        if 1 <= value <= 100:
            filtered.append(num)

    return sorted(set(filtered))


def extract_acronyms(text: str) -> List[str]:

    # Extract likely acronyms from text.
    # Missing acronyms are logged as soft warnings only.

    pattern = r"\b[A-Z]{2,6}\b"
    raw = re.findall(pattern, text)

    filtered = []

    for token in raw:
        if token in UPPERCASE_STOPWORDS:
            continue
        filtered.append(token)

    return sorted(set(filtered))


def check_not_empty(text: str) -> bool:
    # output must contain some text.

    return bool(text and text.strip())


def word_count(text: str) -> int:

    # Count words in a simple and consistent way

    return len(re.findall(r"\b\w+\b", text))


def length_ratio(source_text: str, candidate_text: str) -> float:

    # Return candidate/source word-count ratio

    source_words = max(1, word_count(source_text))
    candidate_words = word_count(candidate_text)
    return candidate_words / source_words


def check_reasonable_length(
    source_text: str,
    candidate_text: str,
    min_ratio: float = 0.5,
) -> bool:

    # Ensure the rewritten output is not suspiciously short

    source_words = max(1, len(source_text.split()))
    candidate_words = len(candidate_text.split())

    return (candidate_words / source_words) >= min_ratio

def validate_llm_output(
    *,
    source_text: str,
    candidate_text: str,
) -> ValidationResult:
    # -----------------------------
    # Hard preserve numbers
    # -----------------------------
    source_percentages = set(extract_percentages(source_text))
    candidate_percentages = set(extract_percentages(candidate_text))
    missing_hard_numbers = sorted(source_percentages - candidate_percentages)

    # -----------------------------
    # Soft warning numbers
    # -----------------------------
    source_years = set(extract_years(source_text))
    candidate_years = set(extract_years(candidate_text))
    missing_years = sorted(source_years - candidate_years)

    source_counts = set(extract_small_counts(source_text))
    candidate_counts = set(extract_small_counts(candidate_text))
    missing_counts = sorted(source_counts - candidate_counts)

    missing_soft_numbers = sorted(set(missing_years + missing_counts))

    # -----------------------------
    # Soft-warning acronyms
    # -----------------------------
    source_acronyms = set(extract_acronyms(source_text))
    candidate_acronyms = set(extract_acronyms(candidate_text))
    missing_acronyms = sorted(source_acronyms - candidate_acronyms)

    # -----------------------------
    # Length diagnostics
    # -----------------------------
    source_words = word_count(source_text)
    candidate_words = word_count(candidate_text)
    ratio = length_ratio(source_text, candidate_text)

    # -----------------------------
    # Hard pass/fail checks
    # -----------------------------
    checks = {
        "not_empty": check_not_empty(candidate_text),
        "reasonable_length": check_reasonable_length(source_text, candidate_text),
        "hard_numbers_preserved": len(missing_hard_numbers) == 0,
    }

    warnings: List[str] = []

    if not checks["not_empty"]:
        warnings.append("LLM output is empty.")

    if not checks["reasonable_length"]:
        warnings.append(
            "LLM output is much shorter than the source summary and may have lost important content. "
            f"Source words={source_words}, candidate words={candidate_words}, ratio={ratio:.2f}."
        )

    if missing_hard_numbers:
        warnings.append(f"Missing critical numbers: {missing_hard_numbers}")

    # Soft warnings only. These are useful for inspection but do not fail validation
    if missing_soft_numbers:
        warnings.append(f"Missing non-critical numbers: {missing_soft_numbers}")

    if missing_acronyms:
        warnings.append(f"Missing acronyms: {missing_acronyms}")

    return ValidationResult(
        passed=all(checks.values()),
        checks=checks,
        warnings=warnings,
        missing_hard_numbers=missing_hard_numbers,
        missing_soft_numbers=missing_soft_numbers,
        missing_acronyms=missing_acronyms,
    )

