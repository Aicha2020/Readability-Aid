from __future__ import annotations
import logging
from datetime import datetime

import shutil
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import pdfplumber
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from src.ingestion.ieee_extractor import extract_ieee_text, IEEEExtractConfig



# ----------------------------
# Data structures
# ----------------------------
@dataclass
class ExtractionQuality:
    passed: bool
    score: float
    warnings: list[str]
    joined_word_ratio: float
    digit_line_ratio: float
    short_line_ratio: float
    average_word_length: float
    extracted_words: int


@dataclass
class ExtractionResult:
    pdf_path: str
    extracted_chars: int
    extracted_words: int
    page_count: int
    pages_processed: int
    is_encrypted: bool
    likely_scanned: bool
    used_method: str
    abstract_extracted: bool
    extraction_quality_score: float = 0.0
    extraction_quality_passed: bool = True
    extraction_quality_warnings: list[str] | None = None


@dataclass
class RunRecord:
    pdf_name: str
    status: str            # SUCCESS / FAILED
    method: str
    extracted_words: int = 0
    likely_scanned: bool = False
    message: str = ""


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pdf_extraction_{ts}.log"

    logger = logging.getLogger("pdf_extractor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console handler (short)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # File handler (detailed)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"Logging to: {log_file}")
    return logger

# ----------------------------
# Validation helpers
# ---------------------------
def is_pdf_encrypted(pdf_path: str) -> bool:

    # Returns True if the PDF is encrypted. Raises errors for missing/corrupt PDFs

    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        reader = PdfReader(str(p))
        return bool(reader.is_encrypted)
    except PdfReadError as e:
        raise ValueError(f"PDF read error (corrupt/unsupported): {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error reading PDF: {e}")


def basic_cleanup_preserve_lines(text: str) -> str:

    # Light cleanup that preserves line boundaries
    # Useful before removing table blocks

    # Fix hyphenation across line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()



def remove_extraction_artifacts(text: str) -> str:
    # Removes common PDF extraction artefacts

    # Removes PDF character mapping artefacts such as cid:180
    text = re.sub(r"\(cid:\d+\)", " ", text)

    # Removes common download/license footer artefacts
    text = re.sub(
        r"Authorized licensed use.*?(?=\n|$)",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"Downloaded on .*?(?=\n|$)",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Removes DOI / URL fragments
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\bdoi\s*:\s*\S+", " ", text, flags=re.IGNORECASE)

    # Removes isolated page numbers on their own line
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", " ", text)

    # Removes repeated punctuation or separator artefacts
    text = re.sub(r"[_=]{3,}", " ", text)

    # Fixes missing spaces after punctuation when extraction merges text
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)

    # Fixes some lower to uppercase merged boundaries:
    # Conservative: only splits lower to uppercase, not all capitals.
    text = re.sub(r"([a-z])([A-Z][a-z])", r"\1 \2", text)

    # Removes isolated single letter noise lines
    text = re.sub(r"(?m)^\s*[A-Za-z]\s*$", " ", text)

    # Normalises spacing but preserve line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_table_like_line(line: str) -> bool:

    # detection for table/statistics like lines

    stripped = line.strip()
    if not stripped:
        return False

    lower = stripped.lower()
    words = stripped.split()
    word_count = len(words)

    # Common table/figure/statistical markers.
    table_keywords = [
        "statistics of sample documents",
        "sentences / document",
        "sentence / summary",
        "sentence/ summary",
        "words / document",
        "words/document summary",
        "avg.",
        "average",
        "standard deviation",
        "std.",
        "precision",
        "recall",
        "f-score",
        "accuracy",
        "dataset",
        "table ",
        "table.",
        "fig.",
        "figure ",
    ]

    if any(k in lower for k in table_keywords):
        return True

    # Common publisher / metadata / footer noise.
    metadata_keywords = [
        "authorized licensed use",
        "downloaded on",
        "copyright",
        "all rights reserved",
        "isbn",
        "issn",
        "doi:",
        "http://",
        "https://",
        "www.",
        "proceedings of",
    ]

    if any(k in lower for k in metadata_keywords):
        return True

    # Section headings alone should not be summarised
    if re.fullmatch(r"(?i)\s*(?:[ivxlcdm]+|\d+)\.?\s+[a-z][a-z\s]{2,40}\s*", stripped):
        return True

    # Isolated page numbers
    if re.fullmatch(r"\d{1,4}", stripped):
        return True

    # Formula like lines
    if "=" in stripped and word_count <= 20:
        return True

    numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", stripped)

    # Many numbers in one line usually means a table row
    if len(numbers) >= 4:
        return True

    # Short line with several numbers usually means numeric clutter
    if word_count <= 8 and len(numbers) >= 2:
        return True

    # Mostly numeric/symbolic line
    alpha_chars = re.findall(r"[A-Za-z]", stripped)
    digit_chars = re.findall(r"\d", stripped)

    if word_count <= 10 and digit_chars and len(digit_chars) > len(alpha_chars):
        return True

    # Mostly uppercase short lines are often table headings
    # conservative to avoid deleting real acronyms
    if alpha_chars:
        uppercase_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)

        if uppercase_ratio > 0.75 and word_count <= 8:
            return True

    # Very long merged tokens often come from bad extraction
    if re.search(r"\b[a-z]{25,}\b", stripped):
        return True

    return False


def remove_table_like_blocks(text: str) -> str:

    # Removes table/statistics/noise lines before flattening

    # This works line-by-line while line breaks still exist
    # It also skips a few lines after a strong table heading
    # because extracted tables often are several lines

    lines = text.splitlines()
    kept_lines = []

    skip_next = 0

    strong_table_headings = [
        "statistics of sample documents",
        "table ",
        "table.",
        "figure ",
        "fig.",
    ]

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        if skip_next > 0:
            skip_next -= 1

            # Continues skipping if the next line still looks table-like.
            if is_table_like_line(line):
                continue

            # If the line looks like normal prose, keeps it
            # This prevents over removing real paragraphs
            if len(stripped.split()) >= 8 and re.search(r"[.!?]$", stripped):
                kept_lines.append(line)
                continue

            # Otherwise skips short fragments after table headings
            if len(stripped.split()) <= 8:
                continue

        if is_table_like_line(line):
            # If this is a strong table/figure heading, skips a few following fragments too
            if any(marker in lower for marker in strong_table_headings):
                skip_next = 4
            continue

        kept_lines.append(line)

    cleaned = "\n".join(kept_lines)

    # Preserves paragraph boundaries but removes excessive blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()

def flatten_to_paragraphs(text: str) -> str:

    # Final cleanup for downstream readability and summarisation

    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_references_section(text: str) -> str:

    # Removes the references/bibliography section from extracted academic text
    # This is done before flattening, while section headings are still easier to detect

    patterns = [
        # References
        r"(?im)^\s*references\s*$",
        r"(?im)^\s*(?:\d+|[IVXLCDM]+)\.?\s+references\s*$",

        # Bibliography
        r"(?im)^\s*bibliography\s*$",
        r"(?im)^\s*(?:\d+|[IVXLCDM]+)\.?\s+bibliography\s*$",

        # Reference list
        r"(?im)^\s*reference list\s*$",
        r"(?im)^\s*(?:\d+|[IVXLCDM]+)\.?\s+reference list\s*$",
    ]

    matches = []

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            matches.append(match.start())

    if matches:
        cut_position = min(matches)
        return text[:cut_position].strip()

    return text.strip()

def extract_abstract(text: str) -> Optional[str]:

    # Heuristic abstract extraction. Works for many papers but not all
    # Returns None if no clear abstract block is found

    # Abstract Index Terms / Keywords / Introduction
    m = re.search(
        r"\bAbstract\b\s*(?:—|-|:)\s*(.+?)(?:\bIndex Terms\b|\bKeywords\b|\bIntroduction\b|\bI\.\s*INTRODUCTION\b|\b1\.\s*Introduction\b)",
        text,
        re.IGNORECASE,
    )
    if m:
        abstract = m.group(1).strip()
        return abstract[:4000].strip()
    return None


# ----------------------------
# Core extraction
# ----------------------------
def extract_text_pdfplumber_basic(pdf_path: str, max_pages: Optional[int] = None) -> Tuple[str, int, int]:

    # Basic extraction using pdfplumber's extract_text (may scramble two-column IEEE PDFs).

    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
        for page in pages:
            chunks.append(page.extract_text() or "")
    text = "\n".join(chunks)
    pages_processed = len(pages)
    return text, page_count, pages_processed


def likely_scanned_pdf(extracted_text: str, min_chars_threshold: int = 200) -> bool:

    # If we extract almost no characters, it's likely scanned or extraction blocked.

    return len(extracted_text.strip()) < min_chars_threshold

def assess_extraction_quality(text: str) -> ExtractionQuality:

    # Estimate whether extracted PDF text is good enough for summarisation

    # This catches cases where text exists but is badly damaged, for example:
    # - words merged together with missing spaces
    # - many standalone page/layout digits
    # - many very short broken lines
    # - unusually long average word length

    warnings: list[str] = []

    stripped = text.strip()
    words = re.findall(r"\b\w+\b", stripped)
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]

    extracted_words = len(words)

    if extracted_words == 0:
        return ExtractionQuality(
            passed=False,
            score=0.0,
            warnings=["No words extracted."],
            joined_word_ratio=1.0,
            digit_line_ratio=1.0,
            short_line_ratio=1.0,
            average_word_length=0.0,
            extracted_words=0,
        )

    # Very long alphabetic tokens often indicate missing spaces:

    alpha_words = [w for w in words if re.fullmatch(r"[A-Za-z]+", w)]
    long_joined_words = [w for w in alpha_words if len(w) >= 22]
    joined_word_ratio = len(long_joined_words) / max(1, len(alpha_words))

    # Lines that are only numbers are often page numbers or layout artefacts
    digit_lines = [ln for ln in lines if re.fullmatch(r"\d{1,4}", ln)]
    digit_line_ratio = len(digit_lines) / max(1, len(lines))

    # Many very short lines can indicate broken layout extraction
    short_lines = [ln for ln in lines if len(ln.split()) <= 2]
    short_line_ratio = len(short_lines) / max(1, len(lines))

    average_word_length = sum(len(w) for w in words) / max(1, len(words))

    score = 1.0

    if extracted_words < 150:
        warnings.append(f"Very few words extracted ({extracted_words}).")
        score -= 0.35

    if joined_word_ratio > 0.05:
        warnings.append(
            f"Very high joined-word ratio ({joined_word_ratio:.3f}); extraction likely unreliable."
        )
        score -= 0.50
    elif joined_word_ratio > 0.025:
        warnings.append(
            f"High joined-word ratio ({joined_word_ratio:.3f}); possible missing spaces."
        )
        score -= 0.35
    elif joined_word_ratio > 0.015:
        warnings.append(
            f"Moderate joined-word ratio ({joined_word_ratio:.3f}); possible extraction noise."
        )
        score -= 0.20

    if digit_line_ratio > 0.08:
        warnings.append(
            f"High digit-line ratio ({digit_line_ratio:.3f}); possible page/layout artefacts."
        )
        score -= 0.20

    if short_line_ratio > 0.45:
        warnings.append(
            f"High short-line ratio ({short_line_ratio:.3f}); possible broken layout."
        )
        score -= 0.15

    if average_word_length > 8.5:
        warnings.append(
            f"High average word length ({average_word_length:.2f}); possible merged words."
        )
        score -= 0.20

    score = max(0.0, score)
    passed = score >= 0.60

    return ExtractionQuality(
        passed=passed,
        score=score,
        warnings=warnings,
        joined_word_ratio=joined_word_ratio,
        digit_line_ratio=digit_line_ratio,
        short_line_ratio=short_line_ratio,
        average_word_length=average_word_length,
        extracted_words=extracted_words,
    )


def extract_pdf(
    pdf_path: str,
    *,
    max_pages: int = 3,
    abstract_only: bool = False,
    strip_references: bool = True,
    method: str = "ieee",  # "ieee" (default) or "basic"
) -> Tuple[str, str, str, ExtractionResult]:

    # If method="ieee", the function first tries the IEEE layout-aware extractor,
    # then also tries the basic pdfplumber extractor as a fallback
    # The extraction result with the better quality score is used.

    encrypted = is_pdf_encrypted(pdf_path)
    if encrypted:
        raise ValueError(f"PDF appears to be encrypted/restricted: {pdf_path}")

    # ----------------------------
    # Extract text
    # ----------------------------
    extraction_attempts = []

    if method.lower() == "ieee":
        # Attempt 1: IEEE/layout-aware extraction
        raw_text_ieee, page_count_ieee, pages_processed_ieee = extract_ieee_text(
            pdf_path,
            cfg=IEEEExtractConfig(max_pages=max_pages),
        )
        quality_ieee = assess_extraction_quality(raw_text_ieee)

        extraction_attempts.append(
            (
                raw_text_ieee,
                page_count_ieee,
                pages_processed_ieee,
                "ieee_layout_aware_pdfplumber",
                quality_ieee,
            )
        )

        # Attempt 2: basic pdfplumber extraction fallback
        raw_text_basic, page_count_basic, pages_processed_basic = extract_text_pdfplumber_basic(
            pdf_path,
            max_pages=max_pages,
        )
        quality_basic = assess_extraction_quality(raw_text_basic)

        extraction_attempts.append(
            (
                raw_text_basic,
                page_count_basic,
                pages_processed_basic,
                "pdfplumber_basic",
                quality_basic,
            )
        )

        # Select the extraction attempt with the highest quality score
        raw_text, page_count, pages_processed, used_method, quality = max(
            extraction_attempts,
            key=lambda item: item[4].score,
        )

    else:
        # Basic extraction only
        raw_text, page_count, pages_processed = extract_text_pdfplumber_basic(
            pdf_path,
            max_pages=max_pages,
        )
        used_method = "pdfplumber_basic"
        quality = assess_extraction_quality(raw_text)

    scanned = likely_scanned_pdf(raw_text)

    # Optional: fail  only if extraction is extremely poor

    if quality.score < 0.65 or quality.joined_word_ratio > 0.05:
        raise ValueError(
            "PDF extraction quality too low for reliable summarisation. "
            f"Method={used_method}; score={quality.score:.2f}; "
            f"warnings={quality.warnings}"
        )
    # ----------------------------
    # Cleanup pipeline
    # ----------------------------
    cleaned = basic_cleanup_preserve_lines(raw_text)

    # Removes general PDF extraction artefacts before table filtering
    cleaned = remove_extraction_artifacts(cleaned)

    if strip_references:
        cleaned = strip_references_section(cleaned)

    # Remove table/statistics-like material before flattening
    structured_cleaned = remove_table_like_blocks(cleaned)


    # ----------------------------
    # Optional abstract extraction
    # ----------------------------
    abstract_text: Optional[str] = None
    if abstract_only:
        abstract_text = extract_abstract(structured_cleaned)

    # Keeps structured version for section-aware summarisation
    structured_text = (
        abstract_text
        if abstract_only and abstract_text is not None
        else structured_cleaned
    )

    # Flatten only for final cleaned extracted text / readability
    final_text = flatten_to_paragraphs(structured_text)

    result = ExtractionResult(
        pdf_path=str(pdf_path),
        extracted_chars=len(final_text),
        extracted_words=len(final_text.split()),
        page_count=page_count,
        pages_processed=pages_processed,
        is_encrypted=encrypted,
        likely_scanned=scanned,
        used_method=used_method,
        abstract_extracted=bool(abstract_text) if abstract_only else False,
        extraction_quality_score=quality.score,
        extraction_quality_passed=quality.passed,
        extraction_quality_warnings=quality.warnings,
    )

    return final_text, structured_text, raw_text, result

# ----------------------------
# Output + preview
# ----------------------------
def save_outputs(text: str, meta: ExtractionResult, out_dir: str) -> Tuple[str, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    stem = Path(meta.pdf_path).stem
    txt_file = out_path / f"{stem}.txt"
    json_file = out_path / f"{stem}.json"

    txt_file.write_text(text, encoding="utf-8")
    json_file.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    return str(txt_file), str(json_file)


def log_text_preview(logger, text: str, max_lines: int = 12, max_chars: int = 800) -> None:
    if not text.strip():
        logger.info("[Preview unavailable: empty text]")
        return

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    preview = "\n".join(lines[:max_lines])[:max_chars]

    logger.info("--- Text preview ---\n" + preview + "\n--- End preview ---")


def log_readability_profile(logger, label: str, profile) -> None:

    # Logs readability metrics for one pipeline stage

    logger.info(
        f"{label} readability: "
        f"SMOG={profile.smog_index:.2f}, "
        f"FKGL={profile.flesch_kincaid_grade:.2f}, "
        f"Fog={profile.gunning_fog:.2f}, "
        f"Coleman-Liau={profile.coleman_liau_index:.2f}, "
        f"ARI={profile.automated_readability_index:.2f}"
    )

def clear_directory(directory: Path, logger=None) -> None:

    # Clears contents of a directory without deleting the directory itself

    directory.mkdir(parents=True, exist_ok=True)

    for item in directory.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except PermissionError as e:
            msg = f"Could not delete {item} (in use). Close it and re-run. ({e})"
            if logger:
                logger.warning(msg)
            else:
                print(msg)

