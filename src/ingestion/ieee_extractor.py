from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import re

import pdfplumber


@dataclass
class IEEEExtractConfig:

    # Config tuned for typical IEEE two-column PDFs with asy to djust margins/tolerances
    max_pages: Optional[int] = 3

    # Ignore header/footer area: fraction of page height
    header_footer_margin: float = 0.08  # 8% top and bottom

    # Group words into lines if their y-position differs by <= this many points
    line_y_tolerance: float = 3.0

    # If the right column has too few words, treat as single-column page
    min_right_col_words: int = 25


def _basic_cleanup_preserve_newlines(text: str) -> str:

    # Conservative cleanup that preserves line breaks temporarily
    # final cleanup later in pdf_extractor.py

    # Fix hyphenation across line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Normalise Windows newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _group_words_into_lines(words: List[dict], y_tol: float) -> List[str]:

    # Group extracted words into lines using their top coordinate
    # then sort within each line by x-coordinate

    if not words:
        return []

    # Sort by y then x
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))

    lines: List[List[dict]] = []
    current: List[dict] = [words[0]]
    current_y = float(words[0]["top"])

    for w in words[1:]:
        y = float(w["top"])
        if abs(y - current_y) <= y_tol:
            current.append(w)
        else:
            lines.append(current)
            current = [w]
            current_y = y
    lines.append(current)

    out_lines: List[str] = []
    for line_words in lines:
        line_words = sorted(line_words, key=lambda w: w["x0"])
        out_lines.append(" ".join(w["text"] for w in line_words))
    return out_lines


def _split_into_columns(words: List[dict], page_width: float, min_right_words: int) -> Tuple[List[dict], List[dict]]:

    # Split words into left/right columns using a midpoint rule for IEEE two column

    if not words:
        return [], []

    midpoint = page_width / 2.0
    left = [w for w in words if float(w["x0"]) < midpoint]
    right = [w for w in words if float(w["x0"]) >= midpoint]

    # If right column is too small, treat as single column
    if len(right) < min_right_words:
        return words, []
    return left, right


# Extract text from IEEE style PDFs with basic two column reconstruction
def extract_ieee_text(pdf_path: str, cfg: Optional[IEEEExtractConfig] = None) -> Tuple[str, int, int]:

    cfg = cfg or IEEEExtractConfig()
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    chunks: List[str] = []

    with pdfplumber.open(str(path)) as pdf:
        page_count = len(pdf.pages)
        pages = pdf.pages if cfg.max_pages is None else pdf.pages[: cfg.max_pages]
        pages_processed = len(pages)

        for page in pages:
            w = float(page.width)
            h = float(page.height)

            # Crop away headers/footers
            top_cut = h * cfg.header_footer_margin
            bottom_cut = h * (1 - cfg.header_footer_margin)
            cropped = page.crop((0, top_cut, w, bottom_cut))

            # Extract words with coordinates
            words = cropped.extract_words(
                keep_blank_chars=False,
                use_text_flow=True
            )

            # Split columns
            left_words, right_words = _split_into_columns(words, page_width=w, min_right_words=cfg.min_right_col_words)

            # Rebuild each column line-by-line
            left_lines = _group_words_into_lines(left_words, y_tol=cfg.line_y_tolerance)
            right_lines = _group_words_into_lines(right_words, y_tol=cfg.line_y_tolerance)

            page_text = "\n".join(left_lines + ([""] if right_lines else []) + right_lines)
            chunks.append(page_text)

    raw = "\n\n".join(chunks)
    cleaned = _basic_cleanup_preserve_newlines(raw)
    return cleaned, page_count, pages_processed

