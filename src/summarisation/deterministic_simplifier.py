from __future__ import annotations

# Regular expressions are used for pattern matching
import re
from dataclasses import dataclass

from typing import List, Tuple


# -----------------------------------------------
# Result structure returned after simplification
#-----------------------------------------------
@dataclass
class SimplificationResult:

    # Container for the output of the deterministic simplification stage

    simplified_text: str
    operations: List[str]
    replacements_made: int


# ---------------------------------------------------------
# Dictionary of academic - simpler replacements
# ---------------------------------------------------------
# These replacements target polysyllabic or academic words
# that increase readability scores such as SMOG
SIMPLIFY_MAP = {
    "utilize": "use",
    "approximately": "about",
    "demonstrate": "show",
    "additional": "extra",
    "numerous": "many",
    "individuals": "people",
    "assistance": "help",
    "methodology": "method",
    "significant": "important",
    "consequently": "so",
    "therefore": "so",
    "however": "but",
    "furthermore": "also",
    "regarding": "about",
    "in order to": "to",
}


# ---------------------------------------------------------
# Citation patterns that appear in academic text
# ---------------------------------------------------------
# These are removed because they artificially increase readability metrics

_CITATION_PATTERNS = [
    r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]",  # Matches [1] or [1, 2]
    r"\(\s*\d{4}\s*\)",              # Matches (2021)
]


# ---------------------------------------------------------
# Remove citation markers
# ---------------------------------------------------------
def _remove_citations(text: str) -> Tuple[str, bool]:

    # Removes common citation patterns from the text.

    before = text

    # Removes each citation pattern
    for pat in _CITATION_PATTERNS:
        text = re.sub(pat, "", text)

    # Normalizes spacing after removal
    text = re.sub(r"\s+", " ", text).strip()

    return text, (text != before)


# ---------------------------------------------------------
# Split very long sentences
# ---------------------------------------------------------
def _split_long_sentences(text: str, max_words: int = 22) -> Tuple[str, bool]:

    # Attempts to split very long sentences into smaller ones

    # Strategy:
    # - Identify sentences exceeding max_words
    # - Try splitting near the middle
    # - Prefer comma or semicolon boundaries

    changed = False

    # Simple sentence segmentation
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    out = []

    for s in sentences:
        words = s.split()

        # If sentence is already short enough, it is kept
        if len(words) <= max_words:
            out.append(s)
            continue

        # Identifies possible split positions (comma/semicolon)
        cuts = [m.start() for m in re.finditer(r"[;,]\s+", s)]

        if not cuts:
            # If no safe split points exist, leave sentence unchanged
            out.append(s)
            continue

        # Chooses split point closest to the middle
        mid = len(s) // 2
        split_at = min(cuts, key=lambda i: abs(i - mid))

        left = s[:split_at].strip()
        right = s[split_at+1:].strip()

        # Constructs two shorter sentences
        if left and right:
            if not left.endswith((".", "!", "?")):
                left += "."

            # Capitalizes start of new sentence
            right = right[0].upper() + right[1:] if right else right

            out.append(left + " " + right)
            changed = True
        else:
            out.append(s)

    return " ".join(out), changed


# ---------------------------------------------------------
# Replace complex academic words with simpler alternatives
# ---------------------------------------------------------
def _smog_targeted_replace(text: str) -> Tuple[str, int]:

    # The function includes protections to avoid replacing:
    # - acronyms (e.g., AI, NLP)
    # - proper nouns
    # - hyphenated technical terms

    replacements_made = 0

    def should_protect(token: str) -> bool:

        # Determines whether a token should not be modified

        # Protects acronyms (e.g., AI)
        if token.isupper():
            return True

        # Protects proper nouns (e.g., Smith)
        if re.search(r"[A-Z][a-z]", token):
            return True

        # Protects hyphenated technical terms
        if "-" in token:
            return True

        return False

    # Sort keys by length to match longer phrases first
    keys = sorted(SIMPLIFY_MAP.keys(), key=len, reverse=True)

    pattern = r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b"

    def repl(m: re.Match) -> str:

        # Replacement logic executed during regex substitution

        nonlocal replacements_made

        original = m.group(0)

        # Skips protected tokens
        if should_protect(original):
            return original

        lower = original.lower()
        replacement = SIMPLIFY_MAP.get(lower)

        if not replacement:
            return original

        replacements_made += 1

        # Preserves capitalization
        if original[0].isupper():
            return replacement.capitalize()

        return replacement

    out = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # Normalizes whitespace
    out = re.sub(r"\s+", " ", out).strip()

    return out, replacements_made


# ---------------------------------------------------------
# Main deterministic simplification pipeline
# ---------------------------------------------------------
def simplify_text_deterministic(seed_summary: str) -> SimplificationResult:

    # The pipeline consists of three stages:

    # 1. Citation removal
    # 2. SMOG-targeted word replacement
    # 3. Sentence splitting


    ops: List[str] = []
    text = seed_summary.strip()

    if not text:
        raise ValueError("Seed summary is empty")

    # -----------------------------------------------------
    # Step 1: Remove citation markers
    # -----------------------------------------------------
    text, changed = _remove_citations(text)

    if changed:
        ops.append("Removed citation markers")

    # -----------------------------------------------------
    # Step 2: Replace complex academic words
    # -----------------------------------------------------
    text, nrep = _smog_targeted_replace(text)

    if nrep:
        ops.append("Applied SMOG-targeted replacements")

    # -----------------------------------------------------
    # Step 3: Split long sentences
    # -----------------------------------------------------
    text2, split_changed = _split_long_sentences(text, max_words=22)

    if split_changed:
        ops.append("Split long sentences")

    # -----------------------------------------------------
    # Return structured result
    # -----------------------------------------------------
    return SimplificationResult(
        simplified_text=text2,
        operations=ops,
        replacements_made=nrep,
    )
