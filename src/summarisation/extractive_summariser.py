from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class ExtractiveSummaryResult:
    summary_text: str
    selected_sentences: List[str]
    compression_ratio: float
    debug_scores: Optional[List[Dict[str, Any]]] = None


@dataclass
class SectionedSentence:
    text: str
    section: str
    index: int


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "while",
    "to", "of", "in", "on", "for", "with", "as", "by", "at", "from", "into", "over",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "they", "their", "we", "our",
    "can", "could", "should", "would", "may", "might", "must",
}


_CLAIM_PATTERNS = [
    r"\bthis paper\b",
    r"\bthis study\b",
    r"\bthis article\b",
    r"\bin this work\b",
    r"\bour work\b",
    r"\bwe propose\b",
    r"\bwe present\b",
    r"\bwe introduce\b",
    r"\bwe develop\b",
    r"\bwe investigate\b",
    r"\bwe examine\b",
    r"\bwe analyse\b",
    r"\bwe analyze\b",
    r"\bwe find\b",
    r"\bit finds that\b",
    r"\bthe aim of this work\b",
    r"\bthe purpose of this paper\b",
    r"\bthe objective\b",
    r"\bthe goal\b",
    r"\bthe major contributions\b",
    r"\bour contributions\b",
    r"\bcontributions are\b",
]


_RESULT_PATTERNS = [
    r"\bresults show\b",
    r"\bthe results show\b",
    r"\bexperiments show\b",
    r"\bthe experiments show\b",
    r"\bwe demonstrate\b",
    r"\bwe show\b",
    r"\bwe found\b",
    r"\bfindings indicate\b",
    r"\bthe findings\b",
    r"\bsignificantly\b",
    r"\boutperform\b",
    r"\boutperforms\b",
    r"\bimprove\b",
    r"\bimproves\b",
    r"\bimproved\b",
    r"\bachieve\b",
    r"\bachieves\b",
    r"\bevaluation\b",
    r"\bperformance\b",
]


_CONCLUSION_PATTERNS = [
    r"\bin conclusion\b",
    r"\bto conclude\b",
    r"\bfinally\b",
    r"\boverall\b",
    r"\bin summary\b",
    r"\bthis suggests\b",
    r"\bthese findings suggest\b",
    r"\bfuture work\b",
    r"\bwe conclude\b",
    r"\bconcludes that\b",
]


_CONTRAST_PATTERNS = [
    r"\bwhile\b",
    r"\bhowever\b",
    r"\bbut\b",
    r"\bin contrast\b",
    r"\brather than\b",
    r"\byet\b",
    r"\balthough\b",
    r"\binstead\b",
    r"\bwhereas\b",
    r"\bcompared with\b",
    r"\bcompared to\b",
]


_BACKGROUND_PATTERNS = [
    r"\bin recent years\b",
    r"\bin the last decade\b",
    r"\bhas captured global attention\b",
    r"\bhas become increasingly important\b",
    r"\bover the past\b",
    r"\bthere are two main approaches\b",
    r"\bcan be defined as\b",
    r"\bis defined as\b",
    r"\baccording to\b",
    r"\brelated work\b",
    r"\brelated works\b",
]


_SECTION_HEADINGS = {
    "abstract": "abstract",
    "introduction": "introduction",
    "method": "method",
    "methodology": "method",
    "proposed methodology": "method",
    "proposed method": "method",
    "approach": "method",
    "experiments": "results",
    "experiment": "results",
    "results": "results",
    "evaluation": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "future work": "conclusion",
    "related work": "related_work",
    "related works": "related_work",
    "references": "references",
    "bibliography": "references",
}

_SECTION_WEIGHTS = {
    "abstract": 3.0,
    "introduction": 1.5,
    "method": 1.0,
    "results": 2.0,
    "discussion": 2.0,
    "conclusion": 3.0,
    "related_work": 0.5,
    "background": 0.5,
    "unknown": 0.0,
    "references": -5.0,
}


def _normalise_section_heading(line: str) -> Optional[str]:

    # Detects section headings while line breaks still exist

    cleaned = line.strip()
    if not cleaned:
        return None

    # Removes numbering
    cleaned = re.sub(
        r"^\s*(?:[IVXLCDM]+|\d+(?:\.\d+)*)\.?\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Removes punctuation around the heading
    cleaned = cleaned.strip(" :-–—").lower()

    # Avoids treating long sentences as headings
    if len(cleaned.split()) > 5:
        return None

    return _SECTION_HEADINGS.get(cleaned)


def _remove_references_from_structured_text(text: str) -> str:

    # Safety backup to remove references inside the summariser

    lines = text.splitlines()
    kept_lines = []

    for line in lines:
        section = _normalise_section_heading(line)

        if section == "references":
            break

        kept_lines.append(line)

    return "\n".join(kept_lines).strip()


def _sentence_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
# splits the extracted paper text into a list of sentences
    protected = {
        "e.g.": "EGTOKEN",
        "i.e.": "IETOKEN",
        "et al.": "ETALTOKEN",
        "Fig.": "FIGTOKEN",
        "Eq.": "EQTOKEN",
        "Dr.": "DRTOKEN",
        "Prof.": "PROFTOKEN",
    }

    for k, v in protected.items():
        text = text.replace(k, v)

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)

    out = []
    for s in sentences:
        for k, v in protected.items():
            s = s.replace(v, k)
        s = s.strip()
        if s:
            out.append(s)

    return out

def _sectioned_sentence_split(text: str) -> List[SectionedSentence]:

    # Splits structured text into sentences while preserving section labels

    # This works before flattening, so headings such as Abstract, Introduction,
    # Results, and Conclusion can influence sentence scoring.

    text = _remove_references_from_structured_text(text)

    sectioned: List[SectionedSentence] = []
    current_section = "unknown"
    sentence_index = 0

    lines = text.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        detected_section = _normalise_section_heading(line)

        if detected_section == "references":
            break

        if detected_section is not None:
            current_section = detected_section
            continue

        for sentence in _sentence_split(line):
            sentence = sentence.strip()
            if not sentence:
                continue

            sectioned.append(
                SectionedSentence(
                    text=sentence,
                    section=current_section,
                    index=sentence_index,
                )
            )
            sentence_index += 1

    return sectioned

def _tokenize(text: str) -> List[str]:
    # extracts the meaningful words from a sentence so the summariser can calculate:
    # word frequencies, title overlap, and sentence importance
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def _word_frequencies(sentences: List[str]) -> dict[str, int]:
    # Counts how often each meaningful word appears across candidate sentences
    freq: dict[str, int] = {}
    for s in sentences:
        for w in _tokenize(s):
            freq[w] = freq.get(w, 0) + 1
    return freq


def _title_keywords(title: str) -> set[str]:
    # Extracts meaningful title words for title-based sentence scoring
    if not title:
        return set()
    return set(_tokenize(title))


def _title_overlap_score(sentence: str, title_terms: set[str]) -> float:
    # Scores a sentence by how many title keywords it contains
    if not title_terms:
        return 0.0
    sent_terms = set(_tokenize(sentence))
    return float(len(sent_terms & title_terms))


def _matched_patterns(sentence: str, patterns: List[str]) -> List[str]:
    # Returns the patterns that match a sentence
    s = sentence.lower()
    return [p for p in patterns if re.search(p, s)]


def _pattern_bonus(sentence: str, patterns: List[str], bonus: float) -> float:
    # Adds a fixed score bonus when a sentence matches important patterns
    return bonus if _matched_patterns(sentence, patterns) else 0.0


def _detect_section(sentence: str, previous_section: str) -> str:

    # This function updates the active section if a heading like sentence is detected

    clean = sentence.strip().lower()
    clean = re.sub(r"^[ivxlcdm]+\.\s*", "", clean)
    clean = re.sub(r"^\d+\.?\s*", "", clean)
    clean = clean.strip(":.- ")

    # Exact or near exact heading
    if clean in _SECTION_HEADINGS:
        return _SECTION_HEADINGS[clean]

    # Heading appears at start of sentence
    for heading, section in _SECTION_HEADINGS.items():
        if clean.startswith(heading + " "):
            return section

    return previous_section


def _section_bonus(section: str) -> float:
    if section == "abstract":
        return 4.0
    if section == "conclusion":
        return 3.5
    if section == "results":
        return 3.0
    if section == "introduction":
        return 1.5
    if section == "method":
        return 1.0
    if section == "related_work":
        return -2.0
    if section == "references":
        return -10.0
    return 0.0


def _position_bonus(index: int, total: int) -> float:
    if total <= 1:
        return 0.0

    ratio = index / max(1, total - 1)

    if ratio <= 0.15:
        return 2.0
    if ratio <= 0.30:
        return 1.0
    if ratio >= 0.85:
        return 1.5
    return 0.0

def _is_result_sentence(sentence: str) -> bool:

    # Detects sentences that contain meaningful evaluation/result information
    # To avoid removing them if they contain numbers

    lower = sentence.lower()

    result_terms = [
        "result",
        "results",
        "showed",
        "shows",
        "found",
        "obtained",
        "score",
        "percentage",
        "percent",
        "practicality",
        "validity",
        "validation",
        "evaluation",
        "test",
        "assessment",
        "feasibility",
        "effectiveness",
        "accuracy",
    ]

    has_result_term = any(term in lower for term in result_terms)
    has_percentage = bool(re.search(r"\b\d+(?:\.\d+)?\s*%", sentence))

    return has_result_term and has_percentage


def _is_metadata_or_header(sentence: str) -> bool:

   # Detects metadata, headers, affiliations, author details, and publisher artefacts.
    # These should not be selected for the seed summary

    s = sentence.strip()
    lower = s.lower()

    if not s:
        return True

    # Author/contact metadata
    metadata_patterns = [
        r"\bdepartment of\b",
        r"\buniversity\b",
        r"\binstitute\b",
        r"\bfaculty\b",
        r"\bschool of\b",
        r"\bcollege\b",
        r"\bcorresponding author\b",
        r"\baccepted\b",
        r"\breceived\b",
        r"\brevised\b",
        r"\bpublished\b",
        r"\bcopyright\b",
        r"\ball rights reserved\b",
        r"\blicensed use\b",
        r"\bdownloaded on\b",
        r"\bdoi\b",
        r"\bissn\b",
        r"\bisbn\b",
        r"\bproceedings\b",
        r"\bconference\b",
        r"\bjournal\b",
    ]

    if any(re.search(pattern, lower) for pattern in metadata_patterns):
        return True

    # Email addresses
    if re.search(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", s):
        return True

    # URLs
    if re.search(r"https?://|www\.", lower):
        return True

    # Mostly a title/header line rather than a sentence
    words = s.split()
    if len(words) <= 12 and not re.search(r"[.!?]$", s):
        title_case_words = sum(1 for w in words if w[:1].isupper())
        if words and title_case_words / len(words) > 0.6:
            return True

    # Standalone section headings
    if re.fullmatch(
        r"(?i)\s*(abstract|introduction|related work|method|methodology|results|discussion|conclusion|references)\s*",
        s,
    ):
        return True

    # Numbered headings
    if re.fullmatch(
        r"(?i)\s*(?:\d+|[ivxlcdm]+)\.?\s+(abstract|introduction|related work|method|methodology|results|discussion|conclusion|references)\s*",
        s,
    ):
        return True

    # Very short metadata like fragments
    if len(words) <= 5 and any(
        term in lower for term in ["author", "article", "copyright", "journal", "conference"]
    ):
        return True

    return False


def _noise_penalty(sentence: str) -> float:
    nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", sentence)
    words = sentence.split()
    lower = sentence.lower()

    penalty = 0.0

    if len(nums) >= 4:
        penalty += 5.0

    if len(nums) >= 2 and len(words) <= 18:
        penalty += 3.0

    if len(words) < 6:
        penalty += 2.0

    table_markers = [
        "table", "figure", "fig.", "eq.", "equation", "statistics",
        "avg.", "average", "doi", "isbn", "issn", "copyright",
        "authorized licensed use", "downloaded on", "ieee", "springer",
        "references", "bibliography"
    ]

    if any(k in lower for k in table_markers):
        penalty += 4.0

    # Citation heavy sentences often come from related work/background
    citation_count = len(re.findall(r"\[[0-9,\s]+\]|\([A-Z][A-Za-z]+ et al\.", sentence))
    if citation_count >= 2:
        penalty += 2.0

    # Mostly uppercase headings / table labels
    alpha_chars = re.findall(r"[A-Za-z]", sentence)
    if alpha_chars:
        uppercase_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if uppercase_ratio > 0.65:
            penalty += 3.0

    return penalty


def _ensure_sentence_end(sentence: str) -> str:

    # Ensures selected summary units end with sentence punctuation

    sentence = sentence.strip()

    if not sentence:
        return sentence

    if sentence[-1] not in ".!?":
        sentence += "."

    return sentence


def _classify_slot(sentence: str, section: str) -> str:

    # Assigns a sentence to a coverage slot
    # Used later to ensure the summary covers aim, method, results, and conclusion

    if _matched_patterns(sentence, _CONCLUSION_PATTERNS) or section == "conclusion":
        return "conclusion"

    if _matched_patterns(sentence, _RESULT_PATTERNS) or section == "results":
        return "result"

    if _matched_patterns(sentence, _CLAIM_PATTERNS):
        return "claim"

    lower = sentence.lower()
    if section == "method" or any(k in lower for k in ["method", "approach", "model", "dataset", "features", "framework"]):
        return "method"

    if section in {"abstract", "introduction"}:
        return "aim"

    return "general"


def _score_sentence(
    sentence: str,
    freq: dict[str, int],
    *,
    index: int,
    total_sentences: int,
    title_terms: set[str],
    section: str,
    max_len_words: int = 45,
) -> Tuple[float, Dict[str, Any]]:
    words = _tokenize(sentence)
    if not words:
        return 0.0, {}

    base = sum(freq.get(w, 0) for w in words)
    length = len(sentence.split())

    if length > max_len_words:
        base *= (max_len_words / float(length))

    title_score = _title_overlap_score(sentence, title_terms) * 3.0
    claim_bonus = _pattern_bonus(sentence, _CLAIM_PATTERNS, 5.0)
    result_bonus = _pattern_bonus(sentence, _RESULT_PATTERNS, 5.0)
    conclusion_bonus = _pattern_bonus(sentence, _CONCLUSION_PATTERNS, 4.0)
    contrast_bonus = _pattern_bonus(sentence, _CONTRAST_PATTERNS, 3.0)
    position = _position_bonus(index, total_sentences)
    section_score = _section_bonus(section)
    background_penalty = _pattern_bonus(sentence, _BACKGROUND_PATTERNS, 2.5)
    noise = _noise_penalty(sentence)

    score = (
        base
        + title_score
        + claim_bonus
        + result_bonus
        + conclusion_bonus
        + contrast_bonus
        + position
        + section_score
        - background_penalty
        - noise
    )

    debug = {
        "index": index,
        "sentence": sentence,
        "score": round(score, 3),
        "section": section,
        "slot": _classify_slot(sentence, section),
        "base_frequency": round(base, 3),
        "title_score": round(title_score, 3),
        "claim_bonus": claim_bonus,
        "result_bonus": result_bonus,
        "conclusion_bonus": conclusion_bonus,
        "contrast_bonus": contrast_bonus,
        "position_bonus": position,
        "section_bonus": section_score,
        "background_penalty": background_penalty,
        "noise_penalty": noise,
        "claim_matches": _matched_patterns(sentence, _CLAIM_PATTERNS),
        "result_matches": _matched_patterns(sentence, _RESULT_PATTERNS),
        "conclusion_matches": _matched_patterns(sentence, _CONCLUSION_PATTERNS),
        "contrast_matches": _matched_patterns(sentence, _CONTRAST_PATTERNS),
    }

    return score, debug


def _word_overlap_ratio(a: str, b: str) -> float:
    # Measures how much two sentences overlap based on shared meaningful words
    a_words = set(_tokenize(a))
    b_words = set(_tokenize(b))

    if not a_words or not b_words:
        return 0.0

    return len(a_words & b_words) / max(1, min(len(a_words), len(b_words)))


def _is_too_similar(candidate: str, selected: List[str], threshold: float = 0.65) -> bool:
    # Checks whether a candidate sentence is too similar to already selected sentences
    return any(_word_overlap_ratio(candidate, s) >= threshold for s in selected)


def _select_with_slots(
    scored: List[Tuple[float, int, str, str, Dict[str, Any]]],
    *,
    max_sentences: int,
) -> List[Tuple[float, int, str, str, Dict[str, Any]]]:

    # Selects sentences using coverage slots
    # This prevents the summary from selecting only background or only method sentences.

    selected: List[Tuple[float, int, str, str, Dict[str, Any]]] = []

    # Desired minimum coverage
    slot_targets = {
        "aim": 2,
        "claim": 2,
        "method": 2,
        "result": 2,
        "conclusion": 1,
    }

    for slot, target in slot_targets.items():
        candidates = [item for item in scored if item[3] == slot]
        for item in candidates:
            if len([s for s in selected if s[3] == slot]) >= target:
                break
            if len(selected) >= max_sentences:
                break

            sentence = item[2]
            if not _is_too_similar(sentence, [s[2] for s in selected]):
                selected.append(item)

    # Fills remaining with best scoring sentences, preserving diversity
    for item in scored:
        if len(selected) >= max_sentences:
            break

        sentence = item[2]
        if item in selected:
            continue

        if not _is_too_similar(sentence, [s[2] for s in selected]):
            selected.append(item)

    # If diversity filtering was too strict, fill anyway
    for item in scored:
        if len(selected) >= max_sentences:
            break
        if item not in selected:
            selected.append(item)

    selected.sort(key=lambda x: x[1])
    return selected


def extractive_summary(
    text: str,
    *,
    max_sentences: int = 15,
    min_sentence_words: int = 8,
    max_sentence_words: int = 55,
    min_summary_words: int = 220,
    title: Optional[str] = None,
    return_debug: bool = False,
) -> ExtractiveSummaryResult:
    text = _remove_references_from_structured_text(text)
    original_words = len(text.split())
    if original_words == 0:
        raise ValueError("Cannot summarise empty text")

    sectioned_sentences = _sectioned_sentence_split(text)

    sentences = [item.text for item in sectioned_sentences]
    total_sentences = len(sentences)
    title_terms = _title_keywords(title or "")

    sentence_sections: Dict[int, str] = {
        item.index: item.section for item in sectioned_sentences
    }

    candidates = []
    for idx, s in enumerate(sentences):
        wc = len(s.split())
        section = sentence_sections.get(idx, "unknown")

        if section == "references":
            continue

        if min_sentence_words <= wc <= max_sentence_words:
            if _is_metadata_or_header(s):
                continue

            if _noise_penalty(s) < 8.0 or _is_result_sentence(s):
                candidates.append(s)

    if not candidates:
        fallback = [
                       s for s in sentences
                       if _noise_penalty(s) < 8.0
                   ][:max_sentences]

        summary_text = " ".join(fallback).strip()

        return ExtractiveSummaryResult(
            summary_text=summary_text,
            selected_sentences=fallback,
            compression_ratio=len(summary_text.split()) / max(1, original_words),
            debug_scores=None,
        )

    freq = _word_frequencies(candidates)

    scored: List[Tuple[float, int, str, str, Dict[str, Any]]] = []
    all_debug: List[Dict[str, Any]] = []

    for idx, s in enumerate(sentences):
        if s not in candidates:
            continue

        # Extra safety check: not score metadata/header artefacts
        if _is_metadata_or_header(s):
            continue

        section = sentence_sections.get(idx, "unknown")
        score, debug = _score_sentence(
            s,
            freq,
            index=idx,
            total_sentences=total_sentences,
            title_terms=title_terms,
            section=section,
        )

        slot = debug.get("slot", _classify_slot(s, section))
        scored.append((score, idx, s, slot, debug))
        all_debug.append(debug)

    scored.sort(key=lambda x: x[0], reverse=True)

    selected_items = _select_with_slots(
        scored,
        max_sentences=max_sentences,
    )

    # ---------------------------------------------------------
    # Safety check 1:
    # If diversity selection produced too few sentences
    # relaxes selection and adds more high scoring sentences
    # ---------------------------------------------------------
    MIN_SELECTED_SENTENCES = max(8, max_sentences // 2)

    if len(selected_items) < MIN_SELECTED_SENTENCES:
        for item in scored:
            if len(selected_items) >= MIN_SELECTED_SENTENCES:
                break

            if item not in selected_items:
                selected_items.append(item)

    # Keeps selected sentences in original document order
    selected_items.sort(key=lambda x: x[1])

    selected = [s.strip() for _, _, s, _, _ in selected_items]
    summary_text = " ".join(selected).strip()

    # ---------------------------------------------------------
    # Safety check 2:
    # If the selected summary is still too short, adds more
    # high-scoring sentences until a minimum word count is reached
    # or the maximum number of sentences is reached
    # ---------------------------------------------------------
    MIN_SUMMARY_WORDS = min_summary_words

    if len(summary_text.split()) < MIN_SUMMARY_WORDS:
        for item in scored:
            if len(selected_items) >= max_sentences:
                break

            if item not in selected_items:
                selected_items.append(item)

                # Keeps document order after adding each sentence
                selected_items.sort(key=lambda x: x[1])

                selected = [s for _, _, s, _, _ in selected_items]
                summary_text = " ".join(selected).strip()

            if len(summary_text.split()) >= MIN_SUMMARY_WORDS:
                break

    # Final selected sentence list after all fallback checks
    selected_items.sort(key=lambda x: x[1])
    selected = [s.strip() for _, _, s, _, _ in selected_items]
    summary_text = " ".join(selected).strip()
    selected_indexes = {idx for _, idx, _, _, _ in selected_items}

    if return_debug:
        for d in all_debug:
            d["selected"] = d["index"] in selected_indexes
        debug_scores = sorted(all_debug, key=lambda d: d["score"], reverse=True)
    else:
        debug_scores = None

    return ExtractiveSummaryResult(
        summary_text=summary_text,
        selected_sentences=selected,
        compression_ratio=len(summary_text.split()) / max(1, original_words),
        debug_scores=debug_scores,
    )

