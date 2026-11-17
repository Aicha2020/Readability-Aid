from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptBundle:
    system_prompt: str
    user_prompt: str


def build_readability_rewrite_prompt(
    *,
    simplified_summary: str,
    target_audience: str = "undergraduate students, especially students with reading difficulties",
    target_smog: float = 12.0,
) -> PromptBundle:

    # Prompt for local LLM rewriting with readability awareness.

    system_prompt = (
        "You rewrite scientific summaries in simpler English. "
        "Your task is to rewrite, not shorten aggressively. "
        "Keep all main ideas from the input summary. "
        "Do not remove claims, findings, conclusions, methods, or important context. "
        "Make the text easier to read by using shorter sentences and simpler words. "
        "Do not add new facts. "
        "Do not invent numbers. "
        "Keep important technical meaning. "
        "Do not use bullet points. "
        "Return only the rewritten summary."
    )

    user_prompt = f"""
Rewrite the summary below for {target_audience}.

Main goal:
- Make it easier to read.
- Aim for a SMOG level close to {target_smog:.1f} or lower.
- Rewrite the summary; do not create a much shorter summary.

Important rules:
- Keep every main idea from the input summary.
- Keep the central claim, main findings, conclusion, and important context.
- Do not remove whole ideas just to make the text shorter.
- The rewritten summary should be at least about half as long as the input summary.
- Use short sentences.
- Use simple, familiar words.
- Prefer common words over academic words.
- Do not add new information.
- Do not invent or guess missing facts.
- Keep important numbers if they are part of the main message.
- Keep important technical terms if needed for meaning.
- Remove table-like statistics or numeric lists only if they are not essential.
- Do not use bullet points.
- Do not explain what you changed.
- Return only the rewritten summary.

Examples of simpler wording:
- "use" instead of "utilize"
- "help" instead of "facilitate"
- "about" instead of "approximately"
- "show" instead of "demonstrate"
- "many" instead of "numerous"

Summary:
\"\"\"
{simplified_summary}
\"\"\"
""".strip()

    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
