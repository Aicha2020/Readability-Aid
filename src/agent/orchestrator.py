from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

from src.agent.llm_client import LLMClient
from src.agent.prompts import build_readability_rewrite_prompt
from src.validation.output_checks import validate_llm_output
from src.evaluation.readability import evaluate_readability


@dataclass
class OrchestrationResult:

    # Final result returned by the orchestration layer.

    final_text: str
    final_source: str

    needs_llm_rewrite: bool
    llm_attempted: bool
    llm_success: bool
    llm_error: str
    llm_validation_passed: bool
    llm_validation_warnings: list[str]

    deterministic_profile: Any
    llm_profile: Optional[Any]
    final_profile: Any
    rejected_llm_text: Optional[str]


class SummaryOrchestrator:

    # Responsibility:
    # - decide whether deterministic output is good enough
    # - call the LLM only when needed
    # - validate the LLM output
    # - compare readability
    # - choose the final output safely

    def __init__(
        self,
        *,
        llm_client: Optional[LLMClient],
        logger: Optional[logging.Logger] = None,
        target_smog: float = 12.0,
        enable_llm: bool = False,
        target_audience: str = "undergraduate students, especially students with reading difficulties",
    ) -> None:
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)
        self.target_smog = target_smog
        self.enable_llm = enable_llm
        self.target_audience = target_audience

    # Run the orchestration logic
    def run(
        self,
        *,
        deterministic_summary: str,
    ) -> OrchestrationResult:

        # -----------------------------------------------------
        # 1. Evaluate deterministic summary
        # ----------------------------------------------------
        deterministic_profile = evaluate_readability(deterministic_summary)
        needs_llm_rewrite = deterministic_profile.smog_index > self.target_smog

        if needs_llm_rewrite:
            if self.enable_llm:
                self.logger.warning(
                    f"Deterministic SMOG {deterministic_profile.smog_index:.2f} "
                    f"> target {self.target_smog:.1f} → invoking LLM rewrite"
                )
            else:
                self.logger.info(
                    f"Deterministic SMOG {deterministic_profile.smog_index:.2f} "
                    f"> target {self.target_smog:.1f} → LLM stage skipped (disabled)"
                )
        else:
            self.logger.info(
                f"Deterministic SMOG {deterministic_profile.smog_index:.2f} "
                f"≤ target {self.target_smog:.1f} → deterministic summary accepted"
            )

        # Defaults
        llm_attempted = False
        llm_success = False
        llm_error = ""
        llm_validation_passed = False
        llm_validation_warnings: list[str] = []
        llm_profile = None
        llm_output_text = None
        rejected_llm_text = None

        # -----------------------------------------------------
        # 2. Optional LLM rewrite
        # -----------------------------------------------------
        if needs_llm_rewrite and self.enable_llm and self.llm_client is not None:
            llm_attempted = True

            prompt_bundle = build_readability_rewrite_prompt(
                simplified_summary=deterministic_summary,
                target_audience=self.target_audience,
                target_smog=self.target_smog,
            )

            response = self.llm_client.rewrite_text(
                system_prompt=prompt_bundle.system_prompt,
                user_prompt=prompt_bundle.user_prompt,
            )

            if response.success and response.text.strip():
                candidate_llm_text = response.text.strip()

                validation = validate_llm_output(
                    source_text=deterministic_summary,
                    candidate_text=candidate_llm_text,
                )

                llm_validation_passed = validation.passed
                llm_validation_warnings = validation.warnings

                if validation.passed:
                    llm_output_text = candidate_llm_text
                    llm_profile = evaluate_readability(llm_output_text)
                    llm_success = True

                    self.logger.info(
                        f"LLM SMOG {llm_profile.smog_index:.2f} "
                        f"(deterministic was {deterministic_profile.smog_index:.2f})"
                    )
                else:
                    llm_error = "Validation failed: " + "; ".join(validation.warnings)
                    self.logger.warning(f"LLM output rejected - {llm_error}")
                    rejected_llm_text = candidate_llm_text
            else:
                llm_error = response.error_message
                self.logger.warning(f"LLM rewrite failed or returned empty output ({llm_error})")

        # -----------------------------------------------------
        # 3. Choose final output
        # -----------------------------------------------------
        final_text = deterministic_summary
        final_profile = deterministic_profile
        final_source = "deterministic"

        # Using LLM output only if it passed validation and improved SMOG
        if llm_success and llm_profile is not None and llm_output_text is not None:
            if llm_profile.smog_index < deterministic_profile.smog_index:
                final_text = llm_output_text
                final_profile = llm_profile
                final_source = "llm"
                self.logger.info("Using LLM output as final summary")
            else:
                self.logger.info("Keeping deterministic summary because LLM did not improve SMOG")

        return OrchestrationResult(
            final_text=final_text,
            final_source=final_source,
            needs_llm_rewrite=needs_llm_rewrite,
            llm_attempted=llm_attempted,
            llm_success=llm_success,
            llm_error=llm_error,
            llm_validation_passed=llm_validation_passed,
            llm_validation_warnings=llm_validation_warnings,
            deterministic_profile=deterministic_profile,
            llm_profile=llm_profile,
            final_profile=final_profile,
            rejected_llm_text=rejected_llm_text,
        )

