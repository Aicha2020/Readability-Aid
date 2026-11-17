from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src.agent.llm_client import load_llm_config, LLMClient
from src.agent.orchestrator import SummaryOrchestrator
from src.evaluation.readability import evaluate_readability
from src.ingestion.pdf_extractor import (
    extract_pdf,
    save_outputs,
    log_text_preview,
    clear_directory,
    setup_logger,
    RunRecord,
    log_readability_profile,
)
from src.summarisation.deterministic_simplifier import simplify_text_deterministic
from src.summarisation.extractive_summariser import extractive_summary


# ----------------------------
# Batch runner
# ----------------------------

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    input_dir = PROJECT_ROOT / "data" / "raw_pdfs"
    output_dir = PROJECT_ROOT / "data" / "extracted_text"
    log_dir = PROJECT_ROOT / "data" / "logs"

    logger = setup_logger(log_dir)

    TARGET_SMOG = 12.0
    ENABLE_LLM = True

    # --- LLM setup ---
    config = load_llm_config()
    llm_client = LLMClient(config, logger=logger)

    # --- Orchestrator setup ---
    orchestrator = SummaryOrchestrator(
        llm_client=llm_client,
        logger=logger,
        target_smog=TARGET_SMOG,
        enable_llm=ENABLE_LLM,
        target_audience="undergraduate students, especially students with reading difficulties",
    )

    # ---------- Clear output folder ----------
    CLEAR_OUTPUT = True
    if CLEAR_OUTPUT:
        clear_directory(output_dir, logger)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Find PDFs ----------
    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in: {input_dir}")

    run_records: list[RunRecord] = []

    # ---------- Process PDFs ----------
    for pdf_path in pdf_files:
        logger.info(f"Processing: {pdf_path.name}")

        try:
            # =========================================================
            # 1. Extract and clean PDF text
            # =========================================================
            final_text, structured_text, raw_text, meta = extract_pdf(
                pdf_path=str(pdf_path),
                max_pages=3,
                abstract_only=False,
                strip_references=True,
                method="ieee",
            )

            # Warns if extraction quality is low
            if meta.likely_scanned or meta.extracted_words < 150:
                logger.warning(
                    f"{pdf_path.name}: low text extracted "
                    f"(words={meta.extracted_words}, likely_scanned={meta.likely_scanned})"
                )

            # Preview RAW extraction for debugging
            log_text_preview(logger, raw_text)

            # Saves cleaned extracted text + metadata
            txt_out, json_out = save_outputs(final_text, meta, out_dir=str(output_dir))
            logger.info(f"Saved text: {txt_out}")
            logger.info(f"Saved meta: {json_out}")

            logger.info(
                f"{pdf_path.name}: extraction method={meta.used_method}; "
                f"quality_score={meta.extraction_quality_score:.2f}; "
                f"quality_passed={meta.extraction_quality_passed}"
            )

            if meta.extraction_quality_warnings:
                logger.warning(
                    f"{pdf_path.name}: extraction quality warnings: "
                    f"{meta.extraction_quality_warnings}"
                )

            # =========================================================
            # 2. Deterministic Stage 1: Extractive seed summary
            # =========================================================
            seed = extractive_summary(
                final_text,
                max_sentences=15,
                min_sentence_words=8,
                max_sentence_words=55,
                min_summary_words=220,
                title=Path(meta.pdf_path).stem,
            )
            seed_text = seed.summary_text

            seed_path = output_dir / f"{Path(meta.pdf_path).stem}_seed_summary.txt"
            seed_path.write_text(seed_text, encoding="utf-8")
            logger.info(f"Saved seed summary: {seed_path}")

            # =========================================================
            # 3. Deterministic Stage 2: Rule based simplification
            # =========================================================
            simplified = simplify_text_deterministic(seed_text)
            simplified_text = simplified.simplified_text

            simplified_path = output_dir / f"{Path(meta.pdf_path).stem}_simplified_summary.txt"
            simplified_path.write_text(simplified_text, encoding="utf-8")
            logger.info(
                f"Saved simplified summary: {simplified_path} "
                f"(ops={simplified.operations}, replacements={simplified.replacements_made})"
            )

            # =========================================================
            # 4. Readability scoring before orchestration
            # =========================================================
            before_profile = evaluate_readability(final_text)
            seed_profile = evaluate_readability(seed_text)

            log_readability_profile(logger, "Original extracted text", before_profile)
            log_readability_profile(logger, "Seed summary", seed_profile)

            # =========================================================
            # 5. Orchestration (SMOG gate + optional LLM + validation + selection)
            # =========================================================
            orchestration = orchestrator.run(
                deterministic_summary=simplified_text,
            )

            final_output_text = orchestration.final_text
            final_output_source = orchestration.final_source

            deterministic_profile = orchestration.deterministic_profile
            llm_profile = orchestration.llm_profile
            final_profile = orchestration.final_profile

            log_readability_profile(logger, "Deterministic summary", deterministic_profile)

            if llm_profile is not None:
                log_readability_profile(logger, "LLM summary", llm_profile)
            else:
                logger.info("LLM summary readability: not available")

            log_readability_profile(logger, "Final selected output", final_profile)

            needs_llm_rewrite = orchestration.needs_llm_rewrite
            llm_attempted = orchestration.llm_attempted
            llm_success = orchestration.llm_success
            llm_error = orchestration.llm_error
            llm_validation_passed = orchestration.llm_validation_passed
            validation_warnings = orchestration.llm_validation_warnings
            rejected_llm_text = orchestration.rejected_llm_text

            # Saves LLM output only if it exists and was accepted
            if llm_success and llm_profile is not None and final_output_source == "llm":
                llm_path = output_dir / f"{Path(meta.pdf_path).stem}_llm_summary.txt"
                llm_path.write_text(final_output_text, encoding="utf-8")
                logger.info(f"Saved LLM summary: {llm_path}")

            # Saves rejected LLM output if validation failed
            if rejected_llm_text:
                rejected_path = output_dir / f"{Path(meta.pdf_path).stem}_llm_rejected.txt"
                rejected_path.write_text(rejected_llm_text, encoding="utf-8")
                logger.info(f"Saved rejected LLM summary: {rejected_path}")

            # Saves final selected output
            final_output_path = output_dir / f"{Path(meta.pdf_path).stem}_final_summary.txt"
            final_output_path.write_text(final_output_text, encoding="utf-8")
            logger.info(f"Saved final summary: {final_output_path}")

            # =========================================================
            # 6. Save comparison JSON
            # =========================================================
            comparison = {
                "pdf": Path(meta.pdf_path).name,
                "method": meta.used_method,
                "extracted_words": meta.extracted_words,
                "seed_compression_ratio": seed.compression_ratio,
                "simplification_ops": simplified.operations,
                "replacements_made": simplified.replacements_made,
                "target_smog": TARGET_SMOG,

                # LLM tracking
                "needs_llm_rewrite": needs_llm_rewrite,
                "llm_enabled": ENABLE_LLM,
                "llm_attempted": llm_attempted,
                "llm_success": llm_success,
                "llm_error": llm_error,

                # Validation
                "llm_validation_passed": llm_validation_passed,
                "llm_validation_warnings": validation_warnings,

                # Final decision
                "final_output_source": final_output_source,
                "llm_used": final_output_source == "llm",

                # Readability profiles
                "readability_original": before_profile.__dict__,
                "readability_seed": seed_profile.__dict__,
                "readability_deterministic": deterministic_profile.__dict__,
                "readability_llm": llm_profile.__dict__ if llm_profile is not None else None,
                "readability_final": final_profile.__dict__,

                # Helpful evaluation flag
                "llm_improved_smog": (
                    llm_profile.smog_index < deterministic_profile.smog_index
                    if llm_profile is not None else None
                ),

                "delta_before_to_seed": {
                    "fkgl": seed_profile.flesch_kincaid_grade - before_profile.flesch_kincaid_grade,
                    "smog": seed_profile.smog_index - before_profile.smog_index,
                    "fog": seed_profile.gunning_fog - before_profile.gunning_fog,
                    "coleman_liau": seed_profile.coleman_liau_index - before_profile.coleman_liau_index,
                    "ari": seed_profile.automated_readability_index - before_profile.automated_readability_index,
                },

                "delta_seed_to_deterministic": {
                    "fkgl": deterministic_profile.flesch_kincaid_grade - seed_profile.flesch_kincaid_grade,
                    "smog": deterministic_profile.smog_index - seed_profile.smog_index,
                    "fog": deterministic_profile.gunning_fog - seed_profile.gunning_fog,
                    "coleman_liau": deterministic_profile.coleman_liau_index - seed_profile.coleman_liau_index,
                    "ari": deterministic_profile.automated_readability_index - seed_profile.automated_readability_index,
                },

                "delta_deterministic_to_final": {
                    "fkgl": final_profile.flesch_kincaid_grade - deterministic_profile.flesch_kincaid_grade,
                    "smog": final_profile.smog_index - deterministic_profile.smog_index,
                    "fog": final_profile.gunning_fog - deterministic_profile.gunning_fog,
                    "coleman_liau": final_profile.coleman_liau_index - deterministic_profile.coleman_liau_index,
                    "ari": final_profile.automated_readability_index - deterministic_profile.automated_readability_index,
                },

                "delta_before_to_deterministic": {
                    "fkgl": deterministic_profile.flesch_kincaid_grade - before_profile.flesch_kincaid_grade,
                    "smog": deterministic_profile.smog_index - before_profile.smog_index,
                    "fog": deterministic_profile.gunning_fog - before_profile.gunning_fog,
                    "coleman_liau": deterministic_profile.coleman_liau_index - before_profile.coleman_liau_index,
                    "ari": deterministic_profile.automated_readability_index - before_profile.automated_readability_index,
                },

                "delta_before_to_final": {
                    "fkgl": final_profile.flesch_kincaid_grade - before_profile.flesch_kincaid_grade,
                    "smog": final_profile.smog_index - before_profile.smog_index,
                    "fog": final_profile.gunning_fog - before_profile.gunning_fog,
                    "coleman_liau": final_profile.coleman_liau_index - before_profile.coleman_liau_index,
                    "ari": final_profile.automated_readability_index - before_profile.automated_readability_index,
                },
            }

            cmp_path = output_dir / f"{Path(meta.pdf_path).stem}_readability_comparison.json"
            cmp_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
            logger.info(f"Saved readability comparison: {cmp_path}")

            logger.info(
                f"{pdf_path.name}: final output source = {final_output_source}; "
                f"final SMOG = {final_profile.smog_index:.2f}"
            )

            # =========================================================
            # 7. Record run status
            # =========================================================
            run_records.append(
                RunRecord(
                    pdf_name=pdf_path.name,
                    status="SUCCESS",
                    method=meta.used_method,
                    extracted_words=meta.extracted_words,
                    likely_scanned=meta.likely_scanned,
                    message="",
                )
            )

        except Exception as e:
            logger.error(f"FAILED: {pdf_path.name} - {e}")
            logger.exception(f"Traceback for: {pdf_path.name}")

            run_records.append(
                RunRecord(
                    pdf_name=pdf_path.name,
                    status="FAILED",
                    method="ieee",
                    message=str(e),
                )
            )
            continue

    # ---------- Save run report ----------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"_extraction_run_{ts}.json"
    report_path.write_text(json.dumps([asdict(r) for r in run_records], indent=2), encoding="utf-8")

    logger.info(f"Run report saved to: {report_path}")
    logger.info(
        f"Done. Total={len(run_records)} "
        f"Success={sum(r.status == 'SUCCESS' for r in run_records)} "
        f"Fail={sum(r.status == 'FAILED' for r in run_records)}"
    )

