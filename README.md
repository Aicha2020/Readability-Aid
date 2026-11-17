# Readability Aid for Academic Papers

## Project Overview

This project implements an evaluation-driven pipeline for improving the readability of academic and scientific papers for student readers, especially students with reading difficulties such as dyslexia.

The aim is to automatically extract, summarise, simplify, and evaluate academic research text while preserving the main meaning of the original paper.

The system combines extractive summarisation, deterministic text simplification, readability evaluation, and optional local LLM rewriting using Ollama.

## Project Structure

```text
readability_aid/
├── data/
│   ├── raw_pdfs/                  # Input PDF papers
│   ├── extracted_text/            # Generated summaries and results
│   └── logs/                      # Pipeline logs
│
├── src/
│   ├── main.py                    # Main pipeline entry point
│   ├── ingestion/
│   │   ├── pdf_extractor.py       # PDF extraction and quality checks
│   │   └── ieee_extractor.py      # Layout-aware PDF extraction
│   ├── summarisation/
│   │   ├── extractive_summariser.py
│   │   └── deterministic_simplifier.py
│   ├── evaluation/
│   │   └── readability.py         # Readability metrics
│   ├── validation/
│   │   └── output_checks.py       # LLM output validation
│   └── agent/
│       ├── llm_client.py          # Ollama API client
│       ├── orchestrator.py        # Pipeline orchestration
│       └── prompts.py             # LLM prompts
│
├── .gitignore
├── requirements.txt
└── README.md
```

**Note:** The `data/` folders are used locally and are excluded from GitHub through `.gitignore`.

## Requirements

- Python 3
- Ollama (for optional LLM rewriting)
- A locally installed Ollama model, such as `gemma3:4b`

### Python Dependencies

The required Python packages are:

- `pdfplumber`
- `pypdf`
- `textstat`
- `python-dotenv`
- `requests`

Install the dependencies from the project root:

```bash
pip install -r requirements.txt
```

## Ollama Setup

The system uses a locally hosted language model through Ollama.

1. Download and install [Ollama](https://ollama.com/).
2. Download the model:

   ```bash
   ollama pull gemma3:4b
   ```

3. Ensure the Ollama service is running locally.

Ollama normally runs at `http://localhost:11434`.

## Environment Configuration

Configure the following environment variables in your local environment or `.env` file:

```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=gemma3:4b
LLM_TIMEOUT_SECONDS=420
LLM_MAX_RETRIES=2
LLM_TEMPERATURE=0.2
```

The `LLM_API_KEY` value is a placeholder for the standard local Ollama setup. Local Ollama normally does not require a real API key, but this project's client requires a non-empty value.

The `.env` file should remain local and must not be committed to GitHub.

## Running the Project

Place the academic PDF files you want to process inside:

```text
data/raw_pdfs/
```

From the project root, run:

```bash
python -m src.main
```

The pipeline processes the PDF files and generates summaries and readability evaluation results.

## Processing Pipeline

```text
PDF Input
    |
    v
Text Extraction and Cleaning
    |
    v
Extraction Quality Assessment
    |
    v
Extractive Seed Summary
    |
    v
Deterministic Simplification
    |
    v
Readability Evaluation
    |
    v
Conditional Ollama LLM Rewriting
    |
    v
Output Validation and Fallback
    |
    v
Final Summary
```

The system uses the SMOG Index as its primary readability decision metric, alongside Flesch–Kincaid Grade Level, Gunning Fog, Coleman–Liau Index, and Automated Readability Index.

If extraction quality is too low, the paper is rejected. If LLM rewriting fails validation, the deterministic summary is retained.

## Limitations

- Complex PDF layouts can affect extraction accuracy.
- Scanned PDFs require additional OCR support.
- Readability scores do not directly measure reader comprehension.
- LLM rewriting may produce inconsistent results.
- Further evaluation with students would help assess practical accessibility.

## Project Status

This project is a working research prototype developed to investigate the use of hybrid summarisation and simplification techniques for improving the readability of academic papers.