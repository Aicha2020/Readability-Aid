# Readability Aid for Academic Papers

# Project Overview

This project implements an evaluation driven pipeline for improving the readability of academic and scientific papers for student readers, especially students with reading difficulties such as dyslexia.

The aim of this project is to automatically extract, summarise, simplify, and evaluate academic research text while preserving the main meaning of the original paper.


# Project structure:

readability_aid/
│
├── data/
│   ├── raw_pdfs/
│   │   └── input PDF papers
│   │
│   ├── extracted_text/
│   │   └── generated summaries, metadata, and comparison files
│   │
│   └── logs/
│       └── extraction and pipeline logs
│
├── src/
│   ├── main.py
│   │
│   ├── ingestion/
│   │   ├── pdf_extractor.py
│   │   └── ieee_extractor.py
│   │
│   ├── summarisation/
│   │   ├── extractive_summariser.py
│   │   └── deterministic_simplifier.py
│   │
│   ├── evaluation/
│   │   └── readability.py
│   │
│   ├── validation/
│   │   └── output_checks.py
│   │
│   └── agent/
│       ├── llm_client.py
│       ├── orchestrator.py
│       └── prompts.py
│
├── .env
├── requirements.txt
└── README.md

# Required Python Packages
pdfplumber
pypdf
textstat
python-dotenv
requests

The system uses a locally hosted LLM through Ollama
Download and install Ollama from the official Ollama website.
Pull a model: ollama pull gemma3:4b
Start Ollama: http://localhost:11434
Environment Variables:
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=gemma3:4b
LLM_TIMEOUT=240
LLM_MAX_RETRIES=3
