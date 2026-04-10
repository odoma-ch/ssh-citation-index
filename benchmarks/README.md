# Benchmarks

Evaluation scripts, datasets, and utilities for measuring the citation extraction, parsing, and linking pipelines.

## Prerequisites

Install the main project in editable mode from the repository root:

```bash
pip install -r requirements.txt
pip install -e .
```

Most runner scripts expect an LLM endpoint (vLLM-compatible) or a GROBID server. Set at least:

```bash
export LLM_API_KEY="..."          # or DEEPSEEK_API_KEY
export LLM_ENDPOINT="http://localhost:8000/v1"
```

All commands below assume you run them **from the repository root**.

---

## Folder structure

```
benchmarks/
├── excite/              # EXCITE corpus (SSH, German/English)
├── cex/                 # CEX corpus (multi-category SSH)
├── linkedbook/          # LinkedBook corpus (multilingual CoNLL-derived)
├── citation_linking/    # API search benchmarks (OpenAlex, Wikidata, …)
├── finetune/            # Finetuning dataset generation
├── brill/               # Brill Knowledge Graph sampling + LLM parse (used for citation linking)
├── error_analysis/      # Stored outputs for manual error inspection
├── dataset_statistics.py
├── detailed_dataset_analysis.md
└── benchmarking_result.md
```

Each dataset folder follows a common layout:

- `all_pdfs/` — source PDF files
- `all_xml/` or `all_xmls/` — TEI/XML ground-truth references
- `*_helper.py` — data loading, preprocessing, evaluation helpers
- `run_*_bench.py` — CLI benchmark runner
- `*_report.ipynb` — Jupyter notebook for result analysis
- `outputs/` — benchmark run artifacts (pickles, CSVs, JSONs)

---

## Running benchmarks

### EXCITE

Supports three tasks: `extraction`, `parsing`, and `extraction_and_parsing`.

```bash
# Full extraction + parsing, using Marker extractor and LLM parser
python benchmarks/excite/run_excite_bench.py \
    --task extraction_and_parsing \
    --method 1 \
    --extractor marker \
    --parser llm \
    --model_name "mistralai/Mistral-Small-3.2-24B-Instruct-2506" \
    --api_base http://localhost:8000/v1 \
    --output_path benchmarks/excite/outputs \
    --limit 10  # quick test run

# Parsing only, with GROBID
python benchmarks/excite/run_excite_bench.py \
    --task parsing \
    --parser grobid \
    --grobid_endpoint http://localhost:8070

# Re-evaluate saved responses without calling the LLM
python benchmarks/excite/run_excite_bench.py \
    --task parsing \
    --responses_path benchmarks/excite/outputs/some_run.pkl
```

Key flags: `--method` (1–5, extraction strategies), `--per_class` (breakdown by document class), `--focus_fields` (which fields to score), `--mode` (exact/fuzzy/soft_fuzzy).

### CEX

Same interface as EXCITE, tuned for CEX categories:

```bash
python benchmarks/cex/run_cex_bench.py \
    --task extraction_and_parsing \
    --method 1 \
    --extractor marker \
    --parser llm \
    --model_name "mistralai/Mistral-Small-3.2-24B-Instruct-2506" \
    --api_base http://localhost:8000/v1 \
    --output_path benchmarks/cex/outputs \
    --per_category  # breakdown by document category
```

**Silver standard generation** — build a silver-standard reference set from CEX using LLM extraction:

```bash
python benchmarks/cex/generate_cex_silver_standard.py \
    --extractor marker \
    --model_name "google/gemma-3-27b-it" \
    --api_base http://localhost:8000/v1 \
    --output_path benchmarks/cex/silver_standard
```

### LinkedBook

Parsing-only benchmark on multilingual reference strings:

```bash
# Single-reference mode
python benchmarks/linkedbook/run_linkedbook_bench.py \
    --mode single \
    --parser llm \
    --model_name "mistralai/Mistral-Small-3.2-24B-Instruct-2506" \
    --api_base http://localhost:8001/v1 \
    --output_path benchmarks/linkedbook/outputs

# Grouped (batch) mode with per-language breakdown
python benchmarks/linkedbook/run_linkedbook_bench.py \
    --mode grouped \
    --per_category \
    --detailed_analysis
```

### Citation linking

First build the test set from the three corpora, then run the API search benchmark:

```bash
# Build test set
python benchmarks/citation_linking/build_test_set.py

# Run search benchmark against OpenAlex, OpenCitations, Wikidata, Matilda
python benchmarks/citation_linking/run_api_search_benchmark.py \
    --test-set benchmarks/citation_linking/api_search_test_set.jsonl \
    --email your@email.com \
    --top-k 3 \
    --limit 50

# Legal study benchmark (LLaMoRe XML data)
python benchmarks/citation_linking/run_legal_study_benchmark.py
```

Use `--skip-apis` to exclude specific APIs (e.g. `--skip-apis opencitations matilda`).

---

## Finetuning dataset generation

Produces chat-format JSONL from all three corpora for supervised finetuning:

```bash
python benchmarks/finetune/generate_finetuning_dataset.py
# Optional: --output-suffix "_v2"
```

Outputs `finetuning_data.jsonl` and `finetuning_references_metadata.jsonl`. Stable document sampling is controlled by `finetuning_used_ids.json`.



## Common flags

Most runner scripts share these options:

| Flag | Description |
|------|-------------|
| `--limit N` | Process only N documents (for quick test runs) |
| `--max_workers N` | Concurrent LLM/API requests |
| `--verbose` / `-v` | Debug logging |
| `--fuzzy_threshold` | Matching threshold for evaluation (0–100) |
| `--mode` | Evaluation mode: `exact`, `fuzzy`, or `soft_fuzzy` |
| `--focus_fields` | Fields to include in scoring (e.g. `full_title authors publication_date`) |
| `--output_path` | Where to write results |
| `--skip_save` | Run evaluation without writing output files |
| `--save_scores PATH` | Append a one-line summary to a shared scores file |

Run any script with `--help` for the full list of options.
