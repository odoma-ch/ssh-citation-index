# CEX Silver Standard Generator

This script generates a silver standard from the CEX dataset using reference extraction. It's designed as a one-time-use tool to create training data for downstream tasks.

## What it does

1. **Runs reference extraction** on all PDFs in the CEX dataset
2. **Generates a silver standard** similar to `all_references.json` but with extracted references instead of ground truth
3. **Provides statistics** on over/under/exact extraction patterns
4. **Saves detailed results** to `tmp.txt` for analysis

## Output Files

- **Silver Standard JSON**: `silver_standard_{model}_{extractor}_{timestamp}.json`
  - Contains all paper metadata with extracted references replacing ground truth
  - Includes extraction metadata (model used, timestamp, etc.)
- **Latest Copy**: `silver_standard_latest.json` (always points to most recent run)
- **Detailed Results**: `tmp.txt` with comprehensive extraction statistics

## Usage

### Basic Usage
```bash
python generate_cex_silver_standard.py
```

### With Custom Parameters
```bash
python generate_cex_silver_standard.py \
    --model_name "google/gemma-3-27b-it" \
    --extractor "marker" \
    --prompt_name "reference_extraction.md" \
    --max_workers 25 \
    --output_path "benchmarks/cex/silver_standard" \
    --per_category
```

### Quick Test Run (limit documents)
```bash
python generate_cex_silver_standard.py --limit 10
```

## Command Line Arguments

### LLM Configuration
- `--model_name`: LLM model to use (default: "google/gemma-3-27b-it")
- `--api_key`: API key for LLM endpoint (defaults to DEEPSEEK_API_KEY env var)
- `--api_base`: Base URL for LLM API (default: "http://localhost:8000/v1")

### Extraction Configuration
- `--extractor`: PDF text extractor ("pymupdf", "marker", "mineru")
- `--prompt_name`: Prompt file in prompts/ directory
- `--max_workers`: Maximum concurrent LLM requests (default: 25)

### Output Configuration
- `--output_path`: Directory to save results (default: "benchmarks/cex/silver_standard")
- `--per_category`: Show per-category breakdown statistics
- `--limit`: Limit number of documents for testing

### Other Options
- `-v, --verbose`: Enable debug logging

## Example Output

### Console Summary
```
============================================================
CEX SILVER STANDARD GENERATION SUMMARY
============================================================
Total documents processed: 108
Over-extracted: 45
Under-extracted: 38
Exact matches: 25

Percentages:
  Over-extracted: 41.7%
  Under-extracted: 35.2%
  Exact matches: 23.1%

Results saved to: benchmarks/cex/silver_standard
============================================================
```

### tmp.txt Contents
```
================================================================================
CEX SILVER STANDARD GENERATION RESULTS
================================================================================
Generated: 2024-01-15 14:30:25
Model: google/gemma-3-27b-it
Extractor: marker
Prompt: reference_extraction.md
Total documents processed: 108
================================================================================

EXTRACTION SUMMARY STATISTICS
----------------------------------------
Over-extracted: 45
Under-extracted: 38
Exact matches: 25

PERCENTAGES
--------------------
Over-extracted: 41.7%
Under-extracted: 35.2%
Exact matches: 23.1%

DETAILED EXTRACTION RESULTS
----------------------------------------
Format: file_id | category | GT_count | Pred_count | Difference | %Diff | Type
--------------------------------------------------------------------------------
AGR-BIO-SCI_1 | AGR-BIO-SCI | 35 | 42 | +7 | +20.0% | over
AGR-BIO-SCI_2 | AGR-BIO-SCI | 28 | 25 | -3 | -10.7% | under
...
```

## Silver Standard Format

The generated silver standard follows the same structure as `all_references.json`:

```json
{
  "AGR-BIO-SCI_1": {
    "title": "Original paper title...",
    "category": "AGR-BIO-SCI",
    "references": [
      "Extracted reference string 1",
      "Extracted reference string 2",
      "..."
    ],
    "extraction_info": {
      "extraction_timestamp": "2024-01-15T14:30:25.123456",
      "model_used": "google/gemma-3-27b-it",
      "extractor_used": "marker",
      "prompt_used": "reference_extraction.md",
      "gt_reference_count": 35,
      "extracted_reference_count": 42,
      "extraction_difference": 7
    }
  }
}
```

## Requirements

- Python 3.7+
- citation_index package and dependencies
- Access to LLM API endpoint
- CEX dataset in `benchmarks/cex/` directory

## Notes

- This script reuses extracted text from markdown files if available
- Results are saved with timestamps to avoid overwriting previous runs
- The `tmp.txt` file provides comprehensive analysis for debugging extraction quality
- Use `--limit` for testing before running on the full dataset
