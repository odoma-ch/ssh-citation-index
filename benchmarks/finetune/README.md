# Finetuning Datasets for Reference Parsing

Two complementary datasets for training reference parsing models, using **identical source documents** for consistency.

## Datasets

| Dataset | Task | Training | Validation | Size |
|---------|------|----------|------------|------|
| **Single** | Parse individual references | 25,772 examples | 1,235 examples | 42 MB |
| **Group** | Extract & parse from documents | 725 examples | 31 examples | 14 MB |

Both use the same 16 CEX + 35 EXCITE documents, reserving 412 documents (96 CEX + 316 EXCITE) for testing.

## Data Sources

- **LinkedBook**: 24,615 train / 1,055 valid tagged references (Italian, English, French, German, Spanish)
- **CEX**: 112 academic papers (27 categories), 16 used in train/valid, 96 reserved for test
- **EXCITE**: 351 papers (3 reference location classes), 35 used in train/valid, 316 reserved for test

## Format

Conversation-style JSON with system, user, and assistant messages. Assistant outputs structured JSON with reference fields (authors, title, journal, volume, pages, date, publisher, etc.).

```json
{
  "messages": [
    {"role": "system", "content": "<instructions>"},
    {"role": "user", "content": "<reference text or PDF>"},
    {"role": "assistant", "content": "{\"references\": [...]}"}
  ],
  "source": "linkedbook|cex|excite",
  "ref_count": 36
}
```

## Prompts

- **Single**: 10 variants from `prompts/reference_parsing_variants.yaml` (512-1K tokens)
- **Group**: 10 parsing variants (LinkedBook) + 3 extraction variants (CEX/EXCITE) from both YAML files (8K-32K tokens)

## ID Tracking

`finetuning_used_ids.json` tracks all CEX/EXCITE documents used in training/validation to exclude them from test sets.

## Usage

### Generation
```bash
# Single reference dataset
python generate_finetuning_dataset_single.py

# Group reference dataset
python generate_finetuning_dataset_group.py
```

### Training Strategy
1. Pre-train on **single** dataset (25K examples) → learn parsing accuracy
2. Finetune on **group** dataset (725 examples) → learn document-level extraction
3. Test on reserved 412 documents → measure performance

## Files

```
finetuning_train_single.json      42 MB   (25,772 examples)
finetuning_valid_single.json      2.1 MB  (1,235 examples)
finetuning_train_group.json       14 MB   (725 examples)
finetuning_valid_group.json       1.0 MB  (31 examples)
finetuning_used_ids.json          1.8 KB  (51 document IDs)
dataset_generation_summary.txt    (generation log)
```

## Notes

- Random seed 42 for reproducibility
- Stratified sampling by category (CEX) and class (EXCITE)
- Group dataset: LinkedBook refs grouped 1-200 per batch, CEX/EXCITE use full PDFs
- PDF extraction via PyMuPDF (max 100K chars)

