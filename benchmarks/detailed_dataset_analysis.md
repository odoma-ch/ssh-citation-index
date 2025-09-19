# Dataset Analysis Summary

## Overview
This analysis covers three different citation datasets used for benchmarking:

1. **CEX (Citation Extraction)** - 112 documents from 27 academic categories
2. **EXCITE** - 351 documents in German and English  
3. **LinkedBook** - 1,194 individual reference entries (test set)

## Key Statistics

### CEX Dataset
- **Total Documents**: 112
- **Total References**: 5,160
- **Average References per Document**: 46.07
- **Document Length**: 14.12 pages average (1,581 total pages)
- **Categories**: 27 different academic fields (evenly distributed with 4 docs each, except VET with 8)
- **Language**: English
- **Reference Range**: 10-113 references per document

### EXCITE Dataset  
- **Total Documents**: 351
- **Total References**: 10,171
- **Average References per Document**: 28.98
- **Document Length**: 22.91 pages average (8,041 total pages)
- **Languages**: German (71.5%) and English (28.5%)
- **Reference Range**: 2-1,026 references per document (high variance)
- **Note**: Contains some outliers with very high reference counts

### LinkedBook Dataset (Test Set)
- **Total Reference Entries**: 1,194 individual references (no full documents)
- **Languages**: 7 languages with detailed distribution:
  - Italian (IT): 863 references (72.3%)
  - English (EN): 219 references (18.3%)
  - German (DE): 51 references (4.3%)
  - French (FR): 47 references (3.9%)
  - Spanish (ES): 7 references (0.6%)
  - Portuguese (PT): 6 references (0.5%)
  - Dutch (NL): 1 reference (0.1%)
- **Average Reference Length**: 107 characters, 18 words
- **Structure**: Individual reference entries rather than full documents
- **Purpose**: Reference parsing and structuring evaluation

## Dataset Comparison

| Metric | CEX | EXCITE | LinkedBook |
|--------|-----|--------|------------|
| Documents | 112 | 351 | N/A* |
| Total References | 5,160 | 10,171 | 1,194 |
| Avg Refs/Doc | 46.07 | 28.98 | N/A* |
| Avg Pages/Doc | 14.12 | 22.91 | N/A |
| Languages | EN | DE (71.5%), EN (28.5%) | IT (72.3%), EN (18.3%), DE (4.3%), FR (3.9%), others (1.0%) |
| Domain Coverage | 27 categories | Mixed | Mixed |

*LinkedBook contains only individual reference entries, not full documents

## Key Insights

1. **CEX** has the most consistent reference distribution and focuses on comprehensive academic coverage across disciplines.

2. **EXCITE** has the largest document collection and highest page count, with significant variance in reference counts (some documents have over 1000 references).

3. **LinkedBook** serves a different purpose - it contains only individual reference entries (not full documents) for reference parsing evaluation, with a strong Italian language focus (72.3% of references).

4. **Document Complexity**: 
   - CEX: Medium-length academic papers (14 pages avg)
   - EXCITE: Longer documents (23 pages avg) with high variability
   - LinkedBook: Focus on reference-level analysis

5. **Language Diversity**:
   - CEX: English only
   - EXCITE: Primarily German (71.5%) with English (28.5%)
   - LinkedBook: 7 European languages, dominated by Italian (72.3%), followed by English (18.3%)

## Use Cases

- **CEX**: Best for evaluating extraction across diverse academic domains
- **EXCITE**: Good for testing scalability and handling variable document lengths
- **LinkedBook**: Ideal for reference parsing and multi-language evaluation (particularly Italian references)

## Generated Files

All files are located in the `benchmarks/` folder:

1. `dataset_statistics.py` - The analysis script (run with `python dataset_statistics.py`)
2. `dataset_summary.csv` - Tabular summary statistics
3. `dataset_statistics.png` - Visual comparison charts
4. `detailed_dataset_analysis.md` - This comprehensive analysis

## Usage

To regenerate the analysis, run from the `benchmarks/` directory:
```bash
conda activate citation_index
python dataset_statistics.py
```
