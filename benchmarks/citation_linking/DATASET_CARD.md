# Citation Linking

> Private release candidate. The builder generates the files and split statistics
> described below. Public release still requires DOI review and license information.

## 1. Graphia Citation Index API

SSH Citation Index modules are a collection of AI modules for extracting, parsing and disambiguating bibliographic references from publications in the Social Sciences and Humanities (SSH).

(Work on these modules is part of Odoma's contribution to deliverables [D4.2](https://operas.atlassian.net/wiki/spaces/GRAPHIA/pages/821329922/D4.2+-+Report+on+existing+datasets+methods+and+tools+for+the+SSH+Citation+Index) and [D4.4](https://operas.atlassian.net/wiki/spaces/GRAPHIA/pages/820838444/D4.4+-+Deployment+of+AI+modules+for+the+SSH+Citation+Index) in [WP4](https://operas.atlassian.net/wiki/spaces/GRAPHIA/pages/818087981/WP4+Artificial+Intelligence+Solutions+for+the+SSH+KG).)

## 2. Citation linking task and data

Citation linking connects a bibliographic reference to the corresponding record
in a target index. In candidate-based experiments, a model receives a reference
and a set of candidate records and selects the matching record, or indicates that
none of the candidates matches. Candidate sets differ between indexes: an
OpenAlex task and a Wikidata task for the same reference are separate examples.

This dataset provides annotated gold targets for OpenAlex, Matilda, and Wikidata,
using references from CEX, EXCITE, LinkedBook, Brill, and the MPILHLT legal-study
corpus. It includes both references with a confirmed target and references for
which no matching record was found during annotation. Each retained reference
has an annotation for all three indexes.

The release contains reference text, available parsed references, and gold IDs
and DOIs. Candidate lists and baseline predictions are excluded so that future
experiments can use their own retrieval and linking methods.

## 3. Data format

**One row represents one reference and one target index.** The same reference
appears in multiple rows with a shared `ref_id` and index-specific gold targets.

| Field | Type | Description |
|---|---|---|
| `ref_id` | string | Stable reference identifier shared across target indexes. |
| `source` | string | Original corpus: `cex`, `excite`, `linkedbook`, `brill`, or `legal_study_mpilhlt`. |
| `document_id` | string or null | Identifier of the citing document, scoped by `source`. |
| `reference` | string | Original reference text. |
| `parsed_reference` | struct or null | Provided structured reference, such as title, authors, and year; parsing provenance is documented separately. |
| `index` | string | Target index: `openalex`, `matilda`, or `wikidata`. |
| `gold_id` | string or null | Reviewed target ID in this index. Null means no match was found during annotation, or an unavailable target was cleared during review. |
| `gold_doi` | string or null | DOI from metadata of a human-confirmed match, reconciled across indexes. Conflicting or unknown values are null; null does not imply that the work has no DOI. |

`(ref_id, index)` uniquely identifies a row. Unresolved annotations and references
containing multiple publications are excluded from the finalized benchmark.
The DOI identifies the cited work and may be known even when `gold_id` is null for
a particular index. Provided parsed references are inputs, not necessarily
human-verified metadata; experiments may use them or parse the original text.
DOIs are normalized to lowercase bare identifiers. A DOI from a rejected
candidate is never used. Conflicting DOI evidence between indexes was reviewed
manually; resolutions are recorded as `REVIEW_OVERRIDES` in the build script. Where
both DOIs are valid for the same work, `gold_doi` is null and the case is listed in
`build_report.json`. Other DOI fields are not independently adjudicated.

## 4. Splits and how to use the dataset

Two Hugging Face configurations expose the same gold rows. Each has
`train`, `validation` (dev), and `test` splits, targeting **60% / 20% / 20%**.
The configurations are alternative evaluation protocols; their rows must not be
combined into a larger training or evaluation set.

| Configuration | Assignment unit | What the test evaluates |
|---|---|---|
| `reference_disjoint` **(default)** | `ref_id`; all target-index rows stay together | Linking references not seen during training, across the supported indexes. |
| `reference_index_disjoint` | `(ref_id, index)`; index-specific tasks can be assigned separately | Linking held-out reference–index combinations; the reference may have appeared in training against another index. |

### `reference_disjoint`

All index rows for a reference belong to one split. Confirmed duplicate references
or references to the same work are grouped before assignment. Balance source
proportions and gold-target availability for each index as closely as grouping and
sample size permit. Citing documents may occur in multiple splits; this protocol
does not claim evaluation on unseen documents.

This is the primary protocol for comparisons involving a shared model trained
across indexes, and provides aligned reference sets for per-index comparisons.

### `reference_index_disjoint`

Assign reference–index tasks within strata defined by `source`, `index`, and
whether `gold_id` is present. For example, a reference's OpenAlex task may be in
training while its Wikidata task is in test. Their candidate sets and gold IDs are
index-specific, but their reference text and cited work are shared.

This is a supplementary protocol for studying performance on new index-specific
tasks. For a model trained across indexes, report test results separately for
references seen and unseen in its training data. Independent per-index models
only see the training rows for their own index. This protocol alone does not
establish generalization to an entirely unseen index.

### Shared rules

- Preserve approximate observed source and target-availability proportions;
  do not oversample or force equal positive and negative counts.
- Rare strata may not occur in every split. Publish achieved counts and
  proportions alongside the release.
- Freeze assignments using a documented seed and release the split files.
  `reference_disjoint` uses five-fold grouped stratification, assigning three folds to
  training and one each to validation and test. It selects the most balanced
  assignment across 16 seeded trials and all validation/test fold combinations,
  using only source, index, and gold-target availability, never model results.
  `reference_index_disjoint` uses a seeded shuffle within each stratum and
  largest-remainder allocation: each split's stratum count is within one row
  of its 60/20/20 target. Rounding ties are broken using the seed.
- All candidate sets and candidate-pair rows derived from a task inherit that
  task's split. Never divide a ranking example's candidates across splits.
- Report the configuration used with every result. Results from the two
  protocols should be presented separately.

### Load the data

Load either configuration with Hugging Face Datasets:

```python
from datasets import load_dataset

# Authenticate with `hf auth login` using an account with access to the private dataset.
dataset_path = "yurui983/citation_linking"
data = load_dataset(dataset_path, name="reference_disjoint", token=True)
train, dev, test = data["train"], data["validation"], data["test"]

# Optional: select one target index for a per-index experiment.
openalex_train = train.filter(lambda row: row["index"] == "openalex")

# Alternative protocol, used as a separate experiment.
task_data = load_dataset(dataset_path, name="reference_index_disjoint", token=True)
```

### Train and evaluate a linker

For each row, retrieve candidates from its target `index`, then use `gold_id` to
identify the correct candidate. Use `reference`, `parsed_reference`, or your own
parsing of the original text as input. Keep `gold_id` and `gold_doi` as supervision
and evaluation labels, rather than model input features. Train on `train`, tune
retrieval and linking settings on `validation`, and report final results on `test`.

Save candidate IDs, metadata, retrieval settings, and retrieval date with the
experiment to make its results reproducible. All candidates for a row inherit
that row's split.

A non-null gold target missing from the candidate list is a retrieval miss.
This differs from a null gold target: no usable target was retained in that index
after annotation and review. Distinguish these cases when reporting retrieval and linking
performance; null annotations do not establish that no record exists in the index.

## 5. Other information

### Parsing and document provenance

Parsed references come from the existing source files and are normalized through
the project's `Reference` model. CEX and EXCITE inputs contain TEI-derived parsing;
LinkedBook inputs contain source annotation tags; Brill inputs contain generated
parsing; legal-study inputs contain TEI-derived parsing. Contributor names are
stored as lists of strings; missing parsing stays null. Original input hashes
provide provenance for the exact source snapshots.

`document_id` uses source document identifiers for CEX, EXCITE, Brill, and legal
studies. LinkedBook document identifiers are unavailable in these inputs and stay
null: the grouped LinkedBook file contains synthetic batches, not citing documents.

## Credits

This work is carried out in the context of the EU-funded [GRAPHIA project](https://graphia-ssh.eu/) (grant ID: [101188018](https://cordis.europa.eu/project/id/101188018)). This dataset and is being developed by [Yurui Zhu](https://github.com/RuiaRui) ([Odoma](https://github.com/odoma-ch)) with the help of annotations inside the GRAPHIA project. 
