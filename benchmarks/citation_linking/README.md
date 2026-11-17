# Citation linking benchmark

Linking a bibliographic reference to its record in OpenAlex, Matilda, Wikidata and
OpenCitations Meta. The folder follows the life of the benchmark: build annotation
tasks, collect the annotations, release the gold dataset, run experiments on it.

```
pre_annotation/   sample references, run the baseline search, create the Argilla tasks
annotation/       pull the annotations out of Argilla, publish and analyse them
dataset/          reviewed gold dataset with search candidates
experiments/      rerankers, decision models and GLiNER fine-tuning on the dataset
data/             local data; only data/sources/ is in git
```

Run everything from the repository root.

## 1. `pre_annotation/`: annotation tasks

| File | Purpose |
|---|---|
| `run_benchmarks.py` | `build-test-set`, `api-search`, `legal-study`, `dasi-ebda`: sample the references and run the baseline search shown to annotators. |
| `build_citation_linking_dataset.ipynb` | Create the OpenAlex, Matilda and Wikidata Argilla datasets. |
| `opencitations_linking.py`, `publish_datasets.py` | The OpenCitations round (November 2026): baseline search and upload to Argilla. |

Baseline search results are not kept in the repository; these scripts read and write
them under `logs/citation_linking_pre_annotation/`.

## 2. `annotation/`: annotations

```bash
# Argilla -> data/annotations/raw/ -> Hugging Face yurui983/citation_linking_annotations
python benchmarks/citation_linking/annotation/export_argilla_annotations.py --push

# metric tables of the baseline shown to annotators -> data/annotations/analysis/
python benchmarks/citation_linking/annotation/analyze_argilla_annotations.py
```

The raw export is published exactly as submitted. `analyze_argilla_annotations.py`
reads the corrected copy that step 3 writes. `ANNOTATION_GUIDELINES.md` is what the
annotators followed.

## 3. `dataset/`: gold dataset

```bash
# raw annotations -> corrections -> gold + candidates -> Hugging Face yurui983/citation_linking
python benchmarks/citation_linking/dataset/build_annotation_dataset.py --output-dir <new dir> --push
```

| File | Purpose |
|---|---|
| `build_annotation_dataset.py` | The build. Holds every manual correction (`ANNOTATION_FIXES`, `REVIEW_OVERRIDES`). `--search <index>` searches an index again first. |
| `candidate_pool.py` | Search candidates: the store, the metadata-only search of the four indexes, the local pools. |
| `release_splits.py` | Train / validation / test assignment; published splits are frozen. |
| `DATASET_CARD.md` | README of the Hugging Face dataset. |
| `CORRECTIONS.md` | Every correction to the gold, with its reason. |

Besides the release, a build writes `data/annotations/corrected/annotations.jsonl` and
the local candidate pools `logs/citation_link_tryings/four_index/search_*.jsonl`.

## 4. `experiments/`

`citation_link_tryings.ipynb` is the workbench: retrieval quality, pretrained
rerankers and decision models. `linking_eval.py`, `rerankers.py` and `decisions.py`
are its helpers. `gliner_finetuning.md` documents the GLiNER listwise fine-tuning
(`gliner_data.py`, `finetune_gliner_listwise.py`, `gliner_eval.py`).

## `data/`

| Folder | In git | Content |
|---|---|---|
| `sources/` | yes | The original references: CEX / EXCITE / LinkedBook sample, Brill sample, legal-study TEI files, DASI / EBDA bibliographies. |
| `annotations/raw/` | no | Output of step 2, identical to the Hugging Face annotations dataset. |
| `annotations/corrected/` | no | Raw annotations with the answer fixes applied. `annotations.jsonl` is written by step 3; `dasi_ebda.jsonl` carries one fix made by hand. |
| `annotations/analysis/` | no | Metric tables per split. |

Search candidates (`logs/citation_link_tryings/candidates.jsonl`), search checkpoints,
releases and training exports stay under `logs/`. The candidates are also published
in the dataset's `candidates` configuration.
