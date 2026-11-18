# GLiNER multilingual listwise fine-tuning

The objective is one choice among `candidate_1 … candidate_N` and `none`, using
`fastino/GLiNER2.5-multi-Decide` at revision
`e173bca1f0c4217d7a55b3b0c705d5ece8a0218d`. Candidate descriptions and citation text
use the pretrained listwise formatter. Candidate IDs stay in sidecar metadata.
The single-label cross-entropy loss uses the same encoder, schema embeddings,
classifier and softmax as the baseline `Classifier` API; the runner checks this
agreement before updating weights.

## Code layout

| Module | Role |
|---|---|
| `linking_eval.py` | Candidate caching, exact-ID scoring, offline pool inputs, checkpointed inference and evaluation |
| `rerankers.py` | Inference adapters, including GLiNER listwise selection and the none-logit bias |
| `decisions.py` | Search-first / API-matching controls, comparison table, rejection thresholds |
| `gliner_data.py` | `export`, `full`, `official` dataset exports |
| `finetune_gliner_listwise.py` | Training |
| `gliner_eval.py` | Bias sweep, `predict`, `summarize` |

## Data and label evidence

From the repository root, in the `citation_index` Conda environment:

```bash
python -m benchmarks.citation_linking.experiments.gliner_data export \
  --output-dir logs/citation_link_tryings/finetuning/gliner_listwise/data_v2
```

Choose a fresh output directory for a new export. The exporter refuses to
overwrite an existing run. It saves GLiNER `input`/`output` JSONL, aligned
`.metadata.jsonl` files, all-task `audit.jsonl`, `needs_review.jsonl`, and an
input/output checksum manifest. It performs no network calls.

The export holds 929 eligible tasks from 441 references: 423 with a retrieved gold
candidate and 506 `none` targets (489 submitted `no_record_in_index` annotations and
17 known-gold retrieval misses with nonempty pools). A null gold alone never supplies
negative supervision. Evidence must have consistent submitted responses, matching
citation text, no review flag, and agreement with the release's reviewed
corrections. Historical annotations have not been re-adjudicated against every
candidate in the current pool.

Five unresolved OpenAlex tasks are withheld. There are 511 empty-pool tasks, two of
which are also unresolved. All 1,443 tasks remain in the audit; empty pools reject
without model inference. Link metrics include approved-gold empty-pool misses in
their denominator. No oracle injection is accepted for training. All 929 exported
examples pass GLiNER schema validation and tokenization (maximum 2,110 tokens, no
truncation). The shared baseline formatter still clips individual fields at 1,000
characters.

### Fourth index and DASI/EbDA references

The numbers above describe the three-index export `data_v2`, which the previous model
`full_final_v1` was trained on. The current model `v4_select_v1` uses `data_v4_dasi_ebda`. `dataset/build_annotation_dataset.py` writes the pools of all four indexes
and all 680 references to `logs/citation_link_tryings/four_index/` (at most ten
candidates per task), and `export` reads that pool and both corrected annotation files
by default:

```bash
python -m benchmarks.citation_linking.experiments.gliner_data export \
  --output-dir logs/citation_link_tryings/finetuning/gliner_listwise/data_v4_dasi_ebda
```

Pass `--pool logs/citation_link_tryings/search_top10.jsonl` for the three-index pool.
Splits come from `reference_index_disjoint` of dataset revision
`0e298f73dac892aae81330104f72a6fa7068f317`. For `official`, use the `reference_disjoint`
files of that revision; the older official split has no OpenCitations or DASI/EbDA
tasks.

The export holds 1,663 eligible tasks (train 987, validation 342, test 334; 801 with
a retrieved gold, 862 `none`). The previous four-index export without DASI/EbDA
is kept as `data_v3_4index` (1,249 tasks).

## Training

The isolated runtime on psnc is `/mnt/data/yzhu/gliner-listwise/venv` (no `activate`
script; call `venv/bin/python` directly). Run checkouts are copies of the repository
code, not git clones: `run-20261028` (`full_final_v1`) and `run-20261117`
(`v4_select_v1`, commit in `GIT_COMMIT`). Run from the checkout with
`HF_HOME=/mnt/data/yzhu/gliner-listwise/hf-cache HF_HUB_OFFLINE=1 PYTHONPATH=$PWD/src:$PWD`;
select a GPU explicitly and check `nvidia-smi` first. Library versions: `gliner2==2.1.0`, `torch==2.7.1`
(CUDA 12.6 build), `transformers==4.56.1`. The runner records resolved model/weight
hashes and package versions.

Training uses one complete candidate list per optimizer step, float32, encoder LR
`1e-5`, classifier LR `1e-4`, AdamW weight decay `0.01`, gradient clipping `1.0`,
seed 42. Each epoch reshuffles candidates and remaps target and candidate IDs.
Native stochastic schema augmentation is bypassed so it cannot drop labels. A finite
token budget causes an error instead of silent truncation; errors fail the run.

Useful trainer flags (defaults are what the final model used):

| Flag | Effect |
|---|---|
| `--train-only` | Fixed epochs, keep the last epoch, no validation. |
| `--select-by recall_policy` (default) | With a validation split, rank epochs by max precision at TP ≥ pretrained TP over a none-logit bias sweep. `loss` selects by validation loss. |
| `--none-weight w` | Scales the loss of `none`-target examples. |
| `--rank-weight w` | Adds a candidates-only softmax when a real candidate is correct. |
| `--label-smoothing e` | Limits overconfidence. |
| `--freeze-encoder` | Trains the head only. |
| `--smoke-size n` | Overfits `n` train examples as a plumbing check, not a score. |

## Current model: `v4_select_v1` (four indexes, train split)

Trained on the 987 train examples of `data_v4_dasi_ebda` with the default flags (up to 5
epochs, patience 2, `recall_policy` selection on the 342 validation examples). Epoch 4 was
selected with none-logit bias **+7.737**; validation loss was lowest at epoch 2. Training
took about 10 minutes on one A100.

```bash
CUDA_VISIBLE_DEVICES=0 venv/bin/python -m benchmarks.citation_linking.experiments.finetune_gliner_listwise \
  --data-dir logs/citation_link_tryings/finetuning/gliner_listwise/data_v4_dasi_ebda \
  --output-dir logs/citation_link_tryings/finetuning/gliner_listwise/v4_select_v1 --device cuda:0
venv/bin/python -m benchmarks.citation_linking.experiments.gliner_eval predict \
  --data-dir .../data_v4_dasi_ebda --run-dir .../v4_select_v1 --device cuda:0
venv/bin/python -m benchmarks.citation_linking.experiments.gliner_eval summarize \
  --run-dirs .../v4_select_v1 --output-dir .../v4_select_v1/test_summary
```

Checkpoint: private Hub repo `yurui983/gliner2.5-multi-decide-citation-linking`, also on
psnc in `run-20261117/.../v4_select_v1/best` and locally in `v4_select_v1/best/`;
`model.safetensors` SHA-256
`20deaed79508d22c1a65986d47ae319031596e53360f27b77ff057b953189406`.

`checkpoint` must be a local directory, so download the snapshot first (needs `HF_TOKEN`
while the repo is private):

```python
from huggingface_hub import snapshot_download

path = snapshot_download("yurui983/gliner2.5-multi-decide-citation-linking")
model = GlinerListwiseReranker(checkpoint=Path(path), device="cuda:0", none_logit_bias=7.737)
```

The Hub snapshot reproduces the psnc test predictions (first 20 examples on CPU: same
choices, maximum probability difference 4e-5).

Test split (`reference_index_disjoint`, 543 tasks including empty pools, 179 gold
targets; bias frozen from validation):

| Test policy | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Pretrained, native | 40.1% | 70.4% | 51.1% | 126 | 188 | 53 |
| Fine-tuned, dev bias +7.74 | 81.6% | 67.0% | 73.6% | 120 | 27 | 59 |

On the 179 gold tasks the fine-tuned model is correct on 120 vs 126 for pretrained (6
only after, 12 only before, McNemar p=0.24). On the 364 no-gold tasks correct abstentions
rise from 215 to 346. Per index F1: OpenAlex 80.0%, Matilda 75.4%, Wikidata 72.7%,
OpenCitations 64.5%; OpenCitations misses 24 of its gold links.

The notebook rescores every other saved method on the same 543 tasks
(`v4_select_v1/held_out_comparison.csv`). The fine-tuned row ranks fifth: D3 listwise leads
at 78.3% F1, then GLiDE listwise (75.9%), D3 pairwise (75.1%) and GLiDE pairwise (74.3%).

This split keeps each `(reference, index)` task in one split, so 250 of 334 test examples
have a reference seen in training under another index. Treat the scores as optimistic for
unseen references; the bias also moved widely between epochs (−10.7 to +7.7).

## Previous model: `full_final_v1` (three indexes, all data)

`full_final_v1` trains on all 929 eligible tasks (441 references) for a fixed 3
epochs and keeps the last epoch. Train loss was 1.236, 0.852, 0.656. Checkpoint
reload reproduced the decisions on the first 40 training examples (a plumbing check
only; there is no held-out score).

```bash
python -m benchmarks.citation_linking.experiments.gliner_data full \
  --source-dir logs/citation_link_tryings/finetuning/gliner_listwise/data_v2 \
  --output-dir logs/citation_link_tryings/finetuning/gliner_listwise/full_v1
CUDA_VISIBLE_DEVICES=1 python -m benchmarks.citation_linking.experiments.finetune_gliner_listwise \
  --data-dir .../full_v1 --output-dir .../full_final_v1 \
  --device cuda:0 --epochs 3 --train-only
```

Checkpoint (psnc, 1.1 GB): `/mnt/data/yzhu/gliner-listwise/run-20261028/logs/citation_link_tryings/finetuning/gliner_listwise/full_final_v1/best`,
`model.safetensors` SHA-256
`5c23efabdc23dbe2d1d59d27b92d0997792d371ef59d651409faf367b1b70bb0`. No local copy;
copy the complete `best/` directory and verify the checksum before local inference.
`full_final_v1/manifest.json` is saved locally.

The **none-logit bias** is not trained. It is a constant added to the `none` logit at
inference, so it is passed with the checkpoint; a positive bias rejects more (higher
precision, lower recall) and candidate ranking is unchanged. Its scale is
model-specific and adjusted scores are not calibrated probabilities.

```python
model = GlinerListwiseReranker(
    checkpoint=FULL_FINAL_DIR / "best", device="cuda:0", none_logit_bias=7.1413
)
run_model("gliner_listwise_final", model, threshold=None)
```

The +7.14 was selected on the validation set of the official-split model (78 gold
targets), not on `full_final_v1`, so it may sit off for this model. No data remains to
check that; fresh annotations would. A lower bias trades precision for recall.

## Held-out evidence: official `reference_disjoint` split

The published split (`yurui983/citation_linking`, config `reference_disjoint`,
revision `93ed9c3dd7e671fb5af140b52356652304d7b204`) was downloaded to
`logs/citation_link_tryings/reference_disjoint_split/` with the token from
`HF_TOKEN`. It covers the same 1,443 tasks, gold IDs match the pool exactly, and the
three splits share no references. The same recipe was trained on its train split
(`gliner_data official`, exported as `official_rd_v1`: train 549, validation
191, test 189 model examples; test has 94 gold targets among 291 tasks including empty
pools) with epoch 2 selected on validation, then scored once on test with
`gliner_eval predict` and `summarize`. This is the recipe's evidence, not a score
of `full_final_v1`.

| Test policy | Precision | Recall | F1 | Correct links | Wrong | Spurious | Missed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pretrained, native | 38.5% | 73.4% | 50.5% | 69 | 20 | 90 | 5 |
| Fine-tuned, dev-selected bias (+7.14) | 87.2% | 72.3% | 79.1% | 68 | 0 | 10 | 26 |

Per task against pretrained on the 94 gold tasks: bias +7.14 gives 68 vs 69 correct
(5 only after, 6 only before, exact McNemar p=1.0), so recall is not distinguishable
from pretrained. On 197 no-gold tasks, correct abstentions rise from 107 to 187. The
bias comes from a single 78-gold validation set and no test label selected it, but
that set is small, so treat the operating point as approximate.

The notebook's final summary rescores every other saved method on these same 291
test tasks (`held_out_comparison.csv` in `cv_runs/official_rd_A/`). The fine-tuned
checkpoint at +7.14 has F1 79.1%, in line with D3/GLiDE (about 79–80%) and ahead of
Jev listwise (76.8%); it is not the best row. D3 pairwise leads at 80.2%. The other
methods are not fine-tuned, but GTE and `_reject` thresholds were tuned on the old
dev split, which shares some references with this test.

## Checks

```bash
python -m pytest tests/test_gliner_recall_policy.py \
  tests/test_gliner_training_data.py tests/test_gliner_finetune.py --disable-warnings -q
```

37 tests pass. The broader existing reranking tests have unrelated failures in Jev
notebook credential tests and the GLiDE hosted-cell test; the targeted checks do not
imply a clean full suite.

Upstream references: [model card](https://huggingface.co/fastino/GLiNER2.5-multi-Decide)
and [GLiNER2 implementation](https://github.com/fastino-ai/GLiNER2).
