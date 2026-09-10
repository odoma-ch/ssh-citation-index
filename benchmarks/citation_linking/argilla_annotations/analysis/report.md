# Citation linking — annotation analysis

Hand-written from the CSVs in this directory, produced by
`benchmarks/citation_linking/analyze_argilla_annotations.py`. Annotation round complete:
**500 references × 3 citation indexes = 1500 human decisions**, all `completed` in Argilla.

| | |
|---|---|
| References | 500 (100 each from `cex`, `excite`, `linkedbook`, `brill`, `legal_study_mpilhlt`) |
| Indexes | `openalex`, `matilda`, `wikidata` (`opencitations` not annotated in this round) |
| Excluded before analysis | **19 references** (57 rows) marked `[SKIP]` — the reference text packs more than one publication into a single context, so no single candidate can be right or wrong |
| Manually corrected | **6 rows** — reviewed contradictions and one unsupported "record exists", corrected once in the exported data and documented in §5 |
| Evaluable decisions | **1443** — 481 references × 3 indexes, the same references for every index |
| Rows still flagged | none evaluable (see §5) |

## How the metrics are defined

Each row is one (reference, index) decision, classified five ways. The two axes are
independent: *did the linker return a candidate* and *does the index actually hold the work*.

| outcome | linker returned | index holds the work | overall n |
|---|---|---|---|
| `correct_link` | yes, and it is right | yes | 315 |
| `wrong_link` | yes, wrong work | yes | 39 |
| `spurious_link` | yes | no | 305 |
| `missed_link` | no | yes | 105 |
| `correct_abstain` | no | no | 679 |

Collapsed for precision/recall: `TP = correct_link`, `FP = wrong_link + spurious_link`,
`FN = missed_link + wrong_link`, `TN = correct_abstain`. A wrong candidate is counted as both
a false answer and a missed true one, so the four do not sum to N.

Coverage is kept separate from linker quality:

- **`coverage_actual`** — share of references the index actually holds (annotator's verdict).
  The ceiling; independent of the linker.
- **`coverage_achieved`** — share actually linked correctly.
- **`coverage_gap`** — the difference: recall lost by the linker, not by the index.

## Headline

- Pooled performance is **precision 0.478, recall 0.686, F1 0.564, accuracy 0.689**
  (95% CI: precision 0.440–0.516, recall 0.642–0.727). Roughly one in two returned links is
  wrong.
- **The dominant error is `spurious_link` (305 of 344 false positives, 89%)** — the linker
  returns a candidate for works the index does not contain at all. Fixing abstention, not
  ranking, is where the precision is.
- **Coverage, not the linker, is the binding constraint.** Only 31.8% of decisions concern a
  work the index actually holds; 46.2% of references exist in *none* of the three indexes.
- Querying all three indexes raises the ceiling to **53.8%** of references found somewhere,
  of which we currently link **39.3%** correctly.
- **The existing `is_match_by_similarity` heuristic is strong and unused**: as an accept gate
  it scores precision 0.830 / recall 0.914 against 0.478 for accepting every candidate.

## 1. Per index

| index | n | TP | FP | FN | TN | precision | recall | F1 | accuracy | MCC | link rate | abstain precision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| openalex | 481 | 166 | 188 | 44 | 106 | 0.469 | 0.790 | **0.589** | 0.565 | 0.163 | 0.736 | 0.835 |
| matilda | 481 | 103 | 130 | 27 | 235 | 0.442 | 0.792 | **0.567** | 0.703 | 0.385 | 0.484 | 0.948 |
| wikidata | 481 | 46 | 26 | 73 | 338 | 0.639 | 0.387 | **0.482** | 0.798 | 0.381 | 0.150 | 0.826 |

Full columns (Wilson CIs, specificity, balanced accuracy, macro-F1 over corpus sources) in
`metrics_by_index.csv`; five-way counts in `outcome_counts.csv`.

**OpenAlex leads on F1 (0.589)** and on absolute volume — 166 correct links, more than the
other two combined. It gets there by answering aggressively: a candidate for 73.6% of
references, which also produces 165 spurious links and an MCC of 0.163, barely better than
guessing. **Wikidata is the opposite** — it answers only 15.0% of the time, so its precision
is the highest (0.639) but it misses 71 works it actually holds; its `coverage_gap` of 0.152
is the worst of the three. **Matilda sits in between** with the best-calibrated abstention:
when it says "not found" it is right 94.8% of the time.

Accuracy is not comparable across these rows and should not be used to rank them — it is
dominated by `correct_abstain`, so the index that answers least looks best.

**Read the OpenAlex–Matilda ordering as a tie, not a ranking.** The exclusion is symmetric in
application but not in origin — 18 of the 19 `[SKIP]` marks were made by the annotator
working on OpenAlex, and removing those references took out 16 OpenAlex errors and *zero*
OpenAlex correct links, against 4 of Matilda's. In the run before the exclusions the two were
level (0.561 vs 0.560); the gap now is the size of that differential effect, and the precision
intervals overlap (OpenAlex 0.418–0.521, Matilda 0.380–0.506). What is robust is the volume:
166 correct links, more than the other two indexes combined.

Excluding only the marked rows instead of the whole reference (`--skip-scope row`) leaves the
indexes on unequal reference sets — matilda 499 rows, openalex 482, wikidata 500 — and barely
moves the result (F1 0.563 / 0.588 / 0.470), so references are dropped whole.

## 2. Per corpus source

Pooled over the three indexes, so each reference contributes 3 decisions
(`metrics_by_source.csv`). Reference counts after the multi-publication exclusions:
`excite` 100, `cex` 98, `linkedbook` 98, `brill` 99, `legal_study_mpilhlt` 86.

| source | n | precision | recall | F1 | coverage_actual | coverage_achieved |
|---|---|---|---|---|---|---|
| cex | 294 | 0.790 | 0.778 | **0.784** | 0.690 | 0.537 |
| brill | 297 | 0.485 | 0.696 | **0.571** | 0.310 | 0.215 |
| excite | 300 | 0.380 | 0.514 | **0.437** | 0.247 | 0.127 |
| legal_study_mpilhlt | 258 | 0.307 | 0.652 | **0.417** | 0.256 | 0.167 |
| linkedbook | 294 | 0.138 | 0.500 | **0.216** | 0.082 | 0.041 |

**Corpus matters more than index choice.** The spread across sources (F1 0.216–0.784) is
five times the spread across indexes (0.482–0.589). `cex` is journal-article-heavy and well
covered; `linkedbook` is monograph citations, which the indexes barely hold (coverage 8.2%) —
nearly everything returned there is spurious.

F1 per index × source (`metrics_by_index_source.csv`, n = 86–100 per cell):

| source | matilda | openalex | wikidata |
|---|---|---|---|
| brill | 0.530 | 0.627 | 0.513 |
| cex | 0.851 | 0.817 | 0.615 |
| excite | 0.408 | 0.490 | 0.261 |
| legal_study_mpilhlt | 0.262 | 0.553 | 0.091 |
| linkedbook | — (no TP) | 0.219 | 0.500 |

Cells worth acting on:

- **`openalex` × `legal_study_mpilhlt`: link rate 0.988** — it returns a candidate for
  virtually every legal-studies reference, yielding 47 spurious links, exactly **1**
  `correct_abstain` in 86 decisions, and accuracy 0.407. Abstention is effectively broken for
  this corpus, and the 13 packed-footnote references now excluded were only part of the story.
- **`matilda` × `linkedbook`: 0 true positives** in 98 references (21 spurious links).
  Matilda holds 1 of those 98 works. Not worth querying for monographs.
- **`matilda` × `legal_study_mpilhlt`: precision 0.160** — 41 spurious links against 8 correct
  ones, the weakest precision cell in the study.
- `wikidata` × `legal_study_mpilhlt` holds 17 of 86 works but links 1 correctly (F1 0.091) —
  a retrieval problem, not a coverage problem.

## 3. Coverage comparison

All three indexes over the same 481 references (`coverage_by_index.csv`):

| index | holds the work | share | unique to it | linked correctly | share |
|---|---|---|---|---|---|
| openalex | 210 | 0.437 | 77 | 166 | 0.345 |
| matilda | 130 | 0.270 | 12 | 103 | 0.214 |
| wikidata | 119 | 0.247 | 33 | 46 | 0.096 |

How many indexes hold each work (`coverage_histogram.csv`):

| indexes holding the work | references | share |
|---|---|---|
| 0 | 222 | 0.462 |
| 1 | 122 | 0.254 |
| 2 | 74 | 0.154 |
| 3 | 63 | 0.131 |

Pairwise overlap (`coverage_overlap.csv`):

| pair | both | only A | only B | Jaccard |
|---|---|---|---|---|
| matilda / openalex | 114 | 16 | 96 | 0.504 |
| matilda / wikidata | 67 | 63 | 52 | 0.368 |
| openalex / wikidata | 82 | 128 | 37 | 0.332 |

**OpenAlex is the primary index and largely subsumes Matilda**: of the 130 works Matilda
holds, OpenAlex holds 114, leaving 16 that Matilda has and OpenAlex does not — and only 12 of
Matilda's are unique across all three indexes. **Wikidata is the genuine complement**: it is
the smallest index but contributes 33 works unique across all three, nearly three times
Matilda's unique contribution, and it overlaps OpenAlex least (Jaccard 0.332).

Union across all three (`coverage_union.csv`): **259 / 481 references (53.8%)** exist in at
least one index; **189 (39.3%)** are currently linked correctly by at least one. So 14
percentage points are lost to linking rather than coverage, and the remaining 46.2% are out
of reach of these three indexes entirely.

Per source (`coverage_by_source.csv`, holds / linked correctly per index):

| source | refs | matilda | openalex | wikidata | any index | linked by any |
|---|---|---|---|---|---|---|
| cex | 98 | 73 / 63 | 74 / 67 | 56 / 28 | 82 | 72 |
| brill | 99 | 30 / 22 | 40 / 32 | 22 / 10 | 58 | 41 |
| excite | 100 | 15 / 10 | 43 / 25 | 16 / 3 | 50 | 29 |
| legal_study_mpilhlt | 86 | 11 / 8 | 38 / 34 | 17 / 1 | 48 | 37 |
| linkedbook | 98 | 1 / 0 | 15 / 8 | 8 / 4 | 21 | 10 |

OpenAlex is the best single index on correct links for every corpus. `cex` is the one place
Matilda is competitive — 63 correct links against OpenAlex's 67, and a better F1 (0.851 vs
0.817) because it reaches them at a much lower link rate (0.765 vs 0.918).

## 4. `is_match_by_similarity` as an accept gate

The pipeline already computes this heuristic and the annotation treats it as context only.
Scored against the gold labels over the 659 returned candidates (`similarity_heuristic.csv`):

| index | candidates | precision | recall | F1 | accuracy | precision if accepting all |
|---|---|---|---|---|---|---|
| openalex | 354 | 0.824 | 0.928 | 0.873 | 0.873 | 0.469 |
| matilda | 233 | 0.805 | 0.961 | 0.876 | 0.880 | 0.442 |
| wikidata | 72 | 0.946 | 0.761 | 0.843 | 0.819 | 0.639 |
| **all** | 659 | 0.830 | 0.914 | 0.870 | 0.869 | 0.478 |

Gating on the existing flag would raise link precision from 0.48 to 0.83 while keeping 91%
of the true links — the single highest-value change suggested by this data, and it needs no
new model.

## 5. Data handling: exclusions and manual corrections

The analysis code applies exactly one special case: a `correct_id` beginning with `[SKIP]`
labels the row `multi_ref_context` and excludes it. Everything else the annotators entered is
taken at face value. Where a row could not be taken at face value it was corrected **once,
directly in the exported data** — the answer and its underlying `<question>.responses` entry
were both rewritten, so re-deriving `label` reproduces the fix and no code carries a special
case. Each patched row keeps a `manual_fix` column with the change and the reason;
`summary.json` reports the count (`n_manually_fixed_rows`).

### Multi-publication references — 19 excluded

Annotators typed `[SKIP]` (sometimes with a note such as *"multiple references packed in one
single context"*) when the reference text bundles several publications, so no single candidate
can be judged. The defect is in the reference text and therefore applies to every index, so
the whole reference is dropped from all three (57 rows). By corpus:
`legal_study_mpilhlt` 14, `cex` 2, `linkedbook` 2, `brill` 1 — which is why the legal-studies
denominator falls to 86.

Typical examples: *"31 For example, Law Reform (Miscellaneous Provisions) Act 1970 (U.K.);
Domestic Relations Act 1975 (N.Z.); Marriage Act Amendment Act 1976 (Cwth.)"*, and
*"32 G.S. Frost, Promises Broken … (1995); Thornton, op. cit. (1996), n. 4"*. Use
`--skip-scope row` to see the row-level variant instead of the reference-level default.

Eighteen of the 19 marks came from the OpenAlex annotator, who was the only one using the
convention; the nineteenth (`10.1111_1467-6478.00057_instance-71`) was added on review — its
text bundles the Harvard Law Review editors' *Sexual Orientation and the Law* (1990),
*Dean v. District of Columbia* (1995) and a 1997 newspaper article.

### Manual corrections — 6 rows

Five rows carried answer combinations that cannot all be true (`is_match_correct` and
`no_match` both `true` while a candidate was returned, or `is_match_correct=true` with no
candidate at all); two of those fell inside `[SKIP]`-excluded references and needed nothing.
Each remaining row was read against the same reference's rows in the other indexes, which
resolved the intent in every case.

| index | reference | correction | why |
|---|---|---|---|
| openalex | `brill_a15e3650-…_634` — Seneca, 1774 translation | `no_match` false → **true** | No candidate was returned, yet `is_match_correct` was `true` and `no_match` `false`. The matilda and wikidata rows for the same reference both read "correctly not found". Now a true negative. |
| matilda | `cex_COM-SCI_28_20` — `Magic Leap. 2018` | `is_match_correct` true → **false** | Candidate is *Pixeldust Studios Reptopia Magic Leap Experience* (ACM 2020), a different work. Matches how the openalex annotator answered the same reference. |
| wikidata | `cex_COM-SCI_28_20` | `is_match_correct` true → **false** | Candidate `Q18351488` is Magic Leap *the company*, not a publication record. |
| matilda | `10.1515_zfrs-1980-0104_instance-7` | `is_match_correct` true → **false** | Candidate is an unrelated PSSRI volume on the China–Pakistan Economic Corridor. |
| openalex | `10.1515_zfrs-1980-0104_instance-63` — Clinard, *Comparative Crime Victimization Surveys*, 1978 | `no_match` false → **true** | Annotator rejected the candidate and marked the work as present in OpenAlex but supplied no ID. OpenAlex holds 91 Marshall Clinard works, none on victimization, and no title match for the article — the work is absent, so the returned candidate is a spurious link, not a missed one. |
| matilda | `10.1111_1467-6478.00057_instance-71` | `correct_id` → **`[SKIP] …`** | Reference bundles three publications (see above), so it joins the excluded set rather than being scored against a single candidate. |

Two judgement calls worth naming. `Magic Leap. 2018` (and `Hellblade. 2018`, `Data citation
index`) are product, software and database citations rather than bibliographic ones; treating
the Wikidata company item as "not a match" is the strict reading, and a linker aimed at
entities rather than works would score that row differently. And the Clinard correction
overrules an annotator on the strength of an API search rather than a second annotation.

**The corrections live only in the exported files.** A fresh pull from Argilla will bring the
originals back, so the durable fix is to correct these six records in the Argilla UI.

### Remaining flags — none

No evaluable row is left flagged. One contradictory row survives inside an excluded reference
(`wikidata` / `cex_NEU_82_11`, where `correct_id` was `INCORRECT`); `summary.json` reports both
the total and the evaluable count. **All 144 evaluable rows where an index holds the work but
the linker did not return it correctly now carry a usable gold ID** (150 across the whole
export) — a ready-made retrieval evaluation set.

## 6. Caveats

- `opencitations` was created in Argilla but not annotated in this round; it is commented out
  of `INDEXES` in `export_argilla_annotations.py`. All "union" and "any index" figures
  therefore describe three indexes, not four.
- Single annotator per record, so no inter-annotator agreement is measurable. The annotators
  also did not share conventions — only the OpenAlex one used `[SKIP]` — so further packed
  references may remain in the pool unmarked.
- 500 of 1110 references were sampled (100 per corpus, seed 42) and 19 were then excluded, so
  per-cell n is 86–100 and per-source CIs are wide — check the `*_lo` / `*_hi` columns in the
  CSVs before treating a single cell as settled.
- `coverage_actual` in `metrics_by_index.csv` and `coverage_by_index.csv` agree, because every
  index has the same 481 evaluable references.

## 7. Suggested next steps

1. Gate returned candidates on `is_match_by_similarity` (or its underlying score with a tuned
   threshold) before emitting a link.
2. Fix abstention for `openalex` × `legal_study_mpilhlt` — 1 correct abstention in 86
   decisions at a 0.988 link rate is a bug-shaped number.
3. Stop querying Matilda for `linkedbook`-style monograph references; keep Wikidata in the
   fan-out despite its low volume, since it holds 33 works nothing else does.
4. Give annotators a first-class "multiple publications in this context" answer rather than a
   free-text `[SKIP]`, and apply it consistently across indexes.
5. Annotate the `opencitations` split to complete the coverage picture.

## Files

| file | contents |
|---|---|
| `summary.json` | run bookkeeping: row/reference counts, exclusions, manual fixes, flags by reason and index |
| `overall.csv` | pooled metrics, all indexes and sources together |
| `outcome_counts.csv` | five-way outcome counts per index |
| `metrics_by_index.csv` | full metric set per index (+ Wilson CIs, MCC, macro-F1) |
| `metrics_by_index_source.csv` | same, per index × corpus source |
| `metrics_by_source.csv` | same, per corpus source, indexes pooled |
| `coverage_by_index.csv` | coverage and unique contribution per index |
| `coverage_histogram.csv` | how many indexes hold each work |
| `coverage_overlap.csv` | pairwise overlap and Jaccard |
| `coverage_union.csv` | union ceiling across indexes |
| `coverage_by_source.csv` | coverage per corpus source × index |
| `coverage_dropped_refs.csv` | references excluded from the complete-case pivot (empty: none) |
| `similarity_heuristic.csv` | `is_match_by_similarity` scored as an accept gate |

Regenerate the CSVs with:

```bash
python benchmarks/citation_linking/analyze_argilla_annotations.py
```
