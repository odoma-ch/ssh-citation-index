# Citation linking — annotation analysis

- rows: 1500  (500 references x 3 indexes)
- evaluable rows: 1403
- excluded rows: {'unannotated': 97}
- needs_review rows: 13

Outcome taxonomy (five-way, unambiguous)
----------------------------------------
    correct_link      candidate returned and it is the right work
    wrong_link        candidate returned, wrong work, but the index does hold the work
    spurious_link     candidate returned, and the index holds no record at all
    missed_link       no candidate returned, but the index does hold the work
    correct_abstain   no candidate returned, and the index holds no record


TP = correct_link, FP = wrong_link + spurious_link, FN = missed_link + wrong_link,
TN = correct_abstain. wrong_link counts in both FP and FN, so the four do not sum to N.
A blank f1 means no true positives and zero recall, not a failed computation.

Tables below show headline columns only — the CSVs carry every metric.

## Overall (all indexes pooled)

| n_eval | n_excluded | TP | FP | FN | TN | precision | recall | f1 | accuracy | abstain_precision | coverage_actual | coverage_achieved | coverage_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1403 | 97 | 286 | 328 | 150 | 684 | 0.466 | 0.656 | 0.545 | 0.691 | 0.867 | 0.311 | 0.204 | 0.107 |


## Outcome counts (five-way, per index)

| index | correct_link | wrong_link | spurious_link | missed_link | correct_abstain |
| --- | --- | --- | --- | --- | --- |
| matilda | 107 | 15 | 124 | 14 | 240 |
| openalex | 132 | 27 | 132 | 18 | 94 |
| wikidata | 47 | 3 | 27 | 73 | 350 |


## Metrics per citation index

_coverage_* here use each index's own evaluable rows (openalex has 97 rows still pending, so its denominator is 403 vs 500). For an apples-to-apples cross-index comparison use 'Coverage per index' below — complete-case, same 403 refs for all._

| index | n_eval | n_excluded | TP | FP | FN | TN | precision | recall | f1 | accuracy | abstain_precision | coverage_actual | coverage_achieved | coverage_gap | macro_f1_over_sources |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matilda | 500 | 0 | 107 | 139 | 29 | 240 | 0.435 | 0.787 | 0.560 | 0.694 | 0.945 | 0.272 | 0.214 | 0.058 | 0.519 |
| openalex | 403 | 97 | 132 | 159 | 45 | 94 | 0.454 | 0.746 | 0.564 | 0.561 | 0.839 | 0.439 | 0.328 | 0.112 | 0.515 |
| wikidata | 500 | 0 | 47 | 30 | 76 | 350 | 0.610 | 0.382 | 0.470 | 0.794 | 0.827 | 0.246 | 0.094 | 0.152 | 0.387 |


## Metrics per index x corpus source

| index | source | n_eval | n_excluded | TP | FP | FN | TN | precision | recall | f1 | accuracy | abstain_precision | coverage_actual | coverage_achieved | coverage_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matilda | brill | 100 | 0 | 22 | 31 | 8 | 44 | 0.415 | 0.733 | 0.530 | 0.660 | 0.936 | 0.300 | 0.220 | 0.080 |
| matilda | cex | 100 | 0 | 64 | 13 | 10 | 17 | 0.831 | 0.865 | 0.848 | 0.810 | 0.739 | 0.740 | 0.640 | 0.100 |
| matilda | excite | 100 | 0 | 10 | 24 | 5 | 65 | 0.294 | 0.667 | 0.408 | 0.750 | 0.985 | 0.150 | 0.100 | 0.050 |
| matilda | legal_study_mpilhlt | 100 | 0 | 11 | 49 | 5 | 37 | 0.183 | 0.688 | 0.289 | 0.480 | 0.925 | 0.160 | 0.110 | 0.050 |
| matilda | linkedbook | 100 | 0 | 0 | 22 | 1 | 77 | 0.000 | 0.000 |  | 0.770 | 0.987 | 0.010 | 0.000 | 0.010 |
| openalex | brill | 86 | 14 | 29 | 25 | 5 | 31 | 0.537 | 0.853 | 0.659 | 0.698 | 0.969 | 0.395 | 0.337 | 0.058 |
| openalex | cex | 82 | 18 | 54 | 21 | 8 | 4 | 0.720 | 0.871 | 0.788 | 0.707 | 0.571 | 0.756 | 0.659 | 0.098 |
| openalex | excite | 80 | 20 | 17 | 26 | 18 | 25 | 0.395 | 0.486 | 0.436 | 0.525 | 0.676 | 0.438 | 0.212 | 0.225 |
| openalex | legal_study_mpilhlt | 74 | 26 | 26 | 47 | 7 | 1 | 0.356 | 0.788 | 0.491 | 0.365 | 1.000 | 0.446 | 0.351 | 0.095 |
| openalex | linkedbook | 81 | 19 | 6 | 40 | 7 | 33 | 0.130 | 0.462 | 0.203 | 0.481 | 0.943 | 0.160 | 0.074 | 0.086 |
| wikidata | brill | 100 | 0 | 10 | 7 | 12 | 71 | 0.588 | 0.455 | 0.513 | 0.810 | 0.855 | 0.220 | 0.100 | 0.120 |
| wikidata | cex | 100 | 0 | 29 | 8 | 28 | 36 | 0.784 | 0.509 | 0.617 | 0.650 | 0.571 | 0.570 | 0.290 | 0.280 |
| wikidata | excite | 100 | 0 | 3 | 4 | 13 | 80 | 0.429 | 0.188 | 0.261 | 0.830 | 0.860 | 0.160 | 0.030 | 0.130 |
| wikidata | legal_study_mpilhlt | 100 | 0 | 1 | 6 | 19 | 75 | 0.143 | 0.050 | 0.074 | 0.760 | 0.806 | 0.200 | 0.010 | 0.190 |
| wikidata | linkedbook | 100 | 0 | 4 | 5 | 4 | 88 | 0.444 | 0.500 | 0.471 | 0.920 | 0.967 | 0.080 | 0.040 | 0.040 |


## Metrics per corpus source (all indexes pooled)

_pooled over index decisions — each reference contributes 3 rows (one per index), so n_eval counts decisions, not references._

| source | n_eval | n_excluded | TP | FP | FN | TN | precision | recall | f1 | accuracy | abstain_precision | coverage_actual | coverage_achieved | coverage_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brill | 286 | 14 | 61 | 63 | 25 | 146 | 0.492 | 0.709 | 0.581 | 0.724 | 0.901 | 0.301 | 0.213 | 0.087 |
| cex | 282 | 18 | 147 | 42 | 46 | 57 | 0.778 | 0.762 | 0.770 | 0.723 | 0.613 | 0.684 | 0.521 | 0.163 |
| excite | 280 | 20 | 30 | 54 | 36 | 170 | 0.357 | 0.455 | 0.400 | 0.714 | 0.867 | 0.236 | 0.107 | 0.129 |
| legal_study_mpilhlt | 274 | 26 | 38 | 102 | 31 | 113 | 0.271 | 0.551 | 0.364 | 0.551 | 0.843 | 0.252 | 0.139 | 0.113 |
| linkedbook | 281 | 19 | 10 | 67 | 12 | 198 | 0.130 | 0.455 | 0.202 | 0.740 | 0.971 | 0.078 | 0.036 | 0.043 |


## Coverage per index (complete-case refs)

_complete-case: only refs annotated in every index. Dropped refs are listed in coverage_dropped_refs.csv._

| index | n_refs | exists | exists_share | unique_to_index | linked_correct | linked_correct_share |
| --- | --- | --- | --- | --- | --- | --- |
| matilda | 403 | 111 | 0.275 | 12 | 87 | 0.216 |
| openalex | 403 | 177 | 0.439 | 69 | 132 | 0.328 |
| wikidata | 403 | 98 | 0.243 | 28 | 35 | 0.087 |


## How many indexes hold each work

| n_indexes_holding_work | n_refs | share |
| --- | --- | --- |
| 0 | 181 | 0.449 |
| 1 | 109 | 0.270 |
| 2 | 62 | 0.154 |
| 3 | 51 | 0.127 |


## Pairwise coverage overlap

| index_a | index_b | both | only_a | only_b | neither | either | jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| matilda | openalex | 94 | 17 | 83 | 209 | 194 | 0.485 |
| matilda | wikidata | 56 | 55 | 42 | 250 | 153 | 0.366 |
| openalex | wikidata | 65 | 112 | 33 | 193 | 210 | 0.310 |


## Union ceiling across indexes

| scope | n_refs | exists_any | exists_any_share | linked_correct_any | linked_correct_any_share |
| --- | --- | --- | --- | --- | --- |
| all indexes (union) | 403 | 222 | 0.551 | 154 | 0.382 |


## Coverage per corpus source x index

| source | n_refs | matilda_exists | matilda_correct | openalex_exists | openalex_correct | wikidata_exists | wikidata_correct | exists_any | linked_correct_any |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brill | 86 | 25 | 19 | 34 | 29 | 18 | 9 | 49 | 36 |
| cex | 82 | 61 | 52 | 62 | 54 | 46 | 23 | 70 | 61 |
| excite | 80 | 14 | 9 | 35 | 17 | 13 | 2 | 42 | 21 |
| legal_study_mpilhlt | 74 | 10 | 7 | 33 | 26 | 16 | 0 | 43 | 29 |
| linkedbook | 81 | 1 | 0 | 13 | 6 | 5 | 1 | 18 | 7 |


## is_match_by_similarity scored as an auto-accept gate

| index | TP | FP | FN | TN | precision | recall | f1 | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matilda | 100 | 24 | 7 | 115 | 0.806 | 0.935 | 0.866 | 0.874 |
| openalex | 123 | 29 | 9 | 130 | 0.809 | 0.932 | 0.866 | 0.869 |
| wikidata | 36 | 3 | 11 | 27 | 0.923 | 0.766 | 0.837 | 0.818 |
| ALL | 259 | 56 | 27 | 272 | 0.822 | 0.906 | 0.862 | 0.865 |

