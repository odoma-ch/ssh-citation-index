Here’s a crisp summary you can share, plus simple pseudocode.

# Reference Matching: Hybrid Blocking + Scoring (with smart fallbacks)

**Goal:** link a structured reference object (title, authors, year, etc.) to the correct record in external sources (OpenAlex, Crossref, Wikidata/OpenCitations…) with high precision and strong recall on messy data.

**Core idea:** combine (1) deterministic, high-precision rules to “fast-accept” obvious matches, with (2) fuzzy, feature-based scoring among a small, well-chosen candidate set. Candidate search (“blocking”) is done in adaptive passes—strict first, then progressively looser only when needed.

## Pipeline (overview)

1. **Normalize & enrich once**

   * Produce normalized views of title (basic, token-set, “head” before colon), author family names, year, pages/article-number, volume/issue, ISSN/ISBN, DOI.
   * Keep both raw and normalized forms.

2. **Candidate search (blocking)**

   * Query multiple backends in parallel; cap per-backend Top-K (e.g., 5) and deduplicate.
   * Use adaptive passes:

     * **Pass A (strict-ish):** title phrase/search + filters (year ±1, first author family, volume, begin-page/article-number, venue ID).
     * **Pass B (relaxed):** title keywords (IDF/longest tokens), year ±2, drop weaker filters.
     * **Pass C (minimal/venue-anchored):** title keywords only, but require author family **or** ISSN/ISBN.
   * By default run **A only**, and promote to **B/C** as **fallbacks**; if input is weak (missing year/author or very short titles), include **B/C** up front.

3. **Deterministic fast-accept (paper-style)**

   * If any candidate satisfies:
     DOI exact → accept;
     or `year + volume + (begin-page|article-number)`;
     or `year + first-author family + (begin-page|article-number)`;
     or `year + first-author family + volume`;
     or `year + ISSN/ISBN + (begin-page|article-number)` → accept immediately.

4. **Feature scoring (hybrid)**

   * For remaining candidates, compute a small set of interpretable features: title similarity (partial/token-set/head), author family exact/fuzzy, year distance, volume/page/ISSN flags, backend search score, DOI match.
   * Linear weighted score; **decision bands**: accept (≥ τ_high), review (≥ τ_low), otherwise reject. Also accept if Top-1 beats Top-2 by a safe margin (e.g., ≥ 0.15) and ≥ τ_low.

5. **Fallback loop**

   * If nothing meets τ_low, widen search progressively (run Pass B then Pass C) and repeat fast-accept + scoring. If still nothing, fail gracefully.

6. **Output & audit**

   * Return decision, chosen match, score, fired rules, feature values, pass used, top alternatives, retries attempted—so results are explainable and debuggable.

---

## Simple pseudocode (logic only)

```pseudo
function match_reference(ref):
    canon = normalize(ref)  // titles, authors, numbers, IDs

    passes = decide_initial_passes(canon)  // ["A"], ["A","B"], or ["A","B","C"] based on signal

    candidates = []
    for p in passes:
        candidates += query_all_backends(canon, pass=p, topk_per_backend=5)
    candidates = dedupe(candidates)

    // 1) deterministic fast-accept
    hit = fast_accept(canon, candidates)
    if hit:
        return ACCEPT(hit, reason="fast_rule")

    // 2) score candidates
    scored = score_all(canon, candidates)  // compute features -> weighted score
    best, second = top2(scored)

    if best.score >= TH_ACCEPT:
        return ACCEPT(best, reason="score_high")
    if best.score >= TH_REVIEW and (best.score - second.score) >= TOP_GAP_MARGIN:
        return ACCEPT(best, reason="score_gap")
    if "B" not in passes:
        return retry_with(["B"], canon)
    if "C" not in passes:
        return retry_with(["C"], canon)

    return FAIL(scored, reasons="below_threshold")

function retry_with(additional_passes, canon):
    more = []
    for p in additional_passes:
        more += query_all_backends(canon, pass=p, topk_per_backend=5)
    more = dedupe(more)
    hit = fast_accept(canon, more)
    if hit: return ACCEPT(hit, reason="fast_rule")
    scored = score_all(canon, more)
    decide as above...
```

**Key defaults to share**

* Top-K per backend per pass: 5 (total pool usually ≤ 25/pass).
* Thresholds: `TH_ACCEPT ≈ 0.85`, `TH_REVIEW ≈ 0.70`, `TOP_GAP_MARGIN ≈ 0.15`.
* Year tolerance: ±1 in Pass A, ±2 in Pass B; handle pages vs article numbers symmetrically.
* DOI exact match short-circuits to accept.

This gives you the **speed and precision** of deterministic rules, the **recall** of fuzzy scoring, and **bounded cost** via adaptive fallbacks.
