# Annotation Guidelines: Citation Linking

## What the fields mean
- ref_id: Internal identifier of the reference.
- original_ref_string: The raw reference text extracted from the document.
- matched_id: Provider-specific ID of the candidate match (or "Not Found").
  - OpenAlex: W1234567890 (last segment of /works/W…)
  - Wikidata: Q12345
  - Matilda: last path segment in /work/<id>
  - OpenCitations: OMID last path segment (e.g., br/06210459208)
- matched_doi: DOI of the candidate match (if available).
- matched_result: Compact summary of the candidate (title, first_author, year, journal).
- is_match_by_similarity:  Model's heuristic confidence score. Use as a soft signal only — always verify by comparing the reference with the candidate yourself.
- matched_link: Link to view the candidate on the provider site. Also use this to navigate to the provider homepage if you need to search manually.

## What you need to do
1) Check if the candidate is the same work as the reference.
   - Compare title, first author surname, year, and DOI (if present).
   - Use matched_link to verify details on the provider page.
   - Minor formatting or casing differences are fine — it must be the same work.
   - **Translations**: A translated edition of a work is **not** the same record as the original. If the reference cites a translation (e.g., a French translation of an English book), match it only if the provider has a record for that specific translation. Do not match it to the original-language edition.
2) **If the candidate is incorrect OR absent** (`matched_id` = "Not Found" or `matched_result` is empty), search for the correct record manually in the **same provider** (use `matched_link` to reach the provider site) and paste its ID in `correct_id`.
3) **If no record exists** for this reference in this provider, select "No match".

## How to answer
There are four possible outcomes depending on whether a candidate was provided and whether the correct record exists in the provider.
 
### A candidate was provided
 
| Scenario | is_match_correct | No match | correct_id |
|---|---|---|---|
| Candidate is correct | true | false | *(leave blank)* |
| Candidate is wrong, but correct record exists | false | false | Provider-specific ID |
| Candidate is wrong, and no record exists | false | true | *(leave blank)* |
 
### No candidate was provided
 
(i.e., `matched_id` = "Not Found" or `matched_result` is empty)
 
| Scenario | is_match_correct | No match | correct_id |
|---|---|---|---|
| You found the correct record manually | false | false | Provider-specific ID |
| You confirmed no record exists | true | true | *(leave blank)* |
 
> **Why is `is_match_correct = true` when there is no candidate and no record?**
> Because the system's output ("Not Found") was correct — the reference genuinely has no match in this provider.

### ID format reminder
 
Provide only the ID, not a full URL:
 
- OpenAlex: `W…`
- Wikidata: `Q…`
- Matilda: work id
- OpenCitations: `br/…`

## Provider-specific notes
 
### Matilda
 
When searching manually, use Matilda's **advanced search** to filter by title, author, and year for more precise results: [Matilda advanced search FAQ](https://matilda.science/faq?l=en#i207).
 
### Wikidata
 
Wikidata may contain separate entries for an **original work** and a **review** of that work. Make sure you match the reference to the correct type:
- If the reference cites the original work (book, article, etc.), match it to the Wikidata item for the **original work** — not a review of it.
- If the reference cites a review, match it to the Wikidata item for the **review**.
- Check the item's "instance of" (`P31`) property to confirm: look for values like "scholarly article", "book", "review article", etc.

## Decision FLow you should follow: 
![Decision Flow](https://raw.githubusercontent.com/odoma-ch/ssh-citation-index/refs/heads/main/benchmarks/citation_linking/argilla-decision-flow-TD.png " Decision Flow")
