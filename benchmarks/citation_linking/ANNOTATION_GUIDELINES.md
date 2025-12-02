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
- is_match_by_similarity: Model’s heuristic guess (for context only).
- matched_link: Link to view the candidate on the provider site.

## What you need to do
1) Check if the candidate is the same work as the reference.
   - Compare title, first author surname, year, and DOI (if present).
   - Use matched_link to verify details on the provider page.
2) If the candidate is incorrect OR empty (matched_id = "Not Found" or no details shown), find the correct record in the same provider and paste its ID in correct_id.
3) If you believe no record exists for this reference in this provider, select “No match”.

## How to answer
- Candidate is correct:
  - is_match_correct = true
  - No match = false
  - correct_id = (leave blank)
- Candidate is incorrect but correct record exists:
  - is_match_correct = false
  - No match = false
  - correct_id = provider-specific ID (OpenAlex W…, Wikidata Q…, Matilda work id, OpenCitations OMID br/…)
- No record in this provider:
  - is_match_correct = false
  - No match = true
  - correct_id = (leave blank)

Notes:
- Minor formatting/casing differences are fine; it must be the same work.
- Provide only the ID (not a URL).


## Decision FLow you should follow: 
![Decision Flow](https://raw.githubusercontent.com/odoma-ch/ssh-citation-index/refs/heads/main/benchmarks/citation_linking/argilla-decision-flow-TD.png " Decision Flow")
