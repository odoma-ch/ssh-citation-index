# """
# Wikidata SPARQL connector for reference search and disambiguation.
# Focuses on books and publications not covered by academic databases.
# """

# import requests
# from typing import Dict, List, Any, Optional
# import json
# import urllib.parse

# from .base import BaseConnector
# from ..models.reference import Reference
# from ..models.references import References


# class WikidataConnector(BaseConnector):
#     """Connector for Wikidata SPARQL query service."""
    
#     def __init__(self, timeout: int = 30):
#         """
#         Initialize Wikidata connector.
        
#         Args:
#             timeout: Request timeout in seconds
#         """
#         self.base_url = "https://query.wikidata.org/sparql"
#         self.timeout = timeout
#         self.headers = {
#             'User-Agent': 'Citation-Index/1.0 (https://github.com/your-repo) wikidata-connector'
#         }
    
#     def search(self, reference: Reference, top_k: int = 10, **kwargs) -> References:
#         """
#         Search for books and publications on Wikidata.
        
#         Args:
#             reference: Reference object with search criteria
#             top_k: Maximum number of results to return
#             **kwargs: Additional search parameters:
#                 - book_only: Search only for books (Q571)
#                 - written_work_only: Search only for written works (Q47461344)
#                 - include_isbn: Include ISBN search
#                 - language: Language filter (e.g., 'en', 'fr')
        
#         Returns:
#             References object containing search results
#         """
#         # Build SPARQL query based on reference content
#         sparql_query = self._build_sparql_query(reference, top_k, **kwargs)
        
#         # Execute query
#         results = self._execute_sparql_query(sparql_query)
        
#         # Convert to References format
#         references = self._sparql_to_references(results)
        
#         return references
    
#     def _build_sparql_query(self, reference: Reference, top_k: int, **kwargs) -> str:
#         """Build SPARQL query based on reference data."""
        
#         # Common prefixes
#         query_parts = [
#             "SELECT DISTINCT ?item ?itemLabel ?isbn ?authorLabel ?publisherLabel ?publicationDate ?description",
#             "WHERE {",
#         ]
        
#         # Determine item types to search
#         book_only = kwargs.get('book_only', False)
#         written_work_only = kwargs.get('written_work_only', False)
        
#         if book_only:
#             query_parts.append("  ?item wdt:P31 wd:Q571 .  # book")
#         elif written_work_only:
#             query_parts.append("  ?item wdt:P31 wd:Q47461344 .  # written work")
#         else:
#             query_parts.append("  { ?item wdt:P31 wd:Q571 } UNION { ?item wdt:P31 wd:Q47461344 } .  # book or written work")
        
#         # Title search
#         if reference.full_title:
#             title_escaped = reference.full_title.replace('"', '\\"')
#             query_parts.extend([
#                 f'  ?item rdfs:label ?itemLabel .',
#                 f'  FILTER(CONTAINS(LCASE(?itemLabel), LCASE("{title_escaped}"))) .',
#             ])
        
#         # Author search
#         if reference.authors and len(reference.authors) > 0:
#             author_name = reference.authors[0].replace('"', '\\"')
#             query_parts.extend([
#                 "  ?item wdt:P50 ?author .",
#                 "  ?author rdfs:label ?authorLabel .",
#                 f'  FILTER(CONTAINS(LCASE(?authorLabel), LCASE("{author_name}"))) .',
#             ])
        
#         # Publisher search
#         if reference.publisher:
#             publisher_escaped = reference.publisher.replace('"', '\\"')
#             query_parts.extend([
#                 "  ?item wdt:P123 ?publisher .",
#                 "  ?publisher rdfs:label ?publisherLabel .",
#                 f'  FILTER(CONTAINS(LCASE(?publisherLabel), LCASE("{publisher_escaped}"))) .',
#             ])
        
#         # Publication date search
#         if reference.publication_date:
#             try:
#                 # Try to extract year from publication date
#                 year = int(reference.publication_date[:4])
#                 query_parts.append(f"  ?item wdt:P577 ?publicationDate .")
#                 query_parts.append(f"  FILTER(YEAR(?publicationDate) = {year}) .")
#             except (ValueError, TypeError):
#                 pass
        
#         # ISBN search
#         if kwargs.get('include_isbn', True):
#             query_parts.append("  OPTIONAL { ?item wdt:P212 ?isbn } .")
        
#         # Optional fields
#         query_parts.extend([
#             "  OPTIONAL { ?item wdt:P50 ?author } .",
#             "  OPTIONAL { ?item wdt:P123 ?publisher } .",
#             "  OPTIONAL { ?item wdt:P577 ?publicationDate } .",
#             "  OPTIONAL { ?item schema:description ?description . FILTER(LANG(?description) = 'en') } .",
#         ])
        
#         # Language filter
#         language = kwargs.get('language', 'en')
#         query_parts.append(f'  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{language}" }} .')
        
#         query_parts.extend([
#             "}",
#             f"LIMIT {top_k}"
#         ])
        
#         return "\n".join(query_parts)
    
#     def search_by_isbn(self, isbn: str, top_k: int = 10) -> References:
#         """
#         Search for books by ISBN.
        
#         Args:
#             isbn: ISBN to search
#             top_k: Maximum number of results
            
#         Returns:
#             References object containing search results
#         """
#         isbn_clean = isbn.replace('-', '').replace(' ', '')
        
#         sparql_query = f"""
#         SELECT DISTINCT ?item ?itemLabel ?isbn ?authorLabel ?publisherLabel ?publicationDate ?description
#         WHERE {{
#           {{ ?item wdt:P31 wd:Q571 }} UNION {{ ?item wdt:P31 wd:Q47461344 }} .
#           ?item wdt:P212 ?isbn .
#           FILTER(CONTAINS(?isbn, "{isbn_clean}")) .
          
#           OPTIONAL {{ ?item wdt:P50 ?author }} .
#           OPTIONAL {{ ?item wdt:P123 ?publisher }} .
#           OPTIONAL {{ ?item wdt:P577 ?publicationDate }} .
#           OPTIONAL {{ ?item schema:description ?description . FILTER(LANG(?description) = 'en') }} .
          
#           SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }} .
#         }}
#         LIMIT {top_k}
#         """
        
#         results = self._execute_sparql_query(sparql_query)
#         return self._sparql_to_references(results)
    
#     def search_books_by_author(self, author: str, top_k: int = 10) -> References:
#         """
#         Search for books by a specific author.
        
#         Args:
#             author: Author name to search
#             top_k: Maximum number of results
            
#         Returns:
#             References object containing search results
#         """
#         author_escaped = author.replace('"', '\\"')
        
#         sparql_query = f"""
#         SELECT DISTINCT ?item ?itemLabel ?isbn ?authorLabel ?publisherLabel ?publicationDate ?description
#         WHERE {{
#           {{ ?item wdt:P31 wd:Q571 }} UNION {{ ?item wdt:P31 wd:Q47461344 }} .
#           ?item wdt:P50 ?author .
#           ?author rdfs:label ?authorLabel .
#           FILTER(CONTAINS(LCASE(?authorLabel), LCASE("{author_escaped}"))) .
          
#           OPTIONAL {{ ?item wdt:P212 ?isbn }} .
#           OPTIONAL {{ ?item wdt:P123 ?publisher }} .
#           OPTIONAL {{ ?item wdt:P577 ?publicationDate }} .
#           OPTIONAL {{ ?item schema:description ?description . FILTER(LANG(?description) = 'en') }} .
          
#           SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }} .
#         }}
#         ORDER BY ?itemLabel
#         LIMIT {top_k}
#         """
        
#         results = self._execute_sparql_query(sparql_query)
#         return self._sparql_to_references(results)
    
#     def search_by_title_fuzzy(self, title: str, top_k: int = 10) -> References:
#         """
#         Fuzzy search for books by title using CONTAINS.
        
#         Args:
#             title: Title to search
#             top_k: Maximum number of results
            
#         Returns:
#             References object containing search results
#         """
#         title_escaped = title.replace('"', '\\"')
        
#         sparql_query = f"""
#         SELECT DISTINCT ?item ?itemLabel ?isbn ?authorLabel ?publisherLabel ?publicationDate ?description
#         WHERE {{
#           {{ ?item wdt:P31 wd:Q571 }} UNION {{ ?item wdt:P31 wd:Q47461344 }} .
#           ?item rdfs:label ?itemLabel .
#           FILTER(CONTAINS(LCASE(?itemLabel), LCASE("{title_escaped}"))) .
          
#           OPTIONAL {{ ?item wdt:P212 ?isbn }} .
#           OPTIONAL {{ ?item wdt:P50 ?author }} .
#           OPTIONAL {{ ?item wdt:P123 ?publisher }} .
#           OPTIONAL {{ ?item wdt:P577 ?publicationDate }} .
#           OPTIONAL {{ ?item schema:description ?description . FILTER(LANG(?description) = 'en') }} .
          
#           SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }} .
#         }}
#         ORDER BY ?itemLabel
#         LIMIT {top_k}
#         """
        
#         results = self._execute_sparql_query(sparql_query)
#         return self._sparql_to_references(results)
    
#     def _execute_sparql_query(self, query: str) -> Dict[str, Any]:
#         """Execute SPARQL query against Wikidata."""
#         params = {
#             'query': query,
#             'format': 'json'
#         }
        
#         try:
#             response = requests.get(
#                 self.base_url,
#                 params=params,
#                 headers=self.headers,
#                 timeout=self.timeout
#             )
#             response.raise_for_status()
#             return response.json()
#         except requests.exceptions.RequestException as e:
#             print(f"Error executing SPARQL query: {e}")
#             return {'results': {'bindings': []}}
    
#     def _sparql_to_references(self, sparql_results: Dict[str, Any]) -> References:
#         """Convert SPARQL results to References format."""
#         references = []
        
#         for binding in sparql_results.get('results', {}).get('bindings', []):
#             # Extract basic info
#             wikidata_id = binding.get('item', {}).get('value', '').split('/')[-1]
#             title = binding.get('itemLabel', {}).get('value', '')
#             description = binding.get('description', {}).get('value', '')
            
#             # Extract author info
#             authors = []
#             if 'authorLabel' in binding:
#                 author_name = binding['authorLabel'].get('value', '')
#                 if author_name and author_name not in authors:
#                     authors.append(author_name)
            
#             # Extract publisher
#             publisher = None
#             if 'publisherLabel' in binding:
#                 publisher = binding['publisherLabel'].get('value', '')
            
#             # Extract publication date
#             publication_date = None
#             if 'publicationDate' in binding:
#                 pub_date = binding['publicationDate'].get('value', '')
#                 if pub_date:
#                     # Extract year from datetime
#                     try:
#                         publication_date = pub_date[:4]
#                     except:
#                         publication_date = pub_date
            
#             # Extract ISBN
#             isbn = None
#             if 'isbn' in binding:
#                 isbn = binding['isbn'].get('value', '')
            
#             # Create Reference object
#             reference = Reference(
#                 full_title=title,
#                 authors=authors if authors else None,
#                 publisher=publisher,
#                 publication_date=publication_date,
#                 isbn=isbn,
#                 source="wikidata",
#                 wikidata_id=wikidata_id,
#                 description=description
#             )
            
#             references.append(reference)
        
#         return References(references=references)
    
#     def map_to_references(self, results: List[Dict[str, Any]]) -> References:
#         """
#         Map raw API results to References format.
#         This method is called by the base class.
#         """
#         # This connector handles mapping in _sparql_to_references
#         # This method is required by the base class interface
#         return References(references=[])


# # Example usage:
# # connector = WikidataConnector()
# # 
# # # Search by title
# # reference = Reference(full_title="The Great Gatsby")
# # results = connector.lookup_ref(reference, top_k=5)
# # 
# # # Search by ISBN
# # isbn_results = connector.search_by_isbn("978-0-7432-7356-5")
# # 
# # # Search by author
# # author_results = connector.search_books_by_author("F. Scott Fitzgerald")
# # 
# # for ref in results.references:
# #     print(f"Title: {ref.full_title}")
# #     print(f"Authors: {ref.authors}")
# #     print(f"Publisher: {ref.publisher}")
# #     print(f"Year: {ref.publication_date}")
# #     print(f"ISBN: {ref.isbn}")
# #     print(f"Wikidata ID: {ref.wikidata_id}")
# #     print("---")

# wikidata_lookup.py
from __future__ import annotations
import requests
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import re
import time

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WDQS_ENDPOINT = "https://query.wikidata.org/sparql"
UA = "Graphia-SSH-CitationIndex/1.0 (contact: your-email@example.org)"

# -----------------------------
# Data model for input
# -----------------------------
@dataclass
class Ref:
    title: str                                # mandatory
    authors: Optional[List[str]] = None       # ["Ada Lovelace", "A. Turing"]
    year: Optional[int] = None
    container: Optional[str] = None           # journal/book/proceedings
    limit: int = 10                           # max candidates to return

# -----------------------------
# Utilities
# -----------------------------
def _norm(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _surname_set(authors: Optional[List[str]]) -> set:
    if not authors:
        return set()
    surnames = set()
    for a in authors:
        a = _norm(a)
        # heuristic: take last token that has letters
        parts = [p for p in re.split(r"[\s\-]", a) if p]
        if parts:
            surnames.add(parts[-1].lower())
    return surnames

def _headers_json() -> Dict[str, str]:
    return {"User-Agent": UA, "Accept": "application/json"}

def _post_sparql(query: str) -> Dict[str, Any]:
    r = requests.post(
        WDQS_ENDPOINT,
        data={"query": query},
        headers={"User-Agent": UA, "Accept": "application/sparql-results+json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

# -----------------------------
# Method 1: Elastic search API + hydrate via SPARQL
# -----------------------------
def wikidata_search_elastic(ref: Ref) -> Dict[str, Any]:
    """
    Use wbsearchentities to get top QIDs for the title, then hydrate with SPARQL
    to pull bibliographic properties. Optionally filter/rank by year/authors/container.
    Returns a JSON dict: { 'source': 'wikidata-elastic', 'query': {...}, 'candidates': [...] }
    """
    if not ref.title:
        raise ValueError("title is required")

    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "type": "item",
        "limit": max(1, min(ref.limit, 50)),
        "search": ref.title,
    }
    resp = requests.get(WIKIDATA_API, params=params, headers=_headers_json(), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    hits = data.get("search", [])

    qids = [h["id"] for h in hits if h.get("id", "").startswith("Q")]
    if not qids:
        return {"source": "wikidata-elastic", "query": asdict(ref), "candidates": []}

    # Hydrate properties for those QIDs via one SPARQL call
    values_block = " ".join(f"wd:{qid}" for qid in qids)

    sparql = f"""
    SELECT ?item ?itemLabel ?title ?date ?year ?containerLabel ?doi ?isbn13 ?isbn10 ?pmid ?pmcid ?arxiv ?issn
           (GROUP_CONCAT(DISTINCT ?authLabel; separator="; ") AS ?authorsLinked)
           (GROUP_CONCAT(DISTINCT ?authStr;   separator="; ") AS ?authorsString)
    WHERE {{
      VALUES ?item {{ {values_block} }}
      OPTIONAL {{ ?item wdt:P1476 ?title. }}
      OPTIONAL {{ ?item wdt:P577 ?date. BIND(YEAR(?date) AS ?year) }}
      OPTIONAL {{ ?item wdt:P1433 ?container. ?container rdfs:label ?containerLabel FILTER(LANG(?containerLabel) = "en") }}
      OPTIONAL {{ ?item wdt:P356 ?doi. }}
      OPTIONAL {{ ?item wdt:P212 ?isbn13. }}
      OPTIONAL {{ ?item wdt:P957 ?isbn10. }}
      OPTIONAL {{ ?item wdt:P698 ?pmid. }}
      OPTIONAL {{ ?item wdt:P932 ?pmcid. }}
      OPTIONAL {{ ?item wdt:P818 ?arxiv. }}
      OPTIONAL {{ ?item wdt:P236 ?issn. }}
      OPTIONAL {{ ?item wdt:P50 ?a. ?a rdfs:label ?authLabel FILTER(LANG(?authLabel) = "en") }}
      OPTIONAL {{ ?item wdt:P2093 ?authStr. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    GROUP BY ?item ?itemLabel ?title ?date ?year ?containerLabel ?doi ?isbn13 ?isbn10 ?pmid ?pmcid ?arxiv ?issn
    """

    hydrated = _post_sparql(sparql)
    rows = hydrated.get("results", {}).get("bindings", [])

    # Basic, transparent scoring to help you downstream (not strictly necessary)
    want_year = ref.year
    want_surnames = _surname_set(ref.authors)
    want_container = (_norm(ref.container).lower() if ref.container else None)

    def ext(v: Dict[str, Any], key: str) -> Optional[str]:
        return v.get(key, {}).get("value")

    candidates = []
    for r in rows:
        qid = ext(r, "item").rsplit("/", 1)[-1]
        item = {
            "qid": qid,
            "label": ext(r, "itemLabel"),
            "title": ext(r, "title"),
            "date": ext(r, "date"),
            "year": int(ext(r, "year")) if ext(r, "year") else None,
            "container": ext(r, "containerLabel"),
            "identifiers": {
                "doi": ext(r, "doi"),
                "isbn13": ext(r, "isbn13"),
                "isbn10": ext(r, "isbn10"),
                "pmid": ext(r, "pmid"),
                "pmcid": ext(r, "pmcid"),
                "arxiv": ext(r, "arxiv"),
                "issn": ext(r, "issn"),
            },
            "authors_linked": [a.strip() for a in (ext(r, "authorsLinked") or "").split(";") if a.strip()],
            "authors_string": [a.strip() for a in (ext(r, "authorsString") or "").split(";") if a.strip()],
        }

        # Simple score: title presence + year proximity + surname overlap + container hint
        score = 0.0
        if item["title"]:
            # Very lightweight token overlap with query title
            t_query = set(_norm(ref.title).lower().split())
            t_item = set(_norm(item["title"]).lower().split())
            inter = len(t_query & t_item)
            score += min(1.0, inter / max(3, len(t_query))) * 0.55

        if want_year and item["year"]:
            dy = abs(want_year - item["year"])
            score += (1.0 if dy == 0 else 0.6 if dy == 1 else 0.0) * 0.15

        if want_surnames:
            names = " ".join(item["authors_linked"] + item["authors_string"]).lower()
            surname_hits = sum(1 for s in want_surnames if s in names)
            score += min(1.0, surname_hits / max(1, len(want_surnames))) * 0.2

        if want_container and item["container"]:
            if want_container in item["container"].lower():
                score += 0.1

        item["score_hint"] = round(score, 3)
        candidates.append(item)

    # Attach wbsearch rank as a nudge (keep your own scoring primary)
    rank_by_qid = {h["id"]: i for i, h in enumerate(hits)}
    for c in candidates:
        c["wbsearch_rank"] = rank_by_qid.get(c["qid"], 999)

    # Sort: higher score, then better wbsearch rank
    candidates.sort(key=lambda x: (-x["score_hint"], x["wbsearch_rank"]))

    # Optional: trim to ref.limit
    if ref.limit and len(candidates) > ref.limit:
        candidates = candidates[:ref.limit]

    return {"source": "wikidata-elastic", "query": asdict(ref), "candidates": candidates}

# -----------------------------
# Method 2: SPARQL-only (uses mwapi:Search inside WDQS)
# -----------------------------
def wikidata_search_sparql(ref: Ref) -> Dict[str, Any]:
    """
    Use WDQS with SERVICE wikibase:mwapi Search to find candidates by title,
    then pull structured bibliographic fields in the same query.
    Returns a JSON dict like the elastic version.
    """
    if not ref.title:
        raise ValueError("title is required")

    # Prepare author surname and year filters (optional, soft filters)
    surname_filters = ""
    if ref.authors:
        # Build FILTERs to match any surname in linked or string authors
        surnames = [re.escape(s) for s in _surname_set(ref.authors)]
        if surnames:
            pattern = "|".join(surnames)
            surname_filters = f"""
            BIND( IF(BOUND(?authNameLower) && REGEX(?authNameLower, "(?:{pattern})"), 1, 0) AS ?authHit )
            """

    year_filter = ""
    if ref.year:
        y = ref.year
        year_filter = f"""
        BIND( IF(BOUND(?year) && (?year = {y} || ?year = {y-1} || ?year = {y+1}), 1, 0) AS ?yearHit )
        """

    container_filter = ""
    if ref.container:
        cont = _norm(ref.container).lower().replace('"', '\\"')
        container_filter = f"""
        BIND(LCASE(COALESCE(?containerLabel, "")) AS ?containerLower)
        BIND( IF(CONTAINS(?containerLower, "{cont}"), 1, 0) AS ?containerHit )
        """

    limit = max(1, min(ref.limit or 10, 50))
    title_q = ref.title.replace('"', '\\"')

    sparql = f"""
    SELECT ?item ?itemLabel ?title ?date ?year ?containerLabel ?doi ?isbn13 ?isbn10 ?pmid ?pmcid ?arxiv ?issn
           (GROUP_CONCAT(DISTINCT ?authName; separator="; ") AS ?authors)
           (COALESCE(?yearHit, 0) AS ?yScore)
           (COALESCE(?authHit, 0) AS ?aScore)
           (COALESCE(?containerHit, 0) AS ?cScore)
    WHERE {{
      # 1) Candidate generation via Elastic search (inside WDQS)
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:endpoint "www.wikidata.org";
                         wikibase:api "Search";
                         mwapi:search "{title_q}";
                         mwapi:language "en".
        ?item wikibase:apiOutputItem mwapi:item .
      }}

      # 2) Pull bibliographic fields
      OPTIONAL {{ ?item wdt:P1476 ?title. }}
      OPTIONAL {{ ?item wdt:P577 ?date. BIND(YEAR(?date) AS ?year) }}
      OPTIONAL {{ ?item wdt:P1433 ?container. ?container rdfs:label ?containerLabel FILTER(LANG(?containerLabel)="en") }}
      OPTIONAL {{ ?item wdt:P356 ?doi. }}
      OPTIONAL {{ ?item wdt:P212 ?isbn13. }}
      OPTIONAL {{ ?item wdt:P957 ?isbn10. }}
      OPTIONAL {{ ?item wdt:P698 ?pmid. }}
      OPTIONAL {{ ?item wdt:P932 ?pmcid. }}
      OPTIONAL {{ ?item wdt:P818 ?arxiv. }}
      OPTIONAL {{ ?item wdt:P236 ?issn. }}
      OPTIONAL {{
        {{ ?item wdt:P50 ?a. ?a rdfs:label ?authName FILTER(LANG(?authName)="en") }}
        UNION
        {{ ?item wdt:P2093 ?authName }}
      }}
      BIND(LCASE(COALESCE(?authName, "")) AS ?authNameLower)

      # 3) Soft signals
      {surname_filters}
      {year_filter}
      {container_filter}

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    GROUP BY ?item ?itemLabel ?title ?date ?year ?containerLabel ?doi ?isbn13 ?isbn10 ?pmid ?pmcid ?arxiv ?issn ?yScore ?aScore ?cScore
    LIMIT {limit}
    """

    js = _post_sparql(sparql)
    rows = js.get("results", {}).get("bindings", [])

    def ext(v: Dict[str, Any], key: str) -> Optional[str]:
        return v.get(key, {}).get("value")

    cands = []
    for r in rows:
        qid = ext(r, "item").rsplit("/", 1)[-1]
        item = {
            "qid": qid,
            "label": ext(r, "itemLabel"),
            "title": ext(r, "title"),
            "date": ext(r, "date"),
            "year": int(ext(r, "year")) if ext(r, "year") else None,
            "container": ext(r, "containerLabel"),
            "identifiers": {
                "doi": ext(r, "doi"),
                "isbn13": ext(r, "isbn13"),
                "isbn10": ext(r, "isbn10"),
                "pmid": ext(r, "pmid"),
                "pmcid": ext(r, "pmcid"),
                "arxiv": ext(r, "arxiv"),
                "issn": ext(r, "issn"),
            },
            "authors": [a.strip() for a in (ext(r, "authors") or "").split(";") if a.strip()],
        }
        # Light composite score (0–1): year hit + author hit + container hit + title token overlap
        score = 0.0
        score += (int(float(ext(r, "yScore") or "0"))) * 0.25
        score += (int(float(ext(r, "aScore") or "0"))) * 0.35
        score += (int(float(ext(r, "cScore") or "0"))) * 0.15

        # Title overlap (client-side)
        if item["title"]:
            t_query = set(_norm(ref.title).lower().split())
            t_item = set(_norm(item["title"]).lower().split())
            inter = len(t_query & t_item)
            score += min(1.0, inter / max(3, len(t_query))) * 0.25

        item["score_hint"] = round(min(1.0, score), 3)
        cands.append(item)

    # Sort by score
    cands.sort(key=lambda x: -x["score_hint"])

    return {"source": "wikidata-sparql", "query": asdict(ref), "candidates": cands}

# -----------------------------
# Tiny CLI test (optional)
# -----------------------------
if __name__ == "__main__":
    example = Ref(
        title="A New Evaluation Model For Intellectual Capital Based On Computing With Linguistic Variable",
        authors=["C. Z. something", "Zadeh"],  # put real authors if known
        year=2010,
        container=None,
        limit=10,
    )

    print("Elastic + hydrate:")
    out1 = wikidata_search_elastic(example)
    print(out1)

    # Friendly pause to be nice to endpoints
    time.sleep(1.2)

    print("\nSPARQL (mwapi inside):")
    out2 = wikidata_search_sparql(example)
    print(out2)
