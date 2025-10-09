"""
Wikidata SPARQL connector for reference search and disambiguation.
Focuses on books and publications not covered by academic databases.
"""

import requests
import re
import time
from typing import Dict, List, Any, Optional
from urllib.parse import quote

from .base import BaseConnector
from ..models.reference import Reference
from ..models.references import References


class WikidataConnector(BaseConnector):
    """Connector for Wikidata SPARQL query service."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: int = 30):
        """
        Initialize Wikidata connector.
        
        Args:
            api_key: Not used for Wikidata (kept for interface compatibility)
            base_url: Base URL for Wikidata SPARQL endpoint
            timeout: Request timeout in seconds
        """
        super().__init__(api_key, base_url)
        self.base_url = base_url or "https://query.wikidata.org/sparql"
        self.api_url = "https://www.wikidata.org/w/api.php"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CitationIndex/1.0 (https://github.com/citation-index) wikidata-connector",
            "Accept": "application/json"
        })
    
    def search(self, reference: Reference, top_k: int = 10, method: str = "elastic", **kwargs) -> List[Dict[str, Any]]:
        """
        Search for books and publications on Wikidata.
        
        Args:
            reference: Reference object with search criteria (title is mandatory)
            top_k: Maximum number of results to return
            method: Search method - "elastic" (default) or "sparql"
            **kwargs: Additional search parameters:
                - book_only: Search only for books (Q571)
                - written_work_only: Search only for written works (Q47461344)
                - language: Language filter (default: 'en')
        
        Returns:
            List of raw API response data
            
        Raises:
            ValueError: If reference title is missing
        """
        self._validate_reference(reference)
        
        if method == "elastic":
            return self._search_elastic(reference, top_k, **kwargs)
        elif method == "sparql":
            return self._search_sparql(reference, top_k, **kwargs)
        else:
            raise ValueError(f"Unknown search method: {method}. Use 'elastic' or 'sparql'")
    
    def _search_elastic(self, reference: Reference, top_k: int, **kwargs) -> List[Dict[str, Any]]:
        """
        Use wbsearchentities to get top QIDs for the title, then hydrate with SPARQL.
        
        This is generally faster and more reliable than pure SPARQL search.
        """
        if not reference.full_title:
            return []
        
        # Step 1: Get candidate QIDs using elastic search
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": kwargs.get("language", "en"),
            "type": "item",
            "limit": max(1, min(top_k, 50)),
            "search": reference.full_title,
        }
        
        try:
            response = self.session.get(self.api_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            hits = data.get("search", [])
            
            qids = [h["id"] for h in hits if h.get("id", "").startswith("Q")]
            if not qids:
                return []
            
            # Step 2: Hydrate properties for those QIDs via SPARQL
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
            
            hydrated = self._execute_sparql(sparql)
            rows = hydrated.get("results", {}).get("bindings", [])
            
            # Attach search rank for sorting
            rank_by_qid = {h["id"]: i for i, h in enumerate(hits)}
            for row in rows:
                qid = row.get("item", {}).get("value", "").rsplit("/", 1)[-1]
                row["wbsearch_rank"] = rank_by_qid.get(qid, 999)
            
            # Score and sort results
            scored_rows = self._score_results(rows, reference)
            return scored_rows[:top_k]
            
        except requests.RequestException as e:
            print(f"Warning: Wikidata elastic search failed: {e}")
            return []
    
    def _search_sparql(self, reference: Reference, top_k: int, **kwargs) -> List[Dict[str, Any]]:
        """
        Use WDQS with SERVICE wikibase:mwapi Search to find candidates.
        
        This method does everything in one SPARQL query but may be slower.
        """
        if not reference.full_title:
            return []
        
        title_escaped = reference.full_title.replace('"', '\\"')
        limit = max(1, min(top_k, 50))
        
        # Build optional filters
        year_filter = ""
        if reference.publication_date:
            year = self._extract_year(reference.publication_date)
            if year:
                year_filter = f"""
                BIND( IF(BOUND(?year) && (?year = {year} || ?year = {year-1} || ?year = {year+1}), 1, 0) AS ?yearHit )
                """
        
        author_filter = ""
        if reference.authors:
            surnames = self._extract_surnames(reference.authors)
            if surnames:
                pattern = "|".join(re.escape(s) for s in surnames)
                author_filter = f"""
                BIND( IF(BOUND(?authNameLower) && REGEX(?authNameLower, "(?:{pattern})"), 1, 0) AS ?authHit )
                """
        
        sparql = f"""
        SELECT ?item ?itemLabel ?title ?date ?year ?containerLabel ?doi ?isbn13 ?isbn10 ?pmid ?pmcid ?arxiv ?issn
               (GROUP_CONCAT(DISTINCT ?authName; separator="; ") AS ?authors)
               (COALESCE(?yearHit, 0) AS ?yScore)
               (COALESCE(?authHit, 0) AS ?aScore)
        WHERE {{
          # Candidate generation via Elastic search
          SERVICE wikibase:mwapi {{
            bd:serviceParam wikibase:endpoint "www.wikidata.org";
                             wikibase:api "Search";
                             mwapi:search "{title_escaped}";
                             mwapi:language "en".
            ?item wikibase:apiOutputItem mwapi:item .
          }}

          # Pull bibliographic fields
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

          # Soft signals
          {author_filter}
          {year_filter}

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        GROUP BY ?item ?itemLabel ?title ?date ?year ?containerLabel ?doi ?isbn13 ?isbn10 ?pmid ?pmcid ?arxiv ?issn ?yScore ?aScore
        LIMIT {limit}
        """
        
        try:
            result = self._execute_sparql(sparql)
            rows = result.get("results", {}).get("bindings", [])
            
            # Score results
            scored_rows = self._score_results(rows, reference)
            return scored_rows[:top_k]
            
        except requests.RequestException as e:
            print(f"Warning: Wikidata SPARQL search failed: {e}")
            return []
    
    def _execute_sparql(self, query: str) -> Dict[str, Any]:
        """Execute SPARQL query against Wikidata."""
        try:
            response = self.session.post(
                self.base_url,
                data={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error executing SPARQL query: {e}")
            return {"results": {"bindings": []}}
    
    def _score_results(self, rows: List[Dict[str, Any]], reference: Reference) -> List[Dict[str, Any]]:
        """Score and sort search results based on relevance."""
        want_year = None
        if reference.publication_date:
            want_year = self._extract_year(reference.publication_date)
        
        want_surnames = set()
        if reference.authors:
            want_surnames = self._extract_surnames(reference.authors)
        
        for row in rows:
            score = 0.0
            
            # Title similarity
            title = self._extract_value(row, "title") or self._extract_value(row, "itemLabel")
            if title and reference.full_title:
                query_tokens = set(self._normalize(reference.full_title).lower().split())
                item_tokens = set(self._normalize(title).lower().split())
                if query_tokens and item_tokens:
                    overlap = len(query_tokens & item_tokens)
                    score += min(1.0, overlap / max(3, len(query_tokens))) * 0.55
            
            # Year proximity
            if want_year:
                year_str = self._extract_value(row, "year")
                if year_str:
                    try:
                        year = int(year_str)
                        dy = abs(want_year - year)
                        score += (1.0 if dy == 0 else 0.6 if dy == 1 else 0.0) * 0.15
                    except (ValueError, TypeError):
                        pass
            
            # Author match
            if want_surnames:
                authors_linked = self._extract_value(row, "authorsLinked") or ""
                authors_string = self._extract_value(row, "authorsString") or ""
                authors_all = self._extract_value(row, "authors") or ""
                author_text = " ".join([authors_linked, authors_string, authors_all]).lower()
                
                surname_hits = sum(1 for s in want_surnames if s in author_text)
                if surname_hits > 0:
                    score += min(1.0, surname_hits / max(1, len(want_surnames))) * 0.2
            
            # Use SPARQL scores if available
            y_score = self._extract_value(row, "yScore")
            if y_score:
                try:
                    score += float(y_score) * 0.05
                except (ValueError, TypeError):
                    pass
            
            a_score = self._extract_value(row, "aScore")
            if a_score:
                try:
                    score += float(a_score) * 0.05
                except (ValueError, TypeError):
                    pass
            
            row["score_hint"] = round(score, 3)
        
        # Sort by score (desc), then by wbsearch_rank (asc) if available
        rows.sort(key=lambda x: (-x.get("score_hint", 0), x.get("wbsearch_rank", 999)))
        return rows
    
    def map_to_references(self, raw_results: List[Dict[str, Any]]) -> References:
        """
        Transform Wikidata SPARQL results to Reference objects.
        
        Args:
            raw_results: List of SPARQL binding results
            
        Returns:
            References object containing mapped Reference instances
        """
        references = References()
        
        for binding in raw_results:
            try:
                # Extract QID
                qid = self._extract_value(binding, "item")
                if qid:
                    qid = qid.rsplit("/", 1)[-1]
                
                # Extract title
                title = self._extract_value(binding, "title") or self._extract_value(binding, "itemLabel")
                if not title:
                    continue
                
                # Extract authors
                authors = []
                authors_linked = self._extract_value(binding, "authorsLinked") or ""
                authors_string = self._extract_value(binding, "authorsString") or ""
                authors_all = self._extract_value(binding, "authors") or ""
                
                for auth_field in [authors_linked, authors_string, authors_all]:
                    if auth_field:
                        for author in auth_field.split(";"):
                            author = author.strip()
                            if author and author not in authors:
                                authors.append(author)
                
                # Extract publication info
                year_str = self._extract_value(binding, "year")
                date_str = self._extract_value(binding, "date")
                pub_date = None
                if year_str:
                    pub_date = str(year_str)
                elif date_str:
                    try:
                        pub_date = date_str[:4] if len(date_str) >= 4 else date_str
                    except:
                        pub_date = date_str
                
                # Extract venue/container
                container = self._extract_value(binding, "containerLabel")
                
                # Extract identifiers
                doi = self._extract_value(binding, "doi")
                isbn13 = self._extract_value(binding, "isbn13")
                isbn10 = self._extract_value(binding, "isbn10")
                pmid = self._extract_value(binding, "pmid")
                pmcid = self._extract_value(binding, "pmcid")
                arxiv = self._extract_value(binding, "arxiv")
                issn = self._extract_value(binding, "issn")
                
                # Prefer ISBN-13 over ISBN-10
                isbn = isbn13 or isbn10
                
                # Create Reference object
                reference = Reference(
                    full_title=title,
                    authors=authors if authors else None,
                    journal_title=container,
                    publication_date=pub_date,
                    isbn=isbn,
                    doi=doi,
                    source="wikidata",
                    wikidata_id=qid
                )
                
                references.append(reference)
                
            except Exception as e:
                print(f"Warning: Could not map Wikidata result to Reference: {e}")
                continue
        
        return references
    
    def search_by_isbn(self, isbn: str, top_k: int = 10) -> References:
        """
        Search for books by ISBN.
        
        Args:
            isbn: ISBN to search (10 or 13 digits)
            top_k: Maximum number of results
            
        Returns:
            References object containing search results
        """
        isbn_clean = isbn.replace("-", "").replace(" ", "")
        
        sparql = f"""
        SELECT DISTINCT ?item ?itemLabel ?isbn ?authorLabel ?publisherLabel ?publicationDate ?description
        WHERE {{
          {{ ?item wdt:P31 wd:Q571 }} UNION {{ ?item wdt:P31 wd:Q47461344 }} .
          {{ ?item wdt:P212 ?isbn }} UNION {{ ?item wdt:P957 ?isbn }} .
          FILTER(CONTAINS(?isbn, "{isbn_clean}")) .
         
          OPTIONAL {{ ?item wdt:P50 ?author }} .
          OPTIONAL {{ ?item wdt:P123 ?publisher }} .
          OPTIONAL {{ ?item wdt:P577 ?publicationDate }} .
          OPTIONAL {{ ?item schema:description ?description . FILTER(LANG(?description) = 'en') }} .
         
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }} .
        }}
        LIMIT {top_k}
        """
        
        result = self._execute_sparql(sparql)
        rows = result.get("results", {}).get("bindings", [])
        return self.map_to_references(rows)
    
    def search_by_doi(self, doi: str) -> Optional[Reference]:
        """
        Search for a work by DOI.
        
        Args:
            doi: DOI to search for
            
        Returns:
            Reference object if found, None otherwise
        """
        if not doi:
            return None
        
        # Clean DOI format
        clean_doi = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
        
        sparql = f"""
        SELECT ?item ?itemLabel ?title ?date ?year ?containerLabel ?doi ?isbn13 ?isbn10
               (GROUP_CONCAT(DISTINCT ?authName; separator="; ") AS ?authors)
        WHERE {{
          ?item wdt:P356 "{clean_doi}" .
          
          OPTIONAL {{ ?item wdt:P1476 ?title. }}
          OPTIONAL {{ ?item wdt:P577 ?date. BIND(YEAR(?date) AS ?year) }}
          OPTIONAL {{ ?item wdt:P1433 ?container. ?container rdfs:label ?containerLabel FILTER(LANG(?containerLabel)="en") }}
          OPTIONAL {{ ?item wdt:P212 ?isbn13. }}
          OPTIONAL {{ ?item wdt:P957 ?isbn10. }}
          OPTIONAL {{
            {{ ?item wdt:P50 ?a. ?a rdfs:label ?authName FILTER(LANG(?authName)="en") }}
            UNION
            {{ ?item wdt:P2093 ?authName }}
          }}
          
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        GROUP BY ?item ?itemLabel ?title ?date ?year ?containerLabel ?doi ?isbn13 ?isbn10
        LIMIT 1
        """
        
        try:
            result = self._execute_sparql(sparql)
            rows = result.get("results", {}).get("bindings", [])
            
            if rows:
                mapped = self.map_to_references(rows)
                return mapped.references[0] if mapped.references else None
                
        except Exception as e:
            print(f"Warning: Wikidata DOI search failed: {e}")
        
        return None
    
    def search_books_by_author(self, author: str, top_k: int = 10) -> References:
        """
        Search for books by a specific author.
        
        Args:
            author: Author name to search
            top_k: Maximum number of results
            
        Returns:
            References object containing search results
        """
        author_escaped = author.replace('"', '\\"')
        
        sparql = f"""
        SELECT DISTINCT ?item ?itemLabel ?isbn ?authorLabel ?publisherLabel ?publicationDate ?description
        WHERE {{
          {{ ?item wdt:P31 wd:Q571 }} UNION {{ ?item wdt:P31 wd:Q47461344 }} .
          ?item wdt:P50 ?author .
          ?author rdfs:label ?authorLabel .
          FILTER(CONTAINS(LCASE(?authorLabel), LCASE("{author_escaped}"))) .
         
          OPTIONAL {{ ?item wdt:P212 ?isbn }} .
          OPTIONAL {{ ?item wdt:P123 ?publisher }} .
          OPTIONAL {{ ?item wdt:P577 ?publicationDate }} .
          OPTIONAL {{ ?item schema:description ?description . FILTER(LANG(?description) = 'en') }} .
         
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }} .
        }}
        ORDER BY ?itemLabel
        LIMIT {top_k}
        """
        
        result = self._execute_sparql(sparql)
        rows = result.get("results", {}).get("bindings", [])
        return self.map_to_references(rows)
    
    # Utility methods
    
    @staticmethod
    def _extract_value(binding: Dict[str, Any], key: str) -> Optional[str]:
        """Extract value from SPARQL binding."""
        return binding.get(key, {}).get("value")
    
    @staticmethod
    def _normalize(s: str) -> str:
        """Normalize string by collapsing whitespace."""
        return re.sub(r"\s+", " ", s.strip())
    
    @staticmethod
    def _extract_year(date_str: str) -> Optional[int]:
        """Extract a 4-digit year from a date string."""
        if not date_str:
            return None
        match = re.search(r'\b(19|20)\d{2}\b', str(date_str))
        if match:
            try:
                return int(match.group(0))
            except (ValueError, TypeError):
                pass
        return None
    
    @staticmethod
    def _extract_surnames(authors: List[str]) -> set:
        """Extract surnames from author list for matching."""
        surnames = set()
        for author in authors:
            if isinstance(author, str):
                author = WikidataConnector._normalize(author)
                # Heuristic: take last token that has letters
                parts = [p for p in re.split(r"[\s\-]", author) if p]
                if parts:
                    surnames.add(parts[-1].lower())
        return surnames


# Example usage:
# connector = WikidataConnector()
# 
# # Search by title (elastic method - faster)
# reference = Reference(full_title="The Great Gatsby")
# results = connector.lookup_ref(reference, top_k=5)
# 
# # Search using SPARQL method
# results = connector.lookup_ref(reference, top_k=5, method="sparql")
# 
# # Search by ISBN
# isbn_results = connector.search_by_isbn("978-0-7432-7356-5")
# 
# # Search by author
# author_results = connector.search_books_by_author("F. Scott Fitzgerald")
# 
# # Search by DOI
# doi_ref = connector.search_by_doi("10.1234/example")
# 
# for ref in results.references:
#     print(f"Title: {ref.full_title}")
#     print(f"Authors: {ref.authors}")
#     print(f"Year: {ref.publication_date}")
#     print(f"ISBN: {ref.isbn}")
#     print(f"Wikidata ID: {ref.wikidata_id}")
#     print("---")
