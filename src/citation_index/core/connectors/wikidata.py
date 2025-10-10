"""Wikidata SPARQL connector for reference search and disambiguation."""

import logging
import re
import time
from typing import Any, Dict, List, Optional

import requests

from .base import BaseConnector
from ..models.reference import Reference
from ..models.references import References


logger = logging.getLogger(__name__)


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

    def search_by_id(
        self,
        identifier: str,
        identifier_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        if not identifier:
            return []

        inferred = identifier_type or self._guess_identifier_type(identifier)
        if not inferred:
            logger.warning("Could not determine identifier type for %s", identifier)
            return []

        limit = kwargs.get("top_k") or kwargs.get("limit") or 10
        query = self._build_identifier_query(inferred.lower(), identifier, limit)
        if not query:
            logger.warning("Unsupported identifier type for Wikidata: %s", inferred)
            return []

        result = self._execute_sparql(query)
        if not result:
            return []
        return result.get("results", {}).get("bindings", [])
    
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
            
        except requests.RequestException as exc:
            logger.warning("Wikidata elastic search failed: %s", exc)
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
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX bd: <http://www.bigdata.com/rdf#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX schema: <http://schema.org/>

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
            
        except requests.RequestException as exc:
            logger.warning("Wikidata SPARQL search failed: %s", exc)
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
        except requests.RequestException as exc:
            logger.error("Error executing SPARQL query: %s", exc)
            return {"results": {"bindings": []}}

    def _build_identifier_query(self, id_type: str, identifier: str, limit: int) -> Optional[str]:
        try:
            limit_value = max(1, min(int(limit), 50))
        except (TypeError, ValueError):
            limit_value = 10

        filter_clause = self._identifier_filter_clause(id_type, identifier)
        if not filter_clause:
            return None

        return f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX schema: <http://schema.org/>

        SELECT ?item ?itemLabel ?title ?date ?year ?containerLabel ?doi ?isbn13 ?isbn10 ?pmid ?pmcid ?arxiv ?issn
               (GROUP_CONCAT(DISTINCT ?authName; separator="; ") AS ?authors)
        WHERE {{
          {filter_clause}
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
          OPTIONAL {{
            {{ ?item wdt:P50 ?auth. ?auth rdfs:label ?authName FILTER(LANG(?authName) = "en") }}
            UNION
            {{ ?item wdt:P2093 ?authName }}
          }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        GROUP BY ?item ?itemLabel ?title ?date ?year ?containerLabel ?doi ?isbn13 ?isbn10 ?pmid ?pmcid ?arxiv ?issn
        LIMIT {limit_value}
        """

    def _identifier_filter_clause(self, id_type: str, identifier: str) -> Optional[str]:
        literal = identifier.strip()

        if id_type == "doi":
            clean = self._clean_doi(literal)
            if not clean:
                return None
            escaped = self._escape_literal(clean)
            return f'?item wdt:P356 "{escaped}".'

        if id_type == "isbn":
            clean = self._clean_isbn(literal)
            if not clean:
                return None
            escaped = self._escape_literal(clean)
            return f"""
          {{
            ?item wdt:P212 ?isbn13Value.
            BIND(REPLACE(STR(?isbn13Value), "-", "") AS ?isbn13Clean)
            FILTER(?isbn13Clean = "{escaped}")
          }}
          UNION
          {{
            ?item wdt:P957 ?isbn10Value.
            BIND(REPLACE(STR(?isbn10Value), "-", "") AS ?isbn10Clean)
            FILTER(?isbn10Clean = "{escaped}")
          }}
        """

        if id_type == "issn":
            clean = self._clean_issn(literal)
            if not clean:
                return None
            escaped = self._escape_literal(clean)
            return f"""
          ?item wdt:P236 ?issnValue.
          BIND(REPLACE(STR(?issnValue), "-", "") AS ?issnClean)
          FILTER(?issnClean = "{escaped}")
        """

        property_map = {
            "pmid": "P698",
            "pmcid": "P932",
            "arxiv": "P818",
        }
        if id_type in property_map:
            escaped = self._escape_literal(literal)
            return f'?item wdt:{property_map[id_type]} "{escaped}".'

        if id_type in {"wikidata", "qid"}:
            qid = literal.upper()
            if not qid.startswith("Q"):
                return None
            return f"VALUES ?item {{ wd:{qid} }}"

        return None

    @staticmethod
    def _guess_identifier_type(identifier: str) -> Optional[str]:
        token = identifier.strip()
        lower = token.lower()
        digits = re.sub(r"[^0-9xX]", "", token)

        if lower.startswith("10.") or lower.startswith("doi:") or "doi.org" in lower:
            return "doi"
        if token.upper().startswith("Q") and token[1:].isdigit():
            return "wikidata"
        if re.match(r"^\d{4}\.[0-9]{4,5}$", lower) or "arxiv" in lower:
            return "arxiv"
        if lower.startswith("pmcid") or lower.startswith("pmc"):
            return "pmcid"
        if lower.startswith("pmid"):
            return "pmid"
        if digits.isdigit() and len(digits) in {13, 10}:
            return "isbn"
        if digits and len(digits) == 8:
            return "issn"
        if digits.isdigit() and 5 <= len(digits) <= 8:
            return "pmid"
        return None

    @staticmethod
    def _clean_doi(doi: str) -> str:
        cleaned = doi.strip()
        prefixes = [
            "https://doi.org/",
            "http://doi.org/",
            "http://dx.doi.org/",
            "doi:",
        ]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix) :]
                break
        return cleaned

    @staticmethod
    def _clean_isbn(isbn: str) -> str:
        return re.sub(r"[^0-9Xx]", "", isbn)

    @staticmethod
    def _clean_issn(issn: str) -> str:
        digits = re.sub(r"[^0-9Xx]", "", issn)
        return digits

    @staticmethod
    def _escape_literal(value: str) -> str:
        return value.replace('"', '\\"')
    
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
                
            except Exception as exc:
                logger.warning("Could not map Wikidata result: %s", exc)
                continue
        
        return references
    
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
# # Search by title
# query = Reference(full_title="The Great Gatsby")
# raw = connector.search(query, top_k=5)
# mapped = connector.map_to_references(raw)
# 
# # Identifier lookup (e.g. DOI)
# doi_rows = connector.search_by_id("10.1234/example", "doi")
# doi_refs = connector.map_to_references(doi_rows)
