"""OpenCitations Meta API connector for reference search and disambiguation."""

import logging
import re
import time
from random import uniform
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from SPARQLWrapper import SPARQLWrapper, JSON
    SPARQL_AVAILABLE = True
except ImportError:
    SPARQL_AVAILABLE = False

from .base import BaseConnector
from ..models.reference import Reference
from ..models.references import References
from ...utils.reference_matching import (
    calculate_matching_score,
    extract_family_name,
    extract_year,
    normalize_title,
)


logger = logging.getLogger(__name__)


class OpenCitationsConnector(BaseConnector):
    """Connector for OpenCitations Meta API."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, 
                 sparql_endpoint: Optional[str] = None, max_retries: int = 5,
                 default_timeout: int = 60, title_only_timeout: int = 180):
        """Initialize OpenCitations connector.
        
        Args:
            api_key: Optional OpenCitations Access Token for higher rate limits
            base_url: Base URL for OpenCitations Meta API (defaults to official API)
            sparql_endpoint: SPARQL endpoint URL (defaults to OpenCitations Meta SPARQL)
            max_retries: Maximum number of retries for SPARQL queries (default: 5)
            default_timeout: Timeout in seconds for primary queries (default: 60s = 1 minute)
            title_only_timeout: Timeout in seconds for title-only fallback queries (default: 180s = 3 minutes)
        """
        super().__init__(api_key, base_url)
        self.base_url = base_url or "https://api.opencitations.net/meta/v1"
        self.sparql_endpoint = sparql_endpoint or "https://opencitations.net/meta/sparql"
        self.max_retries = max_retries
        self.default_timeout = default_timeout
        self.title_only_timeout = title_only_timeout
        self.session = requests.Session()
        
        # Set headers for API requests
        headers = {
            "User-Agent": "citation-index/1.0 (mailto:your-email@domain.com)",
            "Accept": "application/json"
        }
        if self.api_key:
            headers["authorization"] = self.api_key
        
        self.session.headers.update(headers)
        
        # Initialize SPARQL wrapper if available
        self.sparql = None
        if SPARQL_AVAILABLE:
            self.sparql = SPARQLWrapper(self.sparql_endpoint)
            self.sparql.setReturnFormat(JSON)
            # Note: Timeout is set per query in _execute_sparql_query
    
    def search(
        self,
        reference: Reference,
        top_k: int = 10,
        include_author: bool = True,
        include_date: bool = True,
        threshold: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search for works using OpenCitations SPARQL endpoint.

        This method uses the SPARQL endpoint to search by title, author, and other metadata.
        Falls back to identifier-based lookup if SPARQL is not available.
        
        Args:
            reference: Reference object containing search criteria
            top_k: Maximum number of results to return (default: 10)
            include_author: Include author in search query (default: True)
            include_date: Include publication date in search query (default: True)
            threshold: Minimum matching score threshold (default: 50)
            
        Returns:
            List of matching results with scores
            
        Raises:
            ValueError: If reference title is missing and no DOI is provided
        """
        self._validate_reference(reference)

        if not SPARQL_AVAILABLE or self.sparql is None:
            logger.warning(
                "SPARQLWrapper not available for OpenCitations. Falling back to identifier lookup."
            )
            if getattr(reference, "doi", None):
                return self.search_by_id(reference.doi, "doi")
            return []

        results_with_scores = []

        # Try multiple query strategies
        query_strategies = self._build_query_strategies(reference, include_author, include_date)
        
        for strategy_name, query in query_strategies:
            if not query:
                continue
            
            # Set timeout based on strategy type
            # Title-only queries need more time as they're less selective
            timeout = self.title_only_timeout if strategy_name == "title_only" else self.default_timeout
            
            raw_results = self._execute_sparql_query(query, timeout=timeout)
            if raw_results:
                for result in raw_results:
                    score = self._score_result(reference, result)
                    if score >= threshold:
                        result["match_score"] = score
                        result["query_strategy"] = strategy_name
                        results_with_scores.append(result)
                
                # If we got good results from this strategy, don't try fallback strategies
                # This avoids unnecessary timeouts on slower queries
                if results_with_scores:
                    logger.info(
                        "Found %d results with strategy '%s', skipping remaining strategies",
                        len(results_with_scores),
                        strategy_name
                    )
                    break
        
        # Deduplicate by DOI and keep highest score
        unique_results = {}
        for result in results_with_scores:
            doi = result.get('doi', {}).get('value', '')
            if doi:
                if doi not in unique_results or result['match_score'] > unique_results[doi]['match_score']:
                    unique_results[doi] = result
            else:
                # If no DOI, use title as key
                title = result.get('title', {}).get('value', '')
                title_key = normalize_title(title)
                if title_key not in unique_results or result['match_score'] > unique_results[title_key]['match_score']:
                    unique_results[title_key] = result
        
        # Sort by score and return top_k
        sorted_results = sorted(unique_results.values(), 
                               key=lambda x: x['match_score'], 
                               reverse=True)
        
        return sorted_results[:top_k]
    
    def search_by_id(
        self,
        identifier: str,
        identifier_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Search for a specific work by identifier.

        Args:
            identifier: The identifier value (e.g., "10.1000/123" for DOI)
            identifier_type: Type of identifier ("doi", "issn", "isbn", "omid"). 
                           If None, will try to auto-detect from identifier format.

        Returns:
            List of raw metadata records matching the identifier
        """
        if not identifier:
            return []

        # Auto-detect identifier type if not provided
        if identifier_type is None:
            if identifier.startswith(("10.", "DOI:", "doi:")):
                identifier_type = "doi"
            elif identifier.startswith(("978", "979")) and len(identifier.replace("-", "")) in [10, 13]:
                identifier_type = "isbn"
            elif len(identifier.replace("-", "")) == 8:
                identifier_type = "issn"
            elif identifier.startswith("br/") or identifier.startswith("omid:"):
                identifier_type = "omid"
            else:
                logger.warning("Could not infer identifier type for %s", identifier)
                return []

        # Clean and format identifier
        clean_id = identifier
        if identifier_type == "doi":
            clean_id = identifier.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
            if not clean_id.startswith("doi:"):
                clean_id = f"doi:{clean_id}"
        elif identifier_type in ["issn", "isbn", "omid"]:
            if not clean_id.startswith(f"{identifier_type}:"):
                clean_id = f"{identifier_type}:{clean_id}"

        url = f"{self.base_url}/metadata/{clean_id}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 404:
                return []
            logger.error("OpenCitations %s lookup failed: %s", identifier_type, exc)
            return []
        except requests.RequestException as exc:
            logger.error("OpenCitations %s lookup failed: %s", identifier_type, exc)
            return []

        data = response.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []
    
    def _build_query_strategies(self, reference: Reference, include_author: bool, 
                                include_date: bool) -> List[Tuple[str, Optional[str]]]:
        """Build simplified SPARQL query strategies.
        
        Strategy:
        1. If author and year available: Title + Author + Year
        2. Otherwise: Title only
        
        Args:
            reference: Reference object with search metadata
            include_author: Whether to include author in queries
            include_date: Whether to include date in queries
            
        Returns:
            List of tuples (strategy_name, sparql_query)
        """
        strategies = []
        
        # Extract metadata
        title = reference.full_title
        title = normalize_title(title)
        if not title:
            return strategies
        
        year = getattr(reference, 'publication_date', None) or getattr(reference, 'year', None)
        year_int = None
        if include_date and year:
            year_int = extract_year(str(year))
        
        # Get first author last name
        first_author_lastname = None
        if include_author and reference.authors and len(reference.authors) > 0:
            # Extract family name from first author using utility function
            author = reference.authors[0]
            if isinstance(author, str):
                first_author_lastname = extract_family_name(author)
        
        # Strategy 1: Title + Author + Year (if author and year available)
        if first_author_lastname and year_int:
            query = self._build_title_author_year_query(title, first_author_lastname, year_int)
            if query:
                strategies.append(("title_author_year", query))
        
        # Strategy 2: Title only (fallback)
        query = self._build_title_only_query(title)
        if query:
            strategies.append(("title_only", query))
        
        return strategies
    
    def _build_title_author_year_query(self, title: str, author_lastname: str, year: int) -> Optional[str]:
        """Build SPARQL query for Title + Author + Year matching.
        
        This is the primary search strategy when author and year are available.
        Uses REGEX for fuzzy title matching and filters by author first for performance.
        DOI requirement removed to avoid filtering out valid results.
        """
        # Build REGEX pattern from title - extract key words
        # Take first 50 chars and create flexible pattern
        title_substr = title[:50] if len(title) > 50 else title
        # Create regex pattern: extract significant words and join with .*
        words = [w.strip().lower() for w in title_substr.split() if len(w.strip()) > 3]
        regex_pattern = ".*".join(words[:4]) if len(words) > 1 else title_substr.lower()
        # Escape special regex characters
        regex_pattern = regex_pattern.replace('"', '\\"').replace('(', '\\(').replace(')', '\\)')
        
        return f"""
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX pro: <http://purl.org/spar/pro/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>
        SELECT DISTINCT ?br ?title ?pub_date ?first_author {{
            ?br dcterms:title ?title;
                prism:publicationDate ?publicationDate;
                pro:isDocumentContextFor ?role.
            
            ?role pro:isHeldBy ?first_author.
            ?first_author foaf:familyName "{author_lastname}".
            
            BIND(STR(?publicationDate) AS ?pub_date)
            
            # Optimize: Year filter first (fast), then REGEX (slow)
            # Year filtering eliminates most candidates before expensive pattern matching
            FILTER((STRSTARTS(?pub_date, "{year}") || 
                    STRSTARTS(?pub_date, "{year-1}") || 
                    STRSTARTS(?pub_date, "{year+1}")) &&
                   REGEX(?title, "{regex_pattern}", "i"))
        }}
        LIMIT 20
        """
    
    def _build_title_only_query(self, title: str) -> Optional[str]:
        """Build SPARQL query for Title-only matching (fallback).
        
        Note: This query may timeout for common terms without author filtering.
        Uses REGEX for fuzzy matching. DOI requirement removed.
        """
        # Build REGEX pattern from title - extract key words
        title_substr = title[:50] if len(title) > 50 else title
        # Create regex pattern: extract significant words and join with .*
        words = [w.strip().lower() for w in title_substr.split() if len(w.strip()) > 3]
        regex_pattern = ".*".join(words[:4]) if len(words) > 1 else title_substr.lower()
        # Escape special regex characters
        regex_pattern = regex_pattern.replace('"', '\\"').replace('(', '\\(').replace(')', '\\)')
        
        return f"""
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>
        SELECT DISTINCT ?br ?title ?pub_date {{
            ?br dcterms:title ?title;
                prism:publicationDate ?publicationDate.
            
            BIND(STR(?publicationDate) AS ?pub_date)
            FILTER(REGEX(?title, "{regex_pattern}", "i"))
        }}
        LIMIT 20
        """
    
    def _execute_sparql_query(self, query: str, timeout: int = 60) -> Optional[List[Dict]]:
        """Execute a SPARQL query with retry mechanism for 503 errors.
        
        Args:
            query: SPARQL query string
            timeout: Timeout in seconds (default: 60)
            
        Returns:
            List of result bindings, or None if query fails
        """
        if not self.sparql:
            return None
        
        # Set timeout for this query
        self.sparql.setTimeout(timeout)
        self.sparql.setQuery(query)
        
        for attempt in range(self.max_retries):
            try:
                results = self.sparql.query().convert()
                return results['results']['bindings']
            except Exception as exc:
                message = str(exc)

                if "503" in message:
                    wait_time = 2 ** attempt + uniform(0, 1)
                    logger.warning(
                        "OpenCitations SPARQL 503 (retry %s/%s). Waiting %.2f seconds...",
                        attempt + 1,
                        self.max_retries,
                        wait_time,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("OpenCitations SPARQL query error: %s", exc)
                    break

        logger.error("Maximum retry attempts reached for OpenCitations SPARQL query")
        return None
    
    def _score_result(self, reference: Reference, result: Dict) -> int:
        """Score a SPARQL result against the reference.
        
        Args:
            reference: Reference object with search criteria
            result: SPARQL result binding
            
        Returns:
            Integer score (0-100)
        """
        # Convert Reference to dict format
        ref_data = {
            'title': reference.full_title,
            'year': getattr(reference, 'publication_date', None) or getattr(reference, 'year', None),
            'volume': getattr(reference, 'volume', None),
            'first_page': getattr(reference, 'pages', '').split('-')[0].strip() if getattr(reference, 'pages', None) else None
        }
        
        # Convert SPARQL result to dict format
        result_data = {
            'title': result.get('title', {}).get('value', ''),
            'pub_date': result.get('pub_date', {}).get('value', ''),
            'volume': result.get('volume_num', {}).get('value', ''),
            'start_page': result.get('start_page', {}).get('value', '')
        }
        
        return calculate_matching_score(ref_data, result_data)
    
    def map_to_references(self, raw_results: List[Dict[str, Any]]) -> References:
        """Transform OpenCitations Meta API results to Reference objects."""

        references = References()

        for result in raw_results:
            try:
                if self._is_sparql_binding(result):
                    ref = self._map_sparql_binding(result)
                else:
                    ref = self._map_metadata_record(result)
            except Exception as exc:
                logger.warning("Could not map OpenCitations result: %s", exc)
                continue

            if ref:
                references.append(ref)

        return references

    def _result_to_reference(self, result: Dict[str, Any]) -> Reference:
        """Convert OpenCitations API result to Reference object."""
        try:
            if self._is_sparql_binding(result):
                ref = self._map_sparql_binding(result)
            else:
                ref = self._map_metadata_record(result)
            
            if ref is None:
                return Reference(full_title="")
            return ref
        except Exception:
            return Reference(full_title="")

    @staticmethod
    def _is_sparql_binding(result: Dict[str, Any]) -> bool:
        value = result.get("title")
        return isinstance(value, dict) and "value" in value

    @staticmethod
    def _binding_value(binding: Dict[str, Any], key: str) -> Optional[str]:
        value = binding.get(key)
        if isinstance(value, dict):
            return value.get("value")
        return value

    def _map_sparql_binding(self, binding: Dict[str, Any]) -> Optional[Reference]:
        title = self._binding_value(binding, "title") or self._binding_value(binding, "br")
        if not title:
            return None

        pub_date = self._binding_value(binding, "pub_date")
        doi = self._binding_value(binding, "doi")

        authors = self._split_semi_colon(self._binding_value(binding, "author"))
        editors = self._split_semi_colon(self._binding_value(binding, "editor"))

        venue = self._binding_value(binding, "venue")
        journal_title = self._strip_bracketed(venue) if venue else None

        publisher_text = self._binding_value(binding, "publisher")
        publisher = self._strip_bracketed(publisher_text) if publisher_text else None

        volume = self._binding_value(binding, "volume") or self._binding_value(binding, "volume_num")
        issue = self._binding_value(binding, "issue")
        pages = self._binding_value(binding, "page") or self._binding_value(binding, "start_page")

        return Reference(
            full_title=title,
            authors=authors or None,
            editors=editors or None,
            journal_title=journal_title,
            publisher=publisher,
            publication_date=pub_date,
            volume=volume,
            issue=issue,
            pages=pages,
            doi=doi,
        )

    def _map_metadata_record(self, record: Dict[str, Any]) -> Optional[Reference]:
        title = record.get("title") or record.get("title_str")
        if not title:
            return None

        pub_date = record.get("pub_date") or record.get("year") or record.get("publication_date")

        authors = self._normalise_people(record.get("author"))
        editors = self._normalise_people(record.get("editor"))

        venue = record.get("venue") or record.get("journal")
        if isinstance(venue, str):
            journal_title = self._strip_bracketed(venue)
        else:
            journal_title = venue

        publisher = record.get("publisher")
        if isinstance(publisher, str):
            publisher = self._strip_bracketed(publisher)

        volume = record.get("volume")
        issue = record.get("issue")
        pages = record.get("page") or record.get("pages")

        doi = record.get("doi")

        return Reference(
            full_title=title,
            authors=authors or None,
            editors=editors or None,
            journal_title=journal_title,
            publisher=publisher,
            publication_date=str(pub_date) if pub_date else None,
            volume=volume,
            issue=issue,
            pages=pages,
            doi=doi,
        )

    @staticmethod
    def _split_semi_colon(value: Optional[str]) -> List[str]:
        if not value:
            return []
        names = []
        for chunk in value.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            names.append(OpenCitationsConnector._strip_bracketed(chunk))
        return names

    @staticmethod
    def _strip_bracketed(value: str) -> str:
        match = re.match(r"^([^[]+)", value)
        return match.group(1).strip() if match else value.strip()

    @staticmethod
    def _normalise_people(raw: Any) -> List[str]:
        people: List[str] = []
        if isinstance(raw, str):
            return OpenCitationsConnector._split_semi_colon(raw)

        if isinstance(raw, list):
            for entry in raw:
                name: Optional[str] = None
                if isinstance(entry, dict):
                    name = (
                        entry.get("name")
                        or " ".join(
                            part
                            for part in [entry.get("given"), entry.get("family")]
                            if part
                        ).strip()
                    )
                elif isinstance(entry, str):
                    name = entry

                if name:
                    cleaned = OpenCitationsConnector._strip_bracketed(name)
                    if cleaned:
                        people.append(cleaned)

        return people


# Example usage:
# 
# # Initialize connector (optional API key for higher rate limits)
# connector = OpenCitationsConnector(api_key="your-access-token")
# 
# # Identifier lookup examples
# doi_results = connector.search_by_id("10.1007/978-1-4020-9632-7", "doi")
# isbn_results = connector.search_by_id("9781402096327", "isbn")
# mapped = connector.map_to_references(doi_results)
# 
# # SPARQL-based search by title/author/metadata (requires SPARQLWrapper)
# from citation_index.core.models import Reference
# 
# # Search by title only
# ref1 = Reference(full_title="Attention Is All You Need")
# results1 = connector.search(ref1, top_k=5, threshold=50)
# 
# # Search by title and author
# ref2 = Reference(
#     full_title="BERT: Pre-training of Deep Bidirectional Transformers",
#     authors=["Devlin", "Chang", "Lee", "Toutanova"]
# )
# results2 = connector.search(ref2, top_k=5, include_author=True, threshold=60)
# 
# # Search with year and other metadata
# ref3 = Reference(
#     full_title="Deep Learning",
#     publication_date="2015",
#     volume="521",
#     pages="436-444"
# )
# results3 = connector.search(ref3, top_k=5, include_date=True, threshold=70)
