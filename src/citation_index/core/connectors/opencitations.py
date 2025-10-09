"""
OpenCitations Meta API connector for reference search and disambiguation.

Supports both REST API (identifier-based lookups) and SPARQL endpoint 
(title/author-based search).
"""

import requests
import time
from random import uniform
from typing import Dict, List, Any, Optional, Tuple
import re

try:
    from SPARQLWrapper import SPARQLWrapper, JSON
    SPARQL_AVAILABLE = True
except ImportError:
    SPARQL_AVAILABLE = False

from .base import BaseConnector
from ..models.reference import Reference
from ..models.references import References
from ...utils.reference_matching import (
    normalize_title,
    extract_year,
    calculate_title_similarity,
    calculate_matching_score
)


class OpenCitationsConnector(BaseConnector):
    """Connector for OpenCitations Meta API."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, 
                 sparql_endpoint: Optional[str] = None, max_retries: int = 5):
        """Initialize OpenCitations connector.
        
        Args:
            api_key: Optional OpenCitations Access Token for higher rate limits
            base_url: Base URL for OpenCitations Meta API (defaults to official API)
            sparql_endpoint: SPARQL endpoint URL (defaults to OpenCitations Meta SPARQL)
            max_retries: Maximum number of retries for SPARQL queries (default: 5)
        """
        super().__init__(api_key, base_url)
        self.base_url = base_url or "https://api.opencitations.net/meta/v1"
        self.sparql_endpoint = sparql_endpoint or "https://opencitations.net/meta/sparql"
        self.max_retries = max_retries
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
            self.sparql.setTimeout(30)
    
    def search(self, reference: Reference, top_k: int = 10, include_author: bool = True, 
               include_date: bool = True, threshold: int = 50) -> List[Dict[str, Any]]:
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
        # If SPARQL is not available, try identifier-based lookup
        if not SPARQL_AVAILABLE or self.sparql is None:
            print("Warning: SPARQLWrapper not available. Trying identifier-based lookup...")
            if hasattr(reference, 'doi') and reference.doi:
                result = self.search_by_doi(reference.doi)
                if result:
                    return [self._reference_to_dict(result)]
            return []
        
        # Validate reference has searchable fields
        if not reference.full_title and not (hasattr(reference, 'doi') and reference.doi):
            raise ValueError("Reference title or DOI is required for search")
        
        results_with_scores = []
        
        # Try multiple query strategies
        query_strategies = self._build_query_strategies(reference, include_author, include_date)
        
        for strategy_name, query in query_strategies:
            if not query:
                continue
                
            raw_results = self._execute_sparql_query(query)
            if raw_results:
                # Score and filter results
                for result in raw_results:
                    score = self._score_result(reference, result)
                    if score >= threshold:
                        result['match_score'] = score
                        result['query_strategy'] = strategy_name
                        results_with_scores.append(result)
        
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
    
    def search_by_id(self, identifier: str, identifier_type: str = None) -> Optional[Reference]:
        """Search for a specific work by identifier.
        
        Args:
            identifier: The identifier value (e.g., "10.1000/123" for DOI)
            identifier_type: Type of identifier ("doi", "issn", "isbn", "omid"). 
                           If None, will try to auto-detect from identifier format.
            
        Returns:
            Reference object if found, None otherwise
        """
        if not identifier:
            return None
        
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
                print(f"Warning: Could not auto-detect identifier type for '{identifier}'")
                return None
        
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
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if data and len(data) > 0:
                mapped = self.map_to_references(data)
                return mapped.references[0] if mapped.references else None
                
        except Exception as e:
            print(f"Warning: {identifier_type.upper()} search failed: {e}")
            
        return None
    
    def search_by_doi(self, doi: str) -> Optional[Reference]:
        """Search for a specific work by DOI.
        
        Args:
            doi: DOI of the work to search for
            
        Returns:
            Reference object if found, None otherwise
        """
        return self.search_by_id(doi, "doi")
    
    def _build_query_strategies(self, reference: Reference, include_author: bool, 
                                include_date: bool) -> List[Tuple[str, Optional[str]]]:
        """Build multiple SPARQL query strategies based on available metadata.
        
        Args:
            reference: Reference object with search metadata
            include_author: Whether to include author in queries
            include_date: Whether to include date in queries
            
        Returns:
            List of tuples (strategy_name, sparql_query)
        """
        strategies = []
        
        # Extract metadata
        doi = getattr(reference, 'doi', None)
        title = reference.full_title
        year = getattr(reference, 'publication_date', None) or getattr(reference, 'year', None)
        volume = getattr(reference, 'volume', None)
        pages = getattr(reference, 'pages', None)
        first_page = pages.split('-')[0].strip() if pages else getattr(reference, 'first_page', None)
        
        # Get first author last name
        first_author_lastname = None
        if include_author and reference.authors and len(reference.authors) > 0:
            # Try to extract last name from first author
            author = reference.authors[0]
            if isinstance(author, str):
                parts = author.split()
                first_author_lastname = parts[-1] if parts else None
        
        # Extract year as integer
        year_int = None
        if include_date and year:
            year_int = extract_year(str(year))
        
        # Strategy 1: DOI + Year (if available)
        if doi and year_int:
            query = self._build_doi_year_query(doi, year_int)
            if query:
                strategies.append(("doi_year", query))
        
        # Strategy 2: Author + Title (if available)
        if first_author_lastname and title:
            query = self._build_author_title_query(first_author_lastname, title)
            if query:
                strategies.append(("author_title", query))
        
        # Strategy 3: Year + Volume + Page (if available)
        if year_int and volume and first_page:
            query = self._build_year_volume_page_query(year_int, volume, first_page)
            if query:
                strategies.append(("year_volume_page", query))
        
        # Strategy 4: Year + Author + Page (if available)
        if year_int and first_author_lastname and first_page:
            query = self._build_year_author_page_query(year_int, first_author_lastname, first_page)
            if query:
                strategies.append(("year_author_page", query))
        
        # Strategy 5: Title only (fallback)
        if title:
            query = self._build_title_only_query(title)
            if query:
                strategies.append(("title_only", query))
        
        return strategies
    
    def _build_doi_year_query(self, doi: str, year: int) -> Optional[str]:
        """Build SPARQL query for DOI + Year matching."""
        return f"""
        PREFIX datacite: <http://purl.org/spar/datacite/>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>
        PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>
        SELECT DISTINCT ?br ?title ?pub_date ?doi {{
            ?identifier literal:hasLiteralValue "{doi}".
            ?br datacite:hasIdentifier ?identifier;
            dcterms:title ?title;
            prism:publicationDate ?publicationDate;
            datacite:hasIdentifier ?doi_id.
            
            ?doi_id datacite:usesIdentifierScheme datacite:doi;
                    literal:hasLiteralValue ?doi.

            BIND(STR(?publicationDate) AS ?pub_date)
            FILTER(STRSTARTS(?pub_date, "{year}") || 
                STRSTARTS(?pub_date, "{year-1}") || 
                STRSTARTS(?pub_date, "{year+1}"))
        }}
        LIMIT 20
        """
    
    def _build_author_title_query(self, author_lastname: str, title: str) -> Optional[str]:
        """Build SPARQL query for Author + Title matching."""
        # Use a substring of the title for broader matching
        title_substr = title[:50] if len(title) > 50 else title
        return f"""
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX pro: <http://purl.org/spar/pro/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX datacite: <http://purl.org/spar/datacite/>
        PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>
        PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>
        SELECT DISTINCT ?br ?title ?pub_date ?first_author ?doi {{
            ?first_author foaf:familyName "{author_lastname}".
            ?role pro:isHeldBy ?first_author.
            ?br pro:isDocumentContextFor ?role;
            dcterms:title ?title;
            prism:publicationDate ?publicationDate;
            datacite:hasIdentifier ?doi_id.

            ?doi_id datacite:usesIdentifierScheme datacite:doi;
            literal:hasLiteralValue ?doi.

            BIND(STR(?publicationDate) AS ?pub_date)
            FILTER(CONTAINS(LCASE(?title), LCASE("{title_substr}")))
        }}
        LIMIT 20
        """
    
    def _build_year_volume_page_query(self, year: int, volume: str, page: str) -> Optional[str]:
        """Build SPARQL query for Year + Volume + Page matching."""
        return f"""
        PREFIX datacite: <http://purl.org/spar/datacite/>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>
        PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>
        PREFIX frbr: <http://purl.org/vocab/frbr/core#>
        PREFIX fabio: <http://purl.org/spar/fabio/>
        SELECT DISTINCT ?br ?title ?pub_date ?volume_num ?start_page ?doi {{
            ?br dcterms:title ?title;
                prism:publicationDate ?publicationDate;
                frbr:embodiment ?embodiment;
                frbr:partOf ?issue;
                datacite:hasIdentifier ?doi_id.

            ?doi_id datacite:usesIdentifierScheme datacite:doi;
                    literal:hasLiteralValue ?doi.

            BIND(STR(?publicationDate) AS ?pub_date)
            FILTER(STRSTARTS(?pub_date, "{year}") || 
                STRSTARTS(?pub_date, "{year-1}") || 
                STRSTARTS(?pub_date, "{year+1}"))
            ?embodiment prism:startingPage "{page}".
            ?issue frbr:partOf ?volume.
            ?volume fabio:hasSequenceIdentifier "{volume}".
            BIND(STR(?volume) AS ?volume_num)
        }}
        LIMIT 20
        """
    
    def _build_year_author_page_query(self, year: int, author_lastname: str, page: str) -> Optional[str]:
        """Build SPARQL query for Year + Author + Page matching."""
        return f"""
        PREFIX datacite: <http://purl.org/spar/datacite/>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>
        PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>
        PREFIX frbr: <http://purl.org/vocab/frbr/core#>
        PREFIX pro: <http://purl.org/spar/pro/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT DISTINCT ?br ?title ?pub_date ?first_author ?start_page ?doi {{
            ?first_author foaf:familyName "{author_lastname}".
            ?role pro:isHeldBy ?first_author.
            ?br pro:isDocumentContextFor ?role;
                dcterms:title ?title;
                prism:publicationDate ?publicationDate;
                frbr:embodiment ?embodiment;
                datacite:hasIdentifier ?doi_id.

            ?doi_id datacite:usesIdentifierScheme datacite:doi;
            literal:hasLiteralValue ?doi.

            BIND(STR(?publicationDate) AS ?pub_date)
            FILTER(STRSTARTS(?pub_date, "{year}") || 
                STRSTARTS(?pub_date, "{year-1}") || 
                STRSTARTS(?pub_date, "{year+1}"))
            ?embodiment prism:startingPage "{page}".
        }}
        LIMIT 20
        """
    
    def _build_title_only_query(self, title: str) -> Optional[str]:
        """Build SPARQL query for Title-only matching (fallback)."""
        # Use a substring for broader matching
        title_substr = title[:50] if len(title) > 50 else title
        return f"""
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>
        PREFIX datacite: <http://purl.org/spar/datacite/>
        PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>
        SELECT DISTINCT ?br ?title ?pub_date ?doi {{
            ?br dcterms:title ?title;
                prism:publicationDate ?publicationDate;
                datacite:hasIdentifier ?doi_id.
            
            ?doi_id datacite:usesIdentifierScheme datacite:doi;
                    literal:hasLiteralValue ?doi.
            
            BIND(STR(?publicationDate) AS ?pub_date)
            FILTER(CONTAINS(LCASE(?title), LCASE("{title_substr}")))
        }}
        LIMIT 20
        """
    
    def _execute_sparql_query(self, query: str) -> Optional[List[Dict]]:
        """Execute a SPARQL query with retry mechanism for 503 errors.
        
        Args:
            query: SPARQL query string
            
        Returns:
            List of result bindings, or None if query fails
        """
        if not self.sparql:
            return None
        
        self.sparql.setQuery(query)
        
        for attempt in range(self.max_retries):
            try:
                results = self.sparql.query().convert()
                return results['results']['bindings']
            except Exception as e:
                error_message = str(e)
                
                if "503" in error_message:
                    wait_time = 2 ** attempt + uniform(0, 1)
                    print(f"503 Error: Retry {attempt + 1}/{self.max_retries}. "
                          f"Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"SPARQL query error: {e}")
                    break
        
        print("Maximum retry attempts reached. Query failed.")
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
    
    def _reference_to_dict(self, reference: Reference) -> Dict[str, Any]:
        """Convert a Reference object to dictionary format.
        
        Args:
            reference: Reference object
            
        Returns:
            Dictionary representation
        """
        return {
            'title': {'value': reference.full_title or ''},
            'pub_date': {'value': getattr(reference, 'publication_date', '') or ''},
            'doi': {'value': getattr(reference, 'doi', '') or ''},
            'match_score': 100  # Perfect match for direct lookups
        }

    
    def map_to_references(self, raw_results: List[Dict[str, Any]]) -> References:
        """Transform OpenCitations Meta API results to Reference objects.
        
        Args:
            raw_results: List of raw OpenCitations API response data
            
        Returns:
            References object containing mapped Reference instances
        """
        references = []
        
        for result in raw_results:
            try:
                # Extract basic information
                title = result.get("title")
                pub_date = result.get("pub_date")
                
                # Parse authors from semicolon-separated string
                authors = []
                author_str = result.get("author", "")
                if author_str:
                    # Split by semicolon and clean up
                    author_parts = [a.strip() for a in author_str.split(";") if a.strip()]
                    for author_part in author_parts:
                        # Extract just the name part (before any bracketed identifiers)
                        name_match = re.match(r"^([^[]+)", author_part)
                        if name_match:
                            authors.append(name_match.group(1).strip())
                
                # Parse editors similarly
                editors = []
                editor_str = result.get("editor", "")
                if editor_str:
                    editor_parts = [e.strip() for e in editor_str.split(";") if e.strip()]
                    for editor_part in editor_parts:
                        name_match = re.match(r"^([^[]+)", editor_part)
                        if name_match:
                            editors.append(name_match.group(1).strip())
                
                # Extract venue information
                venue = result.get("venue", "")
                journal_title = None
                if venue:
                    # Extract venue name (before any bracketed identifiers)
                    venue_match = re.match(r"^([^[]+)", venue)
                    if venue_match:
                        journal_title = venue_match.group(1).strip()
                
                # Extract publisher
                publisher_str = result.get("publisher", "")
                publisher = None
                if publisher_str:
                    publisher_match = re.match(r"^([^[]+)", publisher_str)
                    if publisher_match:
                        publisher = publisher_match.group(1).strip()
                
                # Extract other fields
                volume = result.get("volume")
                issue = result.get("issue")
                pages = result.get("page")
                
                # Create Reference object with proper field mapping
                reference = Reference(
                    full_title=title,
                    authors=authors if authors else None,
                    editors=editors if editors else None,
                    journal_title=journal_title,
                    publisher=publisher,
                    publication_date=pub_date,
                    volume=volume,
                    issue=issue,
                    pages=pages
                )
                
                references.append(reference)
                
            except Exception as e:
                # Log warning but continue processing other results
                print(f"Warning: Could not map OpenCitations result to Reference: {e}")
                continue
        
        # Create References object using append method to avoid constructor validation issues
        result = References()
        for ref in references:
            result.append(ref)
        return result


# Example usage:
# 
# # Initialize connector (optional API key for higher rate limits)
# connector = OpenCitationsConnector(api_key="your-access-token")
# 
# # Method 1: Search by DOI or other identifiers
# doi_ref = connector.search_by_doi("10.1007/978-1-4020-9632-7")
# isbn_ref = connector.search_by_id("9781402096327", "isbn")
# issn_ref = connector.search_by_id("0138-9130", "issn")
# omid_ref = connector.search_by_id("br/0612058700", "omid")
# auto_ref = connector.search_by_id("10.1007/978-1-4020-9632-7")  # Auto-detects as DOI
# 
# # Method 2: SPARQL-based search by title/author/metadata (requires SPARQLWrapper)
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
# 
# # Method 3: Use convenience method (combines search + mapping)
# ref4 = Reference(full_title="Machine Learning")
# mapped_results = connector.lookup_ref(ref4, top_k=3)
# for result_ref in mapped_results.references:
#     print(f"Title: {result_ref.full_title}")
#     print(f"DOI: {result_ref.doi if hasattr(result_ref, 'doi') else 'N/A'}")
# 
# To test all connectors: python tests/test_connectors.py
# To test reference matching utilities: pytest tests/test_reference_matching.py