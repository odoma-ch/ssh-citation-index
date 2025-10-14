"""
Matilda API connector for citation index.

This connector interfaces with the Matilda Science API to search for academic works
using title-based queries.
"""

import base64
import logging
from typing import Any, Dict, List, Optional

import requests

from .base import BaseConnector
from ..models import Reference, References

logger = logging.getLogger(__name__)


class MatildaConnector(BaseConnector):
    """Connector for Matilda Science API."""
    
    def __init__(self, username: str = "matilda_graphia", password: str = "WBXN2948qndi"):
        super().__init__()
        self.base_url = "https://matilda.science/api"
        self.username = username
        self.password = password
        # TODO: Securely handle credentials in production

        # Set up authentication headers
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
        
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Basic {encoded_credentials}',
            'Accept': 'application/json',
            'User-Agent': 'CitationIndex/1.0'
        })
    
    def search(self, reference: Reference, top_k: int = 10, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Search for works using the Matilda API with comprehensive query support.

        Supports the following search capabilities:
        - Title search (query.title)
        - Author search (query.author) 
        - General free-form search (query)
        - Publication date filtering (filter.fromPublishedDate, filter.untilPublishedDate)
        - Publisher filtering (query.publisher)
        - Type filtering (filter.type)
        - Sorting by relevance, creation date, update date, citations
        Note: 'published' sort may cause errors due to Elasticsearch fielddata limitations
        
        Args:
            reference: Reference object containing search criteria
            top_k: Maximum number of results to return
            **kwargs: Additional search parameters:
                - general_query: Free-form search query
                - author_query: Specific author search
                - publisher_query: Publisher search
                - from_date: Filter works published since date (YYYY-MM-DD format)
                - until_date: Filter works published before date (YYYY-MM-DD format)
                - work_type: Filter by work type (e.g., 'journal-article')
                - sort_by: Sort field ('score', 'created', 'updated', 'published', 'citedBy')
                - sort_order: Sort order ('asc' or 'desc')
                - offset: Pagination offset
            
        Returns:
            List of raw API response data
        """
        self._validate_reference(reference)

        params = self._build_search_params(reference, top_k, kwargs)

        try:
            response = self.session.get(
                f"{self.base_url}/works/query",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.error("Matilda API request failed: %s", exc)
            return []

        data = response.json()
        works = data.get("works", [])

        if not isinstance(works, list):
            logger.warning("Unexpected Matilda response shape: 'works' missing")
            return []

        logger.debug("Matilda returned %d works", len(works))
        return works[: top_k or len(works)]

    def map_to_references(self, raw_results: List[Dict[str, Any]]) -> References:
        """
        Map Matilda API response to References object.
        
        Args:
            raw_results: List of raw API responses from Matilda
            
        Returns:
            References object with mapped data
        """
        references = References()

        for work in self._iter_works(raw_results):
            try:
                ref = self._convert_work_to_reference(work)
            except Exception as exc:
                logger.warning("Failed to map Matilda work: %s", exc)
                continue

            if ref:
                references.append(ref)

        return references
    
    def _convert_work_to_reference(self, work: Dict[str, Any]) -> Optional[Reference]:
        """
        Convert a single Matilda work to a Reference object.
        
        Matilda API returns works that contain multiple texts/sources.
        We'll extract the most relevant text from each work.
        
        Args:
            work: Single work from Matilda API response
            
        Returns:
            Reference object or None if conversion fails
        """
        try:
            # Matilda works contain multiple "texts" - pick the first one with good data
            texts = work.get('texts', [])
            if not texts:
                return None
            
            # Find the best text entry (prefer ones with more complete data)
            best_text = None
            for text in texts:
                if isinstance(text, dict) and text.get('title'):
                    best_text = text
                    break
            
            if not best_text:
                best_text = texts[0]  # Fallback to first text
            
            # Extract title from the best text
            title_data = best_text.get('title', [])
            if isinstance(title_data, list) and title_data:
                title = title_data[0].strip()
            elif isinstance(title_data, str):
                title = title_data.strip()
            else:
                return None
            
            if not title:
                return None
            
            # Clean up HTML entities in title
            title = title.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            
            # Extract authors if available
            authors = []
            author_data = best_text.get('author', best_text.get('authors', []))
            if isinstance(author_data, list):
                for author in author_data:
                    if isinstance(author, dict):
                        name = (author.get('name') or 
                               author.get('displayName') or 
                               author.get('fullName') or
                               f"{author.get('firstName', '')} {author.get('lastName', '')}").strip()
                        if name and name != ' ':
                            authors.append(name)
                    elif isinstance(author, str):
                        authors.append(author.strip())
            
            # Extract publisher
            publisher_data = best_text.get('publisher', [])
            publisher = None
            if isinstance(publisher_data, list) and publisher_data:
                publisher = publisher_data[0]
            elif isinstance(publisher_data, str):
                publisher = publisher_data
            
            # Extract DOI
            doi = None
            identifier_data = best_text.get('identifier', [])
            if isinstance(identifier_data, list):
                for identifier in identifier_data:
                    if isinstance(identifier, dict) and 'doi' in identifier:
                        doi_list = identifier['doi']
                        if isinstance(doi_list, list) and doi_list:
                            doi = doi_list[0]
                            break
            
            # Extract date from timestamp or created date
            pub_date = None
            if 'createdDate' in best_text:
                date_str = best_text['createdDate']
                if isinstance(date_str, str) and len(date_str) >= 4:
                    pub_date = date_str[:4]  # Extract year
            elif 'timestamp' in best_text:
                date_str = best_text['timestamp']
                if isinstance(date_str, str) and len(date_str) >= 4:
                    pub_date = date_str[:4]  # Extract year
            
            # Create reference
            reference = Reference(
                full_title=title,
                authors=authors if authors else None,
                journal_title=publisher,
                publication_date=pub_date,
                publisher=publisher,
                doi=doi,
            )

            return reference

        except Exception as e:
            logger.error(f"Error converting Matilda work to reference: {e}")
            return None

    def search_by_id(
        self,
        identifier: str,
        identifier_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Search for works by identifier using Matilda API.
        
        Supported identifier types: doi, isbn, arxiv, repec, pmid, eid, nlmuniqueid,
        rid, pii, pmcid, pmc, mid, bookaccession, versionId, version, medline, pmpid, hal
        
        Args:
            identifier: The identifier value to search for
            identifier_type: Type of identifier (defaults to 'doi')
            **kwargs: Additional search parameters
            
        Returns:
            List of works matching the identifier
        """
        if not identifier:
            return []

        # Supported identifier types from Matilda API
        supported_types = {
            "doi", "isbn", "arxiv", "repec", "pmid", "eid", "nlmuniqueid",
            "rid", "pii", "pmcid", "pmc", "mid", "bookaccession", 
            "versionid", "version", "medline", "pmpid", "hal"
        }
        
        id_type = (identifier_type or "doi").lower()
        if id_type not in supported_types:
            logger.warning(
                "Matilda does not support '%s' identifier type. Supported types: %s",
                id_type,
                ", ".join(sorted(supported_types))
            )
            return []

        # Use query.identifier format (Matilda API expects this format)
        params = {"query.identifier": identifier}
        
        try:
            response = self.session.get(
                f"{self.base_url}/works/query",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger.warning("Matilda identifier lookup failed: %s", exc)
            return []
        except requests.exceptions.RequestException as exc:
            logger.warning("Matilda identifier lookup failed: %s", exc)
            return []

        data = response.json()
        works = data.get("works", [])
        if not isinstance(works, list):
            return []
        return works

    def _extract_year_from_date(self, date_str: str) -> Optional[str]:
        """Extract a 4-digit year from a date string."""
        if not date_str:
            return None
        
        import re
        # Look for 4-digit year patterns
        match = re.search(r'\b(19|20)\d{2}\b', str(date_str))
        return match.group(0) if match else None

    def _build_search_params(
        self,
        reference: Reference,
        top_k: int,
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "query.title": reference.full_title,
        }

        author_name = self._pick_author(reference)
        if author_name:
            params["query.author"] = author_name

        year = self._extract_year_from_date(reference.publication_date or "")
        if year:
            params["filter.fromPublishedDate"] = f"{year}-01-01"
            params["filter.untilPublishedDate"] = f"{year}-12-31"

        publisher = reference.publisher or reference.journal_title
        if publisher:
            params["query.publisher"] = publisher

        if top_k:
            params["size"] = min(top_k, 100)

        allowed_keys = {
            "general_query": "query",
            "author_query": "query.author",
            "publisher_query": "query.publisher",
            "from_date": "filter.fromPublishedDate",
            "until_date": "filter.untilPublishedDate",
            "work_type": "filter.type",
            "sort_by": "sort",
            "sort_order": "order",
            "offset": "from",
        }

        for key, target in allowed_keys.items():
            if key in extra and extra[key] is not None:
                params[target] = extra[key]

        return params

    @staticmethod
    def _pick_author(reference: Reference) -> Optional[str]:
        if not reference.authors:
            return None

        first = reference.authors[0]
        if isinstance(first, str):
            return first.strip() or None

        first_name = getattr(first, "first_name", "") or ""
        surname = getattr(first, "surname", "") or ""
        author_name = f"{first_name} {surname}".strip()
        return author_name or None

    @staticmethod
    def _iter_works(raw_results: List[Dict[str, Any]]):
        for entry in raw_results:
            if isinstance(entry, dict) and "works" in entry:
                works = entry.get("works", [])
                if isinstance(works, list):
                    for work in works:
                        if isinstance(work, dict):
                            yield work
                continue

            if isinstance(entry, dict) and "content" in entry:
                content = entry.get("content", [])
                if isinstance(content, list):
                    for work in content:
                        if isinstance(work, dict):
                            yield work
                continue

            if isinstance(entry, dict):
                yield entry


    @staticmethod
    def _normalize_identifier(identifier: str) -> str:
        cleaned = identifier.strip()
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

 
