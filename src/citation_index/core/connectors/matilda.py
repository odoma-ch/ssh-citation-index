"""
Matilda API connector for citation index.

This connector interfaces with the Matilda Science API to search for academic works
using title-based queries.
"""

import base64
import logging
from typing import Optional, Dict, Any, List
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
    
    def search(self, reference: Reference, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
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
        # Build query parameters
        params = {}
        
        # Title-based search (primary)
        if reference.full_title:
            params['query.title'] = reference.full_title
        
        # Author-based search
        if reference.authors and len(reference.authors) > 0:
            # Use the first author for query, or combine multiple authors
            if isinstance(reference.authors[0], str):
                author_name = reference.authors[0]
            else:
                # Handle Person objects if they exist
                author_obj = reference.authors[0]
                if hasattr(author_obj, 'surname') and hasattr(author_obj, 'first_name'):
                    author_name = f"{author_obj.first_name or ''} {author_obj.surname or ''}".strip()
                else:
                    author_name = str(author_obj)
            
            if author_name:
                params['query.author'] = author_name
        
        # Publication date filtering
        if reference.publication_date:
            year = self._extract_year_from_date(reference.publication_date)
            if year:
                # Search for works published in that year
                params['filter.fromPublishedDate'] = f"{year}-01-01"
                params['filter.untilPublishedDate'] = f"{year}-12-31"
        
        # Publisher search
        if reference.publisher:
            params['query.publisher'] = reference.publisher
        elif reference.journal_title:
            # Use journal as publisher query
            params['query.publisher'] = reference.journal_title
        
        # Additional parameters from kwargs
        if 'general_query' in kwargs:
            params['query'] = kwargs['general_query']
        
        if 'author_query' in kwargs:
            params['query.author'] = kwargs['author_query']
            
        if 'publisher_query' in kwargs:
            params['query.publisher'] = kwargs['publisher_query']
        
        # Date filtering
        if 'from_date' in kwargs:
            params['filter.fromPublishedDate'] = kwargs['from_date']
        if 'until_date' in kwargs:
            params['filter.untilPublishedDate'] = kwargs['until_date']
        
        # Work type filtering
        if 'work_type' in kwargs:
            params['filter.type'] = kwargs['work_type']
        
        # Sorting
        if 'sort_by' in kwargs:
            params['sort'] = kwargs['sort_by']
        if 'sort_order' in kwargs:
            params['order'] = kwargs['sort_order']
        
        # Pagination
        if top_k:
            params['size'] = min(top_k, 100)  # Matilda might have limits
        if 'offset' in kwargs:
            params['from'] = kwargs['offset']
        
        # If no search criteria provided, return empty
        if not any(key.startswith('query') for key in params.keys()) and 'query' not in params:
            logger.warning("Matilda search requires at least one search parameter (title, author, or general query)")
            return []
        
        try:
            # Make the API request
            url = f"{self.base_url}/works/query"
            logger.info(f"Matilda API request: {url} with params: {params}")
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 401:
                logger.error("Matilda API authentication failed")
                return []
            elif response.status_code != 200:
                logger.error(f"Matilda API error: {response.status_code} - {response.text}")
                return []
            
            data = response.json()
            works_count = len(data.get('works', []))
            total_count = data.get('total', works_count)
            logger.info(f"Matilda API returned {works_count} works (total: {total_count})")
            
            # Return the raw data for mapping
            return [data]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Matilda API request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error processing Matilda API response: {e}")
            return []
    
    def map_to_references(self, raw_results: List[Dict[str, Any]]) -> References:
        """
        Map Matilda API response to References object.
        
        Args:
            raw_results: List of raw API responses from Matilda
            
        Returns:
            References object with mapped data
        """
        references = References()
        
        for api_response in raw_results:
            # Handle different response structures
            works = api_response.get('works', api_response.get('content', []))
            if not isinstance(works, list):
                logger.warning("Unexpected Matilda API response structure")
                continue
            
            for work in works:
                try:
                    ref = self._convert_work_to_reference(work)
                    if ref:
                        references.append(ref)
                except Exception as e:
                    logger.warning(f"Failed to convert Matilda work to reference: {e}")
                    continue
        
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
                journal_title=publisher,  # Use publisher as journal for now
                publication_date=pub_date,
                publisher=publisher
            )
            
            return reference
            
        except Exception as e:
            logger.error(f"Error converting Matilda work to reference: {e}")
            return None
    
    def search_by_doi(self, doi: str) -> Optional[Reference]:
        """
        Search for a work by DOI.
        
        Args:
            doi: DOI to search for
            
        Returns:
            Reference object if found, None otherwise
        """
        try:
            # Try searching by DOI if the API supports it
            params = {'query.doi': doi}
            
            response = self.session.get(
                f"{self.base_url}/works/query",
                params=params,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.warning(f"Matilda DOI search failed: {response.status_code}")
                return None
            
            data = response.json()
            references = self.map_to_references(data)
            
            return references.references[0] if references.references else None
            
        except Exception as e:
            logger.error(f"Error in Matilda DOI search: {e}")
            return None
    
    def _extract_year_from_date(self, date_str: str) -> Optional[str]:
        """Extract a 4-digit year from a date string."""
        if not date_str:
            return None
        
        import re
        # Look for 4-digit year patterns
        match = re.search(r'\b(19|20)\d{2}\b', str(date_str))
        return match.group(0) if match else None

 