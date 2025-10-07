"""
OpenAlex API connector for reference search and disambiguation.
"""

import requests
from typing import Dict, List, Any, Optional

from .base import BaseConnector
from ..models.reference import Reference
from ..models.references import References


class OpenAlexConnector(BaseConnector):
    """Connector for OpenAlex API."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize OpenAlex connector.
        
        Args:
            api_key: Optional API key for higher rate limits
            base_url: Base URL for OpenAlex API (defaults to official API)
        """
        super().__init__(api_key, base_url)
        self.base_url = base_url or "https://api.openalex.org"
        self.session = requests.Session()
        
        # Set headers for API requests
        headers = {
            "User-Agent": "citation-index/1.0 (mailto:your-email@domain.com)"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.session.headers.update(headers)
    
    def search(self, reference: Reference, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Search for works using OpenAlex API.
        
        Args:
            reference: Reference object containing search criteria (title is mandatory)
            top_k: Maximum number of results to return (default: 10)
            
        Returns:
            List of raw OpenAlex API response data
            
        Raises:
            ValueError: If reference title is missing
            Exception: If API call fails
        """
        self._validate_reference(reference)
        
        # Build search query using title
        title = reference.full_title.strip()
        
        # Construct API URL and parameters
        url = f"{self.base_url}/works"
        params = {
            "filter": f"title.search:{title}",  # Don't quote here, let requests handle it
            "per-page": min(top_k, 200),  # OpenAlex max is 200 per page
            "sort": "relevance_score:desc"
        }
        
        # Add additional filters if available
        if reference.publication_date:
            # Extract year from publication_date if possible
            year = reference._extract_year(reference.publication_date)
            if year:
                params["filter"] += f",publication_year:{year}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            return results[:top_k]  # Ensure we don't exceed requested top_k
            
        except requests.RequestException as e:
            raise Exception(f"OpenAlex API request failed: {e}")
        except KeyError as e:
            raise Exception(f"Unexpected OpenAlex API response format: {e}")
    
    def map_to_references(self, raw_results: List[Dict[str, Any]]) -> References:
        """Transform OpenAlex API results to Reference objects.
        
        Args:
            raw_results: List of raw OpenAlex API response data
            
        Returns:
            References object containing mapped Reference instances
        """
        references = []
        
        for result in raw_results:
            try:
                # Extract basic information
                title = result.get("title")
                publication_year = result.get("publication_year")
                
                # Extract authors as strings (Reference model expects List[Person | Organization | str])
                authors = []
                authorship = result.get("authorships", [])
                for auth in authorship:
                    author_info = auth.get("author", {})
                    display_name = author_info.get("display_name")
                    if display_name:
                        authors.append(display_name)
                
                # Extract publication venue information
                primary_location = result.get("primary_location", {})
                source = primary_location.get("source", {}) if primary_location else {}
                
                # Map OpenAlex fields to Reference model fields
                journal_title = source.get("display_name") if source else None
                publisher = source.get("host_organization_name") if source else None
                
                # Extract additional publication details
                volume = result.get("biblio", {}).get("volume") if result.get("biblio") else None
                issue = result.get("biblio", {}).get("issue") if result.get("biblio") else None
                
                # Extract page information
                first_page = result.get("biblio", {}).get("first_page") if result.get("biblio") else None
                last_page = result.get("biblio", {}).get("last_page") if result.get("biblio") else None
                pages = None
                if first_page and last_page:
                    pages = f"{first_page}-{last_page}"
                elif first_page:
                    pages = first_page
                
                # Create Reference object with proper field mapping
                reference = Reference(
                    full_title=title,
                    authors=authors if authors else None,
                    journal_title=journal_title,
                    publisher=publisher,
                    publication_date=str(publication_year) if publication_year else None,
                    volume=volume,
                    issue=issue,
                    pages=pages
                )
                
                references.append(reference)
                
            except Exception as e:
                # Log warning but continue processing other results
                print(f"Warning: Could not map OpenAlex result to Reference: {e}")
                continue
        
        # Create References object using append method to avoid constructor validation issues
        result = References()
        for ref in references:
            result.append(ref)
        return result
    
    def search_by_doi(self, doi: str) -> Optional[Reference]:
        """Search for a specific work by DOI.
        
        Args:
            doi: DOI of the work to search for
            
        Returns:
            Reference object if found, None otherwise
        """
        if not doi:
            return None
            
        # Clean DOI format
        clean_doi = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
        
        url = f"{self.base_url}/works"
        params = {
            "filter": f"doi:{clean_doi}"
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            if results:
                mapped = self.map_to_references([results[0]])
                return mapped.references[0] if mapped.references else None
                
        except Exception as e:
            print(f"Warning: DOI search failed: {e}")
            
        return None


# Example usage:
# connector = OpenAlexConnector()
# reference = Reference(full_title="Your Paper Title")
# results = connector.lookup_ref(reference, top_k=5)
# for ref in results.references:
#     print(f"Title: {ref.full_title}")
#     print(f"Authors: {ref.authors}")
#     print(f"Year: {ref.publication_date}")
#     print("---")
#
# To test all connectors: python tests/test_connectors.py


