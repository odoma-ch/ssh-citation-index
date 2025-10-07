"""
OpenCitations Meta API connector for reference search and disambiguation.
"""

import requests
from typing import Dict, List, Any, Optional
import re

from .base import BaseConnector
from ..models.reference import Reference
from ..models.references import References


class OpenCitationsConnector(BaseConnector):
    """Connector for OpenCitations Meta API."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize OpenCitations connector.
        
        Args:
            api_key: Optional OpenCitations Access Token for higher rate limits
            base_url: Base URL for OpenCitations Meta API (defaults to official API)
        """
        super().__init__(api_key, base_url)
        self.base_url = base_url or "https://api.opencitations.net/meta/v1"
        self.session = requests.Session()
        
        # Set headers for API requests
        headers = {
            "User-Agent": "citation-index/1.0 (mailto:your-email@domain.com)",
            "Accept": "application/json"
        }
        if self.api_key:
            headers["authorization"] = self.api_key
        
        self.session.headers.update(headers)
    
    def search(self, reference: Reference, top_k: int = 10, include_author: bool = False, include_date: bool = False) -> List[Dict[str, Any]]:
        """Search for works using OpenCitations Meta API.
        
        OpenCitations Meta API does not support general title-based search.
        It only supports lookup by specific identifiers: "doi", "issn", "isbn", and "omid".
        
        This method returns an empty list. Use search_by_id() instead.
        
        Args:
            reference: Reference object containing search criteria
            top_k: Maximum number of results to return (not used)
            include_author: Not supported for general search
            include_date: Not supported for general search
            
        Returns:
            Empty list (OpenCitations doesn't support general search)
            
        Raises:
            ValueError: If reference title is missing
        """
        self._validate_reference(reference)
        
        print("INFO: OpenCitations Meta API only supports identifier-based lookups.")
        print("      Supported identifiers: 'doi', 'issn', 'isbn', 'omid'")
        print("      Use search_by_id() method for identifier-based lookups.")
        
        return []
    
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
# connector = OpenCitationsConnector(api_key="your-access-token")
# 
# # Search by various identifiers
# doi_ref = connector.search_by_doi("10.1007/978-1-4020-9632-7")
# isbn_ref = connector.search_by_id("9781402096327", "isbn")
# issn_ref = connector.search_by_id("0138-9130", "issn")
# omid_ref = connector.search_by_id("br/0612058700", "omid")
# 
# # Auto-detect identifier type
# auto_ref = connector.search_by_id("10.1007/978-1-4020-9632-7")  # Auto-detects as DOI
#
# To test all connectors: python tests/test_connectors.py