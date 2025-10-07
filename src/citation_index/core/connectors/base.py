"""
Base classes for data source API connectors.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from ..models.reference import Reference
from ..models.references import References


class BaseConnector(ABC):
    """Abstract base class for data source API connectors."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the connector.
        
        Args:
            api_key: API key for authentication (if required)
            base_url: Base URL for the API (if different from default)
        """
        self.api_key = api_key
        self.base_url = base_url
    
    @abstractmethod
    def search(self, reference: Reference, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Search for references using the data source API.
        
        Args:
            reference: Reference object containing search criteria (title is mandatory)
            top_k: Maximum number of results to return
            **kwargs: Additional search parameters specific to each connector
            
        Returns:
            List of raw API response data (JSON objects)
            
        Raises:
            ValueError: If reference title is missing
            Exception: If API call fails
        """
        pass
    
    @abstractmethod
    def map_to_references(self, raw_results: List[Dict[str, Any]]) -> References:
        """Transform raw API results to Reference objects.
        
        Args:
            raw_results: List of raw API response data
            
        Returns:
            References object containing mapped Reference instances
        """
        pass
    
    def lookup_ref(self, reference: Reference, top_k: int = 10, **kwargs) -> References:
        """Search for references and return mapped Reference objects.
        
        This is a convenience method that combines search() and map_to_references().
        
        Args:
            reference: Reference object containing search criteria
            top_k: Maximum number of results to return
            **kwargs: Additional parameters passed to the search method
            
        Returns:
            References object containing mapped Reference instances
        """
        raw_results = self.search(reference, top_k, **kwargs)
        return self.map_to_references(raw_results)
    
    def _validate_reference(self, reference: Reference) -> None:
        """Validate that reference has required fields for search.
        
        Args:
            reference: Reference object to validate
            
        Raises:
            ValueError: If title is missing or empty
        """
        if not reference.full_title or not reference.full_title.strip():
            raise ValueError("Reference title is required for search")