"""Base classes for data source API connectors."""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ..models.reference import Reference
from ..models.references import References


class BaseConnector(ABC):
    """Abstract base class shared by connector implementations."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    def search(self, reference: Reference, top_k: int = 10, **kwargs: Any) -> List[Dict[str, Any]]:
        """Search the remote source using bibliographic metadata.

        Implementations must, at minimum, use the reference title when building
        the query. Additional kwargs can extend connector-specific behaviour.
        """

    @abstractmethod
    def search_by_id(
        self,
        identifier: str,
        identifier_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Fetch records by an identifier such as DOI or ISBN.

        Implementations should accept ``identifier_type`` to disambiguate when
        multiple identifiers are supported. When the type is ``None`` they may
        attempt to infer it from the identifier value.
        """

    @abstractmethod
    def map_to_references(self, raw_results: List[Dict[str, Any]]) -> References:
        """Convert raw API payloads into ``Reference`` instances."""

    def _validate_reference(self, reference: Reference) -> None:
        title = (reference.full_title or "").strip()
        if not title:
            raise ValueError("Reference title is required for search")

    @abstractmethod
    def _result_to_reference(self, result: Dict[str, Any]) -> Reference:
        """Convert a raw API result dictionary to a Reference object.
        
        Each connector must implement this to extract the relevant fields
        from their specific API response format.
        
        Args:
            result: Raw API result dictionary
            
        Returns:
            Reference object with extracted fields
        """

    def _reference_to_string(self, reference: Reference) -> str:
        """Convert a Reference object to a normalized string for matching.
        
        Format: "title | authors | year | publisher | pages"
        All text is lowercased, extra whitespace removed, and punctuation normalized.
        
        Args:
            reference: Reference object to convert
            
        Returns:
            Normalized string representation
        """
        parts = []
        
        # Add title
        title = (reference.full_title or "").strip()
        if title:
            title = re.sub(r"\s+", " ", title.lower())
            title = re.sub(r"[^\w\s]", "", title)  # Remove punctuation
            parts.append(title)
        
        # Add authors (concatenate all author names)
        if reference.authors:
            author_strs = []
            for author in reference.authors:
                if isinstance(author, str):
                    author_strs.append(author.strip())
                elif hasattr(author, "surname") and hasattr(author, "first_name"):
                    name_parts = []
                    if author.surname:
                        name_parts.append(str(author.surname))
                    if author.first_name:
                        name_parts.append(str(author.first_name))
                    if name_parts:
                        author_strs.append(" ".join(name_parts))
                elif hasattr(author, "name"):
                    author_strs.append(str(author.name))
            
            if author_strs:
                authors = " ".join(author_strs)
                authors = re.sub(r"\s+", " ", authors.lower())
                authors = re.sub(r"[^\w\s]", "", authors)
                parts.append(authors)
        
        # Add year
        year = reference.publication_date or ""
        year = re.sub(r"[^\d]", "", str(year))  # Extract digits only
        if year and len(year) >= 4:
            parts.append(year[:4])  # Take first 4 digits (year)
        
        # Add publisher
        publisher = (reference.publisher or "").strip()
        if publisher:
            publisher = re.sub(r"\s+", " ", publisher.lower())
            publisher = re.sub(r"[^\w\s]", "", publisher)
            parts.append(publisher)
        
        # Add pages
        pages = (reference.pages or "").strip()
        if pages:
            pages = re.sub(r"\s+", "", pages.lower())  # Remove all whitespace
            pages = re.sub(r"[^\d\-]", "", pages)  # Keep only digits and hyphens
            if pages:
                parts.append(pages)
        
        return " | ".join(parts) if parts else ""

    def match(
        self, 
        query_ref: Reference, 
        result: Dict[str, Any], 
        threshold: float = 0.9
    ) -> Tuple[bool, float]:
        """Determine if a search result matches the query reference.
        
        Converts both the query reference and search result to normalized
        strings and computes fuzzy string similarity. A match is determined
        if the similarity score exceeds the threshold.
        
        Args:
            query_ref: The original query Reference object
            result: Raw API result dictionary
            threshold: Similarity threshold for match (0-1), default 0.9
            
        Returns:
            Tuple of (is_match: bool, similarity_score: float)
        """
        try:
            # Import rapidfuzz here to avoid making it a hard dependency
            from rapidfuzz import fuzz
        except ImportError:
            raise ImportError(
                "rapidfuzz is required for fuzzy matching. "
                "Install it with: pip install rapidfuzz"
            )
        
        # Convert query reference to string
        query_str = self._reference_to_string(query_ref)
        if not query_str:
            return False, 0.0
        
        # Convert result to Reference, then to string
        try:
            result_ref = self._result_to_reference(result)
            result_str = self._reference_to_string(result_ref)
            if not result_str:
                return False, 0.0
        except Exception:
            # If we can't parse the result, it's not a match
            return False, 0.0
        
        # Compute similarity (rapidfuzz returns 0-100, convert to 0-1)
        similarity = fuzz.ratio(query_str, result_str) / 100.0
        is_match = similarity >= threshold
        
        return is_match, similarity
