"""Base classes for data source API connectors."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

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
