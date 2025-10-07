"""Connectors for reference disambiguation APIs."""

from .base import BaseConnector
from .openalex import OpenAlexConnector
from .opencitations import OpenCitationsConnector
from .matilda import MatildaConnector
from .wikidata import WikidataConnector

__all__ = [
    "BaseConnector",
    "OpenAlexConnector",
    "OpenCitationsConnector",
    "MatildaConnector",
    "WikidataConnector",
]
