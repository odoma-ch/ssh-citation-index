"""Citation Index - A citation extraction and parsing system for academic documents."""

from .core.models import Reference, Person, Organization, References
from .core.extractors import ExtractorFactory, BaseExtractor, ExtractResult
from .core.parsers import TeiBiblParser

__version__ = "0.1.0"

__all__ = [
    "Reference",
    "Person", 
    "Organization",
    "References",
    "ExtractorFactory",
    "BaseExtractor",
    "ExtractResult",
    "TeiBiblParser",
]
