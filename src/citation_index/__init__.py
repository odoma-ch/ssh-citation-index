"""Citation Index - A citation extraction and parsing system for academic documents."""

from .core.models import Reference, Person, Organization, References
from .core.extractors import ExtractorFactory, BaseExtractor, ExtractResult
from .core.parsers import TeiBiblParser
from .core.segmenters.references_locator import extract_all_reference_sections
from .pipelines.reference_extraction import extract_text_references, extract_pdf_references
from .pipelines.reference_parsing import parse_reference_strings, parse_reference_file
from .pipelines.reference_extraction_and_parsing import run_pdf_extract_and_parse

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
    "extract_all_reference_sections",
    "extract_text_references",
    "extract_pdf_references",
    "parse_reference_strings",
    "parse_reference_file",
    "run_pdf_extract_and_parse",
]
