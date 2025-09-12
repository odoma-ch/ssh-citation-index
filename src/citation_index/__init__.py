"""Citation Index - A citation extraction and parsing system for academic documents."""

from .core.models import Reference, Person, Organization, References
from .core.extractors import ExtractorFactory, BaseExtractor, ExtractResult, GrobidExtractor
from .core.parsers import TeiBiblParser
from .core.segmenters.semantic_reference_locator import locate_reference_sections_semantic
from .pipelines.reference_extraction import (
    extract_text_references, extract_text_references_semantic_sections
)
from .pipelines.reference_parsing import parse_reference_strings, parse_reference_file
from .pipelines.reference_extraction_and_parsing import (
    run_pdf_extract_and_parse, run_pdf_semantic_one_step
)
from .llm.grobid_client import GrobidClient

__version__ = "0.1.0"

__all__ = [
    "Reference",
    "Person", 
    "Organization",
    "References",
    "ExtractorFactory",
    "BaseExtractor",
    "ExtractResult",
    "GrobidExtractor",
    "GrobidClient",
    "TeiBiblParser",
    "locate_reference_sections_semantic",
    "extract_text_references",
    "extract_text_references_semantic_sections",
    "parse_reference_strings",
    "parse_reference_file",
    "run_pdf_extract_and_parse",
    "run_pdf_semantic_one_step",
]
