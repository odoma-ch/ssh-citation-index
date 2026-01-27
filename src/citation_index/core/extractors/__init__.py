"""
PDF Extractors Module

Contains all PDF text extraction implementations and factory.
"""

from .base import BaseExtractor, ExtractResult

# Use lazy imports for extractors to avoid issues with optional dependencies
# The factory will handle imports when actually needed
try:
    from .pymupdf import PyMuPDFExtractor
except ImportError:
    PyMuPDFExtractor = None

try:
    from .marker import MarkerExtractor
except ImportError:
    MarkerExtractor = None

try:
    from .mineru import MineruExtractor
except ImportError:
    MineruExtractor = None

try:
    from .grobid import GrobidExtractor
except ImportError:
    GrobidExtractor = None

from .factory import ExtractorFactory

__all__ = [
    "BaseExtractor",
    "ExtractResult", 
    "PyMuPDFExtractor",
    "MarkerExtractor",
    "MineruExtractor",
    "GrobidExtractor",
    "ExtractorFactory"
] 