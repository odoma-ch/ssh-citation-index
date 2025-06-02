"""
Factory for creating PDF extractors.
"""

from .base import BaseExtractor
from .pymupdf import PyMuPDFExtractor
from .marker import MarkerExtractor
from .mineru import MineruExtractor


class ExtractorFactory:
    """Factory class for creating PDF extractors."""
    
    @staticmethod
    def create(extractor_type: str) -> BaseExtractor:
        """Create an extractor instance based on type.
        
        Args:
            extractor_type: Type of extractor ('pymupdf', 'marker', 'mineru')
            
        Returns:
            BaseExtractor instance
            
        Raises:
            ValueError: If extractor type is not supported
        """
        extractor_type = extractor_type.lower()
        
        if extractor_type == 'pymupdf':
            return PyMuPDFExtractor()
        elif extractor_type == 'marker':
            return MarkerExtractor()
        elif extractor_type == 'mineru':
            return MineruExtractor()
        else:
            raise ValueError(f"Unsupported extractor type: {extractor_type}")
    
    @staticmethod
    def get_available_extractors():
        """Get list of available extractor types.
        
        Returns:
            List of available extractor type names
        """
        return ['pymupdf', 'marker', 'mineru'] 