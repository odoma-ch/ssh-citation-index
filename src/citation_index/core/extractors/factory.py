"""
Factory for creating PDF extractors.
"""

from .base import BaseExtractor


class ExtractorFactory:
    """Factory class for creating PDF extractors."""
    
    @staticmethod
    def create(extractor_type: str, **kwargs) -> BaseExtractor:
        """Create an extractor instance based on type.
        
        Args:
            extractor_type: Type of extractor ('pymupdf', 'marker', 'mineru', 'grobid')
            **kwargs: Additional arguments for extractor initialization
            
        Returns:
            BaseExtractor instance
            
        Raises:
            ValueError: If extractor type is not supported
            ImportError: If required dependencies for the extractor are not installed
        """
        extractor_type = extractor_type.lower()
        
        if extractor_type == 'pymupdf':
            from .pymupdf import PyMuPDFExtractor
            return PyMuPDFExtractor()
        elif extractor_type == 'marker':
            from .marker import MarkerExtractor
            return MarkerExtractor()
        elif extractor_type == 'mineru':
            from .mineru import MineruExtractor
            return MineruExtractor()
        elif extractor_type == 'grobid':
            from .grobid import GrobidExtractor
            # Extract GROBID-specific parameters
            grobid_kwargs = {}
            if 'grobid_endpoint' in kwargs:
                grobid_kwargs['endpoint'] = kwargs['grobid_endpoint']
            if 'grobid_timeout' in kwargs:
                grobid_kwargs['timeout'] = kwargs['grobid_timeout']
            if 'grobid_max_retries' in kwargs:
                grobid_kwargs['max_retries'] = kwargs['grobid_max_retries']
            return GrobidExtractor(**grobid_kwargs)
        else:
            raise ValueError(f"Unsupported extractor type: {extractor_type}")
    
    @staticmethod
    def get_available_extractors():
        """Get list of available extractor types.
        
        Returns:
            List of available extractor type names
        """
        return ['pymupdf', 'marker', 'mineru', 'grobid'] 