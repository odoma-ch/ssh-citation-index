"""
Base classes for PDF content extractors.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Optional


class ExtractResult(NamedTuple):
    """Unified result return format for PDF extraction."""
    text: str | List[str]
    metadata: Optional[Dict] = None
    images: Optional[List] = None


class BaseExtractor(ABC):
    """Abstract base class for PDF content extractors."""
    
    @abstractmethod
    def extract(self, filepath: str, save_dir: str = None, **kwargs) -> ExtractResult:
        """Extract content from a PDF file.
        
        Args:
            filepath: Path to the PDF file
            save_dir: Optional directory to save extracted content
            **kwargs: Additional extraction parameters
            
        Returns:
            ExtractResult containing the extracted text and metadata
        """
        pass

    def _read_file(self, filepath: str) -> bytes:
        """Read file contents as bytes.
        
        Args:
            filepath: Path to the file
            
        Returns:
            File contents as bytes
            
        Raises:
            FileNotFoundError: If file does not exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File does not exist: {filepath}")
        with open(filepath, "rb") as f:
            return f.read() 