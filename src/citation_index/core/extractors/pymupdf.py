"""
PyMuPDF-based PDF content extractor.
"""

import os
import pymupdf
import pymupdf4llm
from .base import BaseExtractor, ExtractResult


class PyMuPDFExtractor(BaseExtractor):
    """PDF content extractor using PyMuPDF backend."""
    
    def extract(
        self, 
        filepath: str, 
        save_dir: str = None, 
        markdown: bool = True, 
        **kwargs
    ) -> ExtractResult:
        """Extract content using PyMuPDF.
        
        Args:
            filepath: Path to the PDF file
            save_dir: Optional directory to save extracted content
            markdown: Whether to output in markdown format
            **kwargs: Additional extraction parameters
            
        Returns:
            ExtractResult containing the extracted text
            
        Raises:
            RuntimeError: If extraction fails
        """
        base_name = os.path.basename(filepath).rsplit(".", 1)[0]
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        try:
            if markdown:
                text = pymupdf4llm.to_markdown(filepath)
                if save_dir:
                    with open(os.path.join(save_dir, f"{base_name}_pymupdf.md"), "w") as f:
                        f.write(text)
            else:
                doc = pymupdf.open(filepath)
                text = "".join(page.get_text() for page in doc)
                if save_dir:
                    with open(os.path.join(save_dir, f"{base_name}_pymupdf.txt"), "w") as f:
                        f.write(text)

            return ExtractResult(text=text)
        except Exception as e:
            raise RuntimeError(f"PyMuPDF extraction failed: {str(e)}") from e 