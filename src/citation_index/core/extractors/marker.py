"""
Marker-based PDF content extractor.
"""

import os
from typing import Dict
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
from .base import BaseExtractor, ExtractResult


class MarkerExtractor(BaseExtractor):
    """PDF content extractor using Marker backend."""
    
    def __init__(self, config: Dict = None):
        """Initialize the MarkerExtractor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {
            "use_llm": False,
            'output_format': 'markdown',
            'disable_image_extraction': True,
            'paginate_output': True,
        }

        self.converter = PdfConverter(
            artifact_dict=create_model_dict(),
            config=ConfigParser(self.config).generate_config_dict(),
        )

    def extract(self, filepath: str, save_dir: str = None, **kwargs) -> ExtractResult:
        """Extract content using Marker.
        
        Args:
            filepath: Path to the PDF file
            save_dir: Optional directory to save extracted content
            **kwargs: Additional extraction parameters
            
        Returns:
            ExtractResult containing the extracted text, metadata and images
            
        Raises:
            RuntimeError: If extraction fails
        """
        base_name = os.path.basename(filepath).rsplit(".", 1)[0]
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        try:
            rendered = self.converter(filepath)
            text, metadata, images = text_from_rendered(rendered)
            if save_dir:
                with open(os.path.join(save_dir, f"{base_name}_marker.md"), "w") as f:
                    f.write(text)

            return ExtractResult(text=text, metadata=metadata, images=images)
        except Exception as e:
            raise RuntimeError(f"Marker extraction failed: {str(e)}") from e 