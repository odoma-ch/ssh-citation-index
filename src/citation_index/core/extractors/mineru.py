"""
Mineru-based PDF content extractor.
"""

import os
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from .base import BaseExtractor, ExtractResult


class MineruExtractor(BaseExtractor):
    """PDF content extractor using Mineru backend."""
    
    def __init__(self):
        """Initialize the MineruExtractor."""
        self.reader = FileBasedDataReader("")

    def extract(
        self,
        filepath: str,
        save_dir: str = "output",
        page_split: bool = True,
        **kwargs
    ) -> ExtractResult:
        """Extract content using Mineru.
        
        Args:
            filepath: Path to the PDF file
            save_dir: Directory to save extracted content
            page_split: Whether to split pages
            **kwargs: Additional extraction parameters
            
        Returns:
            ExtractResult containing the extracted text
            
        Raises:
            RuntimeError: If extraction fails
        """
        try:
            base_name = os.path.basename(filepath).rsplit(".", 1)[0]
            image_dir = os.path.join(save_dir, "images")
            
            os.makedirs(image_dir, exist_ok=True)
            
            image_writer = FileBasedDataWriter(image_dir)
            md_writer = FileBasedDataWriter(save_dir)
            
            pdf_bytes = self.reader.read(filepath)
            dataset = PymuDocDataset(pdf_bytes)
            result = dataset.apply(doc_analyze, ocr=False, formula_enable=False)
            
            # Process results and save
            text_content = ""
            for page_id, page_data in result.items():
                if 'md_content' in page_data:
                    text_content += page_data['md_content'] + "\n\n"
                    
                # Save images if any
                if 'images' in page_data:
                    for img_name, img_data in page_data['images'].items():
                        image_writer.write(img_data, img_name)
            
            # Save markdown content
            md_filename = f"{base_name}_mineru.md"
            md_writer.write(text_content, md_filename)
            
            return ExtractResult(text=text_content)
            
        except Exception as e:
            raise RuntimeError(f"Mineru extraction failed: {str(e)}") from e 