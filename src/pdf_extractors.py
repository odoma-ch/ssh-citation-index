"""
PDF extraction utilities providing different methods to extract content from PDF files.
Supports multiple extraction backends including PyMuPDF, Marker, and Mineru.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Optional

import pymupdf
import pymupdf4llm
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze


class ExtractResult(NamedTuple):
    """Unified result return format for PDF extraction."""
    text: str
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


class MarkerExtractor(BaseExtractor):
    """PDF content extractor using Marker backend."""
    
    def __init__(self, config: Dict = None):
        """Initialize the MarkerExtractor.
        
        Args:
            config: Optional configuration dictionary
            processor_list: Optional list of processors to use
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
            # processor_list=self.processor_list
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
            pipe_result = result.pipe_txt_mode(image_writer)

            
            md_content = pipe_result.get_markdown("images")
            pipe_result.dump_md(md_writer, f"{base_name}_mineru.md", "images")
            
            return ExtractResult(text=md_content)
        except Exception as e:
            raise RuntimeError(f"Mineru extraction failed: {str(e)}") from e


class ExtractorFactory:
    """Factory class for creating PDF extractors."""
    
    @staticmethod
    def create(extractor_type: str) -> BaseExtractor:
        """Create an extractor instance of the specified type.
        
        Args:
            extractor_type: Type of extractor to create ('pymupdf', 'marker', or 'mineru')
            
        Returns:
            An instance of the requested extractor
            
        Raises:
            ValueError: If extractor type is not supported
        """
        types = {
            "pymupdf": PyMuPDFExtractor,
            "marker": MarkerExtractor,
            "mineru": MineruExtractor
        }
        if extractor_type.lower() not in types:
            raise ValueError(f"Unsupported extraction type: {extractor_type}")
        return types[extractor_type.lower()]()


if __name__ == "__main__":
    # Example usage
    file_path = 'resources/777_towards_better_citation_intent.pdf'

    extractor = ExtractorFactory.create("marker")
    result = extractor.extract(file_path, save_dir='output/marker')
    print(result.text)
    print('----------------------')

    extractor = ExtractorFactory.create("pymupdf")
    result = extractor.extract(filepath=file_path, save_dir='output/pymupdf')
    print(result.text)
    print('----------------------')

    # extractor = ExtractorFactory.create("mineru")
    # result = extractor.extract(filepath=file_path)
    # print(result.text)