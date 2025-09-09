"""
GROBID client for processing PDFs and extracting references.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import requests


class GrobidError(Exception):
    """Raised when GROBID service encounters an error."""
    pass


class GrobidClient:
    """Client for interacting with GROBID service."""
    
    def __init__(self, endpoint: str, timeout: float = 180.0, max_retries: int = 3):
        """Initialize the GROBID client.
        
        Args:
            endpoint: GROBID server endpoint (e.g., "http://localhost:8070")
            timeout: Maximum time to wait for response (seconds)
            max_retries: Maximum number of retry attempts for failed calls
        """
        self.endpoint = endpoint.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # Set common headers
        self.session.headers.update({
            'User-Agent': 'citation-index-grobid-client/1.0'
        })
    
    def process_references(
        self, 
        pdf_path: str | Path,
        consolidate_references: bool = True,
        include_raw_references: bool = True
    ) -> str:
        """Extract references from a PDF using GROBID processReferences service.
        
        Args:
            pdf_path: Path to the PDF file
            consolidate_references: Enable reference consolidation with external services
            include_raw_references: Include raw reference strings in output
            
        Returns:
            XML string containing extracted references in TEI format
            
        Raises:
            GrobidError: If GROBID service fails or returns an error
            FileNotFoundError: If PDF file doesn't exist
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        url = f"{self.endpoint}/api/processReferences"
        
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                with open(pdf_path, 'rb') as pdf_file:
                    files = {'input': (pdf_path.name, pdf_file, 'application/pdf')}
                    data = {
                        'consolidateReferences': '1' if consolidate_references else '0',
                        'includeRawReferences': '1' if include_raw_references else '0',
                    }
                    
                    response = self.session.post(
                        url, 
                        files=files, 
                        data=data, 
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        xml_content = response.text
                        if not xml_content.strip():
                            raise GrobidError("GROBID returned empty response")
                        return xml_content
                    else:
                        error_msg = f"GROBID service error (HTTP {response.status_code}): {response.text}"
                        raise GrobidError(error_msg)
                        
            except (requests.RequestException, GrobidError) as e:
                last_exception = e
                attempt_info = f"attempt {attempt + 1}/{self.max_retries + 1}"
                
                if isinstance(e, requests.Timeout):
                    logging.warning(f"GROBID timeout on {attempt_info}: {e}")
                elif isinstance(e, GrobidError):
                    logging.warning(f"GROBID service error on {attempt_info}: {e}")
                else:
                    logging.warning(f"GROBID request error on {attempt_info}: {type(e).__name__}: {e}")
                
                if attempt < self.max_retries:
                    wait_time = min(2 ** attempt, 10)  # Exponential backoff with cap
                    logging.info(f"Retrying GROBID request in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"All {self.max_retries + 1} GROBID attempts failed")
                    break
        
        # If all retries failed, raise the last exception
        if last_exception:
            raise last_exception
    
    def process_full_text(
        self, 
        pdf_path: str | Path,
        consolidate_references: bool = True,
        include_raw_references: bool = True,
        tei_coordinates: bool = False,
        consolidate_header: bool = False,
        segment_sentences: bool = False
    ) -> str:
        """Extract full document structure from PDF using GROBID processFulltextDocument service.
        
        Args:
            pdf_path: Path to the PDF file
            consolidate_references: Enable reference consolidation with external services
            include_raw_references: Include raw reference strings in output
            tei_coordinates: Include coordinates in TEI output
            consolidate_header: Consolidate header information
            segment_sentences: Segment sentences in the output
            
        Returns:
            XML string containing full document in TEI format
            
        Raises:
            GrobidError: If GROBID service fails or returns an error
            FileNotFoundError: If PDF file doesn't exist
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        url = f"{self.endpoint}/api/processFulltextDocument"
        
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                with open(pdf_path, 'rb') as pdf_file:
                    files = {'input': (pdf_path.name, pdf_file, 'application/pdf')}
                    data = {
                        'consolidateReferences': '1' if consolidate_references else '0',
                        'includeRawReferences': '1' if include_raw_references else '0',
                        'teiCoordinates': '1' if tei_coordinates else '0',
                        'consolidateHeader': '1' if consolidate_header else '0',
                        'segmentSentences': '1' if segment_sentences else '0',
                    }
                    
                    response = self.session.post(
                        url, 
                        files=files, 
                        data=data, 
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        xml_content = response.text
                        if not xml_content.strip():
                            raise GrobidError("GROBID returned empty response")
                        return xml_content
                    else:
                        error_msg = f"GROBID service error (HTTP {response.status_code}): {response.text}"
                        raise GrobidError(error_msg)
                        
            except (requests.RequestException, GrobidError) as e:
                last_exception = e
                attempt_info = f"attempt {attempt + 1}/{self.max_retries + 1}"
                
                if isinstance(e, requests.Timeout):
                    logging.warning(f"GROBID timeout on {attempt_info}: {e}")
                elif isinstance(e, GrobidError):
                    logging.warning(f"GROBID service error on {attempt_info}: {e}")
                else:
                    logging.warning(f"GROBID request error on {attempt_info}: {type(e).__name__}: {e}")
                
                if attempt < self.max_retries:
                    wait_time = min(2 ** attempt, 10)
                    logging.info(f"Retrying GROBID request in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"All {self.max_retries + 1} GROBID attempts failed")
                    break
        
        if last_exception:
            raise last_exception
    
    def health_check(self) -> bool:
        """Check if GROBID service is available and responding.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/api/isalive", 
                timeout=10.0
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close session."""
        self.session.close()
