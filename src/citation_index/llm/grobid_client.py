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
                    elif response.status_code == 204:
                        # Process completed but no content could be extracted - don't retry
                        raise GrobidError(f"GROBID could not extract content (HTTP 204): Process completed but no content could be extracted and structured")
                    elif response.status_code == 500:
                        # Server error - cannot process this document - don't retry
                        raise GrobidError(f"GROBID cannot process document (HTTP 500): Server error, document cannot be processed")
                    else:
                        error_msg = f"GROBID service error (HTTP {response.status_code}): {response.text}"
                        raise GrobidError(error_msg)
                        
            except (requests.RequestException, GrobidError) as e:
                last_exception = e
                attempt_info = f"attempt {attempt + 1}/{self.max_retries + 1}"
                
                # Check if this is a non-retryable error
                is_non_retryable = (
                    isinstance(e, GrobidError) and 
                    ("HTTP 204" in str(e) or "HTTP 500" in str(e))
                )
                
                if isinstance(e, requests.Timeout):
                    logging.warning(f"GROBID timeout on {attempt_info}: {e}")
                elif isinstance(e, GrobidError):
                    if is_non_retryable:
                        logging.error(f"GROBID non-retryable error: {e}")
                        break  # Don't retry for 204/500 errors
                    else:
                        logging.warning(f"GROBID service error on {attempt_info}: {e}")
                else:
                    logging.warning(f"GROBID request error on {attempt_info}: {type(e).__name__}: {e}")
                
                if attempt < self.max_retries and not is_non_retryable:
                    # Use fixed retry delays: 10s, 30s, 60s
                    retry_delays = [10, 30, 60]
                    wait_time = retry_delays[min(attempt, len(retry_delays) - 1)]
                    logging.info(f"Retrying GROBID request in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    if not is_non_retryable:
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
                    elif response.status_code == 204:
                        # Process completed but no content could be extracted - don't retry
                        raise GrobidError(f"GROBID could not extract content (HTTP 204): Process completed but no content could be extracted and structured")
                    elif response.status_code == 500:
                        # Server error - cannot process this document - don't retry
                        raise GrobidError(f"GROBID cannot process document (HTTP 500): Server error, document cannot be processed")
                    else:
                        error_msg = f"GROBID service error (HTTP {response.status_code}): {response.text}"
                        raise GrobidError(error_msg)
                        
            except (requests.RequestException, GrobidError) as e:
                last_exception = e
                attempt_info = f"attempt {attempt + 1}/{self.max_retries + 1}"
                
                # Check if this is a non-retryable error
                is_non_retryable = (
                    isinstance(e, GrobidError) and 
                    ("HTTP 204" in str(e) or "HTTP 500" in str(e))
                )
                
                if isinstance(e, requests.Timeout):
                    logging.warning(f"GROBID timeout on {attempt_info}: {e}")
                elif isinstance(e, GrobidError):
                    if is_non_retryable:
                        logging.error(f"GROBID non-retryable error: {e}")
                        break  # Don't retry for 204/500 errors
                    else:
                        logging.warning(f"GROBID service error on {attempt_info}: {e}")
                else:
                    logging.warning(f"GROBID request error on {attempt_info}: {type(e).__name__}: {e}")
                
                if attempt < self.max_retries and not is_non_retryable:
                    # Use fixed retry delays: 10s, 30s, 60s
                    retry_delays = [10, 30, 60]
                    wait_time = retry_delays[min(attempt, len(retry_delays) - 1)]
                    logging.info(f"Retrying GROBID request in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    if not is_non_retryable:
                        logging.error(f"All {self.max_retries + 1} GROBID attempts failed")
                    break
        
        if last_exception:
            raise last_exception
    
    def health_check(self, max_retries: int = 2) -> bool:
        """Check if GROBID service is available and responding with retry logic.
        
        Args:
            max_retries: Maximum number of retry attempts (default: 2)
        
        Returns:
            True if service is healthy, False otherwise
        """
        retry_delays = [30, 60]  # 30s for first retry, 60s for second retry
        
        for attempt in range(max_retries + 1):
            try:
                response = self.session.get(
                    f"{self.endpoint}/api/isalive", 
                    timeout=10.0
                )
                if response.status_code == 200:
                    if attempt > 0:
                        logging.info(f"GROBID health check succeeded on attempt {attempt + 1}")
                    return True
                else:
                    logging.warning(f"GROBID health check failed with status {response.status_code} on attempt {attempt + 1}")
                    
            except requests.RequestException as e:
                logging.warning(f"GROBID health check request failed on attempt {attempt + 1}: {type(e).__name__}: {e}")
            
            # If not the last attempt, wait before retrying
            if attempt < max_retries:
                delay = retry_delays[attempt]
                logging.info(f"Retrying GROBID health check in {delay} seconds...")
                time.sleep(delay)
        
        logging.error(f"GROBID health check failed after {max_retries + 1} attempts")
        return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close session."""
        self.session.close()
