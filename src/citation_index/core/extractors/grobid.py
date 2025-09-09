"""
GROBID-based PDF reference extractor.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict

from lxml import etree
from .base import BaseExtractor, ExtractResult
from ...llm.grobid_client import GrobidClient, GrobidError
from ..parsers.tei_bibl_parser import TeiBiblParser


class GrobidExtractor(BaseExtractor):
    """PDF reference extractor using GROBID backend."""
    
    def __init__(self, endpoint: str = "http://localhost:8070", timeout: float = 180.0, max_retries: int = 3):
        """Initialize the GROBID extractor.
        
        Args:
            endpoint: GROBID server endpoint
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.grobid_client = GrobidClient(
            endpoint=endpoint, 
            timeout=timeout, 
            max_retries=max_retries
        )
        self.tei_parser = TeiBiblParser()
    
    def extract(
        self, 
        filepath: str, 
        save_dir: str = None, 
        mode: str = "references",
        parse_references: bool = True,
        consolidate_references: bool = True,
        include_raw_references: bool = True,
        tei_coordinates: bool = False,
        consolidate_header: bool = False,
        segment_sentences: bool = False,
        **kwargs
    ) -> ExtractResult:
        """Extract content using GROBID.
        
        Args:
            filepath: Path to the PDF file
            save_dir: Optional directory to save extracted content
            mode: Extraction mode - 'references' (references only) or 'fulltext' (full document)
            parse_references: Whether to parse references into structured format
            consolidate_references: Enable reference consolidation with external services
            include_raw_references: Include raw reference strings in output
            tei_coordinates: Include coordinates in TEI output (fulltext mode only)
            consolidate_header: Consolidate header information (fulltext mode only)
            segment_sentences: Segment sentences in the output (fulltext mode only)
            **kwargs: Additional extraction parameters
            
        Returns:
            ExtractResult containing the extracted references and metadata
            
        Raises:
            RuntimeError: If extraction fails
            FileNotFoundError: If PDF file doesn't exist
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"PDF file not found: {filepath}")
        
        base_name = filepath.stem
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Check GROBID service health
            if not self.grobid_client.health_check():
                raise RuntimeError(f"GROBID service is not available at {self.grobid_client.endpoint}")
            
            # Extract using appropriate GROBID service
            if mode == "references":
                xml_content = self.grobid_client.process_references(
                    filepath,
                    consolidate_references=consolidate_references,
                    include_raw_references=include_raw_references
                )
            elif mode == "fulltext":
                xml_content = self.grobid_client.process_full_text(
                    filepath,
                    consolidate_references=consolidate_references,
                    include_raw_references=include_raw_references,
                    tei_coordinates=tei_coordinates,
                    consolidate_header=consolidate_header,
                    segment_sentences=segment_sentences
                )
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'references' or 'fulltext'")
            
            # Save raw XML if requested
            if save_dir:
                xml_file = save_dir / f"{base_name}_grobid.xml"
                xml_file.write_text(xml_content, encoding='utf-8')
                logging.info(f"Saved GROBID XML to: {xml_file}")
            
            # Parse references if requested
            references = None
            if parse_references:
                try:
                    references = self._extract_references_from_xml(xml_content)
                    
                    # Save parsed references if requested
                    if save_dir and references:
                        json_file = save_dir / f"{base_name}_grobid_references.json"
                        import json
                        with open(json_file, 'w', encoding='utf-8') as f:
                            json.dump([ref.model_dump() for ref in references], f, indent=2, ensure_ascii=False)
                        logging.info(f"Saved parsed references to: {json_file}")
                
                except Exception as e:
                    logging.warning(f"Failed to parse references from GROBID XML: {e}")
                    references = None
            
            # Prepare metadata
            metadata = {
                'extractor': 'grobid',
                'mode': mode,
                'endpoint': self.grobid_client.endpoint,
                'references_parsed': references is not None,
                'reference_count': len(references) if references else 0,
            }
            
            # Return XML content as text, with references in metadata
            return ExtractResult(
                text=xml_content,
                metadata=metadata,
                images=None  # GROBID doesn't extract images in this context
            )
            
        except GrobidError as e:
            raise RuntimeError(f"GROBID extraction failed: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"GROBID extraction failed: {str(e)}") from e
    
    def _extract_references_from_xml(self, xml_content: str) -> list:
        """Extract structured references from GROBID XML.
        
        Args:
            xml_content: TEI XML content from GROBID
            
        Returns:
            List of Reference objects
        """
        try:
            # Parse XML
            root = etree.fromstring(xml_content.encode('utf-8'))
            
            # Find bibliography section using TEI parser's namespace handling
            # Use the TEI parser's from_xml method for better compatibility
            references_lists = self.tei_parser.from_xml(xml_str=xml_content)
            
            # Flatten the list of lists into a single list
            references = []
            for ref_list in references_lists:
                references.extend(ref_list)
            
            if not references:
                # Fallback: try direct parsing if from_xml doesn't work
                logging.info("No references found via TEI parser, trying direct parsing...")
                references = self._fallback_parse_references(root)
            
            logging.info(f"Extracted {len(references)} references from GROBID XML")
            return references
            
        except etree.XMLSyntaxError as e:
            raise ValueError(f"Invalid XML from GROBID: {e}")
        except Exception as e:
            logging.warning(f"TEI parser failed: {e}, trying fallback parsing")
            try:
                root = etree.fromstring(xml_content.encode('utf-8'))
                return self._fallback_parse_references(root)
            except Exception as fallback_e:
                raise RuntimeError(f"Failed to parse GROBID XML: {e}, fallback also failed: {fallback_e}")
    
    def _fallback_parse_references(self, root: etree._Element) -> list:
        """Fallback method to parse references when TEI parser fails."""
        references = []
        namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
        
        # Try different XPath expressions for finding references
        bibl_elements = []
        
        # Method 1: Look for listBibl (bibliography list)
        list_bibl = root.xpath('//tei:listBibl', namespaces=namespaces)
        if list_bibl:
            for lb in list_bibl:
                # Use TEI parser for each listBibl element
                try:
                    refs = self.tei_parser.to_references(lb, raise_empty_error=False)
                    references.extend(refs)
                except Exception as e:
                    logging.warning(f"Failed to parse listBibl: {e}")
                    # Fallback to individual biblStruct elements
                    bibl_elements.extend(lb.xpath('.//tei:biblStruct', namespaces=namespaces))
        
        # Method 2: Look for biblStruct directly
        if not references and not bibl_elements:
            bibl_elements = root.xpath('//tei:biblStruct', namespaces=namespaces)
        
        # Parse individual biblStruct elements
        for bibl_elem in bibl_elements:
            try:
                refs = self.tei_parser.to_references(bibl_elem, raise_empty_error=False)
                references.extend(refs)
            except Exception as e:
                logging.warning(f"Failed to parse biblStruct element: {e}")
                # Create minimal reference from raw text
                try:
                    raw_text = etree.tostring(bibl_elem, method='text', encoding='unicode').strip()
                    if raw_text:
                        from ..models.reference import Reference
                        ref = Reference(raw_reference=raw_text)
                        references.append(ref)
                except Exception:
                    continue
        
        # Method 3: Look for bibl elements (less structured) as last resort
        if not references:
            bibl_elements = root.xpath('//tei:bibl', namespaces=namespaces)
            for bibl_elem in bibl_elements:
                try:
                    raw_text = etree.tostring(bibl_elem, method='text', encoding='unicode').strip()
                    if raw_text:
                        from ..models.reference import Reference
                        ref = Reference(raw_reference=raw_text)
                        references.append(ref)
                except Exception:
                    continue
        
        return references
    
    def extract_references_only(
        self, 
        filepath: str, 
        save_dir: str = None, 
        consolidate_references: bool = True,
        include_raw_references: bool = True,
        **kwargs
    ) -> list:
        """Convenience method to extract only references as structured objects.
        
        Args:
            filepath: Path to the PDF file
            save_dir: Optional directory to save extracted content
            consolidate_references: Enable reference consolidation with external services
            include_raw_references: Include raw reference strings in output
            **kwargs: Additional extraction parameters
            
        Returns:
            List of Reference objects
        """
        result = self.extract(
            filepath, 
            save_dir=save_dir, 
            mode="references", 
            parse_references=True,
            consolidate_references=consolidate_references,
            include_raw_references=include_raw_references,
            **kwargs
        )
        
        if result.metadata and result.metadata.get('references_parsed'):
            # References are stored in the XML, need to re-parse
            return self._extract_references_from_xml(result.text)
        else:
            return []
