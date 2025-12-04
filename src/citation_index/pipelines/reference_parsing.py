"""Pipelines for parsing reference strings into structured References using an LLM or GROBID."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from citation_index.core.models import References
from citation_index.llm.client import LLMClient
from citation_index.llm.grobid_client import GrobidClient
from citation_index.llm.prompt_loader import ReferenceParsingPrompt
from citation_index.utils.json_helper import safe_json_parse


def parse_reference_strings(
    reference_lines: List[str],
    llm_client: LLMClient,
    prompt_name: str = "prompts/reference_parsing.md",
    include_schema: bool = True,
    temperature: float = 0.0,
    use_streaming: bool = True,
) -> References:
    """Parse a list of reference strings into structured References via LLM."""
    text = "\n".join(reference_lines)
    prompt_obj = ReferenceParsingPrompt(
        prompt=prompt_name, input_text=text, include_json_schema=include_schema
    )
    response = llm_client.call(
        prompt_obj.prompt,
        json_output=True,
        json_schema=prompt_obj.json_schema,
        temperature=temperature,
        use_streaming=use_streaming,
    )
    parsed = safe_json_parse(response)
    if isinstance(parsed, list):
        data = parsed
    elif isinstance(parsed, dict):
        data = parsed.get("references") or parsed.get("parsed_references") or parsed.get("refs")
        if data is None:
            data = [parsed]
    else:
        data = []
    return References.from_dict(data) if data else References(references=[])


def parse_reference_file(
    path: str | Path,
    llm_client: LLMClient,
    prompt_name: str = "prompts/reference_parsing.md",
    include_schema: bool = True,
    temperature: float = 0.0,
) -> References:
    """Parse a text file with one reference per line into structured References."""
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    return parse_reference_strings(
        lines,
        llm_client=llm_client,
        prompt_name=prompt_name,
        include_schema=include_schema,
        temperature=temperature,
    )


def parse_reference_strings_grobid(
    reference_lines: List[str],
    grobid_client: GrobidClient,
    include_raw: bool = False,
    batch_mode: bool = True,
) -> References:
    """Parse a list of reference strings into structured References using GROBID.
    
    This provides a GROBID-based alternative to the LLM-based parsing in parse_reference_strings().
    GROBID is free, open-source, and typically faster than LLM calls, making it suitable for
    high-volume parsing tasks.
    
    Args:
        reference_lines: List of raw reference strings to parse
        grobid_client: GrobidClient instance connected to a GROBID server
        include_raw: Include raw reference strings in the TEI XML output
        batch_mode: If True, use processCitationList for better performance (default).
                   If False, process citations one by one.
    
    Returns:
        References object containing parsed Reference instances
        
    Raises:
        GrobidError: If GROBID service fails
        
    Example:
        >>> from citation_index.llm.grobid_client import GrobidClient
        >>> client = GrobidClient(endpoint="http://localhost:8070")
        >>> refs = ["Smith, J. (2020). Article. Journal, 10, 1-5."]
        >>> result = parse_reference_strings_grobid(refs, client)
    """
    if not reference_lines:
        return References(references=[])
    
    # Filter out empty lines
    reference_lines = [line.strip() for line in reference_lines if line and line.strip()]
    if not reference_lines:
        return References(references=[])
    
    # Use batch processing for better performance
    if batch_mode:
        xml_content = grobid_client.process_citation_list(
            reference_lines,
            include_raw_citations=include_raw
        )
    else:
        # Process individually and concatenate (slower, but more fault-tolerant)
        xml_parts = []
        for line in reference_lines:
            try:
                # Process as single-item list
                xml_part = grobid_client.process_citation_list(
                    [line],
                    include_raw_citations=include_raw
                )
                xml_parts.append(xml_part)
            except Exception as e:
                # Log and continue on error
                import logging
                logging.warning(f"Failed to parse citation '{line[:50]}...': {e}")
                continue
        
        # Combine individual biblStruct elements into a single TEI document
        if xml_parts:
            xml_content = _combine_bibl_structs(xml_parts)
        else:
            return References(references=[])
    
    # Parse TEI XML into References using existing parser
    try:
        references = References.from_xml(xml_str=xml_content)
        return references
    except Exception as e:
        import logging
        logging.error(f"Failed to parse GROBID XML output: {e}")
        return References(references=[])


def parse_reference_file_grobid(
    path: str | Path,
    grobid_client: GrobidClient,
    include_raw: bool = False,
    batch_mode: bool = True,
) -> References:
    """Parse a text file with one reference per line into structured References using GROBID.
    
    Args:
        path: Path to text file with one reference per line
        grobid_client: GrobidClient instance connected to a GROBID server
        include_raw: Include raw reference strings in output
        batch_mode: Use batch processing for better performance
        
    Returns:
        References object containing parsed Reference instances
    """
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    return parse_reference_strings_grobid(
        lines,
        grobid_client=grobid_client,
        include_raw=include_raw,
        batch_mode=batch_mode,
    )


def _combine_bibl_structs(xml_parts: List[str]) -> str:
    """Combine multiple biblStruct XML fragments into a single TEI document.
    
    Args:
        xml_parts: List of XML strings, each containing a biblStruct or TEI fragment
        
    Returns:
        Combined TEI XML document with all biblStructs in a listBibl
    """
    from lxml import etree
    
    # Create TEI document with listBibl
    nsmap = {None: "http://www.tei-c.org/ns/1.0"}
    tei = etree.Element("TEI", nsmap=nsmap)
    list_bibl = etree.SubElement(tei, "listBibl")
    
    # Parse and extract biblStruct elements from each part
    for xml_part in xml_parts:
        try:
            # Parse the XML part
            root = etree.fromstring(xml_part.encode('utf-8'))
            
            # Find biblStruct elements
            ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
            bibl_structs = root.xpath('.//tei:biblStruct | .//biblStruct', namespaces=ns)
            
            # If no biblStruct found, check if root itself is biblStruct
            if not bibl_structs:
                if etree.QName(root).localname == 'biblStruct':
                    bibl_structs = [root]
            
            # Add each biblStruct to our listBibl
            for bibl_struct in bibl_structs:
                list_bibl.append(bibl_struct)
                
        except Exception as e:
            import logging
            logging.warning(f"Failed to parse XML fragment: {e}")
            continue
    
    # Convert to string
    xml_bytes = etree.tostring(tei, encoding='utf-8', xml_declaration=True)
    return xml_bytes.decode('utf-8')


