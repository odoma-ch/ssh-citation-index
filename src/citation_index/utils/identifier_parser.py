"""Identifier parsing utilities for citation processing."""

import re
from typing import Optional
from pydantic import BaseModel


class Identifier(BaseModel):
    """A bibliographic identifier with scheme, value, and normalized form.
    
    Examples:
        - DOI: scheme="doi", value="10.1234/abc", normalized="10.1234/abc"
        - ISBN: scheme="isbn", value="978-0-123-45678-9", normalized="9780123456789"
        - arXiv: scheme="arxiv", value="1234.5678", normalized="1234.5678"
    """
    
    scheme: str  # "doi", "isbn", "issn", "url", "pmid", "arxiv", "wikidata", etc.
    value: str
    normalized: Optional[str] = None


def parse_identifier(text: str, type_attr: Optional[str] = None) -> Optional[Identifier]:
    """Parse an identifier from text with optional type attribute.
    
    Handles both:
    - Typed: <idno type="doi">10.1234/abc</idno> (type_attr="doi", text="10.1234/abc")
    - Inline: <idno>DOI: 10.1234/abc</idno> (type_attr=None, text="DOI: 10.1234/abc")
    
    Args:
        text: The identifier text content
        type_attr: Optional type/scheme attribute from XML
        
    Returns:
        Identifier object or None if parsing fails
    """
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # If we have a type attribute, use it directly
    if type_attr:
        scheme = type_attr.lower().strip()
        value = text
        normalized = _normalize_identifier(scheme, value)
        return Identifier(scheme=scheme, value=value, normalized=normalized)
    
    # Try to parse inline format "SCHEME: value" or "SCHEME value"
    # Common patterns: "DOI: 10.1234", "ISBN: 978-...", "arXiv:1234.5678"
    inline_match = re.match(r"^([a-zA-Z][\w-]*)\s*[:\s]\s*(.+)$", text)
    if inline_match:
        scheme = inline_match.group(1).lower().strip()
        value = inline_match.group(2).strip()
        normalized = _normalize_identifier(scheme, value)
        return Identifier(scheme=scheme, value=value, normalized=normalized)
    
    # Try to detect identifier type from content
    detected_scheme = _detect_identifier_type(text)
    if detected_scheme:
        normalized = _normalize_identifier(detected_scheme, text)
        return Identifier(scheme=detected_scheme, value=text, normalized=normalized)
    
    # If all else fails, treat as generic identifier
    return None


def _detect_identifier_type(text: str) -> Optional[str]:
    """Detect identifier type from text content.
    
    Returns the identifier scheme if detected, None otherwise.
    """
    text = text.strip()
    text_lower = text.lower()
    
    # DOI: starts with "10." and contains "/"
    if text.startswith("10.") and "/" in text:
        return "doi"
    
    # arXiv: matches pattern like "1234.5678" or "math/0601001" or "2101.12345"
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", text) or re.match(r"^[a-z-]+/\d{7}$", text):
        return "arxiv"
    
    # PMID: "PMID" followed by digits or just digits (8-9 digits typical)
    if text_lower.startswith("pmid") or (text.isdigit() and 6 <= len(text) <= 10):
        if text_lower.startswith("pmid"):
            return "pmid"
    
    # ISBN: contains ISBN-like pattern (with or without hyphens)
    isbn_clean = re.sub(r"[^0-9X]", "", text.upper())
    if len(isbn_clean) in [10, 13]:
        return "isbn"
    
    # ISSN: ISSN-like pattern (8 digits with hyphen: 1234-5678)
    if re.match(r"^\d{4}-?\d{3}[\dxX]$", text):
        return "issn"
    
    # Wikidata QID: Q followed by digits
    if re.match(r"^Q\d+$", text.upper()):
        return "wikidata"
    
    # URL: starts with http:// or https://
    if text_lower.startswith(("http://", "https://")):
        return "url"
    
    return None


def _normalize_identifier(scheme: str, value: str) -> Optional[str]:
    """Normalize an identifier value based on its scheme.
    
    Args:
        scheme: The identifier scheme (doi, isbn, etc.)
        value: The raw identifier value
        
    Returns:
        Normalized value or None if normalization not applicable
    """
    scheme = scheme.lower()
    value = value.strip()
    
    if scheme == "doi":
        return _normalize_doi(value)
    elif scheme == "isbn":
        return _normalize_isbn(value)
    elif scheme == "issn":
        return _normalize_issn(value)
    elif scheme == "pmid":
        return _normalize_pmid(value)
    elif scheme == "arxiv":
        return _normalize_arxiv(value)
    elif scheme == "wikidata":
        return value.upper()  # QIDs are uppercase
    elif scheme == "url":
        return value  # URLs kept as-is
    else:
        # For unknown schemes, return the value as-is
        return value


def _normalize_doi(doi: str) -> str:
    """Normalize a DOI by removing common prefixes."""
    clean = doi.strip()
    prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "http://dx.doi.org/",
        "https://dx.doi.org/",
        "doi:",
        "DOI:",
    ]
    for prefix in prefixes:
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break
        elif clean.lower().startswith(prefix.lower()):
            clean = clean[len(prefix):]
            break
    return clean.strip()


def _normalize_isbn(isbn: str) -> str:
    """Normalize an ISBN by removing all non-alphanumeric characters."""
    return re.sub(r"[^0-9X]", "", isbn.upper())


def _normalize_issn(issn: str) -> str:
    """Normalize an ISSN to standard format with hyphen: 1234-5678."""
    clean = re.sub(r"[^0-9X]", "", issn.upper())
    if len(clean) == 8:
        return f"{clean[:4]}-{clean[4:]}"
    return clean


def _normalize_pmid(pmid: str) -> str:
    """Normalize a PMID to just the numeric portion."""
    if pmid.lower().startswith("pmid"):
        pmid = pmid[4:].strip()
    # Remove any non-digit characters
    return re.sub(r"\D", "", pmid)


def _normalize_arxiv(arxiv: str) -> str:
    """Normalize an arXiv ID."""
    # Remove "arXiv:" prefix if present
    if arxiv.lower().startswith("arxiv:"):
        arxiv = arxiv[6:].strip()
    return arxiv
