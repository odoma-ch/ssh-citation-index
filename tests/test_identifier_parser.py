#!/usr/bin/env python
"""Comprehensive tests for identifier parser."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from citation_index.utils.identifier_parser import (
    Identifier,
    parse_identifier,
    _detect_identifier_type,
    _normalize_doi,
    _normalize_isbn,
    _normalize_issn,
    _normalize_pmid,
    _normalize_arxiv,
)


class TestIdentifierModel:
    """Test the Identifier Pydantic model."""
    
    def test_create_identifier(self):
        """Test creating an Identifier."""
        id_obj = Identifier(scheme="doi", value="10.1234/abc", normalized="10.1234/abc")
        assert id_obj.scheme == "doi"
        assert id_obj.value == "10.1234/abc"
        assert id_obj.normalized == "10.1234/abc"
    
    def test_identifier_optional_normalized(self):
        """Test that normalized is optional."""
        id_obj = Identifier(scheme="custom", value="test123")
        assert id_obj.normalized is None


class TestParseIdentifier:
    """Test the parse_identifier function."""
    
    def test_parse_typed_doi(self):
        """Test parsing with type attribute."""
        result = parse_identifier("10.1234/test", "doi")
        assert result is not None
        assert result.scheme == "doi"
        assert result.value == "10.1234/test"
        assert result.normalized == "10.1234/test"
    
    def test_parse_inline_doi(self):
        """Test parsing DOI: prefix."""
        result = parse_identifier("DOI: 10.1234/test", None)
        assert result is not None
        assert result.scheme == "doi"
        assert result.value == "10.1234/test"
        assert result.normalized == "10.1234/test"
    
    def test_parse_inline_isbn(self):
        """Test parsing ISBN: prefix."""
        result = parse_identifier("ISBN: 978-0-123-45678-9", None)
        assert result is not None
        assert result.scheme == "isbn"
        assert result.normalized == "9780123456789"
    
    def test_parse_inline_arxiv(self):
        """Test parsing arXiv: prefix."""
        result = parse_identifier("arXiv:1234.5678", None)
        assert result is not None
        assert result.scheme == "arxiv"
        assert result.value == "1234.5678"
    
    def test_parse_auto_detect_doi(self):
        """Test auto-detection of DOI."""
        result = parse_identifier("10.1000/xyz123", None)
        assert result is not None
        assert result.scheme == "doi"
    
    def test_parse_auto_detect_wikidata(self):
        """Test auto-detection of Wikidata QID."""
        result = parse_identifier("Q12345", None)
        assert result is not None
        assert result.scheme == "wikidata"
        assert result.value == "Q12345"
    
    def test_parse_empty_string(self):
        """Test that empty string returns None."""
        result = parse_identifier("", None)
        assert result is None
    
    def test_parse_whitespace(self):
        """Test that whitespace returns None."""
        result = parse_identifier("   ", None)
        assert result is None


class TestDetectIdentifierType:
    """Test identifier type detection."""
    
    def test_detect_doi(self):
        """Test DOI detection."""
        assert _detect_identifier_type("10.1234/abc") == "doi"
        assert _detect_identifier_type("10.5555/test.2020.001") == "doi"
    
    def test_detect_arxiv(self):
        """Test arXiv detection."""
        assert _detect_identifier_type("1234.5678") == "arxiv"
        assert _detect_identifier_type("2101.12345") == "arxiv"
        assert _detect_identifier_type("math/0601001") == "arxiv"
    
    def test_detect_isbn(self):
        """Test ISBN detection."""
        assert _detect_identifier_type("978-0-123-45678-9") == "isbn"
        assert _detect_identifier_type("9780123456789") == "isbn"
        assert _detect_identifier_type("0-123-45678-X") == "isbn"
    
    def test_detect_issn(self):
        """Test ISSN detection."""
        assert _detect_identifier_type("1234-5678") == "issn"
        assert _detect_identifier_type("12345678") == "issn"
    
    def test_detect_wikidata(self):
        """Test Wikidata QID detection."""
        assert _detect_identifier_type("Q12345") == "wikidata"
        assert _detect_identifier_type("q98765") == "wikidata"
    
    def test_detect_url(self):
        """Test URL detection."""
        assert _detect_identifier_type("https://example.com/paper") == "url"
        assert _detect_identifier_type("http://doi.org/10.1234/abc") == "url"
    
    def test_detect_unknown(self):
        """Test that unknown formats return None."""
        assert _detect_identifier_type("random text") is None
        assert _detect_identifier_type("123") is None


class TestNormalizationFunctions:
    """Test identifier normalization functions."""
    
    def test_normalize_doi(self):
        """Test DOI normalization."""
        assert _normalize_doi("https://doi.org/10.1234/abc") == "10.1234/abc"
        assert _normalize_doi("http://doi.org/10.1234/abc") == "10.1234/abc"
        assert _normalize_doi("http://dx.doi.org/10.1234/abc") == "10.1234/abc"
        assert _normalize_doi("doi:10.1234/abc") == "10.1234/abc"
        assert _normalize_doi("DOI:10.1234/abc") == "10.1234/abc"
        assert _normalize_doi("10.1234/abc") == "10.1234/abc"
    
    def test_normalize_isbn(self):
        """Test ISBN normalization."""
        assert _normalize_isbn("978-0-123-45678-9") == "9780123456789"
        assert _normalize_isbn("0-123-45678-X") == "012345678X"
        assert _normalize_isbn("ISBN 978-0-123-45678-9") == "9780123456789"
    
    def test_normalize_issn(self):
        """Test ISSN normalization."""
        assert _normalize_issn("1234-5678") == "1234-5678"
        assert _normalize_issn("12345678") == "1234-5678"
        assert _normalize_issn("ISSN 1234-5678") == "1234-5678"
    
    def test_normalize_pmid(self):
        """Test PMID normalization."""
        assert _normalize_pmid("12345678") == "12345678"
        assert _normalize_pmid("PMID 12345678") == "12345678"
        assert _normalize_pmid("pmid:12345678") == "12345678"
    
    def test_normalize_arxiv(self):
        """Test arXiv normalization."""
        assert _normalize_arxiv("arXiv:1234.5678") == "1234.5678"
        assert _normalize_arxiv("1234.5678") == "1234.5678"
        assert _normalize_arxiv("ARXIV:1234.5678") == "1234.5678"


if __name__ == "__main__":
    # Run tests without pytest
    import sys
    
    test_classes = [
        TestIdentifierModel,
        TestParseIdentifier,
        TestDetectIdentifierType,
        TestNormalizationFunctions,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                passed_tests += 1
                print(f"✓ {test_class.__name__}.{method_name}")
            except AssertionError as e:
                failed_tests.append(f"{test_class.__name__}.{method_name}: {e}")
                print(f"✗ {test_class.__name__}.{method_name}: {e}")
            except Exception as e:
                failed_tests.append(f"{test_class.__name__}.{method_name}: {e}")
                print(f"✗ {test_class.__name__}.{method_name}: {e}")
    
    print(f"\n{passed_tests}/{total_tests} tests passed")
    
    if failed_tests:
        print("\nFailed tests:")
        for failure in failed_tests:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print("\n✓ ALL TESTS PASSED!")
        sys.exit(0)
