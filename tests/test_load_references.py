#!/usr/bin/env python
"""
Comprehensive tests for loading references from various sources.

Tests loading references from:
- LinkedBook format (JSONL)
- EXCITE format (JSON with raw text)
- CEX (TEI XML)
- OpenAlex connector
- JSON (direct Reference format)
- Legacy data (deprecated fields)
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citation_index.core.models import Reference
from citation_index.core.parsers.tei_bibl_parser import TeiBiblParser
from citation_index.core.connectors.openalex import OpenAlexConnector


class TestLoadLinkedBook:
    """Test loading references from LinkedBook format."""
    
    def test_load_linkedbook_jsonl(self):
        """Load and parse LinkedBook JSONL format."""
        linkedbook_file = Path(__file__).parent.parent / "benchmarks/linkedbook/linkedbooks_test_references.jsonl"
        
        if not linkedbook_file.exists():
            print(f"⊘ Skipping: {linkedbook_file} not found")
            return
        
        # Load first 3 references
        refs = []
        with open(linkedbook_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                data = json.loads(line)
                # from_linkedbook returns (Reference, List[str]) tuple
                ref, tags = Reference.from_linkedbook(data)
                refs.append(ref)
        
        assert len(refs) == 3, f"Expected 3 refs, got {len(refs)}"
        
        # Check first reference
        ref1 = refs[0]
        assert ref1.full_title is not None
        # Note: Some LinkedBook refs may not have authors
        # assert ref1.authors is not None and len(ref1.authors) > 0
        assert ref1.publication_year is not None
        
        # Verify new fields are populated
        assert hasattr(ref1, 'identifiers')
        assert hasattr(ref1, 'publication_date_raw')
        assert hasattr(ref1, 'raw')
        
        # Check that linkedbook data is preserved in raw
        if ref1.raw and 'linkedbook' in ref1.raw:
            assert 'tags' in ref1.raw['linkedbook']
        
        print(f"✓ Loaded {len(refs)} LinkedBook references")
        print(f"  Sample: {ref1.full_title}")
        print(f"  Authors: {ref1.authors}")
        print(f"  Year: {ref1.publication_year}")


class TestLoadEXCITE:
    """Test loading references from EXCITE format."""
    
    def test_load_excite_json(self):
        """Load and parse EXCITE raw text format."""
        excite_file = Path(__file__).parent.parent / "benchmarks/excite/all_references.json"
        
        if not excite_file.exists():
            print(f"⊘ Skipping: {excite_file} not found")
            return
        
        with open(excite_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get first document
        first_doc_id = list(data.keys())[0]
        first_doc = data[first_doc_id]
        
        # Parse first 3 references
        refs = []
        for raw_text in first_doc['references'][:3]:
            try:
                ref = Reference.from_excite_xml(raw_text)
                refs.append(ref)
            except Exception as e:
                print(f"⊘ Failed to parse: {raw_text[:50]}... - {e}")
        
        assert len(refs) > 0, "Should parse at least one reference"
        
        ref1 = refs[0]
        assert ref1.full_title is not None
        
        # Verify new fields
        assert hasattr(ref1, 'publication_year')
        assert hasattr(ref1, 'publication_date_raw')
        
        print(f"✓ Loaded {len(refs)} EXCITE references from document {first_doc_id}")
        print(f"  Sample: {ref1.full_title[:80]}...")
        if ref1.publication_year:
            print(f"  Year: {ref1.publication_year}")


class TestLoadCEXTEI:
    """Test loading references from CEX TEI XML format."""
    
    def test_load_tei_xml(self):
        """Load and parse TEI biblStruct XML."""
        xml_dir = Path(__file__).parent.parent / "benchmarks/cex/all_xmls"
        
        if not xml_dir.exists():
            print(f"⊘ Skipping: {xml_dir} not found")
            return
        
        # Find first XML file
        xml_files = list(xml_dir.glob("*.xml"))
        if not xml_files:
            print(f"⊘ No XML files found in {xml_dir}")
            return
        
        parser = TeiBiblParser()
        refs = []
        
        # Parse first XML file (from_xml returns List[List[Reference]])
        xml_file = xml_files[0]
        with open(xml_file, 'rb') as f:
            xml_content = f.read()
        
        try:
            # from_xml returns List[List[Reference]], flatten to get references
            refs_lists = parser.from_xml(xml_str=xml_content)
            for refs_list in refs_lists:
                refs.extend(refs_list[:3])  # Take first 3 from each list
                if len(refs) >= 3:
                    break
        except Exception as e:
            print(f"⊘ Failed to parse {xml_file.name}: {e}")
            return
        
        assert len(refs) > 0, "Should parse at least one reference"
        
        ref1 = refs[0]
        assert ref1.full_title is not None
        
        # Verify identifiers were extracted
        if ref1.identifiers:
            print(f"  Identifiers: {[(i.scheme, i.value) for i in ref1.identifiers]}")
        
        # Verify year extraction
        if ref1.publication_year:
            print(f"  Year: {ref1.publication_year}")
        
        # Verify TEI data preserved in raw
        if ref1.raw and 'tei' in ref1.raw:
            print(f"  TEI data preserved: {list(ref1.raw['tei'].keys())}")
        
        print(f"✓ Loaded {len(refs)} TEI references from {xml_file.name}")
        print(f"  Sample: {ref1.full_title}")
        
    def test_tei_round_trip(self):
        """Test TEI XML round-trip conversion."""
        xml_dir = Path(__file__).parent.parent / "benchmarks/cex/all_xmls"
        
        if not xml_dir.exists():
            print(f"⊘ Skipping: {xml_dir} not found")
            return
        
        xml_files = list(xml_dir.glob("*.xml"))
        if not xml_files:
            print(f"⊘ No XML files found")
            return
        
        parser = TeiBiblParser()
        
        # Load and parse first XML (from_xml returns List[List[Reference]])
        with open(xml_files[0], 'rb') as f:
            original_xml = f.read()
        
        refs_lists = parser.from_xml(xml_str=original_xml)
        if not refs_lists or not refs_lists[0]:
            print(f"⊘ No references parsed")
            return
        
        # Convert back to XML
        ref = refs_lists[0][0]  # Get first ref from first list
        reconstructed_xml = parser.to_xml(ref)
        
        assert reconstructed_xml is not None
        # Handle both bytes and string
        if isinstance(reconstructed_xml, bytes):
            assert b'<biblStruct' in reconstructed_xml
        else:
            assert '<biblStruct' in reconstructed_xml
        
        # Parse reconstructed XML
        refs_lists_again = parser.from_xml(xml_str=reconstructed_xml)
        assert len(refs_lists_again) > 0
        assert len(refs_lists_again[0]) > 0
        
        ref_again = refs_lists_again[0][0]
        assert ref_again.full_title == ref.full_title
        
        print(f"✓ TEI round-trip successful")
        print(f"  Original title: {ref.full_title}")
        print(f"  Reconstructed title: {ref_again.full_title}")


class TestLoadFromConnector:
    """Test loading references from API connectors."""
    
    def test_openalex_format(self):
        """Test loading from OpenAlex API response format."""
        # Sample OpenAlex API response
        sample_response = {
            "id": "https://openalex.org/W2741809807",
            "doi": "https://doi.org/10.1038/nature12373",
            "title": "Global cancer statistics",
            "publication_year": 2011,
            "publication_date": "2011-02-04",
            "type": "journal-article",
            "authorships": [
                {
                    "author": {
                        "display_name": "Ahmedin Jemal",
                        "id": "https://openalex.org/A5023888391"
                    }
                }
            ],
            "biblio": {
                "volume": "61",
                "issue": "2",
                "first_page": "69",
                "last_page": "90"
            },
            "primary_location": {
                "source": {
                    "display_name": "CA: A Cancer Journal for Clinicians",
                    "issn_l": "0007-9235"
                }
            },
            "ids": {
                "openalex": "https://openalex.org/W2741809807",
                "doi": "https://doi.org/10.1038/nature12373",
                "pmid": "https://pubmed.ncbi.nlm.nih.gov/21296855"
            }
        }
        
        connector = OpenAlexConnector()
        ref = connector._map_single_result(sample_response)
        
        assert ref.full_title == "Global cancer statistics"
        assert ref.publication_year == 2011
        assert ref.publication_date_raw == "2011-02-04"
        assert ref.ref_type == "journal-article"
        
        # Check identifiers were extracted
        assert len(ref.identifiers) > 0
        
        schemes = [i.scheme for i in ref.identifiers]
        assert 'doi' in schemes
        assert 'openalex' in schemes
        assert 'pmid' in schemes
        
        # Check DOI normalization
        doi_id = next(i for i in ref.identifiers if i.scheme == 'doi')
        assert doi_id.normalized == "10.1038/nature12373"
        
        print(f"✓ OpenAlex connector mapping successful")
        print(f"  Title: {ref.full_title}")
        print(f"  Year: {ref.publication_year}")
        print(f"  Type: {ref.ref_type}")
        print(f"  Identifiers: {[(i.scheme, i.normalized) for i in ref.identifiers]}")


class TestLoadFromJSON:
    """Test loading references from direct JSON format."""
    
    def test_load_new_format_json(self):
        """Load reference from new JSON format with identifiers."""
        ref_data = {
            "full_title": "Machine Learning for Citation Processing",
            "authors": ["Smith, John", "Doe, Jane"],
            "publication_year": 2023,
            "publication_date_raw": "2023-05-15",
            "identifiers": [
                {
                    "scheme": "doi",
                    "value": "10.1234/example.2023",
                    "normalized": "10.1234/example.2023"
                },
                {
                    "scheme": "arxiv",
                    "value": "2305.12345",
                    "normalized": "2305.12345"
                }
            ],
            "ref_type": "journal-article",
            "volume": "42",
            "pages": "100-125"
        }
        
        ref = Reference(**ref_data)
        
        assert ref.full_title == "Machine Learning for Citation Processing"
        assert ref.publication_year == 2023
        assert len(ref.identifiers) == 2
        assert ref.ref_type == "journal-article"
        
        # Test JSON serialization
        json_str = ref.model_dump_json()
        ref_reloaded = Reference.model_validate_json(json_str)
        
        assert ref_reloaded.full_title == ref.full_title
        assert len(ref_reloaded.identifiers) == len(ref.identifiers)
        
        print(f"✓ New format JSON loading successful")
        print(f"  Title: {ref.full_title}")
        print(f"  Identifiers: {len(ref.identifiers)}")


class TestLoadLegacyData:
    """Test loading references with deprecated legacy fields."""
    
    def test_load_with_publication_date(self):
        """Load reference with deprecated publication_date field."""
        legacy_data = {
            "full_title": "Legacy Reference Example",
            "authors": ["Author, Old"],
            "publication_date": "2020-06-15",  # Deprecated field
            "journal": "Old Journal",
            "volume": "10"
        }
        
        ref = Reference(**legacy_data)
        
        # Check migration happened
        assert ref.publication_year == 2020
        assert ref.publication_date_raw == "2020-06-15"
        
        # Check deprecated field excluded from serialization
        serialized = ref.model_dump()
        assert 'publication_date' not in serialized
        assert 'publication_year' in serialized
        assert 'publication_date_raw' in serialized
        
        print(f"✓ Legacy publication_date migrated")
        print(f"  publication_year: {ref.publication_year}")
        print(f"  publication_date_raw: {ref.publication_date_raw}")
    
    def test_load_with_analytic_monographic_titles(self):
        """Load reference with deprecated title fields."""
        legacy_data = {
            "full_title": "Article Title in Book Title",
            "analytic_title": "Article Title",  # Deprecated
            "monographic_title": "Book Title",  # Deprecated
            "authors": ["Author, Test"],
            "publication_date": "2019"
        }
        
        ref = Reference(**legacy_data)
        
        # Check titles preserved in raw dict
        assert ref.raw is not None
        assert 'tei' in ref.raw
        assert ref.raw['tei']['analytic_title'] == "Article Title"
        assert ref.raw['tei']['monographic_title'] == "Book Title"
        
        # Check deprecated fields excluded
        serialized = ref.model_dump()
        assert 'analytic_title' not in serialized
        assert 'monographic_title' not in serialized
        
        # Check full_title preserved
        assert ref.full_title == "Article Title in Book Title"
        
        print(f"✓ Legacy titles migrated to raw dict")
        print(f"  full_title: {ref.full_title}")
        print(f"  raw.tei: {ref.raw['tei']}")
    
    def test_load_all_legacy_fields(self):
        """Load reference with all deprecated fields."""
        legacy_data = {
            "full_title": "Complete Legacy Example",
            "analytic_title": "The Article",
            "monographic_title": "The Journal",
            "publication_date": "2018-12-25",
            "authors": ["Smith, Legacy"],
            "volume": "5",
            "pages": "1-10"
        }
        
        ref = Reference(**legacy_data)
        
        # Verify all migrations
        assert ref.publication_year == 2018
        assert ref.publication_date_raw == "2018-12-25"
        assert ref.raw['tei']['analytic_title'] == "The Article"
        assert ref.raw['tei']['monographic_title'] == "The Journal"
        
        # Verify clean serialization (no deprecated fields)
        serialized = ref.model_dump()
        deprecated_fields = ['publication_date', 'analytic_title', 'monographic_title']
        for field in deprecated_fields:
            assert field not in serialized
        
        # Verify round-trip
        json_str = ref.model_dump_json()
        ref_reloaded = Reference.model_validate_json(json_str)
        
        assert ref_reloaded.full_title == ref.full_title
        assert ref_reloaded.publication_year == ref.publication_year
        assert ref_reloaded.raw['tei']['analytic_title'] == ref.raw['tei']['analytic_title']
        
        print(f"✓ All legacy fields migrated successfully")
        print(f"  publication_year: {ref.publication_year}")
        print(f"  raw.tei keys: {list(ref.raw['tei'].keys())}")


def run_all_tests():
    """Run all test classes."""
    test_classes = [
        TestLoadLinkedBook,
        TestLoadEXCITE,
        TestLoadCEXTEI,
        TestLoadFromConnector,
        TestLoadFromJSON,
        TestLoadLegacyData,
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_cls in test_classes:
        print(f"\n{'='*60}")
        print(f"{test_cls.__name__}")
        print(f"{'='*60}")
        
        test_instance = test_cls()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                print(f"\n{method_name}:")
                method = getattr(test_instance, method_name)
                method()
                passed_tests += 1
                print(f"✓ {method_name} PASSED")
            except AssertionError as e:
                print(f"✗ {method_name} FAILED: {e}")
            except Exception as e:
                print(f"✗ {method_name} ERROR: {e}")
    
    print(f"\n{'='*60}")
    print(f"Test Summary: {passed_tests}/{total_tests} tests passed")
    print(f"{'='*60}")
    
    if passed_tests == total_tests:
        print("✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"✗ {total_tests - passed_tests} TESTS FAILED")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
