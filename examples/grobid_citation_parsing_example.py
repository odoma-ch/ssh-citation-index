#!/usr/bin/env python3
"""
Example script demonstrating GROBID citation parsing functionality.

This script shows how to:
1. Parse single citations using GROBID
2. Parse multiple citations in batch
3. Get TEI XML or BibTeX output
4. Use consolidation with external services
5. Parse citations into structured Reference objects
6. Use GROBID as an alternative to LLM-based parsing

Prerequisites:
- GROBID server running (e.g., via Docker: docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0)

Usage:
    python examples/grobid_citation_parsing_example.py
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import citation_index
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citation_index.llm.grobid_client import GrobidClient, GrobidError
from citation_index.core.extractors.grobid import GrobidExtractor
from citation_index.pipelines.reference_parsing import (
    parse_reference_strings_grobid,
    parse_reference_file_grobid
)


def example_1_basic_citation_parsing():
    """Example 1: Basic citation parsing with TEI XML output."""
    print("\n=== Example 1: Basic Citation Parsing (TEI XML) ===")
    
    client = GrobidClient(endpoint="http://localhost:8070")
    
    # Check if GROBID service is available
    if not client.health_check():
        print("❌ GROBID service is not available at http://localhost:8070")
        print("   Please start GROBID server first:")
        print("   docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0")
        return
    
    print("✅ GROBID service is available")
    
    # Parse a single citation
    citation = "Graff, Expert. Opin. Ther. Targets (2002) 6(1): 103-113"
    
    try:
        xml_result = client.process_citation(citation)
        print(f"\n📝 Input citation:\n{citation}")
        print(f"\n📄 TEI XML output:\n{xml_result[:500]}...")  # Show first 500 chars
        
    except GrobidError as e:
        print(f"❌ Error: {e}")


def example_2_bibtex_output():
    """Example 2: Parse citation and get BibTeX output."""
    print("\n=== Example 2: BibTeX Output ===")
    
    client = GrobidClient(endpoint="http://localhost:8070")
    
    if not client.health_check():
        print("❌ GROBID service not available")
        return
    
    citation = "Smith, J., & Doe, A. (2020). Machine Learning Applications. Nature, 580(7803), 245-251."
    
    try:
        bibtex_result = client.process_citation(citation, output_format="bibtex")
        print(f"\n📝 Input citation:\n{citation}")
        print(f"\n📚 BibTeX output:\n{bibtex_result}")
        
    except GrobidError as e:
        print(f"❌ Error: {e}")


def example_3_batch_parsing():
    """Example 3: Parse multiple citations in batch."""
    print("\n=== Example 3: Batch Citation Parsing ===")
    
    client = GrobidClient(endpoint="http://localhost:8070")
    
    if not client.health_check():
        print("❌ GROBID service not available")
        return
    
    citations = [
        "Smith, J. (2020). Article Title. Journal Name, 10(2), 45-67.",
        "Doe, A. (2019). Book Title. New York: Publisher.",
        "Brown, C., & Green, D. (2021). Another Article. Science, 123, 456-789."
    ]
    
    try:
        xml_result = client.process_citation_list(citations)
        print(f"\n📝 Input: {len(citations)} citations")
        print(f"\n📄 TEI XML output (first 800 chars):\n{xml_result[:800]}...")
        
    except GrobidError as e:
        print(f"❌ Error: {e}")


def example_4_consolidation():
    """Example 4: Use consolidation to add metadata from external services."""
    print("\n=== Example 4: Citation Consolidation ===")
    
    client = GrobidClient(endpoint="http://localhost:8070")
    
    if not client.health_check():
        print("❌ GROBID service not available")
        return
    
    # Well-known citation that should have metadata in CrossRef/other services
    citation = "LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444."
    
    try:
        # Without consolidation
        xml_no_consolidate = client.process_citation(citation, consolidate_citations=0)
        print(f"\n📝 Citation:\n{citation}")
        print(f"\n📄 Without consolidation (first 400 chars):\n{xml_no_consolidate[:400]}...")
        
        # With full consolidation (slower, but adds DOI, URLs, etc.)
        xml_consolidate = client.process_citation(citation, consolidate_citations=1)
        print(f"\n📄 With consolidation (first 400 chars):\n{xml_consolidate[:400]}...")
        
        # DOI-only consolidation (medium speed)
        xml_doi_only = client.process_citation(citation, consolidate_citations=2)
        print(f"\n📄 With DOI-only consolidation (first 400 chars):\n{xml_doi_only[:400]}...")
        
    except GrobidError as e:
        print(f"❌ Error: {e}")


def example_5_structured_references():
    """Example 5: Parse citations into structured Reference objects."""
    print("\n=== Example 5: Structured Reference Objects ===")
    
    extractor = GrobidExtractor(endpoint="http://localhost:8070")
    
    if not extractor.grobid_client.health_check():
        print("❌ GROBID service not available")
        return
    
    citations = [
        "Smith, J. (2020). Machine Learning Basics. AI Journal, 15(3), 123-145.",
        "Doe, A., & Brown, C. (2019). Deep Neural Networks. Tech Press.",
        "Green, D. (2021). Data Science Methods. Science, 456, 789-801."
    ]
    
    try:
        # Parse citations into Reference objects
        references = extractor.parse_citations_to_references(citations)
        
        print(f"\n✅ Parsed {len(references)} references:\n")
        for i, ref in enumerate(references, 1):
            print(f"{i}. Title: {ref.full_title}")
            print(f"   Journal: {ref.journal_title or 'N/A'}")
            print(f"   Authors: {len(ref.authors) if ref.authors else 0} author(s)")
            if ref.authors and len(ref.authors) > 0:
                author = ref.authors[0]
                if hasattr(author, 'surname'):
                    print(f"   First author: {author.surname}")
            print(f"   Year: {ref.publication_date_raw or ref.publication_year or 'N/A'}")
            print(f"   Volume: {ref.volume or 'N/A'}")
            print(f"   Pages: {ref.pages or 'N/A'}")
            print()
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_6_pipeline_integration():
    """Example 6: Use GROBID in reference parsing pipeline."""
    print("\n=== Example 6: Pipeline Integration ===")
    
    client = GrobidClient(endpoint="http://localhost:8070")
    
    if not client.health_check():
        print("❌ GROBID service not available")
        return
    
    # Sample reference strings (as might be extracted from a PDF)
    reference_lines = [
        "1. Smith, J. (2020). Article Title. Journal Name, 10, 45-67.",
        "2. Doe, A. (2019). Book Title. Publisher.",
        "3. Brown, C., & Green, D. (2021). Another Article. Science, 123, 456-789.",
        "4. White, E. (2018). Conference Paper. In Proc. ICML, pp. 100-110."
    ]
    
    try:
        # Use GROBID-based pipeline (alternative to LLM-based parsing)
        references = parse_reference_strings_grobid(
            reference_lines,
            grobid_client=client,
            consolidate=0,  # 0=no consolidation, 1=full, 2=DOI only
            include_raw=False,
            batch_mode=True  # Use batch processing for better performance
        )
        
        print(f"\n✅ Parsed {len(references)} references using GROBID pipeline:\n")
        for i, ref in enumerate(references, 1):
            print(f"{i}. {ref.full_title or 'No title'}")
            if ref.publication_date_raw or ref.publication_year:
                print(f"   Year: {ref.publication_date_raw or ref.publication_year}")
            print()
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_7_error_handling():
    """Example 7: Demonstrate error handling."""
    print("\n=== Example 7: Error Handling ===")
    
    client = GrobidClient(endpoint="http://localhost:8070")
    
    if not client.health_check():
        print("❌ GROBID service not available")
        return
    
    # Test various error conditions
    test_cases = [
        ("Empty string", ""),
        ("Whitespace only", "   "),
        ("Invalid consolidation", None),  # Will use special handling
    ]
    
    for description, citation in test_cases:
        print(f"\n🧪 Test: {description}")
        try:
            if description == "Invalid consolidation":
                # Test invalid consolidation value
                client.process_citation("Some citation", consolidate_citations=99)
            else:
                client.process_citation(citation)
            print("   ✅ No error (unexpected)")
        except ValueError as e:
            print(f"   ✅ Caught ValueError: {e}")
        except GrobidError as e:
            print(f"   ✅ Caught GrobidError: {e}")


def example_8_performance_comparison():
    """Example 8: Compare individual vs batch processing."""
    print("\n=== Example 8: Performance Comparison ===")
    
    import time
    
    client = GrobidClient(endpoint="http://localhost:8070")
    
    if not client.health_check():
        print("❌ GROBID service not available")
        return
    
    citations = [
        "Smith, J. (2020). Article 1. Journal, 10, 1-10.",
        "Doe, A. (2019). Article 2. Science, 20, 20-30.",
        "Brown, C. (2021). Article 3. Nature, 30, 30-40.",
    ]
    
    try:
        # Method 1: Individual processing
        start = time.time()
        for citation in citations:
            client.process_citation(citation)
        individual_time = time.time() - start
        print(f"\n⏱️  Individual processing: {individual_time:.2f}s for {len(citations)} citations")
        
        # Method 2: Batch processing
        start = time.time()
        client.process_citation_list(citations)
        batch_time = time.time() - start
        print(f"⏱️  Batch processing: {batch_time:.2f}s for {len(citations)} citations")
        
        speedup = individual_time / batch_time if batch_time > 0 else 0
        print(f"\n🚀 Speedup: {speedup:.1f}x faster with batch processing")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("GROBID Citation Parsing Examples")
    print("=" * 60)
    
    examples = [
        ("Basic Citation Parsing", example_1_basic_citation_parsing),
        ("BibTeX Output", example_2_bibtex_output),
        ("Batch Parsing", example_3_batch_parsing),
        ("Citation Consolidation", example_4_consolidation),
        ("Structured References", example_5_structured_references),
        ("Pipeline Integration", example_6_pipeline_integration),
        ("Error Handling", example_7_error_handling),
        ("Performance Comparison", example_8_performance_comparison),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...\n")
    
    for name, example_func in examples:
        try:
            example_func()
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Example '{name}' failed with error: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

