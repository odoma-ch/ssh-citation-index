#!/usr/bin/env python3
"""
Example script demonstrating GROBID integration with citation-index.

This script shows how to:
1. Use GROBID client directly
2. Use GROBID extractor through the factory
3. Use GROBID via CLI commands

Prerequisites:
- GROBID server running (e.g., via Docker: docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0)
- PDF file to process
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import citation_index
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citation_index import GrobidClient, GrobidExtractor, ExtractorFactory


def example_grobid_client():
    """Example using GROBID client directly."""
    print("=== GROBID Client Example ===")
    
    # Initialize client
    client = GrobidClient(endpoint="https://grobid-graphia-app1-staging.apps.bst2.paas.psnc.pl/", timeout=120.0)
    
    # Check if GROBID service is available
    if not client.health_check():
        print("❌ GROBID service is not available at https://grobid-graphia-app1-staging.apps.bst2.paas.psnc.pl/")
        print("   Please start GROBID server first:")
        print("   docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0")
        return
    
    print("✅ GROBID service is available")
    
    # Process a PDF (replace with your PDF path)
    pdf_path = "/Users/alex/docs/code/Odoma/citation_index/benchmarks/cex/all_pdfs/PSY_100.pdf"
    if not Path(pdf_path).exists():
        print(f"❌ PDF file not found: {pdf_path}")
        print("   Please provide a valid PDF file path")
        return
    
    try:
        # Extract references only
        xml_content = client.process_references(pdf_path)
        print(f"✅ Extracted references XML ({len(xml_content)} characters)")
        
        # Save XML to file
        output_path = Path(pdf_path).stem + "_grobid_references.xml"
        Path(output_path).write_text(xml_content, encoding='utf-8')
        print(f"💾 Saved references XML to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")


def example_grobid_extractor():
    """Example using GROBID extractor."""
    print("\n=== GROBID Extractor Example ===")
    
    # Initialize extractor
    extractor = GrobidExtractor(endpoint="https://grobid-graphia-app1-staging.apps.bst2.paas.psnc.pl/")
    
    # Or create via factory
    extractor = ExtractorFactory.create("grobid", grobid_endpoint="https://grobid-graphia-app1-staging.apps.bst2.paas.psnc.pl/")
    
    pdf_path = "/Users/alex/docs/code/Odoma/citation_index/benchmarks/cex/all_pdfs/PSY_100.pdf"
    if not Path(pdf_path).exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    try:
        # Extract references as structured objects
        references = extractor.extract_references_only(pdf_path)
        print(f"✅ Extracted {len(references)} structured references")
        
        # Print first few references
        for i, ref in enumerate(references[:3]):
            # Get the best available title from the reference
            title = (ref.full_title or ref.analytic_title or ref.monographic_title or 
                    ref.journal_title or ref.raw_reference or 'No title')
            print(f"  {i+1}. {title}...")
        
        if len(references) > 3:
            print(f"  ... and {len(references) - 3} more references")
            
    except Exception as e:
        print(f"❌ Error extracting references: {e}")

  

if __name__ == "__main__":
    print("🔬 GROBID Integration Examples")
    print("=" * 40)
    
    # Run examples
    example_grobid_client()
    example_grobid_extractor()
    
