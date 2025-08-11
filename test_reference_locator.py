#!/usr/bin/env python3
"""
Unified test script to evaluate the reference locator on both CEX and EXCITE datasets.

This script:
1. Loads dataset data (PDFs and ground truth references)
2. Extracts text from PDFs using different extractors
3. Uses the reference locator to find references sections
4. Saves results to JSON for analysis (no evaluation, just raw results)

Supports both CEX and EXCITE datasets with automatic detection or explicit selection.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import sys
import os

# Add the source directory to Python path
sys.path.append('src')
sys.path.append('benchmarks/cex')
sys.path.append('benchmarks/excite')

from citation_index.core.extractors import ExtractorFactory
from citation_index.core.segmenters.references_locator import (
    extract_all_reference_sections
)


def load_dataset_data(dataset: str) -> Tuple[object, Dict]:
    """
    Load data for the specified dataset.
    
    Args:
        dataset: 'cex' or 'excite'
    
    Returns:
        Tuple of (pdf_df, papers_data)
    """
    if dataset.lower() == 'cex':
        try:
            from cex_helper import load_cex_data
            print("Loading CEX data...")
            return load_cex_data()
        except ImportError:
            raise ImportError("CEX helper not found. Make sure you're in the correct directory.")
    elif dataset.lower() == 'excite':
        try:
            from excite_helper import load_excite_data
            print("Loading EXCITE data...")
            return load_excite_data()
        except ImportError:
            raise ImportError("EXCITE helper not found. Make sure you're in the correct directory.")
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'cex' or 'excite'")


def get_dataset_specific_info(dataset: str, row: object) -> Dict[str, Any]:
    """
    Extract dataset-specific information from a row.
    
    Args:
        dataset: 'cex' or 'excite'
        row: DataFrame row
    
    Returns:
        Dictionary with dataset-specific fields
    """
    if dataset.lower() == 'cex':
        return {
            "category": row.get("category", "Unknown"),
            "class": None,
            "lang": None
        }
    elif dataset.lower() == 'excite':
        return {
            "category": None,
            "class": row.get("class", "Unknown"),
            "lang": row.get("lang", "Unknown")
        }
    else:
        return {"category": None, "class": None, "lang": None}


def get_markdown_dir(dataset: str) -> Path:
    """Get the appropriate markdown directory for the dataset."""
    if dataset.lower() == 'cex':
        return Path("benchmarks/cex/all_markdown")
    elif dataset.lower() == 'excite':
        return Path("benchmarks/excite/all_markdown")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def test_reference_locator(
    dataset: str,
    limit: int = None,
    extractors: List[str] = ["pymupdf", "marker", "mineru"],
    output_file: str = None,
    force_extract: bool = False
) -> Dict[str, Any]:
    """
    Test the reference locator on specified dataset.
    
    Args:
        dataset: 'cex' or 'excite'
        limit: Maximum number of documents to process (None for all)
        extractors: List of extractors to test
        output_file: JSON file to save results (auto-generated if None)
        force_extract: Force re-extraction even if markdown files exist
    
    Returns:
        Dictionary with test results
    """
    # Load dataset data
    pdf_df, papers_data = load_dataset_data(dataset)
    
    if limit:
        pdf_df = pdf_df.head(limit)
        print(f"Limited to {limit} documents for testing")
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = f"reference_locator_{dataset.lower()}_test_results.json"
    
    results = {
        "metadata": {
            "dataset": dataset.lower(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_documents": len(pdf_df),
            "extractors_tested": extractors
        },
        "documents": {}
    }
    
    # Get markdown directory for this dataset
    markdown_dir = get_markdown_dir(dataset)
    markdown_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each document
    for _, row in tqdm(pdf_df.iterrows(), total=len(pdf_df), desc=f"Testing reference locator on {dataset.upper()}"):
        file_id = str(row["file_id"])
        file_path = row["file_path"]
        
        # Get dataset-specific information
        dataset_info = get_dataset_specific_info(dataset, row)
        
        # Get ground truth references count
        gt_references = papers_data.get(file_id, {}).get("references", [])
        gt_ref_count = len(gt_references)
        
        doc_results = {
            "file_id": file_id,
            "file_path": file_path,
            "ground_truth_ref_count": gt_ref_count,
            "extractors": {}
        }
        
        # Add dataset-specific fields
        doc_results.update(dataset_info)
        
        # Test each extractor
        for extractor_name in extractors:
            # print(f"Processing {file_id} with {extractor_name}...")
            
            extractor_result = {
                "success": False,
                "error": None,
                "reference_sections": []
            }
            
            try:
                # Check if markdown file exists for reuse
                markdown_path = markdown_dir / f"{file_id}_{extractor_name}.md"
                
                if markdown_path.exists() and not force_extract:
                    # print(f"  Loading existing markdown: {markdown_path}")
                    with open(markdown_path, "r", encoding="utf-8") as md_file:
                        full_text = md_file.read()
                else:
                    # Extract text from PDF
                    # print(f"  Extracting text from PDF with {extractor_name}...")
                    extractor = ExtractorFactory.create(extractor_name)
                    extraction_result = extractor.extract(file_path)
                    
                    if not extraction_result.text.strip():
                        extractor_result["error"] = "No text extracted"
                        doc_results["extractors"][extractor_name] = extractor_result
                        continue
                    
                    full_text = extraction_result.text
                    
                    # Save extracted text to markdown file for reuse
                    with open(markdown_path, "w", encoding="utf-8") as md_file:
                        md_file.write(full_text)
                    # print(f"  Saved markdown to: {markdown_path}")
                
                extractor_result["success"] = True
                
                # Find reference sections - this is the main focus now
                ref_sections = extract_all_reference_sections(full_text,prefer_tokens=False)
                
                # Process each reference section
                for i, section in enumerate(ref_sections):
                    section_info = {
                        "title": section.get("title", ""),
                        "level": section.get("level", 0),
                        "method": section.get("method", ""),
                        "start_line": section.get("start_line", 0),
                        "end_line": section.get("end_line", 0),
                        "text": section.get("text", "")[:2000]  # First 2000 chars of section text
                    }
                    extractor_result["reference_sections"].append(section_info)
                
            except Exception as e:
                extractor_result["error"] = str(e)
                print(f"Error processing {file_id} with {extractor_name}: {e}")
            
            doc_results["extractors"][extractor_name] = extractor_result
        
        results["documents"][file_id] = doc_results
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary statistics
    print_summary_statistics(dataset, results)
    
    return results


def print_summary_statistics(dataset: str, results: Dict[str, Any]):
    """Print summary statistics from the test results."""
    documents = results["documents"]
    extractors = results["metadata"]["extractors_tested"]
    
    print(f"\n{'='*60}")
    print(f"{dataset.upper()} REFERENCE LOCATOR SUMMARY")
    print(f"{'='*60}")
    print(f"Total documents: {len(documents)}")
    print(f"Extractors tested: {', '.join(extractors)}")
    
    # Statistics by extractor
    for extractor in extractors:
        print(f"\n--- {extractor.upper()} ---")
        
        successful = 0
        docs_with_refs = 0
        total_ref_sections = 0
        total_gt_refs = 0
        
        # Stats by dataset-specific categories
        stats_by_category = {}
        
        for doc_id, doc_data in documents.items():
            extractor_data = doc_data["extractors"].get(extractor, {})
            
            # Create category key based on dataset
            if dataset.lower() == 'cex':
                category = doc_data.get("category", "Unknown")
                key = f"category_{category}"
            elif dataset.lower() == 'excite':
                class_num = doc_data.get("class", "Unknown")
                lang = doc_data.get("lang", "Unknown")
                key = f"class_{class_num}_{lang}"
            else:
                key = "unknown"
            
            if key not in stats_by_category:
                stats_by_category[key] = {
                    "successful": 0, "docs_with_refs": 0, 
                    "total_ref_sections": 0, "total_gt_refs": 0, "total_docs": 0
                }
            
            stats_by_category[key]["total_docs"] += 1
            
            if extractor_data.get("success", False):
                successful += 1
                stats_by_category[key]["successful"] += 1
                
                ref_sections = extractor_data.get("reference_sections", [])
                if ref_sections:
                    docs_with_refs += 1
                    total_ref_sections += len(ref_sections)
                    stats_by_category[key]["docs_with_refs"] += 1
                    stats_by_category[key]["total_ref_sections"] += len(ref_sections)
            
            gt_count = doc_data.get("ground_truth_ref_count", 0)
            total_gt_refs += gt_count
            stats_by_category[key]["total_gt_refs"] += gt_count
        
        print(f"Successful extractions: {successful}/{len(documents)} ({100*successful/len(documents):.1f}%)")
        print(f"Found reference sections: {docs_with_refs}/{successful} ({100*docs_with_refs/max(1,successful):.1f}%)")
        print(f"Total reference sections found: {total_ref_sections}")
        print(f"Ground truth references: {total_gt_refs}")
        if docs_with_refs > 0:
            print(f"Avg sections per doc: {total_ref_sections/docs_with_refs:.1f}")
        
        # Print stats by category
        if len(stats_by_category) > 1:  # Only show breakdown if there are multiple categories
            category_label = "category" if dataset.lower() == 'cex' else "class and language"
            print(f"\nBreakdown by {category_label}:")
            for key, stats in stats_by_category.items():
                if stats["total_docs"] > 0:
                    success_rate = 100 * stats["successful"] / stats["total_docs"]
                    ref_found_rate = 100 * stats["docs_with_refs"] / max(1, stats["successful"])
                    print(f"  {key}: {stats['successful']}/{stats['total_docs']} successful ({success_rate:.1f}%), "
                          f"{stats['docs_with_refs']} with refs ({ref_found_rate:.1f}%), "
                          f"{stats['total_ref_sections']} sections, {stats['total_gt_refs']} GT refs")


def detect_available_datasets() -> List[str]:
    """Detect which datasets are available based on directory structure."""
    available = []
    
    if Path("benchmarks/cex").exists():
        available.append("cex")
    
    if Path("benchmarks/excite").exists():
        available.append("excite")
    
    return available


if __name__ == "__main__":
    import argparse
    
    # Detect available datasets
    available_datasets = detect_available_datasets()
    
    parser = argparse.ArgumentParser(description="Test reference locator on CEX and/or EXCITE datasets")
    parser.add_argument("--dataset", type=str, choices=["cex", "excite", "both"], default="both",
                      help="Dataset to test: 'cex', 'excite', or 'both' (default: cex)")
    parser.add_argument("--limit", type=int, default=None, 
                      help="Limit number of documents to process per dataset (default: None)")
    parser.add_argument("--extractors", nargs="+", default=["pymupdf", "marker"],
                      choices=["pymupdf", "marker", "mineru"],
                      help="Extractors to test (default: pymupdf, marker)")
    parser.add_argument("--output-prefix", type=str, default="reference_locator",
                      help="Prefix for output JSON files (default: reference_locator)")
    parser.add_argument("--force-extract", action="store_true",
                      help="Force re-extraction even if markdown files exist")
    
    args = parser.parse_args()
    
    # Determine which datasets to test
    datasets_to_test = []
    if args.dataset == "both":
        datasets_to_test = available_datasets
    else:
        if args.dataset in available_datasets:
            datasets_to_test = [args.dataset]
        else:
            print(f"Error: Dataset '{args.dataset}' not found.")
            print(f"Available datasets: {', '.join(available_datasets)}")
            sys.exit(1)
    
    if not datasets_to_test:
        print("No datasets found. Please ensure benchmarks/cex and/or benchmarks/excite directories exist.")
        sys.exit(1)
    
    print(f"Testing datasets: {', '.join(datasets_to_test)}")
    
    # Run tests for each dataset
    all_results = {}
    for dataset in datasets_to_test:
        output_file = f"{args.output_prefix}_{dataset}_test_results.json"
        
        try:
            results = test_reference_locator(
                dataset=dataset,
                limit=args.limit,
                extractors=args.extractors,
                output_file=output_file,
                force_extract=args.force_extract
            )
            all_results[dataset] = results
            print(f"\n{dataset.upper()} test completed! Results saved to {output_file}")
        except Exception as e:
            print(f"\nError testing {dataset}: {e}")
    
    print(f"\nAll tests completed!")
    if len(datasets_to_test) > 1:
        print("Summary files generated:")
        for dataset in datasets_to_test:
            if dataset in all_results:
                print(f"  - {args.output_prefix}_{dataset}_test_results.json")
