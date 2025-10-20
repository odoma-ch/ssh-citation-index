#!/usr/bin/env python3
"""Build test set by sampling references from CEX, EXCITE, and LinkedBook datasets.

This script samples references from three datasets and creates a unified test set
with both original strings and parsed reference data.
"""

import argparse
import csv
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add parent directory to path to import citation_index
sys.path.insert(0, str(Path(__file__).parent.parent))

from citation_index.core.models import References

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def sample_cex_references(
    pdf_info_file: str,
    references_file: str,
    num_categories: int = 10
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Sample references from CEX dataset.
    
    Args:
        pdf_info_file: Path to pdf_files_info.csv
        references_file: Path to all_references.json
        num_categories: Number of categories to sample from (default: 10)
    
    Returns:
        Tuple of (list of sampled references, metadata dict)
    """
    logger.info(f"Sampling from CEX dataset: {num_categories} categories")
    
    # Load PDF metadata
    pdf_metadata = {}
    with open(pdf_info_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdf_metadata[row['file_id']] = row
    
    # Group by category
    categories = {}
    for file_id, meta in pdf_metadata.items():
        category = meta.get('category', '')
        if category not in categories:
            categories[category] = []
        categories[category].append(file_id)
    
    # Sample num_categories categories
    available_categories = list(categories.keys())
    sampled_categories = random.sample(available_categories, min(num_categories, len(available_categories)))
    
    # Sample one PDF from each category
    sampled_file_ids = []
    for category in sampled_categories:
        file_ids = categories[category]
        sampled_file_ids.append(random.choice(file_ids))
    
    logger.info(f"Sampled categories: {sampled_categories}")
    logger.info(f"Sampled PDFs: {sampled_file_ids}")
    
    # Load all references
    with open(references_file, 'r', encoding='utf-8') as f:
        all_references = json.load(f)
    
    # Extract references from sampled PDFs
    sampled_refs = []
    xml_dir = Path("benchmarks/cex/all_xmls")
    
    for file_id in sampled_file_ids:
        if file_id not in all_references:
            logger.warning(f"No references found for {file_id}")
            continue
        
        file_data = all_references[file_id]
        category = pdf_metadata.get(file_id, {}).get('category', 'unknown')
        ref_strings = file_data.get('references', [])
        
        # Load parsed references from XML
        xml_path = xml_dir / f"{file_id}.xml"
        parsed_refs = []
        if xml_path.exists():
            try:
                refs_obj = References.from_xml(file_path=str(xml_path))
                parsed_refs = [ref.model_dump() for ref in refs_obj]
                logger.debug(f"Loaded {len(parsed_refs)} parsed refs from XML for {file_id}")
            except Exception as e:
                logger.warning(f"Failed to parse XML for {file_id}: {e}")
        
        for idx, ref_str in enumerate(ref_strings):
            ref_entry = {
                "ref_id": f"cex_{file_id}_{idx}",
                "source": "cex",
                "file_id": file_id,
                "category": category,
                "original_string": ref_str,
            }
            
            # Add parsed data if available
            if idx < len(parsed_refs):
                ref_entry["parsed"] = parsed_refs[idx]
            
            sampled_refs.append(ref_entry)
    
    metadata = {
        "categories": sampled_categories,
        "file_ids": sampled_file_ids,
        "total_refs": len(sampled_refs),
        "refs_with_parsed": sum(1 for r in sampled_refs if "parsed" in r)
    }
    
    logger.info(f"Sampled {len(sampled_refs)} references from CEX")
    logger.info(f"  {metadata['refs_with_parsed']} have parsed data")
    return sampled_refs, metadata


def sample_excite_references(
    pdf_info_file: str,
    references_file: str,
    sample_rate_de: float = 0.05,
    sample_rate_en: float = 0.03
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Sample references from EXCITE dataset.
    
    Args:
        pdf_info_file: Path to pdf_files_info.csv
        references_file: Path to all_references.json
        sample_rate_de: Percentage to sample from German PDFs (default: 0.05 = 5%)
        sample_rate_en: Percentage to sample from English PDFs (default: 0.03 = 3%)
    
    Returns:
        Tuple of (list of sampled references, metadata dict)
    """
    logger.info(f"Sampling from EXCITE dataset: {sample_rate_de*100}% German, {sample_rate_en*100}% English")
    
    # Load PDF metadata
    pdf_metadata = {}
    with open(pdf_info_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdf_metadata[row['file_id']] = row
    
    # Group by language
    languages = {}
    for file_id, meta in pdf_metadata.items():
        lang = meta.get('lang', '')
        if lang not in languages:
            languages[lang] = []
        languages[lang].append(file_id)
    
    # Sample from each language with specific rates
    sampled_file_ids = []
    lang_counts = {}
    language_rates = {
        'de': sample_rate_de,
        'en': sample_rate_en
    }
    
    for lang, file_ids in languages.items():
        sample_rate = language_rates.get(lang, 0.0)  # Default to 0 for other languages
        if sample_rate > 0:
            num_to_sample = max(1, int(len(file_ids) * sample_rate))
            sampled = random.sample(file_ids, num_to_sample)
            sampled_file_ids.extend(sampled)
            lang_counts[lang] = len(sampled)
        else:
            lang_counts[lang] = 0
    
    logger.info(f"Sampled PDFs per language: {lang_counts}")
    
    # Load all references
    with open(references_file, 'r', encoding='utf-8') as f:
        all_references = json.load(f)
    
    # Extract references from sampled PDFs
    sampled_refs = []
    xml_dir = Path("benchmarks/excite/all_xml")
    
    for file_id in sampled_file_ids:
        if file_id not in all_references:
            logger.warning(f"No references found for {file_id}")
            continue
        
        file_data = all_references[file_id]
        lang = pdf_metadata.get(file_id, {}).get('lang', 'unknown')
        class_num = pdf_metadata.get(file_id, {}).get('class', 'unknown')
        ref_strings = file_data.get('references', [])
        
        # Load parsed references from XML
        xml_path = xml_dir / f"{file_id}.xml"
        parsed_refs = []
        if xml_path.exists():
            try:
                refs_obj = References.from_excite_xml(str(xml_path))
                parsed_refs = [ref.model_dump() for ref in refs_obj]
                logger.debug(f"Loaded {len(parsed_refs)} parsed refs from XML for {file_id}")
            except Exception as e:
                logger.warning(f"Failed to parse EXCITE XML for {file_id}: {e}")
        
        for idx, ref_str in enumerate(ref_strings):
            ref_entry = {
                "ref_id": f"excite_{file_id}_{idx}",
                "source": "excite",
                "file_id": file_id,
                "language": lang,
                "class": class_num,
                "original_string": ref_str,
            }
            
            # Add parsed data if available
            if idx < len(parsed_refs):
                ref_entry["parsed"] = parsed_refs[idx]
            
            sampled_refs.append(ref_entry)
    
    metadata = {
        "file_ids_per_language": lang_counts,
        "total_refs": len(sampled_refs),
        "refs_with_parsed": sum(1 for r in sampled_refs if "parsed" in r)
    }
    
    logger.info(f"Sampled {len(sampled_refs)} references from EXCITE")
    logger.info(f"  {metadata['refs_with_parsed']} have parsed data")
    return sampled_refs, metadata


def sample_linkedbook_references(
    references_file: str,
    sample_rate: float = 0.15
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Sample references from LinkedBook test dataset.
    
    Args:
        references_file: Path to linkedbooks_test_references.jsonl
        sample_rate: Percentage to sample (default: 0.15 = 15%)
    
    Returns:
        Tuple of (list of sampled references, metadata dict)
    """
    logger.info(f"Sampling from LinkedBook test dataset: {sample_rate*100}%")
    
    # Load all references from JSONL
    all_references = []
    with open(references_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_references.append(json.loads(line))
    
    logger.info(f"Loaded {len(all_references)} total references from LinkedBook")
    
    # Sample references
    num_to_sample = max(1, int(len(all_references) * sample_rate))
    sampled_refs_data = random.sample(all_references, num_to_sample)
    
    # Convert to benchmark format
    sampled_refs = []
    language_counts = {}
    for idx, ref_data in enumerate(sampled_refs_data):
        ref_string = ref_data.get('reference', '')
        lang = ref_data.get('language', 'unknown')
        tags = ref_data.get('tags', {})
        
        sampled_refs.append({
            "ref_id": f"linkedbook_{idx}",
            "source": "linkedbook",
            "language": lang,
            "original_string": ref_string,
            "parsed": tags,  # LinkedBook tags are the parsed fields
        })
        
        language_counts[lang] = language_counts.get(lang, 0) + 1
    
    metadata = {
        "total_available": len(all_references),
        "sampled": num_to_sample,
        "sample_rate": sample_rate,
        "language_distribution": language_counts,
        "total_refs": len(sampled_refs),
        "refs_with_parsed": sum(1 for r in sampled_refs if "parsed" in r and r["parsed"])
    }
    
    logger.info(f"Sampled {len(sampled_refs)} references from LinkedBook")
    logger.info(f"Language distribution: {language_counts}")
    logger.info(f"  {metadata['refs_with_parsed']} have parsed data")
    return sampled_refs, metadata


def save_test_set(
    references: List[Dict[str, Any]],
    output_file: str,
    metadata: Dict[str, Any] = None
):
    """Save test set to JSON and JSONL files.
    
    Args:
        references: List of sampled references
        output_file: Path to output file
        metadata: Optional metadata to include
    """
    output_data = {
        "metadata": metadata or {},
        "references": references
    }
    
    # Add timestamp
    output_data["metadata"]["created_at"] = datetime.utcnow().isoformat()
    output_data["metadata"]["total_references"] = len(references)

    
    # Save as JSONL
    jsonl_path = output_file
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        # Write metadata as first line
        f.write(json.dumps({"_metadata": output_data["metadata"]}, ensure_ascii=False) + "\n")
        # Write each reference as a separate line
        for ref in references:
            f.write(json.dumps(ref, ensure_ascii=False) + "\n")
    logger.info(f"Test set (JSONL) saved to {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build API search test set from CEX, EXCITE, and LinkedBook datasets"
    )
    parser.add_argument(
        "--cex-categories",
        type=int,
        default=12,
        help="Number of CEX categories to sample (default: 12, ~490 refs)"
    )
    parser.add_argument(
        "--excite-sample-rate-de",
        type=float,
        default=0.02,
        help="Percentage to sample from EXCITE German PDFs (default: 0.01 = 1%%, ~2-3 PDFs)"
    )
    parser.add_argument(
        "--excite-sample-rate-en",
        type=float,
        default=0.01,
        help="Percentage to sample from EXCITE English PDFs (default: 0.01 = 1%%, ~1 PDF)"
    )
    parser.add_argument(
        "--linkedbook-sample-rate",
        type=float,
        default=0.09,
        help="Percentage to sample from LinkedBook test set (default: 0.09 = 9%%, ~107 refs)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/api_search_test_set.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")
    
    # Sample references from CEX
    logger.info("=" * 60)
    logger.info("Sampling from CEX dataset")
    logger.info("=" * 60)
    cex_refs, cex_metadata = sample_cex_references(
        pdf_info_file="benchmarks/cex/pdf_files_info.csv",
        references_file="benchmarks/cex/all_references.json",
        num_categories=args.cex_categories
    )
    
    # Sample references from EXCITE
    logger.info("=" * 60)
    logger.info("Sampling from EXCITE dataset")
    logger.info("=" * 60)
    excite_refs, excite_metadata = sample_excite_references(
        pdf_info_file="benchmarks/excite/pdf_files_info.csv",
        references_file="benchmarks/excite/all_references.json",
        sample_rate_de=args.excite_sample_rate_de,
        sample_rate_en=args.excite_sample_rate_en
    )
    
    # Sample references from LinkedBook
    logger.info("=" * 60)
    logger.info("Sampling from LinkedBook test dataset")
    logger.info("=" * 60)
    linkedbook_refs, linkedbook_metadata = sample_linkedbook_references(
        references_file="benchmarks/linkedbook/linkedbooks_test_references.jsonl",
        sample_rate=args.linkedbook_sample_rate
    )
    
    # Combine references
    all_refs = cex_refs + excite_refs + linkedbook_refs
    logger.info("=" * 60)
    logger.info(f"Total references sampled: {len(all_refs)}")
    logger.info("=" * 60)
    
    # Prepare metadata
    final_metadata = {
        "seed": args.seed,
        "cex_samples": cex_metadata,
        "excite_samples": excite_metadata,
        "linkedbook_samples": linkedbook_metadata,
        "total_by_source": {
            "cex": len(cex_refs),
            "excite": len(excite_refs),
            "linkedbook": len(linkedbook_refs)
        }
    }
    
    # Save test set
    save_test_set(all_refs, args.output, metadata=final_metadata)
    
    logger.info("=" * 60)
    logger.info("Test set build complete!")
    logger.info(f"Output files:")
    logger.info(f"  - {args.output}")
    logger.info(f"  - {Path(args.output).with_suffix('.jsonl')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

