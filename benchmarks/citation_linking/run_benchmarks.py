"""Citation-linking benchmark runners as one CLI.

Subcommands:
  build-test-set  sample CEX / EXCITE / LinkedBook references into an API-search test set
  api-search      search the metadata APIs for every reference of a test set
  legal-study     same search on the LLaMoRe legal study XML files
  dasi-ebda       DASI/EBDA bibliography prepare, search and publish stages
"""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from citation_index.core.connectors import (
    MatildaConnector,
    OpenAlexConnector,
    OpenCitationsConnector,
    WikidataConnector,
)
from citation_index.core.models import References
from citation_index.core.models.reference import Reference
from citation_index.core.parsers.tei_bibl_parser import TeiBiblParser
from citation_index.utils.reference_matching import (
    custom_match,
    reference_match_fields,
)

logger = logging.getLogger(__name__)


def sample_cex_references(
    pdf_info_file: str, references_file: str, num_categories: int = 10
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
    with open(pdf_info_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdf_metadata[row["file_id"]] = row

    # Group by category
    categories = {}
    for file_id, meta in pdf_metadata.items():
        category = meta.get("category", "")
        if category not in categories:
            categories[category] = []
        categories[category].append(file_id)

    # Sample num_categories categories
    available_categories = list(categories.keys())
    sampled_categories = random.sample(
        available_categories, min(num_categories, len(available_categories))
    )

    # Sample one PDF from each category
    sampled_file_ids = []
    for category in sampled_categories:
        file_ids = categories[category]
        sampled_file_ids.append(random.choice(file_ids))

    logger.info(f"Sampled categories: {sampled_categories}")
    logger.info(f"Sampled PDFs: {sampled_file_ids}")

    # Load all references
    with open(references_file, "r", encoding="utf-8") as f:
        all_references = json.load(f)

    # Extract references from sampled PDFs
    sampled_refs = []
    xml_dir = Path("benchmarks/cex/all_xmls")

    for file_id in sampled_file_ids:
        if file_id not in all_references:
            logger.warning(f"No references found for {file_id}")
            continue

        file_data = all_references[file_id]
        category = pdf_metadata.get(file_id, {}).get("category", "unknown")
        ref_strings = file_data.get("references", [])

        # Load parsed references from XML
        xml_path = xml_dir / f"{file_id}.xml"
        parsed_refs = []
        if xml_path.exists():
            try:
                refs_obj = References.from_xml(file_path=str(xml_path))
                parsed_refs = [ref.model_dump() for ref in refs_obj]
                logger.debug(
                    f"Loaded {len(parsed_refs)} parsed refs from XML for {file_id}"
                )
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
        "refs_with_parsed": sum(1 for r in sampled_refs if "parsed" in r),
    }

    logger.info(f"Sampled {len(sampled_refs)} references from CEX")
    logger.info(f"  {metadata['refs_with_parsed']} have parsed data")
    return sampled_refs, metadata


def sample_excite_references(
    pdf_info_file: str,
    references_file: str,
    sample_rate_de: float = 0.05,
    sample_rate_en: float = 0.03,
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
    logger.info(
        f"Sampling from EXCITE dataset: {sample_rate_de*100}% German, {sample_rate_en*100}% English"
    )

    # Load PDF metadata
    pdf_metadata = {}
    with open(pdf_info_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdf_metadata[row["file_id"]] = row

    # Group by language
    languages = {}
    for file_id, meta in pdf_metadata.items():
        lang = meta.get("lang", "")
        if lang not in languages:
            languages[lang] = []
        languages[lang].append(file_id)

    # Sample from each language with specific rates
    sampled_file_ids = []
    lang_counts = {}
    language_rates = {"de": sample_rate_de, "en": sample_rate_en}

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
    with open(references_file, "r", encoding="utf-8") as f:
        all_references = json.load(f)

    # Extract references from sampled PDFs
    sampled_refs = []
    xml_dir = Path("benchmarks/excite/all_xml")

    for file_id in sampled_file_ids:
        if file_id not in all_references:
            logger.warning(f"No references found for {file_id}")
            continue

        file_data = all_references[file_id]
        lang = pdf_metadata.get(file_id, {}).get("lang", "unknown")
        class_num = pdf_metadata.get(file_id, {}).get("class", "unknown")
        ref_strings = file_data.get("references", [])

        # Load parsed references from XML
        xml_path = xml_dir / f"{file_id}.xml"
        parsed_refs = []
        if xml_path.exists():
            try:
                refs_obj = References.from_excite_xml(str(xml_path))
                parsed_refs = [ref.model_dump() for ref in refs_obj]
                logger.debug(
                    f"Loaded {len(parsed_refs)} parsed refs from XML for {file_id}"
                )
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
        "refs_with_parsed": sum(1 for r in sampled_refs if "parsed" in r),
    }

    logger.info(f"Sampled {len(sampled_refs)} references from EXCITE")
    logger.info(f"  {metadata['refs_with_parsed']} have parsed data")
    return sampled_refs, metadata


def sample_linkedbook_references(
    references_file: str, sample_rate: float = 0.15
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
    with open(references_file, "r", encoding="utf-8") as f:
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
        ref_string = ref_data.get("reference", "")
        lang = ref_data.get("language", "unknown")
        tags = ref_data.get("tags", {})

        sampled_refs.append(
            {
                "ref_id": f"linkedbook_{idx}",
                "source": "linkedbook",
                "language": lang,
                "original_string": ref_string,
                "parsed": tags,  # LinkedBook tags are the parsed fields
            }
        )

        language_counts[lang] = language_counts.get(lang, 0) + 1

    metadata = {
        "total_available": len(all_references),
        "sampled": num_to_sample,
        "sample_rate": sample_rate,
        "language_distribution": language_counts,
        "total_refs": len(sampled_refs),
        "refs_with_parsed": sum(
            1 for r in sampled_refs if "parsed" in r and r["parsed"]
        ),
    }

    logger.info(f"Sampled {len(sampled_refs)} references from LinkedBook")
    logger.info(f"Language distribution: {language_counts}")
    logger.info(f"  {metadata['refs_with_parsed']} have parsed data")
    return sampled_refs, metadata


def save_test_set(
    references: List[Dict[str, Any]], output_file: str, metadata: Dict[str, Any] = None
):
    """Save test set to JSON and JSONL files.

    Args:
        references: List of sampled references
        output_file: Path to output file
        metadata: Optional metadata to include
    """
    output_data = {"metadata": metadata or {}, "references": references}

    # Add timestamp
    output_data["metadata"]["created_at"] = datetime.utcnow().isoformat()
    output_data["metadata"]["total_references"] = len(references)

    # Save as JSONL
    jsonl_path = output_file
    with open(jsonl_path, "w", encoding="utf-8") as f:
        # Write metadata as first line
        f.write(
            json.dumps({"_metadata": output_data["metadata"]}, ensure_ascii=False)
            + "\n"
        )
        # Write each reference as a separate line
        for ref in references:
            f.write(json.dumps(ref, ensure_ascii=False) + "\n")
    logger.info(f"Test set (JSONL) saved to {jsonl_path}")


def build_test_set_main():
    parser = argparse.ArgumentParser(
        description="Build API search test set from CEX, EXCITE, and LinkedBook datasets"
    )
    parser.add_argument(
        "--cex-categories",
        type=int,
        default=12,
        help="Number of CEX categories to sample (default: 12, ~490 refs)",
    )
    parser.add_argument(
        "--excite-sample-rate-de",
        type=float,
        default=0.02,
        help="Percentage to sample from EXCITE German PDFs (default: 0.01 = 1%%, ~2-3 PDFs)",
    )
    parser.add_argument(
        "--excite-sample-rate-en",
        type=float,
        default=0.01,
        help="Percentage to sample from EXCITE English PDFs (default: 0.01 = 1%%, ~1 PDF)",
    )
    parser.add_argument(
        "--linkedbook-sample-rate",
        type=float,
        default=0.09,
        help="Percentage to sample from LinkedBook test set (default: 0.09 = 9%%, ~107 refs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/citation_linking/api_search_test_set.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
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
        num_categories=args.cex_categories,
    )

    # Sample references from EXCITE
    logger.info("=" * 60)
    logger.info("Sampling from EXCITE dataset")
    logger.info("=" * 60)
    excite_refs, excite_metadata = sample_excite_references(
        pdf_info_file="benchmarks/excite/pdf_files_info.csv",
        references_file="benchmarks/excite/all_references.json",
        sample_rate_de=args.excite_sample_rate_de,
        sample_rate_en=args.excite_sample_rate_en,
    )

    # Sample references from LinkedBook
    logger.info("=" * 60)
    logger.info("Sampling from LinkedBook test dataset")
    logger.info("=" * 60)
    linkedbook_refs, linkedbook_metadata = sample_linkedbook_references(
        references_file="benchmarks/linkedbook/linkedbooks_test_references.jsonl",
        sample_rate=args.linkedbook_sample_rate,
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
            "linkedbook": len(linkedbook_refs),
        },
    }

    # Save test set
    save_test_set(all_refs, args.output, metadata=final_metadata)

    logger.info("=" * 60)
    logger.info("Test set build complete!")
    logger.info("Output files:")
    logger.info(f"  - {args.output}")
    logger.info(f"  - {Path(args.output).with_suffix('.jsonl')}")
    logger.info("=" * 60)


def load_test_set(test_set_file: str) -> Dict[str, Any]:
    """Load test set from JSON or JSONL file.

    Args:
        test_set_file: Path to test set file (.json or .jsonl)

    Returns:
        Dictionary with metadata and references
    """
    test_set_path = Path(test_set_file)

    if test_set_path.suffix == ".jsonl":
        # Load from JSONL
        references = []
        metadata = {}
        with open(test_set_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    data = json.loads(line)
                    if idx == 0 and "_metadata" in data:
                        metadata = data["_metadata"]
                    else:
                        references.append(data)
        return {"metadata": metadata, "references": references}
    else:
        # Load from JSON
        with open(test_set_path, "r", encoding="utf-8") as f:
            return json.load(f)


def extract_doi_from_string(ref_string: str) -> Optional[str]:
    """Try to extract DOI from reference string."""
    import re

    # Simple DOI pattern
    doi_pattern = r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+"
    match = re.search(doi_pattern, ref_string)
    if match:
        doi = match.group(0)
        # Clean up trailing punctuation
        doi = doi.rstrip(".,;)")
        return doi
    return None


def create_reference_from_data(ref_data: Dict[str, Any]) -> Reference:
    """Create a Reference object from test set data.

    Uses parsed data if available, otherwise falls back to original string.
    """
    parsed = ref_data.get("parsed", {})

    if parsed:
        # Use parsed data to create a more complete Reference
        ref_dict = {}

        # Handle different field names across datasets
        if "full_title" in parsed:
            ref_dict["full_title"] = parsed["full_title"]
        elif "title" in parsed:
            ref_dict["full_title"] = parsed["title"]

        if "authors" in parsed:
            ref_dict["authors"] = parsed["authors"]
        elif "author" in parsed:
            # Simple author string, would need more sophisticated parsing
            ref_dict["authors"] = [{"name": parsed["author"]}]

        if "publication_date" in parsed:
            ref_dict["publication_date"] = parsed["publication_date"]
        elif "year" in parsed:
            ref_dict["publication_date"] = parsed["year"]

        if "journal_title" in parsed:
            ref_dict["journal_title"] = parsed["journal_title"]

        if "doi" in parsed:
            ref_dict["doi"] = parsed["doi"]

        # Add other fields as available
        for field in ["volume", "pages", "issue", "publisher"]:
            if field in parsed:
                ref_dict[field] = parsed[field]

        return Reference(**ref_dict)
    else:
        # Fallback to simple parsing from original string
        original = ref_data.get("original_string", "")
        return Reference(full_title=original[:200])


def extract_simplified_result(
    result: Dict[str, Any], connector: Any, connector_name: str
) -> Dict[str, Any]:
    """Extract simplified result with only essential fields and IDs.

    Uses the connector's _result_to_reference() method to leverage existing
    parsing logic, then extracts only the fields we need for output.

    Args:
        result: Raw API result
        connector: Connector instance (to use its mapping method)
        connector_name: Name of the connector (for ID extraction)

    Returns:
        Dictionary with simplified fields: title, first_author, year, journal, ids
    """
    simplified = {
        "title": None,
        "first_author": None,
        "year": None,
        "journal": None,
        "ids": {},
    }

    try:
        # Use connector's mapping to convert to Reference object
        ref = connector._result_to_reference(result)

        simplified.update(reference_match_fields(ref))

        # Extract journal
        if hasattr(ref, "journal") and ref.journal:
            simplified["journal"] = ref.journal
        elif hasattr(ref, "journal_title") and ref.journal_title:
            simplified["journal"] = ref.journal_title

        # Extract IDs - prefer from Reference object when available, else from raw result
        if connector_name == "openalex":
            simplified["ids"]["openalex_id"] = result.get("id")
            simplified["ids"]["doi"] = result.get("doi")
            ids_dict = result.get("ids", {})
            if isinstance(ids_dict, dict):
                simplified["ids"]["isbn"] = ids_dict.get("isbn")

        elif connector_name == "wikidata":
            # Extract Wikidata ID from Reference object (connector stores it there)
            if hasattr(ref, "wikidata_id") and ref.wikidata_id:
                simplified["ids"]["wikidata_id"] = ref.wikidata_id
            else:
                # Fallback: extract from raw result's "id" field
                qid = result.get("id")
                if qid and qid.startswith("Q"):
                    simplified["ids"]["wikidata_id"] = qid

            # DOI and ISBN (if present in raw result)
            # Helper to safely extract value (handles both dict with "value" key and direct values)
            def safe_get_value(data, key):
                val = data.get(key)
                if val is None:
                    return None
                if isinstance(val, dict):
                    return val.get("value")
                return val

            doi_value = safe_get_value(result, "doi")
            if doi_value:
                simplified["ids"]["doi"] = doi_value
            isbn_value = safe_get_value(result, "isbn")
            if isbn_value:
                simplified["ids"]["isbn"] = isbn_value

        elif connector_name == "matilda":
            simplified["ids"]["matilda_id"] = result.get("id")
            # Extract DOI and ISBN from nested structure
            texts = result.get("texts", [])
            if texts and len(texts) > 0:
                identifiers = texts[0].get("identifier", [])
                for identifier in identifiers:
                    if "doi" in identifier:
                        doi_list = identifier.get("doi", [])
                        if doi_list:
                            simplified["ids"]["doi"] = doi_list[0]
                    if "isbn" in identifier:
                        isbn_list = identifier.get("isbn", [])
                        if isbn_list:
                            simplified["ids"]["isbn"] = isbn_list[0]

        elif connector_name == "opencitations":
            # Helper to safely extract value
            def safe_get_value(data, key):
                val = data.get(key)
                if val is None:
                    return None
                if isinstance(val, dict):
                    return val.get("value")
                return val

            omid_direct = result.get("omid")
            br_value = safe_get_value(result, "br")
            simplified["ids"]["omid"] = omid_direct or br_value

            doi_value = safe_get_value(result, "doi")
            if doi_value:
                simplified["ids"]["doi"] = doi_value

    except Exception as e:
        logger.warning(f"Error extracting simplified result from {connector_name}: {e}")

    return simplified


def search_with_connector(
    connector: Any,
    connector_name: str,
    reference: Reference,
    original_string: str,
    top_k: int = 10,
    match_threshold: float = 0.9,
) -> Dict[str, Any]:
    """Search using a connector and evaluate matches.

    Args:
        connector: API connector instance
        connector_name: Name of the connector
        reference: Reference object to search for
        original_string: Original reference string
        top_k: Number of results to retrieve
        match_threshold: Similarity threshold for matching

    Returns:
        Dictionary with search results and match evaluations
    """
    result = {
        "metadata_search": {
            "success": False,
            "num_results": 0,
            "top_result": None,
            "error": None,
        }
    }

    # Try metadata search
    try:
        logger.debug(f"Searching {connector_name} with metadata...")
        raw_results = connector.search(reference, top_k=top_k)
        result["metadata_search"]["success"] = True
        result["metadata_search"]["num_results"] = len(raw_results)

        if raw_results:
            top_result = raw_results[0]

            # Extract simplified result using connector's mapping
            result_simplified = extract_simplified_result(
                top_result, connector, connector_name
            )

            # Evaluate match using custom matching logic
            is_match, match_details = custom_match(reference, result_simplified)

            result["metadata_search"]["top_result"] = {
                "ids": result_simplified["ids"],
                "title": result_simplified["title"],
                "first_author": result_simplified["first_author"],
                "year": result_simplified["year"],
                "journal": result_simplified["journal"],
                "is_match": is_match,
                "match_details": match_details,
            }
    except Exception as e:
        logger.error(f"{connector_name} metadata search failed: {e}")
        result["metadata_search"]["error"] = str(e)

    # DOI search disabled for now
    # # Try ID search if DOI is available
    # doi = extract_doi_from_string(original_string)
    # if not doi and hasattr(reference, 'doi') and reference.doi:
    #     doi = reference.doi
    #
    # if doi:
    #     try:
    #         logger.debug(f"Searching {connector_name} with DOI: {doi}")
    #         raw_results = connector.search_by_id(doi, identifier_type="doi")
    #         result["id_search"]["success"] = True
    #         result["id_search"]["num_results"] = len(raw_results)
    #
    #         if raw_results:
    #             top_result = raw_results[0]
    #
    #             # Evaluate match
    #             is_match, similarity = connector.match(reference, top_result, threshold=match_threshold)
    #
    #             # Extract unique ID
    #             unique_id = extract_unique_id(top_result, connector_name)
    #
    #             result["id_search"]["top_result"] = {
    #                 "id": unique_id,
    #                 "is_match": is_match,
    #                 "similarity_score": similarity,
    #                 "data": top_result
    #             }
    #     except Exception as e:
    #         logger.error(f"{connector_name} ID search failed: {e}")
    #         result["id_search"]["error"] = str(e)

    # Rate limiting - be nice to APIs
    time.sleep(0.3)

    return result


def search_single_api(
    api_name: str,
    connector: Any,
    reference: Reference,
    original_string: str,
    top_k: int,
    match_threshold: float,
) -> tuple[str, Dict[str, Any]]:
    """Search a single API (designed for parallel execution).

    Args:
        api_name: Name of the API
        connector: API connector instance
        reference: Reference object to search for
        original_string: Original reference string
        top_k: Number of results to retrieve
        match_threshold: Similarity threshold for matching

    Returns:
        Tuple of (api_name, search_result)
    """
    logger.info(f"Searching {api_name}...")
    result = search_with_connector(
        connector=connector,
        connector_name=api_name,
        reference=reference,
        original_string=original_string,
        top_k=top_k,
        match_threshold=match_threshold,
    )
    return api_name, result


def process_single_reference(
    ref_data: Dict[str, Any],
    connectors: Dict[str, Any],
    skip_apis: List[str],
    top_k: int,
    match_threshold: float,
    idx: int,
    total: int,
) -> Dict[str, Any]:
    """Process a single reference through all API connectors (with parallel connector searches).

    Args:
        ref_data: Reference dictionary from test set
        connectors: Dictionary of connector instances
        skip_apis: List of API names to skip
        top_k: Number of results to retrieve per search
        match_threshold: Similarity threshold for matching
        idx: Current index (for logging)
        total: Total number of references (for logging)

    Returns:
        Processed reference with search results
    """
    logger.info(f"Processing reference {idx}/{total}: {ref_data['ref_id']}")

    ref_string = ref_data["original_string"]

    # Create Reference object from parsed data
    reference = create_reference_from_data(ref_data)

    # Search with each connector in parallel
    search_results = {}
    active_connectors = {
        name: conn for name, conn in connectors.items() if name not in skip_apis
    }

    # Use ThreadPoolExecutor for parallel API searches
    with ThreadPoolExecutor(max_workers=len(active_connectors)) as executor:
        # Submit all search tasks
        future_to_api = {
            executor.submit(
                search_single_api,
                api_name,
                connector,
                reference,
                ref_string,
                top_k,
                match_threshold,
            ): api_name
            for api_name, connector in active_connectors.items()
        }

        # Collect results as they complete
        for future in as_completed(future_to_api):
            api_name = future_to_api[future]
            try:
                result_api_name, result = future.result()
                search_results[result_api_name] = result
            except Exception as e:
                logger.error(f"Error searching {api_name}: {e}")
                search_results[api_name] = {
                    "metadata_search": {"success": False, "error": str(e)}
                }

    # Compile results
    processed_ref = {**ref_data, "search_results": search_results}

    return processed_ref


def process_references(
    references: List[Dict[str, Any]],
    connectors: Dict[str, Any],
    skip_apis: List[str],
    top_k: int,
    match_threshold: float,
    output_file: str,
    max_workers: int = 1,
) -> List[Dict[str, Any]]:
    """Process all references through all API connectors.

    Args:
        references: List of reference dictionaries from test set
        connectors: Dictionary of connector instances
        skip_apis: List of API names to skip
        top_k: Number of results to retrieve per search
        match_threshold: Similarity threshold for matching
        output_file: Path to output file for periodic saving
        max_workers: Number of parallel workers for processing references

    Returns:
        List of processed references with search results
    """
    processed_refs = []
    total = len(references)

    if max_workers == 1:
        # Sequential processing
        for idx, ref_data in enumerate(references, 1):
            processed_ref = process_single_reference(
                ref_data, connectors, skip_apis, top_k, match_threshold, idx, total
            )
            processed_refs.append(processed_ref)

            # Save intermediate results every 50 references
            if idx % 50 == 0:
                logger.info(f"Saving intermediate results at {idx}/{total}")
                save_results(processed_refs, output_file, is_final=False)
    else:
        # Parallel processing of references
        logger.info(f"Processing references with {max_workers} parallel workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all reference processing tasks
            future_to_idx = {
                executor.submit(
                    process_single_reference,
                    ref_data,
                    connectors,
                    skip_apis,
                    top_k,
                    match_threshold,
                    idx,
                    total,
                ): (idx, ref_data)
                for idx, ref_data in enumerate(references, 1)
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_idx):
                idx, ref_data = future_to_idx[future]
                try:
                    processed_ref = future.result()
                    processed_refs.append(processed_ref)
                    completed += 1

                    # Save intermediate results every 50 completions
                    if completed % 50 == 0:
                        logger.info(
                            f"Saving intermediate results ({completed}/{total} completed)"
                        )
                        save_results(processed_refs, output_file, is_final=False)

                except Exception as e:
                    logger.error(
                        f"Error processing reference {ref_data.get('ref_id', idx)}: {e}"
                    )

    return processed_refs


def save_results(
    references: List[Dict[str, Any]],
    output_file: str,
    is_final: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save results to JSON file (only JSON, no JSONL).

    Args:
        references: List of processed references
        output_file: Path to output file (will be modified to include metadata)
        is_final: Whether this is the final save
        metadata: Optional metadata to include in filename

    Returns:
        Actual output filename used (may be different from input if metadata added)
    """
    # Keep only essential fields (results are already simplified)
    simplified_refs = []
    for ref in references:
        simplified = {
            "ref_id": ref.get("ref_id"),
            "source": ref.get("source"),
            "original_string": ref.get("original_string"),
            "search_results": ref.get("search_results", {}),
        }
        simplified_refs.append(simplified)

    # Generate filename with metadata if final save
    if is_final and metadata:
        # Extract metadata for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        limit = metadata.get("limit", "all")
        apis = "_".join(metadata.get("apis_used", []))

        # Modify output filename to include metadata
        path = Path(output_file)
        new_filename = f"{path.stem}_{timestamp}_limit{limit}_{apis}{path.suffix}"
        output_file = str(path.parent / new_filename)

    # Save JSON only
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(simplified_refs, f, indent=2, ensure_ascii=False)

    logger.info(
        f"{'Final' if is_final else 'Intermediate'} results saved to {output_file}"
    )

    return output_file


def api_search_main():
    parser = argparse.ArgumentParser(
        description="Run API search benchmark on a pre-built test set"
    )
    parser.add_argument(
        "--test-set",
        type=str,
        required=True,
        default="benchmarks/citation_linking/api_search_test_set.jsonl",
        help="Path to test set file (.json or .jsonl)",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for matches (default: 0.9)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/citation_linking/api_search_benchmark_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--skip-apis",
        type=str,
        default="opencitations",
        help="Comma-separated list of APIs to skip (e.g., 'matilda,wikidata')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve per search (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to random N references for testing (default: process all)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing references (default: 1, sequential)",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="Contact email for the OpenAlex User-Agent; authentication uses OPENALEX_API",
    )

    args = parser.parse_args()

    # Parse skip_apis
    skip_apis = [api.strip() for api in args.skip_apis.split(",") if api.strip()]

    # Load test set
    logger.info("=" * 60)
    logger.info("Loading test set")
    logger.info("=" * 60)
    test_set = load_test_set(args.test_set)
    references = test_set.get("references", [])
    test_set_metadata = test_set.get("metadata", {})

    logger.info(f"Loaded {len(references)} references from test set")
    if test_set_metadata:
        logger.info(f"Test set metadata: {test_set_metadata}")

    # Apply limit if specified (random sampling)
    if args.limit and args.limit < len(references):
        logger.info(
            f"Randomly sampling {args.limit} references from {len(references)} total"
        )
        references = random.sample(references, args.limit)
        logger.info(f"Selected {len(references)} references for processing")

    # Initialize connectors
    logger.info("=" * 60)
    logger.info("Initializing API connectors")
    logger.info("=" * 60)
    connectors = {}

    if "openalex" not in skip_apis:
        connectors["openalex"] = OpenAlexConnector(email=args.email)
        if args.email:
            logger.info(f"Initialized OpenAlex connector with email: {args.email}")
        else:
            logger.info(
                "OpenAlex authentication uses OPENALEX_API (legacy fallback: OPENALEX_API_KEY)"
            )
        logger.info("Initialized OpenAlex connector")

    if "opencitations" not in skip_apis:
        connectors["opencitations"] = OpenCitationsConnector()
        logger.info("Initialized OpenCitations connector")

    if "wikidata" not in skip_apis:
        connectors["wikidata"] = WikidataConnector()
        logger.info("Initialized Wikidata connector")

    if "matilda" not in skip_apis:
        connectors["matilda"] = MatildaConnector()
        logger.info("Initialized Matilda connector")

    # Process references
    logger.info("=" * 60)
    logger.info("Processing references through API connectors")
    logger.info("=" * 60)
    processed_refs = process_references(
        references=references,
        connectors=connectors,
        skip_apis=skip_apis,
        top_k=args.top_k,
        match_threshold=args.match_threshold,
        output_file=args.output,
        max_workers=args.max_workers,
    )

    # Save final results
    final_metadata = {
        "test_set_file": args.test_set,
        "test_set_metadata": test_set_metadata,
        "match_threshold": args.match_threshold,
        "top_k": args.top_k,
        "limit": args.limit,
        "max_workers": args.max_workers,
        "apis_used": list(connectors.keys()),
        "total_references": len(processed_refs),
    }

    final_output_file = save_results(
        processed_refs, args.output, is_final=True, metadata=final_metadata
    )

    logger.info("=" * 60)
    logger.info("Benchmark complete!")
    logger.info(f"Results saved to: {final_output_file}")
    logger.info("=" * 60)


LLAMORE_NS = "https://gitlab.mpcdf.mpg.de/dcfidalgo/llamore"


TEI_NS = "http://www.tei-c.org/ns/1.0"


XML_NS = "http://www.w3.org/XML/1998/namespace"


_DEFAULT_DATA_DIR = Path(__file__).parent / "legal_study_data_mpilhlt"


def parse_xml_file(xml_path: Path, tei_parser: TeiBiblParser) -> List[Dict[str, Any]]:
    """Parse a single LLaMoRe XML file and return a list of instance dicts.

    Each dict contains:
        ref_id         – unique identifier: ``{xml_stem}_{instance_xml_id}``
        source         – DOI / URL from the dataset @source attribute
        original_string – raw reference string from ``<input type="raw">``
        reference      – ``Reference`` object parsed from ``<output type="biblstruct">``
                         (may be ``None`` if parsing failed or element is absent)
    """
    try:
        tree = etree.parse(str(xml_path))
    except etree.XMLSyntaxError as exc:
        logger.error(f"XML parse error in {xml_path.name}: {exc}")
        return []

    root = tree.getroot()
    source = root.get("source", xml_path.stem)
    instances: List[Dict[str, Any]] = []

    for instance_el in root.findall(f"{{{LLAMORE_NS}}}instance"):
        xml_id = instance_el.get(f"{{{XML_NS}}}id") or instance_el.get("xml:id", "")
        ref_id = f"{xml_path.stem}_{xml_id}" if xml_id else xml_path.stem

        # Raw reference string -----------------------------------------------
        input_el = instance_el.find(f"{{{LLAMORE_NS}}}input[@type='raw']")
        if input_el is None:
            logger.debug(f"No <input type='raw'> for {ref_id}, skipping")
            continue
        raw_string = (input_el.text or "").strip()
        if not raw_string:
            logger.debug(f"Empty raw string for {ref_id}, skipping")
            continue

        # Structured biblStruct -----------------------------------------------
        output_el = instance_el.find(f"{{{LLAMORE_NS}}}output[@type='biblstruct']")
        reference: Optional[Reference] = None

        if output_el is not None:
            list_bibl_el = output_el.find(f"{{{TEI_NS}}}listBibl")
            if list_bibl_el is not None:
                try:
                    refs = tei_parser.to_references(
                        list_bibl_el, raise_empty_error=False
                    )
                    reference = refs[0] if refs else None
                except Exception as exc:
                    logger.warning(f"Failed to parse biblStruct for {ref_id}: {exc}")

        instances.append(
            {
                "ref_id": ref_id,
                "source": source,
                "original_string": raw_string,
                "reference": reference,
            }
        )

    return instances


def load_legal_study_data(
    data_dir: Path, tei_parser: TeiBiblParser
) -> List[Dict[str, Any]]:
    """Parse all XML files in *data_dir* and return every instance, including failed ones.

    Each dict contains the fields described in :func:`parse_xml_file` plus an
    optional ``drop_reason`` key (``"parse_error"`` or ``"no_title"``) for
    instances that would be excluded by :func:`filter_instances`.

    Returns the **unfiltered** flat list so callers can inspect or serialise
    the full picture before discarding anything.
    """
    xml_files = sorted(data_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {data_dir}")

    all_instances: List[Dict[str, Any]] = []
    for xml_path in xml_files:
        logger.info(f"Parsing {xml_path.name} …")
        instances = parse_xml_file(xml_path, tei_parser)
        logger.info(f"  → {len(instances)} instances extracted")
        all_instances.extend(instances)

    # Annotate instances with a drop_reason without removing them yet.
    for inst in all_instances:
        ref = inst.get("reference")
        if ref is None:
            inst["drop_reason"] = "parse_error"
        elif not (ref.full_title or "").strip():
            inst["drop_reason"] = "no_title"

    return all_instances


def filter_instances(all_instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only instances that are valid for API linking.

    Drops any instance whose ``drop_reason`` key is set, logging a warning for
    each one.  This is intentionally a separate step from :func:`load_legal_study_data`
    so the full parsed set can be serialised first.
    """
    valid: List[Dict[str, Any]] = []
    dropped_parse = 0
    dropped_no_title = 0

    for inst in all_instances:
        reason = inst.get("drop_reason")
        if reason == "parse_error":
            logger.warning(
                f"Dropping {inst['ref_id']}: parse error (no Reference object)"
            )
            dropped_parse += 1
        elif reason == "no_title":
            logger.warning(f"Dropping {inst['ref_id']}: reference has no title")
            dropped_no_title += 1
        else:
            valid.append(inst)

    logger.info(
        f"Filtering: {len(all_instances)} total → {len(valid)} kept, "
        f"{dropped_parse} dropped (parse error), "
        f"{dropped_no_title} dropped (no title)"
    )
    return valid


def create_reference(instance: Dict[str, Any]) -> Reference:
    """Return the parsed ``Reference`` for *instance*, falling back to raw string."""
    ref = instance.get("reference")
    if ref is not None:
        return ref
    raw = instance.get("original_string", "")
    return Reference(full_title=raw[:200])


def process_single_instance(
    instance: Dict[str, Any],
    connectors: Dict[str, Any],
    skip_apis: List[str],
    top_k: int,
    match_threshold: float,
    idx: int,
    total: int,
) -> Dict[str, Any]:
    """Process one reference through all active connectors in parallel."""
    logger.info(f"[{idx}/{total}] {instance['ref_id']}")

    reference = create_reference(instance)
    original_string = instance["original_string"]

    active_connectors = {
        name: conn for name, conn in connectors.items() if name not in skip_apis
    }

    search_results: Dict[str, Any] = {}
    if not active_connectors:
        return {
            "ref_id": instance["ref_id"],
            "source": instance["source"],
            "original_string": original_string,
            "search_results": search_results,
        }

    with ThreadPoolExecutor(max_workers=len(active_connectors)) as executor:
        future_to_api = {
            executor.submit(
                search_single_api,
                api_name,
                connector,
                reference,
                original_string,
                top_k,
                match_threshold,
            ): api_name
            for api_name, connector in active_connectors.items()
        }
        for future in as_completed(future_to_api):
            api_name = future_to_api[future]
            try:
                result_api_name, result = future.result()
                search_results[result_api_name] = result
            except Exception as exc:
                logger.error(f"Error searching {api_name}: {exc}")
                search_results[api_name] = {
                    "metadata_search": {"success": False, "error": str(exc)}
                }

    return {
        "ref_id": instance["ref_id"],
        "source": instance["source"],
        "original_string": original_string,
        "search_results": search_results,
    }


def process_instances(
    instances: List[Dict[str, Any]],
    connectors: Dict[str, Any],
    skip_apis: List[str],
    top_k: int,
    match_threshold: float,
    output_file: str,
    max_workers: int = 1,
) -> List[Dict[str, Any]]:
    """Process all instances, optionally in parallel.  Saves intermediate results every 50."""
    total = len(instances)
    processed: List[Dict[str, Any]] = []

    if max_workers == 1:
        for idx, instance in enumerate(instances, 1):
            processed_ref = process_single_instance(
                instance, connectors, skip_apis, top_k, match_threshold, idx, total
            )
            processed.append(processed_ref)
            if idx % 50 == 0:
                logger.info(f"Saving intermediate results at {idx}/{total}")
                save_legal_results(processed, output_file, is_final=False)
    else:
        logger.info(f"Processing with {max_workers} parallel workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    process_single_instance,
                    instance,
                    connectors,
                    skip_apis,
                    top_k,
                    match_threshold,
                    idx,
                    total,
                ): (idx, instance)
                for idx, instance in enumerate(instances, 1)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                idx, instance = future_to_idx[future]
                try:
                    processed.append(future.result())
                    completed += 1
                    if completed % 50 == 0:
                        logger.info(
                            f"Saving intermediate results ({completed}/{total} completed)"
                        )
                        save_legal_results(processed, output_file, is_final=False)
                except Exception as exc:
                    logger.error(
                        f"Error processing {instance.get('ref_id', idx)}: {exc}"
                    )

    return processed


def build_output_path(
    output_arg: Optional[str], apis: List[str], limit: Optional[int]
) -> str:
    """Generate a timestamped filename if no explicit output path given."""
    if output_arg:
        return output_arg
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    limit_str = str(limit) if limit else "None"
    apis_str = "_".join(apis)
    filename = f"legal_study_results_{timestamp}_limit{limit_str}_{apis_str}.json"
    return str(Path(__file__).parent / filename)


def save_parsed_instances(
    instances: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> str:
    """Serialise parsed instances to a human-readable intermediate JSON file.

    Each record contains:
        ref_id          – unique instance identifier
        source          – dataset DOI / URL
        original_string – raw footnote / reference text
        parsed          – serialised ``Reference`` fields (via ``model_dump``)

    Args:
        instances:   List of instance dicts from :func:`load_legal_study_data`.
        output_path: Destination path.  Defaults to an auto-timestamped file
                     next to this script.

    Returns:
        The path the file was written to.
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(
            Path(__file__).parent / f"legal_study_parsed_{timestamp}.json"
        )

    records = []
    for inst in instances:
        ref = inst.get("reference")
        records.append(
            {
                "ref_id": inst["ref_id"],
                "source": inst["source"],
                "original_string": inst["original_string"],
                "drop_reason": inst.get("drop_reason"),
                "parsed": ref.model_dump() if ref is not None else None,
            }
        )

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    logger.info(
        f"Parsed intermediate file saved to {output_path} ({len(records)} records)"
    )
    return output_path


def save_legal_results(
    references: List[Dict[str, Any]],
    output_file: str,
    is_final: bool = True,
) -> str:
    """Save *references* to *output_file* as JSON."""
    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(references, fh, indent=2, ensure_ascii=False)
    label = "Final" if is_final else "Intermediate"
    logger.info(f"{label} results saved to {output_file}")
    return output_file


def legal_study_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run citation linking benchmark on LLaMoRe legal study XML files"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(_DEFAULT_DATA_DIR),
        help="Path to the folder containing LLaMoRe XML files "
        "(default: legal_study_data_mpilhlt/ next to this script)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: auto-timestamped in the same folder as this script)",
    )
    parser.add_argument(
        "--skip-apis",
        type=str,
        default="",
        help="Comma-separated list of APIs to skip, e.g. 'opencitations,matilda'",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve per connector search (default: 10)",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.9,
        help="Similarity threshold used in match evaluation (default: 0.9)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Randomly sample N instances for a quick test run (default: all)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing instances (default: 1, sequential)",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="Contact email for the OpenAlex User-Agent; authentication uses OPENALEX_API",
    )
    parser.add_argument(
        "--save-parsed",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help=(
            "Save an intermediate JSON with original_string + parsed Reference fields "
            "before running any API searches.  Optionally provide a path; "
            "if the flag is given without a value an auto-timestamped file is created "
            "next to this script."
        ),
    )

    args = parser.parse_args()
    skip_apis = [s.strip() for s in args.skip_apis.split(",") if s.strip()]

    # ------------------------------------------------------------------
    # 1. Parse XML files
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Parsing LLaMoRe XML files")
    logger.info("=" * 60)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    tei_parser = TeiBiblParser()
    all_instances = load_legal_study_data(data_dir, tei_parser)
    logger.info(f"Total instances extracted: {len(all_instances)}")

    parsed_count = sum(1 for i in all_instances if i.get("reference") is not None)
    logger.info(
        f"Successfully parsed biblStruct: {parsed_count}/{len(all_instances)} "
        f"({100 * parsed_count / len(all_instances):.1f}%)"
    )

    # ------------------------------------------------------------------
    # 1b. Optionally save the parsed intermediate file (full, pre-filter)
    # ------------------------------------------------------------------
    if args.save_parsed is not None:
        parsed_path = None if args.save_parsed == "auto" else args.save_parsed
        save_parsed_instances(all_instances, parsed_path)

    # ------------------------------------------------------------------
    # 1c. Filter out parse errors and title-less instances
    # ------------------------------------------------------------------
    instances = filter_instances(all_instances)

    if args.limit and args.limit < len(instances):
        logger.info(f"Randomly sampling {args.limit} of {len(instances)} instances")
        instances = random.sample(instances, args.limit)

    # ------------------------------------------------------------------
    # 2. Initialise connectors
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Initialising API connectors")
    logger.info("=" * 60)

    connectors: Dict[str, Any] = {}

    if "openalex" not in skip_apis:
        connectors["openalex"] = OpenAlexConnector(email=args.email)
        if args.email:
            logger.info(f"OpenAlex connector initialised (email: {args.email})")
        else:
            logger.warning(
                "OpenAlex authentication uses OPENALEX_API "
                "(legacy fallback: OPENALEX_API_KEY)."
            )

    if "opencitations" not in skip_apis:
        connectors["opencitations"] = OpenCitationsConnector()
        logger.info("OpenCitations connector initialised")

    if "wikidata" not in skip_apis:
        connectors["wikidata"] = WikidataConnector()
        logger.info("Wikidata connector initialised")

    if "matilda" not in skip_apis:
        connectors["matilda"] = MatildaConnector()
        logger.info("Matilda connector initialised")

    active_apis = [name for name in connectors]
    output_file = build_output_path(args.output, active_apis, args.limit)

    # ------------------------------------------------------------------
    # 3. Run search
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Running citation linking")
    logger.info("=" * 60)

    processed = process_instances(
        instances=instances,
        connectors=connectors,
        skip_apis=skip_apis,
        top_k=args.top_k,
        match_threshold=args.match_threshold,
        output_file=output_file,
        max_workers=args.max_workers,
    )

    # ------------------------------------------------------------------
    # 4. Save final results
    # ------------------------------------------------------------------
    final_path = save_legal_results(processed, output_file, is_final=True)

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info(f"  Instances processed : {len(processed)}")
    logger.info(f"  Results saved to    : {final_path}")
    logger.info("=" * 60)


ROOT = Path(__file__).resolve().parents[2]


LOGGER = logging.getLogger(__name__)


INDEXES = ("openalex", "matilda", "wikidata")


DEFAULT_OUTPUT = ROOT / "logs/citation_linking/dasi_ebda"


DEFAULT_INPUT = Path(__file__).parent / "DASI_EBDA"


HF_REPO = "yurui983/citation_linking_baseline"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    value = unquote(value.strip())
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value, flags=re.I)
    return value.lower() if re.fullmatch(r"10\.\d{4,9}/\S+", value) else None


def plain(value: str) -> str:
    from pybtex.richtext import Text

    value = re.sub(r"\\(?:textit|textrm)\b", "", value)
    for command, replacement in (
        ("textless", "<"),
        ("textgreater", ">"),
        ("textbackslash", "\\"),
    ):
        value = value.replace("\\" + command + "{}", replacement)
    return " ".join(Text.from_latex(value).render_as("text").split())


def displayed_person(person: Any, harvard: bool) -> tuple[str, dict[str, str]]:
    surname = plain(
        " ".join(person.prelast_names + person.last_names + person.lineage_names)
    )
    given = plain(" ".join(person.first_names + person.middle_names))
    if harvard:
        given = " ".join(
            (
                word
                if re.fullmatch(r"(?:[^\W\d_]\.)+", word)
                else "-".join(part[0] + "." for part in word.split("-") if part)
            )
            for word in given.split()
        )
    parsed = {k: v for k, v in (("surname", surname), ("first_name", given)) if v}
    return ", ".join(filter(None, (surname, given))), parsed


def join_names(names: list[str]) -> str:
    if len(names) < 2:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def names_for_style(
    people: list[tuple[str, dict[str, str]]], harvard: bool, editors: bool = False
) -> str:
    names = []
    for position, (inverted, person) in enumerate(people):
        if harvard or (position == 0 and not editors):
            names.append(inverted)
        else:
            names.append(
                " ".join(
                    filter(None, (person.get("first_name"), person.get("surname")))
                )
            )
    return join_names(names)


def sentence(text: str) -> str:
    return text if text.endswith((".", "?", "!")) else text + "."


def render_entry(entry: Any, style: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    harvard = style == "harvard"
    fields = {
        key: plain(value)
        for key, value in entry.fields.items()
        if key not in {"abstract", "note", "keywords"}
    }
    title, year = fields["title"], fields["year"]
    parsed: dict[str, Any] = {"full_title": title, "publication_date_raw": year}
    if year.isdigit():
        parsed["publication_year"] = int(year)
    raw: dict[str, str] = {}
    authors = [
        displayed_person(person, harvard) for person in entry.persons.get("author", [])
    ]
    if authors:
        parsed["authors"] = [person for _, person in authors]
    names = names_for_style(authors, harvard)
    pieces = (
        [f"{names} ({year})." if harvard and names else f"{sentence(names)} {year}."]
        if names
        else []
    )
    article = entry.type == "article"
    container = fields.get("booktitle") or (
        fields.get("journal") if not article else None
    )
    pieces.append(
        f"‘{title}’."
        if harvard and (article or container)
        else f"“{sentence(title)}”" if article or container else sentence(title)
    )
    if not names:
        pieces.append(f"({year})." if harvard else year + ".")
    if article and fields.get("journal"):
        parsed["journal_title"] = fields["journal"]
        venue = fields["journal"]
        for key, target in (("volume", "volume"), ("number", "issue")):
            if fields.get(key):
                parsed[target] = fields[key]
        if "volume" in parsed:
            venue += " " + parsed["volume"]
        if "issue" in parsed:
            venue += " (" + parsed["issue"] + ")"
        if fields.get("pages"):
            parsed["pages"] = fields["pages"].replace("--", "–")
            venue += (", pp. " if harvard else ": ") + parsed["pages"]
        pieces.append(sentence(venue))
    else:
        if container:
            raw["booktitle"] = container
            text = "In " + container
            editors = [
                displayed_person(person, harvard)
                for person in entry.persons.get("editor", [])
            ]
            if editors:
                parsed["editors"] = [person for _, person in editors]
                text += ", edited by " + names_for_style(editors, harvard, editors=True)
            pieces.append(sentence(text))
        for key, label in (
            ("edition", "Edition"),
            ("series", "Series"),
            ("volume", "Volume"),
            ("number", "Number"),
            ("type", "Format"),
        ):
            if fields.get(key):
                raw[key] = fields[key]
                pieces.append(sentence(f"{label}: {fields[key]}"))
        place, publisher = fields.get("address"), fields.get("publisher")
        if place:
            parsed["publication_place"] = place
        if publisher:
            parsed["publisher"] = publisher
        if place or publisher:
            pieces.append(sentence(": ".join(filter(None, (place, publisher)))))
        if fields.get("pages"):
            parsed["pages"] = fields["pages"].replace("--", "–")
            pieces.append("pp. " + parsed["pages"] + ".")
    if raw:
        parsed["raw"] = raw

    supplied_doi = fields.get("doi")
    doi = normalize_doi(supplied_doi)
    url = fields.get("url", "")
    origin = "doi_field" if supplied_doi else None
    if not supplied_doi:
        doi = normalize_doi(url)
        if not doi:
            match = re.fullmatch(
                r"https?://[^/]+/document/doi/(10\.\d{4,9}/.+?)/(?:html|pdf)", url
            )
            doi = normalize_doi(match.group(1)) if match else None
        if doi:
            origin = "url_field"
    identifiers = []
    if doi:
        link = "https://doi.org/" + doi
        pieces.append(link)
        identifiers.append({"scheme": "doi", "value": link, "normalized": doi})
    elif url:
        pieces.append(("Available from: " if harvard else "") + url)
        identifiers.append({"scheme": "url", "value": url, "normalized": url})
    if harvard and (doi or url):
        if fields.get("urldate"):
            accessed = fields["urldate"]
            pieces.append(f"[Accessed {accessed}].")
            parsed.setdefault("raw", {})["access_date"] = accessed
    if identifiers:
        parsed["identifiers"] = identifiers
    reference = " ".join(pieces)
    audit = {"supplied_doi": supplied_doi, "normalized_doi": doi, "origin": origin}
    if supplied_doi and not doi:
        audit["warning"] = "Invalid supplied DOI; excluded from lookup"
    elif not doi and re.search(r"10\.\d{4,9}/", url):
        audit["warning"] = "Ambiguous DOI-like URL; preserved as URL only"
    if not authors:
        audit["missing_author"] = True
    Reference.model_validate(parsed)
    return reference, parsed, audit


def prepare(input_dir: Path, output: Path) -> list[dict[str, Any]]:
    from pybtex.database import parse_file

    rows, audits, hashes = [], [], {}
    for filename, source in (("DASI.bib", "dasi"), ("EbDA.bib", "ebda")):
        path = input_dir / filename
        entries = parse_file(path).entries
        if len(entries) != 100:
            raise ValueError(f"{filename}: expected 100 entries, found {len(entries)}")
        hashes[filename] = digest(path)
        for position, key in enumerate(sorted(entries)):
            style = "chicago" if position % 2 == 0 else "harvard"
            text, parsed, audit = render_entry(entries[key], style)
            ref_id = f"{source}_{key}"
            rows.append(
                {
                    "ref_id": ref_id,
                    "source": source,
                    "citation_style": style,
                    "original_string": text,
                    "parsed": parsed,
                }
            )
            audits.append({"ref_id": ref_id, **audit})
    manifest = {
        "created_at": utc_now(),
        "input_sha256": hashes,
        "script_sha256": digest(Path(__file__)),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "top_k": 10,
        "method": "doi_first_metadata_fallback",
        "references": len(rows),
    }
    prepared_path = output / "references.json"
    if (output / "search_results.json").exists() and prepared_path.exists():
        if json.loads(prepared_path.read_text()) != rows:
            raise ValueError(
                "Prepared references conflict with existing search checkpoint; use a new output directory"
            )
    write_json(prepared_path, rows)
    write_json(output / "doi_audit.json", audits)
    if not (output / "search_results.json").exists():
        write_json(output / "manifest.json", manifest)
    return rows


def candidate_dois(candidate: dict[str, Any], connector: Any, index: str) -> set[str]:
    values = []
    if index == "openalex":
        values.extend([candidate.get("doi"), (candidate.get("ids") or {}).get("doi")])
    elif index == "wikidata":
        values.extend(
            item.get("mainsnak", {}).get("datavalue", {}).get("value")
            for item in candidate.get("claims", {}).get("P356", [])
        )
    else:
        for text in candidate.get("texts", []) or []:
            for identifier in text.get("identifier", []) or []:
                value = identifier.get("doi")
                values.extend(value if isinstance(value, list) else [value])
    values.extend(
        identifier.normalized or identifier.value
        for identifier in connector._result_to_reference(candidate).identifiers
        if identifier.scheme == "doi"
    )
    return {
        doi
        for value in values
        if isinstance(value, str) and (doi := normalize_doi(value))
    }


def simplify(
    candidate: dict[str, Any], connector: Any, index: str, reference: Reference
) -> dict[str, Any]:
    if not candidate.get("id"):
        raise ValueError(f"{index} candidate is missing its provider ID")
    mapped = connector._result_to_reference(candidate)
    fields = reference_match_fields(mapped)
    is_match, details = custom_match(reference, fields)
    dois = sorted(candidate_dois(candidate, connector, index))
    return {
        **fields,
        "journal": mapped.journal_title,
        "ids": {f"{index}_id": candidate.get("id"), "doi": dois[0] if dois else None},
        "is_match": is_match,
        "match_details": details,
        "doi_identifiers": dois,
    }


def search_pair(
    row: dict[str, Any], connector: Any, index: str, state: dict[str, Any] | None = None
) -> dict[str, Any]:
    state = state if state is not None else {}
    reference = Reference.model_validate(row["parsed"])
    doi = next(
        (
            identifier.normalized or normalize_doi(identifier.value)
            for identifier in reference.identifiers
            if identifier.scheme == "doi"
        ),
        None,
    )
    selected = None
    if doi:
        if not state.get("doi_search", {}).get("success"):
            raw = connector.search_by_id(doi, identifier_type="doi", top_k=10)
            candidates = [
                simplify(candidate, connector, index, reference) for candidate in raw
            ]
            state["doi_search"] = {
                "success": True,
                "queried_doi": doi,
                "num_results": len(raw),
                "candidates": candidates,
                "searched_at": utc_now(),
            }
        selected = next(
            (
                candidate
                for candidate in state["doi_search"]["candidates"]
                if doi in candidate["doi_identifiers"]
            ),
            None,
        )
    if selected is None:
        search_reference = reference.model_copy(update={"identifiers": []})
        raw = connector.search(search_reference, top_k=10)
        selected = simplify(raw[0], connector, index, reference) if raw else None
        state["metadata_search"] = {
            "success": True,
            "num_results": len(raw),
            "top_result": selected,
            "searched_at": utc_now(),
        }
        state["retrieval_method"] = "metadata"
    else:
        state["retrieval_method"] = "doi"
        selected = {**selected, "ids": {**selected["ids"], "doi": doi}}
    state.update(
        {
            "success": True,
            "selected_result": selected,
            "doi_exact_match": bool(
                doi and selected and doi in selected["doi_identifiers"]
            ),
        }
    )
    state.pop("error", None)
    return state


def search(
    output: Path, connectors: dict[str, Any] | None = None, delay: float = 0.3
) -> list[dict[str, Any]]:
    prepared = json.loads((output / "references.json").read_text())
    path = output / "search_results.json"
    results = (
        json.loads(path.read_text())
        if path.exists()
        else [{**row, "search_results": {}} for row in prepared]
    )
    if [{k: row[k] for k in prepared[0]} for row in results] != prepared:
        raise ValueError("Search checkpoint does not match prepared references")
    manifest = json.loads((output / "manifest.json").read_text())
    if (
        manifest["method"] != "doi_first_metadata_fallback"
        or manifest["top_k"] != 10
        or manifest["script_sha256"] != digest(Path(__file__))
    ):
        raise ValueError(
            "Checkpoint script or search parameters differ; use a new output directory"
        )
    if connectors is None:
        openalex = OpenAlexConnector()
        if not openalex.api_key:
            raise RuntimeError(
                "Export OPENALEX_API before running authenticated searches"
            )
        openalex._get(f"{openalex.base_url}/rate-limit")
        connectors = {
            "openalex": openalex,
            "matilda": MatildaConnector(),
            "wikidata": WikidataConnector(),
        }
    for row in results:
        for index in INDEXES:
            state = row["search_results"].setdefault(index, {})
            if state.get("success"):
                continue
            try:
                for attempt in range(3):
                    try:
                        search_pair(row, connectors[index], index, state)
                        break
                    except Exception:
                        if index == "openalex" or attempt == 2:
                            raise
                        time.sleep(2**attempt)
            except Exception as exc:
                state.update(
                    success=False,
                    error={
                        "type": type(exc).__name__,
                        "stage": (
                            "metadata"
                            if state.get("doi_search", {}).get("success")
                            else (
                                "doi"
                                if any(
                                    i.get("scheme") == "doi"
                                    for i in row["parsed"].get("identifiers", [])
                                )
                                else "metadata"
                            )
                        ),
                    },
                )
                write_json(path, results)
                try:
                    raise RuntimeError(
                        f"Search failed; ref_id={row['ref_id']}; index={index}; error_type={type(exc).__name__}; checkpoint saved"
                    ) from None
                except RuntimeError:
                    LOGGER.exception("Benchmark search failed")
                    raise
            write_json(path, results)
            LOGGER.info(
                "Completed %s / %s using %s",
                row["ref_id"],
                index,
                state["retrieval_method"],
            )
            time.sleep(delay)
    manifest["search_completed_at"] = utc_now()
    write_json(output / "manifest.json", manifest)
    write_json(output / "publication_rows.json", publication_rows(results))
    return results


def publication_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(results) != 200 or len({row["ref_id"] for row in results}) != 200:
        raise ValueError("Publication requires 200 unique references")
    rows = []
    for row in results:
        for index in INDEXES:
            state = row["search_results"].get(index, {})
            if not state.get("success"):
                raise ValueError(f"Incomplete search: {row['ref_id']} / {index}")
            top = state.get("selected_result") or {}
            identifier = (top.get("ids") or {}).get(f"{index}_id")
            if identifier:
                identifier = str(identifier).rstrip("/").rsplit("/", 1)[-1]
            link = {
                "openalex": "https://openalex.org/",
                "matilda": "https://matilda.science/?l=en",
                "wikidata": "https://www.wikidata.org/wiki/Wikidata:Main_Page",
            }[index]
            if identifier:
                link = {
                    "openalex": f"https://openalex.org/works/{identifier.lower()}",
                    "matilda": f"https://matilda.science/work/{identifier}",
                    "wikidata": f"https://www.wikidata.org/wiki/{identifier}",
                }[index]
            rows.append(
                {
                    "ref_id": row["ref_id"],
                    "source": row["source"],
                    "original_ref_string": row["original_string"],
                    "index": index,
                    "matched_id": identifier or "Not Found",
                    "matched_doi": (top.get("ids") or {}).get("doi") or "",
                    "matched_result": {
                        key: top.get(key)
                        for key in ("first_author", "journal", "title", "year")
                    },
                    "is_match_by_similarity": bool(top.get("is_match")),
                    "matched_link": link,
                }
            )
    return rows


def merge_rows(
    existing: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged = list(existing)
    by_key = {(row["ref_id"], row["index"]): row for row in existing}
    if len(by_key) != len(existing):
        raise ValueError("Existing split has duplicate reference/index rows")
    for row in incoming:
        key = (row["ref_id"], row["index"])
        if key in by_key:
            if by_key[key] != row:
                raise ValueError(f"Conflicting published row: {key}")
        else:
            merged.append(row)
            by_key[key] = row
    return merged


def argilla_fields(row: dict[str, Any]) -> dict[str, str]:
    fields = {key: value for key, value in row.items() if key != "index"}
    fields["matched_result"] = (
        "```json\n"
        + json.dumps(row["matched_result"], ensure_ascii=False, indent=2)
        + "\n```"
    )
    fields["is_match_by_similarity"] = str(row["is_match_by_similarity"]).lower()
    fields["matched_link"] = f"[{row['matched_link']}]({row['matched_link']})"
    return fields


def settings_snapshot(settings: Any) -> dict[str, Any]:
    """Compare annotation configuration without server-assigned resource IDs."""
    snapshot = settings.serialize()
    for group in ("fields", "questions", "vectors", "metadata"):
        for item in snapshot.get(group, []):
            for key in ("id", "dataset_id", "inserted_at", "updated_at"):
                item.pop(key, None)
    return snapshot


def publish(output: Path, repo: str = HF_REPO) -> dict[str, Any]:
    import argilla as rg
    from datasets import Dataset, load_dataset
    from huggingface_hub import (
        CommitOperationAdd,
        CommitOperationDelete,
        DatasetCard,
        HfApi,
        hf_hub_download,
    )

    results = json.loads((output / "search_results.json").read_text())
    prepared = json.loads((output / "references.json").read_text())
    if [{key: row[key] for key in prepared[0]} for row in results] != prepared:
        raise ValueError("Search results differ from prepared references")
    rows = publication_rows(results)
    write_json(output / "publication_rows.json", rows)
    token = os.environ.get("ARGILLA_API_TOKEN")
    if not token or not token.strip():
        raise RuntimeError("Export a valid ARGILLA_API_TOKEN before publishing")
    client = rg.Argilla(
        api_url=os.getenv("ARGILLA_API_URL") or "https://argilla.graphia-ssh.eu/",
        api_key=token,
    )
    sources, targets = {}, {}
    for index in INDEXES:
        source = client.datasets(name=f"citation_linking_{index}")
        if source is None:
            raise ValueError(
                f"Source Argilla dataset missing: citation_linking_{index}"
            )
        write_json(
            output / "argilla_settings" / f"{index}.json",
            settings_snapshot(source.settings),
        )
        target = client.datasets(
            name=f"citation_linking_dasi_ebda_{index}", workspace=source.workspace.name
        )
        expected = {
            row["ref_id"]: argilla_fields(row) for row in rows if row["index"] == index
        }
        if target is not None:
            if settings_snapshot(target.settings) != settings_snapshot(source.settings):
                raise ValueError(f"Existing Argilla settings conflict: {index}")
            for record in target.records:
                if (
                    record.id not in expected
                    or record.fields.to_dict() != expected[record.id]
                ):
                    raise ValueError(f"Existing Argilla record conflicts: {index}")
        sources[index], targets[index] = source, target

    api = HfApi()
    info = api.dataset_info(repo)
    if not info.private:
        raise ValueError("The baseline HF repository must remain private")
    revision = info.sha
    cache_dir = str(output / "hf_cache")
    existing = load_dataset(repo, revision=revision, token=True, cache_dir=cache_dir)
    files = api.list_repo_files(repo, repo_type="dataset", revision=revision)
    staging = output / "hf_upload"
    operations, expected_splits, statistics = [], {}, []
    for split in ("full", "annotated"):
        dataset = existing[split]
        if set(dataset.column_names) != set(rows[0]):
            raise ValueError(f"Unexpected HF schema in {split}")
        merged = merge_rows(dataset.to_list(), rows)
        expected_splits[split] = merged
        updated = Dataset.from_list(merged, features=dataset.features)
        relative = f"data/{split}-00000-of-00001.parquet"
        local = staging / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        updated.to_parquet(str(local))
        operations.append(
            CommitOperationAdd(path_in_repo=relative, path_or_fileobj=str(local))
        )
        for old_file in files:
            if (
                re.fullmatch(rf"data/{split}-\d+-of-\d+\.parquet", old_file)
                and old_file != relative
            ):
                operations.append(CommitOperationDelete(path_in_repo=old_file))
        statistics.append(
            {
                "name": split,
                "num_examples": len(updated),
                "num_bytes": updated.data.nbytes,
            }
        )

    card_path = hf_hub_download(
        repo,
        "README.md",
        repo_type="dataset",
        revision=revision,
        cache_dir=str(output / "hf_cache/hub"),
    )
    card = DatasetCard(Path(card_path).read_text())
    card_info = card.data.dataset_info
    if not isinstance(card_info, dict):
        raise ValueError("Expected the existing single-configuration dataset card")
    card_info["splits"] = statistics
    card_info["dataset_size"] = sum(item["num_bytes"] for item in statistics)
    card_info["download_size"] = sum(
        (staging / f"data/{split}-00000-of-00001.parquet").stat().st_size
        for split in ("full", "annotated")
    )
    marker = "<!-- dasi-ebda-extension -->"
    end_marker = "<!-- /dasi-ebda-extension -->"
    section = f"{marker}\n## DASI/EbDA extension\n\n200 references (100 DASI and 100 EbDA), each searched against OpenAlex, Matilda, and Wikidata. Reference strings are complete bibliography entries generated from BibTeX, evenly mixing Chicago author–date and Leeds Harvard per corpus. Parsed objects contain only displayed values. Supplied DOIs are unverified.\n\nThis extension uses DOI-first lookup with metadata fallback; historical rows use metadata-only retrieval. Exact DOI hits retain the existing title/author/year similarity flag and require human verification. The annotated split denotes annotation selection, not completed labels. Detailed retrieval provenance and the DOI audit are in `extensions/dasi_ebda/`.\n\nFull: {len(expected_splits['full'])} rows. Annotated: {len(expected_splits['annotated'])} rows.\n{end_marker}"
    if marker in card.text:
        if end_marker not in card.text:
            raise ValueError("Existing extension card section has no closing marker")
        card.text = re.sub(
            re.escape(marker) + r".*?" + re.escape(end_marker),
            lambda _: section,
            card.text,
            flags=re.S,
        )
    else:
        card.text = card.text.rstrip() + "\n\n" + section + "\n"
    readme = staging / "README.md"
    readme.write_text(str(card), encoding="utf-8")
    operations.append(
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(readme))
    )
    for filename in (
        "references.json",
        "search_results.json",
        "publication_rows.json",
        "doi_audit.json",
        "manifest.json",
    ):
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"extensions/dasi_ebda/{filename}",
                path_or_fileobj=str(output / filename),
            )
        )
    commit = api.create_commit(
        repo,
        repo_type="dataset",
        operations=operations,
        parent_commit=revision,
        commit_message="Add DASI and EbDA DOI-first linking extension",
    )
    reloaded = load_dataset(repo, revision=commit.oid, token=True, cache_dir=cache_dir)
    for split, expected in expected_splits.items():
        if reloaded[split].to_list() != expected:
            raise RuntimeError(f"HF readback differs: {split}")
    report: dict[str, Any] = {
        "hf_repo": repo,
        "hf_commit": commit.oid,
        "splits": statistics,
        "argilla": {},
    }
    write_json(output / "publication_report.json", report)
    for index in INDEXES:
        source, target = sources[index], targets[index]
        if target is None:
            target = rg.Dataset(
                name=f"citation_linking_dasi_ebda_{index}",
                workspace=source.workspace,
                settings=rg.Settings.from_json(
                    output / "argilla_settings" / f"{index}.json"
                ),
                client=client,
            )
            target.create()
        expected = {
            row["ref_id"]: argilla_fields(row) for row in rows if row["index"] == index
        }
        existing_ids = {record.id for record in target.records}
        records = [
            rg.Record(id=ref_id, fields=fields)
            for ref_id, fields in expected.items()
            if ref_id not in existing_ids
        ]
        if records:
            target.records.log(records)
        check = client.datasets(name=target.name, workspace=source.workspace.name)
        actual = {record.id: record.fields.to_dict() for record in check.records}
        if actual != expected or settings_snapshot(check.settings) != settings_snapshot(
            source.settings
        ):
            raise RuntimeError(f"Argilla readback differs: {index}")
        report["argilla"][index] = {
            "name": target.name,
            "workspace": source.workspace.name,
            "records": len(actual),
        }
        write_json(output / "publication_report.json", report)
    return report


def dasi_ebda_main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "search", "publish"))
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hf-repo", default=HF_REPO)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.stage == "prepare":
        rows = prepare(args.input_dir, args.output_dir)
        LOGGER.info(
            "Prepared %s full bibliography entries in %s", len(rows), args.output_dir
        )
    elif args.stage == "search":
        search(args.output_dir)
    else:
        publish(args.output_dir, args.hf_repo)


COMMANDS = {
    "build-test-set": build_test_set_main,
    "api-search": api_search_main,
    "legal-study": legal_study_main,
    "dasi-ebda": dasi_ebda_main,
}


def main() -> None:
    """Dispatch ``run_benchmarks.py <command> ...`` to the command's own parser."""
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(f"usage: run_benchmarks.py {{{','.join(COMMANDS)}}} [options]")
    command = sys.argv[1]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    sys.argv = [f"run_benchmarks.py {command}", *sys.argv[2:]]
    COMMANDS[command]()


if __name__ == "__main__":
    main()
