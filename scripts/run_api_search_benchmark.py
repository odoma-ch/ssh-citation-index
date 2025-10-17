#!/usr/bin/env python3
"""Run API search benchmark on a pre-built test set.

This script loads a test set of references and searches for them across
multiple API connectors (OpenAlex, OpenCitations, Wikidata, Matilda),
recording match results.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path to import citation_index
sys.path.insert(0, str(Path(__file__).parent.parent))

from citation_index.core.connectors import (
    OpenAlexConnector,
    OpenCitationsConnector,
    WikidataConnector,
    MatildaConnector,
)
from citation_index.core.models.reference import Reference
from citation_index.utils.reference_matching import (
    calculate_title_similarity,
    extract_family_name,
    extract_year,
    normalize_title
)
from fuzzywuzzy import fuzz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)




def load_test_set(test_set_file: str) -> Dict[str, Any]:
    """Load test set from JSON or JSONL file.
    
    Args:
        test_set_file: Path to test set file (.json or .jsonl)
    
    Returns:
        Dictionary with metadata and references
    """
    test_set_path = Path(test_set_file)
    
    if test_set_path.suffix == '.jsonl':
        # Load from JSONL
        references = []
        metadata = {}
        with open(test_set_path, 'r', encoding='utf-8') as f:
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
        with open(test_set_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def extract_doi_from_string(ref_string: str) -> Optional[str]:
    """Try to extract DOI from reference string."""
    import re
    # Simple DOI pattern
    doi_pattern = r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+'
    match = re.search(doi_pattern, ref_string)
    if match:
        doi = match.group(0)
        # Clean up trailing punctuation
        doi = doi.rstrip('.,;)')
        return doi
    return None


def create_reference_from_data(ref_data: Dict[str, Any]) -> Reference:
    """Create a Reference object from test set data.
    
    Uses parsed data if available, otherwise falls back to original string.
    """
    parsed = ref_data.get('parsed', {})
    
    if parsed:
        # Use parsed data to create a more complete Reference
        ref_dict = {}
        
        # Handle different field names across datasets
        if 'full_title' in parsed:
            ref_dict['full_title'] = parsed['full_title']
        elif 'title' in parsed:
            ref_dict['full_title'] = parsed['title']
        
        if 'authors' in parsed:
            ref_dict['authors'] = parsed['authors']
        elif 'author' in parsed:
            # Simple author string, would need more sophisticated parsing
            ref_dict['authors'] = [{'name': parsed['author']}]
        
        if 'publication_date' in parsed:
            ref_dict['publication_date'] = parsed['publication_date']
        elif 'year' in parsed:
            ref_dict['publication_date'] = parsed['year']
        
        if 'journal_title' in parsed:
            ref_dict['journal_title'] = parsed['journal_title']
        
        if 'doi' in parsed:
            ref_dict['doi'] = parsed['doi']
        
        # Add other fields as available
        for field in ['volume', 'pages', 'issue', 'publisher']:
            if field in parsed:
                ref_dict[field] = parsed[field]
        
        return Reference(**ref_dict)
    else:
        # Fallback to simple parsing from original string
        original = ref_data.get('original_string', '')
        return Reference(full_title=original[:200])


def extract_simplified_result(result: Dict[str, Any], connector_name: str) -> Dict[str, Any]:
    """Extract simplified result with only essential fields and IDs.
    
    Args:
        result: Raw API result
        connector_name: Name of the connector
        
    Returns:
        Dictionary with simplified fields: title, first_author, year, journal, ids
    """
    simplified = {
        "title": None,
        "first_author": None,
        "year": None,
        "journal": None,
        "ids": {}
    }
    
    try:
        if connector_name == "openalex":
            # Extract title
            simplified["title"] = result.get("title")
            
            # Extract first author
            authorships = result.get("authorships", [])
            if authorships and len(authorships) > 0:
                author_data = authorships[0].get("author", {})
                author_name = author_data.get("display_name")
                if author_name:
                    simplified["first_author"] = extract_family_name(author_name) or author_name
            
            # Extract year
            simplified["year"] = result.get("publication_year")
            
            # Extract journal
            primary_location = result.get("primary_location", {})
            if primary_location:
                source = primary_location.get("source", {})
                simplified["journal"] = source.get("display_name")
            if not simplified["journal"]:
                host_venue = result.get("host_venue", {})
                simplified["journal"] = host_venue.get("display_name")
            
            # Extract IDs
            simplified["ids"]["openalex_id"] = result.get("id")
            simplified["ids"]["doi"] = result.get("doi")
            ids_dict = result.get("ids", {})
            if isinstance(ids_dict, dict):
                simplified["ids"]["isbn"] = ids_dict.get("isbn")
            
        elif connector_name == "wikidata":
            # Extract title from Wikidata format
            if "label" in result:
                simplified["title"] = result.get("label", {}).get("value")
            elif "title" in result:
                simplified["title"] = result.get("title", {}).get("value")
            
            # Extract author - Wikidata has different formats
            if "author" in result:
                author_value = result.get("author", {}).get("value")
                if author_value:
                    simplified["first_author"] = extract_family_name(author_value) or author_value
            
            # Extract year from publication date
            pub_date = result.get("publicationDate", {}).get("value") or result.get("pub_date", {}).get("value")
            if pub_date:
                year = extract_year(pub_date)
                simplified["year"] = year
            
            # Extract journal/venue
            if "venue" in result:
                simplified["journal"] = result.get("venue", {}).get("value")
            elif "venueLabel" in result:
                simplified["journal"] = result.get("venueLabel", {}).get("value")
            
            # Extract IDs
            item_uri = result.get("item", {}).get("value", "")
            if item_uri:
                qid = item_uri.rsplit("/", 1)[-1]
                simplified["ids"]["wikidata_id"] = qid
            
            # DOI from Wikidata properties
            if "doi" in result:
                simplified["ids"]["doi"] = result.get("doi", {}).get("value")
            if "isbn" in result:
                simplified["ids"]["isbn"] = result.get("isbn", {}).get("value")
            
        elif connector_name == "matilda":
            # Matilda has nested texts structure
            texts = result.get("texts", [])
            if texts and len(texts) > 0:
                first_text = texts[0]
                
                # Extract title
                title_list = first_text.get("title", [])
                if title_list:
                    simplified["title"] = title_list[0]
                
                # Extract first author
                authors = first_text.get("author", [])
                if authors and len(authors) > 0:
                    first_author = authors[0]
                    full_name = first_author.get("fullName")
                    last_name = first_author.get("lastName", [])
                    if full_name:
                        simplified["first_author"] = extract_family_name(full_name) or full_name
                    elif last_name:
                        simplified["first_author"] = last_name[0] if isinstance(last_name, list) else last_name
                
                # Extract year
                pub_date = first_text.get("publicationDate")
                if pub_date:
                    year = extract_year(pub_date)
                    simplified["year"] = year
                
                # Extract journal
                support = first_text.get("support", [])
                if support and len(support) > 0:
                    support_type = support[0].get("type")
                    if support_type == "title":
                        simplified["journal"] = support[0].get("value")
                
                # Extract IDs - DOI
                identifiers = first_text.get("identifier", [])
                for identifier in identifiers:
                    if "doi" in identifier:
                        doi_list = identifier.get("doi", [])
                        if doi_list:
                            simplified["ids"]["doi"] = doi_list[0]
                    if "isbn" in identifier:
                        isbn_list = identifier.get("isbn", [])
                        if isbn_list:
                            simplified["ids"]["isbn"] = isbn_list[0]
            
            # Matilda work ID
            simplified["ids"]["matilda_id"] = result.get("id")
            
        elif connector_name == "opencitations":
            # OpenCitations SPARQL binding format
            simplified["title"] = result.get("title", {}).get("value")
            
            # Extract author
            author_value = result.get("author", {}).get("value")
            if author_value:
                simplified["first_author"] = extract_family_name(author_value) or author_value
            
            # Extract year
            pub_date = result.get("pub_date", {}).get("value")
            if pub_date:
                year = extract_year(pub_date)
                simplified["year"] = year
            
            # Extract journal
            simplified["journal"] = result.get("venue", {}).get("value")
            
            # Extract IDs
            simplified["ids"]["omid"] = result.get("omid") or result.get("br", {}).get("value")
            if "doi" in result:
                simplified["ids"]["doi"] = result.get("doi", {}).get("value")
    
    except Exception as e:
        logger.warning(f"Error extracting simplified result from {connector_name}: {e}")
    
    return simplified


def custom_match(
    reference: Reference,
    result_simplified: Dict[str, Any]
) -> tuple[bool, Dict[str, Any]]:
    """Custom matching logic comparing title, first author, and year separately.
    
    Matching criteria:
    - Title similarity >= 90 (on 0-100 scale)
    - First author similarity >= 70 (on 0-100 scale) 
    - Year within ±1 year
    
    A match is declared if:
    - Title matches AND (author matches OR year matches)
    
    Args:
        reference: Reference object to match against
        result_simplified: Simplified result dictionary
        
    Returns:
        Tuple of (is_match: bool, match_details: Dict)
    """
    match_details = {
        "title_similarity": 0.0,
        "author_similarity": 0.0,
        "year_match": False,
        "year_diff": None
    }
    
    # Extract reference fields
    ref_title = reference.full_title or ""
    ref_year = None
    ref_first_author = None
    
    # Get year from reference
    if hasattr(reference, 'publication_date') and reference.publication_date:
        ref_year = extract_year(str(reference.publication_date))
    elif hasattr(reference, 'year') and reference.year:
        ref_year = extract_year(str(reference.year))
    
    # Get first author from reference
    if hasattr(reference, 'authors') and reference.authors:
        first_author = reference.authors[0]
        if isinstance(first_author, str):
            ref_first_author = extract_family_name(first_author)
        elif isinstance(first_author, dict):
            author_name = first_author.get('name') or first_author.get('display_name')
            if author_name:
                ref_first_author = extract_family_name(author_name)
        elif hasattr(first_author, 'display_name'):
            ref_first_author = extract_family_name(str(first_author.display_name))
        elif hasattr(first_author, 'surname'):
            ref_first_author = str(first_author.surname)
    
    # Extract result fields
    result_title = result_simplified.get("title") or ""
    result_year = result_simplified.get("year")
    result_first_author = result_simplified.get("first_author") or ""
    
    # Calculate title similarity (returns 0-100)
    if ref_title and result_title:
        match_details["title_similarity"] = calculate_title_similarity(ref_title, result_title)
    
    # Calculate author similarity (returns 0-100)
    if ref_first_author and result_first_author:
        # Normalize author names for comparison
        ref_author_norm = ref_first_author.lower().strip()
        result_author_norm = result_first_author.lower().strip()
        match_details["author_similarity"] = fuzz.ratio(ref_author_norm, result_author_norm)
    
    # Calculate year match
    if ref_year and result_year:
        try:
            ref_year_int = int(ref_year) if not isinstance(ref_year, int) else ref_year
            result_year_int = int(result_year) if not isinstance(result_year, int) else result_year
            year_diff = abs(ref_year_int - result_year_int)
            match_details["year_diff"] = year_diff
            match_details["year_match"] = year_diff <= 1
        except (ValueError, TypeError):
            match_details["year_match"] = False
    
    # Determine if it's a match
    # Title must match AND (author OR year must match)
    title_matches = match_details["title_similarity"] >= 90.0
    author_matches = match_details["author_similarity"] >= 70.0
    year_matches = match_details["year_match"]
    
    is_match = title_matches and (author_matches or year_matches)
    
    return is_match, match_details


def search_with_connector(
    connector: Any,
    connector_name: str,
    reference: Reference,
    original_string: str,
    top_k: int = 10,
    match_threshold: float = 0.9
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
            "error": None
        },
        "id_search": {
            "success": False,
            "num_results": 0,
            "top_result": None,
            "error": None
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
            
            # Extract simplified result
            result_simplified = extract_simplified_result(top_result, connector_name)
            
            # Evaluate match using custom matching logic
            is_match, match_details = custom_match(reference, result_simplified)
            
            result["metadata_search"]["top_result"] = {
                "ids": result_simplified["ids"],
                "title": result_simplified["title"],
                "first_author": result_simplified["first_author"],
                "year": result_simplified["year"],
                "journal": result_simplified["journal"],
                "is_match": is_match,
                "match_details": match_details
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
    match_threshold: float
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
        match_threshold=match_threshold
    )
    return api_name, result


def process_single_reference(
    ref_data: Dict[str, Any],
    connectors: Dict[str, Any],
    skip_apis: List[str],
    top_k: int,
    match_threshold: float,
    idx: int,
    total: int
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
        name: conn for name, conn in connectors.items() 
        if name not in skip_apis
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
                match_threshold
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
                    "metadata_search": {"success": False, "error": str(e)},
                    "id_search": {"success": False, "error": str(e)}
                }
    
    # Compile results
    processed_ref = {
        **ref_data,
        "search_results": search_results
    }
    
    return processed_ref


def process_references(
    references: List[Dict[str, Any]],
    connectors: Dict[str, Any],
    skip_apis: List[str],
    top_k: int,
    match_threshold: float,
    output_file: str,
    max_workers: int = 1
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
                    total
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
                        logger.info(f"Saving intermediate results ({completed}/{total} completed)")
                        save_results(processed_refs, output_file, is_final=False)
                        
                except Exception as e:
                    logger.error(f"Error processing reference {ref_data.get('ref_id', idx)}: {e}")
    
    return processed_refs


def save_results(
    references: List[Dict[str, Any]],
    output_file: str,
    is_final: bool = True,
    metadata: Optional[Dict[str, Any]] = None
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
            "search_results": ref.get("search_results", {})
        }
        simplified_refs.append(simplified)
    
    # Generate filename with metadata if final save
    if is_final and metadata:
        # Extract metadata for filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        limit = metadata.get("limit", "all")
        apis = "_".join(metadata.get("apis_used", []))
        
        # Modify output filename to include metadata
        path = Path(output_file)
        new_filename = f"{path.stem}_{timestamp}_limit{limit}_{apis}{path.suffix}"
        output_file = str(path.parent / new_filename)
    
    # Save JSON only
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simplified_refs, f, indent=2, ensure_ascii=False)
    
    logger.info(f"{'Final' if is_final else 'Intermediate'} results saved to {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Run API search benchmark on a pre-built test set"
    )
    parser.add_argument(
        "--test-set",
        type=str,
        required=True,
        default="scripts/api_search_test_set.jsonl",
        help="Path to test set file (.json or .jsonl)"
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for matches (default: 0.9)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/api_search_benchmark_results.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--skip-apis",
        type=str,
        default="opencitations",
        help="Comma-separated list of APIs to skip (e.g., 'matilda,wikidata')"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve per search (default: 10)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to random N references for testing (default: process all)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing references (default: 1, sequential)"
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="Email address for OpenAlex polite pool access (highly recommended to avoid rate limiting)"
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
        logger.info(f"Randomly sampling {args.limit} references from {len(references)} total")
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
            logger.warning("OpenAlex connector initialized without email - may experience rate limiting. Use --email to avoid this.")
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
        max_workers=args.max_workers
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
        "total_references": len(processed_refs)
    }
    
    final_output_file = save_results(processed_refs, args.output, is_final=True, metadata=final_metadata)
    
    logger.info("=" * 60)
    logger.info("Benchmark complete!")
    logger.info(f"Results saved to: {final_output_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

