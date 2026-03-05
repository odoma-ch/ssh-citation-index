#!/usr/bin/env python3
"""Run citation linking benchmark on LLaMoRe legal study XML files.

Parses all XML files in the legal_study_data_mpilhlt/ folder, extracts raw
reference strings and structured bibliographic data (via TeiBiblParser), then
searches each reference through OpenAlex, OpenCitations, Wikidata, and Matilda.

Results are saved as JSON in the same format as the existing benchmark output,
ready to be consumed by search_results_analysis.ipynb for Argilla upload.

Usage:
    python run_legal_study_benchmark.py
    python run_legal_study_benchmark.py --skip-apis opencitations --email you@example.com
    python run_legal_study_benchmark.py --limit 20 --max-workers 4
"""

import argparse
import importlib.util
import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from lxml import etree

# Add the project root to sys.path so citation_index is importable when running
# this script directly (without editable install).
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from citation_index.core.connectors import (
    MatildaConnector,
    OpenAlexConnector,
    OpenCitationsConnector,
    WikidataConnector,
)
from citation_index.core.models.reference import Reference
from citation_index.core.parsers.tei_bibl_parser import TeiBiblParser
from citation_index.utils.reference_matching import (
    calculate_title_similarity,
    extract_family_name,
    extract_year,
)
from fuzzywuzzy import fuzz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import shared search utilities from run_api_search_benchmark
# ---------------------------------------------------------------------------
# Load the sibling module without requiring it to be on sys.path as a package.
_bench_path = Path(__file__).parent / "run_api_search_benchmark.py"
_spec = importlib.util.spec_from_file_location("run_api_search_benchmark", _bench_path)
_bench_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench_module)

extract_simplified_result = _bench_module.extract_simplified_result
custom_match = _bench_module.custom_match

# ---------------------------------------------------------------------------
# XML namespace constants
# ---------------------------------------------------------------------------
LLAMORE_NS = "https://gitlab.mpcdf.mpg.de/dcfidalgo/llamore"
TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_NS = "http://www.w3.org/XML/1998/namespace"

_DEFAULT_DATA_DIR = Path(__file__).parent / "legal_study_data_mpilhlt"


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

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
                    refs = tei_parser.to_references(list_bibl_el, raise_empty_error=False)
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


def load_legal_study_data(data_dir: Path, tei_parser: TeiBiblParser) -> List[Dict[str, Any]]:
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
            logger.warning(f"Dropping {inst['ref_id']}: parse error (no Reference object)")
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


# ---------------------------------------------------------------------------
# Reference creation
# ---------------------------------------------------------------------------

def create_reference(instance: Dict[str, Any]) -> Reference:
    """Return the parsed ``Reference`` for *instance*, falling back to raw string."""
    ref = instance.get("reference")
    if ref is not None:
        return ref
    raw = instance.get("original_string", "")
    return Reference(full_title=raw[:200])


# ---------------------------------------------------------------------------
# Search utilities  (connector logic stays in run_api_search_benchmark)
# ---------------------------------------------------------------------------

def search_with_connector(
    connector: Any,
    connector_name: str,
    reference: Reference,
    original_string: str,
    top_k: int = 10,
    match_threshold: float = 0.9,
) -> Dict[str, Any]:
    """Search *connector* for *reference* and return a result dict."""
    result: Dict[str, Any] = {
        "metadata_search": {
            "success": False,
            "num_results": 0,
            "top_result": None,
            "error": None,
        }
    }

    try:
        logger.debug(f"Searching {connector_name} …")
        raw_results = connector.search(reference, top_k=top_k)
        result["metadata_search"]["success"] = True
        result["metadata_search"]["num_results"] = len(raw_results)

        if raw_results:
            top_result = raw_results[0]
            result_simplified = extract_simplified_result(top_result, connector, connector_name)
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
    except Exception as exc:
        logger.error(f"{connector_name} search failed: {exc}")
        result["metadata_search"]["error"] = str(exc)

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
    """Thin wrapper for parallel execution – returns *(api_name, result)*."""
    logger.info(f"Searching {api_name} …")
    result = search_with_connector(
        connector=connector,
        connector_name=api_name,
        reference=reference,
        original_string=original_string,
        top_k=top_k,
        match_threshold=match_threshold,
    )
    return api_name, result


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
                save_results(processed, output_file, is_final=False)
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
                        logger.info(f"Saving intermediate results ({completed}/{total} completed)")
                        save_results(processed, output_file, is_final=False)
                except Exception as exc:
                    logger.error(f"Error processing {instance.get('ref_id', idx)}: {exc}")

    return processed


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_output_path(output_arg: Optional[str], apis: List[str], limit: Optional[int]) -> str:
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

    logger.info(f"Parsed intermediate file saved to {output_path} ({len(records)} records)")
    return output_path


def save_results(
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
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
        help="Email for OpenAlex polite-pool access (strongly recommended)",
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
                "OpenAlex connector initialised without email – rate limiting may occur. "
                "Use --email to enable the polite pool."
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
    final_path = save_results(processed, output_file, is_final=True)

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info(f"  Instances processed : {len(processed)}")
    logger.info(f"  Results saved to    : {final_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
