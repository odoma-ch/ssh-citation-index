#!/usr/bin/env python
"""
Migration script for Reference model refactoring.

This script migrates existing Reference JSON data to use the new schema:
- Maps publication_date → publication_year (int) + publication_date_raw (str)
- Moves analytic_title/monographic_title to raw dict
- Extracts identifiers from text fields (DOI:, ISBN:, etc.)
- Populates identifiers list with Identifier objects

Usage:
    python scripts/migrate_references.py input.json output.json
    python scripts/migrate_references.py --in-place data/*.json
    python scripts/migrate_references.py --dir data/references/
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citation_index.core.models import Reference
from citation_index.utils.identifier_parser import parse_identifier, _detect_identifier_type

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_identifiers_from_text(ref_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract identifiers from text fields in reference data.
    
    Looks for patterns like "DOI: 10.1234/foo" in full_title and other text fields.
    """
    identifiers = []
    
    # Fields to search for identifiers
    search_fields = ['full_title', 'journal', 'publisher']
    
    for field in search_fields:
        if field not in ref_data or not ref_data[field]:
            continue
            
        text = ref_data[field]
        
        # Look for inline identifier patterns (DOI:, ISBN:, etc.)
        # Common patterns: "DOI: value", "DOI：value", "doi:value"
        patterns = [
            r'\b(DOI|doi)[:\s：]+([^\s,;]+)',
            r'\b(ISBN|isbn)[:\s：]+([^\s,;]+)',
            r'\b(ISSN|issn)[:\s：]+([^\s,;]+)',
            r'\b(PMID|pmid)[:\s：]+([^\s,;]+)',
            r'\b(arXiv|arxiv)[:\s：]+([^\s,;]+)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                scheme = match.group(1).lower()
                value = match.group(2).strip()
                
                # Parse to get normalized form
                identifier = parse_identifier(value, scheme)
                if identifier:
                    identifiers.append({
                        'scheme': identifier.scheme,
                        'value': identifier.value,
                        'normalized': identifier.normalized
                    })
                    logger.debug(f"Extracted {scheme}: {value} from {field}")
    
    return identifiers


def migrate_reference(ref_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate a single reference dict to new schema.
    
    The Reference model's _migrate_deprecated_fields validator handles most
    of the migration automatically, but we also extract identifiers from text.
    """
    # Let the model handle the basic migration via its validator
    try:
        ref = Reference(**ref_data)
    except Exception as e:
        logger.error(f"Failed to parse reference: {e}")
        logger.debug(f"Reference data: {ref_data}")
        raise
    
    # Extract additional identifiers from text fields if not already present
    if not ref.identifiers:
        extracted = extract_identifiers_from_text(ref_data)
        if extracted:
            # Re-create with extracted identifiers
            updated_data = ref.model_dump()
            updated_data['identifiers'] = extracted
            ref = Reference(**updated_data)
            logger.info(f"Extracted {len(extracted)} identifier(s) from text fields")
    
    # Return serialized form (excludes deprecated fields)
    return ref.model_dump()


def migrate_file(input_path: Path, output_path: Path, in_place: bool = False) -> None:
    """Migrate a single JSON file containing Reference objects."""
    logger.info(f"Processing {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read {input_path}: {e}")
        return
    
    # Handle both single reference and list of references
    if isinstance(data, dict):
        migrated = migrate_reference(data)
    elif isinstance(data, list):
        migrated = []
        for i, ref_data in enumerate(data):
            try:
                migrated.append(migrate_reference(ref_data))
            except Exception as e:
                logger.error(f"Failed to migrate reference {i} in {input_path}: {e}")
                # Continue processing other references
        logger.info(f"Migrated {len(migrated)}/{len(data)} references")
    else:
        logger.error(f"Unknown data format in {input_path}: {type(data)}")
        return
    
    # Write output
    out_path = input_path if in_place else output_path
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(migrated, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote migrated data to {out_path}")
    except Exception as e:
        logger.error(f"Failed to write {out_path}: {e}")


def migrate_directory(dir_path: Path, pattern: str = "*.json", in_place: bool = True) -> None:
    """Migrate all JSON files in a directory."""
    files = list(dir_path.glob(pattern))
    logger.info(f"Found {len(files)} JSON files in {dir_path}")
    
    for file_path in files:
        migrate_file(file_path, file_path, in_place=in_place)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Reference JSON files to new schema"
    )
    parser.add_argument(
        'input',
        nargs='?',
        help='Input JSON file or directory'
    )
    parser.add_argument(
        'output',
        nargs='?',
        help='Output JSON file (if migrating single file)'
    )
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Modify files in place'
    )
    parser.add_argument(
        '--dir',
        help='Directory containing JSON files to migrate'
    )
    parser.add_argument(
        '--pattern',
        default='*.json',
        help='File pattern for directory mode (default: *.json)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Determine mode
    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            logger.error(f"Not a directory: {dir_path}")
            sys.exit(1)
        migrate_directory(dir_path, pattern=args.pattern, in_place=True)
    
    elif args.input:
        input_path = Path(args.input)
        
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)
        
        if input_path.is_dir():
            migrate_directory(input_path, pattern=args.pattern, in_place=args.in_place)
        else:
            # Single file mode
            if args.in_place:
                output_path = input_path
            elif args.output:
                output_path = Path(args.output)
            else:
                logger.error("Output file required unless --in-place is specified")
                sys.exit(1)
            
            migrate_file(input_path, output_path, in_place=args.in_place)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
