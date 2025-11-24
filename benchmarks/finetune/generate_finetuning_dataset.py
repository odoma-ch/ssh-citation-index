#!/usr/bin/env python3
"""
Generate finetuning datasets for reference extraction and parsing.

Supports two modes:
- single: Each reference becomes its own training example (1 ref → 1 example)
- group: Multiple references grouped together (N refs → 1 example)
- both: Generate both single and group datasets (default)
"""

import json
import random
import yaml
import pandas as pd
import pymupdf
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional


# =============================================================================
# CONFIGURATION - Adjust these variables to control dataset sampling
# =============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# CEX dataset sampling (112 total documents)
CEX_TRAIN_RATE = 0.30      # 10% for training (~11 documents)
CEX_TRAIN_MIN = 10          # Minimum training documents
CEX_VALID_MIN = 2           # Minimum validation documents
CEX_VALID_MAX = 5           # Maximum validation documents

# EXCITE dataset sampling (351 total documents)
EXCITE_TRAIN_RATE = 0.30    # 10% for training (~35 documents)
EXCITE_TRAIN_MIN = 30       # Minimum training documents
EXCITE_VALID_MIN = 2        # Minimum validation documents
EXCITE_VALID_MAX = 5        # Maximum validation documents

# LinkedBook dataset sampling
# Note: We filter ~45% of LinkedBook data due to unparsed authors, so we sample more to compensate
LINKEDBOOK_USE_ALL = False          # If True, use all data; if False, use MAX_TRAIN/VALID
LINKEDBOOK_MAX_TRAIN = 1000        # Maximum training examples to use (increased to compensate for ~45% filtering)
LINKEDBOOK_MAX_VALID = 100          # Maximum validation examples to use (increased to compensate for ~45% filtering)
LINKEDBOOK_VALIDATE = True          # Drop broken references (missing critical fields)

# Dataset mixing strategy
SINGLE_MODE_RATIO = 0.3          # 50% of references as single examples, 50% as groups (applies to all datasets)
GROUP_FROM_SINGLE_PROB = 0.1     # 10% chance to create small groups from single-ref datasets

# Reference grouping (for group mode)
GROUP_TINY_PROB = 0.2    # 5% chance of 1-2 references
GROUP_LARGE_PROB = 0.05     # 10% chance of 100+ references (if enough data)
GROUP_NORMAL_MIN = 3       # Normal group size minimum
GROUP_NORMAL_MAX = 10       # Normal group size maximum
GROUP_LARGE_MIN = 15       # Large group size minimum
GROUP_LARGE_MAX = 20       # Large group size maximum

# =============================================================================


def load_linkedbook_jsonl(file_path: Path) -> List[Dict]:
    """Load linkedbook JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def validate_linkedbook_reference(item: Dict) -> tuple[bool, str]:
    """
    Validate a LinkedBook reference entry.
    
    Returns:
        (is_valid, reason): True if valid, False with reason if invalid
    
    Validation rules:
    1. Must have title and title must be >= 5 characters
    2. If has title, must have either author or year
    3. Must not have 'abbreviation' or 'numbered_ref' tag
    """
    tags = item.get('tags', {})
    
    # Check for abbreviation or numbered_ref tags
    if 'abbreviation' in tags or 'numbered_ref' in tags:
        return False, "has_abbreviation_or_numbered_ref_tag"
    
    # Check title
    title = tags.get('title', '').strip()
    if not title or len(title) < 5:
        return False, "missing_or_short_title"
    
    # Check author and year
    author = tags.get('author', '').strip()
    year = tags.get('year', '').strip()
    
    if not author and not year:
        return False, "missing_author_and_year"
    
    return True, "valid"


def filter_linkedbook_data(data: List[Dict], max_count: int = None) -> tuple[List[Dict], Dict]:
    """
    Filter LinkedBook data to remove broken references.
    
    Returns:
        (filtered_data, stats): Filtered list and statistics dictionary
    """
    if not LINKEDBOOK_VALIDATE:
        # No validation, just apply max_count
        if max_count and len(data) > max_count:
            return data[:max_count], {"total": len(data), "kept": max_count, "filtered": 0}
        return data, {"total": len(data), "kept": len(data), "filtered": 0}
    
    filtered = []
    filter_stats = defaultdict(int)
    
    for item in data:
        is_valid, reason = validate_linkedbook_reference(item)
        if is_valid:
            filtered.append(item)
        else:
            filter_stats[reason] += 1
    
    # Apply max_count after filtering
    original_filtered_count = len(filtered)
    if max_count and len(filtered) > max_count:
        filtered = filtered[:max_count]
    
    stats = {
        "total": len(data),
        "kept": len(filtered),
        "filtered": len(data) - original_filtered_count,
        "by_reason": dict(filter_stats)
    }
    
    return filtered, stats


def load_json_dict(file_path: Path) -> Dict:
    """Load JSON file with dictionary structure."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_variants(file_path: Path) -> Dict:
    """Load prompt variants from YAML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_pdf_text(pdf_path: Path, max_chars: int = 100000) -> str:
    """Extract text from PDF using PyMuPDF (max 100K characters)."""
    try:
        doc = pymupdf.open(str(pdf_path))
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) > max_chars:
                break
        doc.close()
        return text[:max_chars]
    except Exception as e:
        print(f"Error extracting PDF {pdf_path}: {e}")
        return ""


def clean_publication_date(date_str: str) -> str:
    """Clean publication date by removing asterisks and normalizing punctuation."""
    if not date_str:
        return ""
    
    # Remove asterisks
    cleaned = date_str.replace('*', '').strip()
    
    # Remove trailing periods (but keep internal ones for ranges like "1990.")
    cleaned = cleaned.rstrip('.')
    
    # Remove extra commas
    while ',,' in cleaned:
        cleaned = cleaned.replace(',,', ',')
    
    return cleaned.strip()


def validate_parsed_reference(reference: Dict, original_text: str = "") -> tuple[bool, str]:
    """
    Validate that a parsed reference meets quality standards.
    
    Returns:
        (is_valid, reason): True if valid, False with reason if invalid
    """
    authors = reference.get('authors', [])
    
    # Rule 1: Must have authors array (not None, not missing)
    if authors is None:
        return False, "authors_is_none"
    
    # Rule 2: Authors must not contain ANY strings (including empty strings)
    # Authors should be dictionaries with first_name, middle_name, surname keys
    for author in authors:
        if isinstance(author, str):
            # No strings allowed in authors array, not even empty strings
            # If no authors, use empty array [] or properly structured empty objects
            return False, "unparsed_author_string"
    
    # Rule 3: If original text clearly has an author, must not be empty
    if original_text:
        # Check if text starts with name-like pattern (Capital Letter followed by lowercase or period)
        import re
        name_pattern = r'^[A-Z][a-z]+[\s,.]|^[A-Z]\.'
        if re.match(name_pattern, original_text.strip()):
            if not authors or (len(authors) == 1 and authors[0] == {"first_name": "", "middle_name": "", "surname": ""}):
                return False, "missing_obvious_author"
    
    return True, "valid"


def convert_linkedbook_tags_to_json(tags: Dict, for_single: bool = False) -> tuple[Dict, bool, str]:
    """Convert linkedbook tags to the expected JSON format.
    
    Args:
        tags: Dictionary of linkedbook tags
        for_single: If True, wrap in single-item list format for single mode
    
    Returns:
        (json_output, is_valid, reason): Output dict, validity flag, and reason if invalid
    """
    from citation_index.utils import parse_author_high_precision
    
    # Use high-precision author parser
    author_str = tags.get('author', '')
    authors = parse_author_high_precision(author_str)
    
    # Clean publication date
    raw_date = tags.get('year', '').strip()
    cleaned_date = clean_publication_date(raw_date)
    
    reference = {
        "authors": authors,
        "full_title": tags.get('title', '').strip(),
        "journal_title": tags.get('journal', tags.get('publicationnumber', '')).strip(),
        "volume": tags.get('volume', '').strip(),
        "issue": tags.get('issue', '').strip(),
        "pages": tags.get('pagination', '').strip(),
        "publication_date": cleaned_date,
        "publisher": tags.get('publisher', '').strip(),
        "publication_place": tags.get('publicationplace', '').strip()
    }
    
    # Validate the reference
    original_text = tags.get('original_reference', '')  # Will be passed in separately
    is_valid, reason = validate_parsed_reference(reference, original_text)
    
    if for_single:
        return {"references": [{"reference": reference}]}, is_valid, reason
    else:
        return {"reference": reference}, is_valid, reason


def group_linkedbook_references(data: List[Dict]) -> List[Dict]:
    """
    Group linkedbook references into batches.
    Uses configuration variables GROUP_* to control size distribution.
    """
    grouped = []
    i = 0
    
    while i < len(data):
        remaining = len(data) - i
        
        # Determine group size with weighted probabilities
        rand = random.random()
        if rand < GROUP_TINY_PROB:  # Chance of tiny group (1-2)
            group_size = random.randint(1, min(2, remaining))
        elif rand < GROUP_LARGE_PROB and remaining >= GROUP_LARGE_MIN:  # Chance of very large group
            group_size = random.randint(GROUP_LARGE_MIN, min(GROUP_LARGE_MAX, remaining))
        elif remaining >= GROUP_NORMAL_MIN:  # Normal group size
            group_size = random.randint(GROUP_NORMAL_MIN, min(GROUP_NORMAL_MAX, remaining))
        else:  # Less than normal minimum remaining, take all
            group_size = remaining
        
        # Ensure we have at least 1
        group_size = max(1, group_size)
        
        group = data[i:i + group_size]
        grouped.append({
            'references': group,
            'size': group_size
        })
        i += group_size
    
    return grouped


def load_cex_parsed_references(file_id: str, xml_dir: Path) -> List[Dict]:
    """Load parsed references from CEX XML file."""
    from citation_index.core.models import References
    
    xml_path = xml_dir / f"{file_id}.xml"
    if not xml_path.exists():
        return []
    
    try:
        refs = References.from_xml(file_path=str(xml_path))
        # Convert to JSON format
        result = []
        for ref in refs:
            ref_dict = {
                "authors": [],
                "full_title": ref.full_title or "",
                "journal_title": ref.journal_title or "",
                "volume": ref.volume or "",
                "issue": ref.issue or "",
                "pages": ref.pages or "",
                "publication_date": ref.publication_date or "",
                "publisher": ref.publisher or "",
                "publication_place": ref.publication_place or ""
            }
            
            # Parse authors
            if ref.authors:
                for author in ref.authors:
                    author_dict = {
                        "first_name": getattr(author, 'first_name', '') or "",
                        "middle_name": getattr(author, 'middle_name', '') or "",
                        "surname": getattr(author, 'surname', '') or getattr(author, 'name', '') or ""
                    }
                    ref_dict["authors"].append(author_dict)
            
            result.append({"reference": ref_dict})
        
        return result
    except Exception as e:
        print(f"Error loading CEX XML for {file_id}: {e}")
        return []


def load_excite_parsed_references(file_id: str, xml_dir: Path) -> List[Dict]:
    """Load parsed references from EXCITE XML file."""
    from citation_index.core.models import References
    
    xml_path = xml_dir / f"{file_id}.xml"
    if not xml_path.exists():
        return []
    
    try:
        refs = References.from_excite_xml(str(xml_path))
        # Convert to JSON format
        result = []
        for ref in refs:
            ref_dict = {
                "authors": [],
                "full_title": ref.full_title or "",
                "journal_title": ref.journal_title or "",
                "volume": ref.volume or "",
                "issue": ref.issue or "",
                "pages": ref.pages or "",
                "publication_date": ref.publication_date or "",
                "publisher": ref.publisher or "",
                "publication_place": ref.publication_place or ""
            }
            
            # Parse authors
            if ref.authors:
                for author in ref.authors:
                    author_dict = {
                        "first_name": getattr(author, 'first_name', '') or "",
                        "middle_name": getattr(author, 'middle_name', '') or "",
                        "surname": getattr(author, 'surname', '') or getattr(author, 'name', '') or ""
                    }
                    ref_dict["authors"].append(author_dict)
            
            result.append({"reference": ref_dict})
        
        return result
    except Exception as e:
        print(f"Error loading EXCITE XML for {file_id}: {e}")
        return []


def stratified_sample_cex(cex_data: Dict, n_samples: Optional[int] = None, sample_rate: float = 0.1) -> List[str]:
    """
    Sample CEX data stratified by category. Returns list of file_ids.
    
    Args:
        cex_data: Dictionary of CEX entries
        n_samples: Exact number of samples to take (if provided, overrides sample_rate)
        sample_rate: Fraction to sample from each category (used if n_samples not provided)
    """
    # Group by category
    by_category = defaultdict(list)
    for file_id, entry in cex_data.items():
        category = entry.get('category', 'UNKNOWN')
        by_category[category].append(file_id)
    
    if n_samples is not None:
        # Take proportional samples from each category to reach n_samples total
        total_docs = len(cex_data)
        sampled = []
        
        # Calculate samples per category proportionally
        for category, file_ids in by_category.items():
            category_proportion = len(file_ids) / total_docs
            category_samples = max(1, int(n_samples * category_proportion))
            category_samples = min(category_samples, len(file_ids))
            sampled.extend(random.sample(file_ids, category_samples))
        
        # If we have too many, randomly drop some
        if len(sampled) > n_samples:
            sampled = random.sample(sampled, n_samples)
        
        return sampled
    else:
        # Original behavior: sample by rate from each category
        sampled = []
        for category, file_ids in by_category.items():
            n = max(1, int(len(file_ids) * sample_rate))
            sampled.extend(random.sample(file_ids, min(n, len(file_ids))))
        
        return sampled


def stratified_sample_excite(excite_pdf_df: pd.DataFrame, excite_data: Dict, 
                             n_samples: Optional[int] = None, sample_rate: float = 0.1) -> List[str]:
    """
    Sample EXCITE data stratified by class. Returns list of file_ids that have references.
    
    Args:
        excite_pdf_df: DataFrame with PDF info
        excite_data: Dictionary of EXCITE entries
        n_samples: Exact number of samples to take (if provided, overrides sample_rate)
        sample_rate: Fraction to sample from each class (used if n_samples not provided)
    """
    # Convert PDF file_ids to strings for comparison
    excite_pdf_df = excite_pdf_df.copy()
    excite_pdf_df['file_id'] = excite_pdf_df['file_id'].astype(str)
    
    # Filter to files that have both PDF and parsed XML
    file_ids_in_data = set(excite_data.keys())
    excite_df_filtered = excite_pdf_df[excite_pdf_df['file_id'].isin(file_ids_in_data)]
    
    if n_samples is not None:
        # Take proportional samples from each class to reach n_samples total
        sampled = []
        classes = sorted(excite_df_filtered['class'].unique())
        
        # Calculate samples per class proportionally
        for class_num in classes:
            class_df = excite_df_filtered[excite_df_filtered['class'] == class_num]
            if len(class_df) == 0:
                continue
            class_proportion = len(class_df) / len(excite_df_filtered)
            class_samples = max(1, int(n_samples * class_proportion))
            class_samples = min(class_samples, len(class_df))
            sampled_ids = class_df.sample(n=class_samples, random_state=42)['file_id'].tolist()
            sampled.extend(sampled_ids)
        
        # If we have too many, randomly drop some
        if len(sampled) > n_samples:
            sampled = random.sample(sampled, n_samples)
        
        return sampled
    else:
        # Original behavior: sample by rate from each class
        sampled = []
        for class_num in excite_df_filtered['class'].unique():
            class_df = excite_df_filtered[excite_df_filtered['class'] == class_num]
            if len(class_df) == 0:
                continue
            n = max(1, int(len(class_df) * sample_rate))
            sampled_ids = class_df.sample(n=min(n, len(class_df)), random_state=42)['file_id'].tolist()
            sampled.extend(sampled_ids)
        
        return sampled


def select_variant_weighted(variants: Dict) -> Dict:
    """Select a variant using weighted probabilities.
    
    Distribution:
    - 40% Variant 1 (detailed)
    - 25% Variant 2 (minimal)
    - 25% Variant 3 (instruction-focused)
    - 5% Variant 4 (ultra-minimal)
    - 5% No prompt (minimal input/output format)
    """
    rand = random.random()
    
    # Get variant keys (might be int or str depending on YAML)
    variant_dict = variants['variants']
    keys = list(variant_dict.keys())
    
    # Ensure we have at least 4 variants
    if len(keys) < 4:
        raise ValueError(f"Expected at least 4 variants, got {len(keys)}")
    
    # Select based on probabilities
    if rand < 0.40:  # 40%
        return variant_dict[keys[0]]
    elif rand < 0.65:  # 25%
        return variant_dict[keys[1]]
    elif rand < 0.90:  # 25%
        return variant_dict[keys[2]]
    elif rand < 0.95:  # 5%
        return variant_dict[keys[3]]
    else:  # 5%
        return None  # No prompt variant


def create_conversation(input_text: str, output_json: Dict, variant: Dict) -> Dict:
    """Create a conversation-style training example."""
    assistant_response = json.dumps(output_json, ensure_ascii=False)
    
    if variant is None:
        # No prompt variant - just Input/Output
        return {
            "messages": [
                {"role": "system", "content": "You are a bibliographic reference parser."},
                {"role": "user", "content": f"Input: {input_text}"},
                {"role": "assistant", "content": assistant_response}
            ]
        }
    
    system_prompt = variant['system'].strip()
    user_prompt = variant['user'].replace('[[input_text]]', input_text).strip()
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def process_dataset_single(linkedbook_data: List[Dict], 
                           cex_file_ids: List[str],
                           cex_data: Dict,
                           cex_xml_dir: Path,
                           excite_file_ids: List[str],
                           excite_data: Dict,
                           excite_xml_dir: Path,
                           excite_pdf_df: pd.DataFrame,
                           variants: Dict,
                           split: str):
    """Process all data and create single-reference training examples.
    
    Args:
        split: Dataset split ('train' or 'valid')
    
    Returns:
        (training_examples, metadata): Lists of examples and metadata entries
    """
    
    all_examples = []
    all_metadata = []
    variant_list = list(variants['variants'].values())
    
    # Create language lookup for EXCITE files
    excite_lang_lookup = dict(zip(
        excite_pdf_df['file_id'].astype(str),
        excite_pdf_df['lang']
    ))
    
    # Process linkedbook data
    print(f"Processing {len(linkedbook_data)} linkedbook examples...")
    linkedbook_filtered = 0
    linkedbook_kept = 0
    filter_reasons = defaultdict(int)
    
    for item in linkedbook_data:
        input_text = item['reference']
        
        # Add reference text to tags for validation
        item['tags']['original_reference'] = input_text
        output_json, is_valid, reason = convert_linkedbook_tags_to_json(item['tags'], for_single=True)
        
        # Filter out invalid references
        if not is_valid:
            linkedbook_filtered += 1
            filter_reasons[reason] += 1
            continue
        
        linkedbook_kept += 1
        
        # Create metadata entry
        metadata = {
            'source': 'linkedbook',
            'split': split,
            'mode': 'single',
            'language': item.get('language', 'UNKNOWN'),
            'reference_text': input_text,
            'parsed_object': output_json
        }
        all_metadata.append(metadata)
        
        # Randomly select a variant
        variant = random.choice(variant_list)
        example = create_conversation(input_text, output_json, variant)
        example['source'] = 'linkedbook'
        example['split'] = split
        example['mode'] = 'single'
        example['language'] = item.get('language', 'UNKNOWN')
        all_examples.append(example)
    
    print(f"  LinkedBook: kept {linkedbook_kept}, filtered {linkedbook_filtered}")
    if filter_reasons:
        for reason, count in filter_reasons.items():
            print(f"    - {reason}: {count}")
    
    # Process CEX samples
    print(f"Processing {len(cex_file_ids)} CEX file_ids...")
    for file_id in cex_file_ids:
        # Get raw references from all_references.json
        cex_entry = cex_data.get(file_id, {})
        raw_refs = cex_entry.get('references', [])
        
        # Load parsed references from XML
        parsed_refs = load_cex_parsed_references(file_id, cex_xml_dir)
        
        # Match raw and parsed references
        # If counts don't match, skip this file
        if len(raw_refs) != len(parsed_refs):
            print(f"Warning: CEX {file_id} has {len(raw_refs)} raw refs but {len(parsed_refs)} parsed refs. Skipping.")
            continue
        
        # Create training examples for each reference
        for raw_ref, parsed_ref_list in zip(raw_refs, parsed_refs):
            output_json = {"references": [parsed_ref_list]}
            
            # Create metadata entry
            metadata = {
                'source': 'cex',
                'split': split,
                'mode': 'single',
                'file_id': file_id,
                'category': cex_entry.get('category', 'UNKNOWN'),
                'language': 'en',
                'reference_text': raw_ref,
                'parsed_object': output_json
            }
            all_metadata.append(metadata)
            
            variant = random.choice(variant_list)
            example = create_conversation(raw_ref, output_json, variant)
            example['source'] = 'cex'
            example['split'] = split
            example['mode'] = 'single'
            example['category'] = cex_entry.get('category', 'UNKNOWN')
            example['file_id'] = file_id
            example['language'] = 'en'  # CEX is all English
            all_examples.append(example)
    
    # Process EXCITE samples
    print(f"Processing {len(excite_file_ids)} EXCITE file_ids...")
    for file_id in excite_file_ids:
        # Get raw references from all_references.json
        excite_entry = excite_data.get(file_id, {})
        raw_refs = excite_entry.get('references', [])
        
        # Load parsed references from XML
        parsed_refs = load_excite_parsed_references(file_id, excite_xml_dir)
        
        # Match raw and parsed references
        # If counts don't match, skip this file
        if len(raw_refs) != len(parsed_refs):
            print(f"Warning: EXCITE {file_id} has {len(raw_refs)} raw refs but {len(parsed_refs)} parsed refs. Skipping.")
            continue
        
        # Create training examples for each reference
        for raw_ref, parsed_ref_list in zip(raw_refs, parsed_refs):
            output_json = {"references": [parsed_ref_list]}
            
            # Create metadata entry
            metadata = {
                'source': 'excite',
                'split': split,
                'mode': 'single',
                'file_id': file_id,
                'language': excite_lang_lookup.get(str(file_id), 'UNKNOWN'),
                'reference_text': raw_ref,
                'parsed_object': output_json
            }
            all_metadata.append(metadata)
            
            variant = random.choice(variant_list)
            example = create_conversation(raw_ref, output_json, variant)
            example['source'] = 'excite'
            example['split'] = split
            example['mode'] = 'single'
            example['file_id'] = file_id
            example['language'] = excite_lang_lookup.get(str(file_id), 'UNKNOWN')
            all_examples.append(example)
    
    print(f"Processed {len(all_examples)} examples for {split} split (single mode)")
    print(f"  - Linkedbook: {sum(1 for ex in all_examples if ex['source'] == 'linkedbook')}")
    print(f"  - CEX: {sum(1 for ex in all_examples if ex['source'] == 'cex')}")
    print(f"  - EXCITE: {sum(1 for ex in all_examples if ex['source'] == 'excite')}")
    
    return all_examples, all_metadata


def process_dataset_group(linkedbook_groups: List[Dict],
                          cex_file_ids: List[str],
                          cex_data: Dict,
                          cex_pdf_dir: Path,
                          cex_xml_dir: Path,
                          excite_file_ids: List[str],
                          excite_data: Dict,
                          excite_pdf_dir: Path,
                          excite_xml_dir: Path,
                          excite_pdf_df: pd.DataFrame,
                          parsing_variants: Dict,
                          extraction_variants: Dict,
                          split: str,
                          include_pdf: bool = True):
    """Process all data and create grouped-reference training examples.
    
    Args:
        split: Dataset split ('train' or 'valid')
        include_pdf: If True, include CEX and EXCITE PDF-based examples.
                     If False, only include linkedbook reference-only examples.
    
    Returns:
        (training_examples, metadata): Lists of examples and metadata entries
    """
    
    all_examples = []
    all_metadata = []
    parsing_variant_list = list(parsing_variants['variants'].values())
    extraction_variant_list = list(extraction_variants['variants'].values()) if extraction_variants else []
    
    # Create language lookup for EXCITE files
    excite_lang_lookup = dict(zip(
        excite_pdf_df['file_id'].astype(str),
        excite_pdf_df['lang']
    ))
    
    # Process linkedbook groups
    print(f"Processing {len(linkedbook_groups)} linkedbook groups...")
    linkedbook_groups_filtered = 0
    linkedbook_groups_kept = 0
    group_filter_reasons = defaultdict(int)
    
    for group in linkedbook_groups:
        refs = group['references']
        
        # Convert all references to JSON format and validate
        output_refs = []
        valid_ref_texts = []
        all_valid = True
        
        for ref in refs:
            # Add reference text for validation
            ref['tags']['original_reference'] = ref['reference']
            output, is_valid, reason = convert_linkedbook_tags_to_json(ref['tags'], for_single=False)
            
            if not is_valid:
                # Skip entire group if any reference is invalid
                all_valid = False
                group_filter_reasons[reason] += 1
                break
            
            output_refs.append(output)
            valid_ref_texts.append(ref['reference'])
        
        if not all_valid:
            linkedbook_groups_filtered += 1
            continue
        
        linkedbook_groups_kept += 1
        
        # Concatenate reference strings
        input_text = "\n".join(valid_ref_texts)
        output_json = {"references": output_refs}
        
        # Create metadata entry
        metadata = {
            'source': 'linkedbook',
            'split': split,
            'mode': 'group',
            'ref_count': len(refs),
            'languages': list(set(ref.get('language', 'UNKNOWN') for ref in refs)),
            'reference_text': input_text,
            'parsed_object': output_json
        }
        all_metadata.append(metadata)
        
        # Use parsing variants for linkedbook (references already extracted)
        variant = random.choice(parsing_variant_list)
        example = create_conversation(input_text, output_json, variant)
        example['source'] = 'linkedbook'
        example['split'] = split
        example['mode'] = 'group'
        example['ref_count'] = len(refs)
        example['languages'] = list(set(ref.get('language', 'UNKNOWN') for ref in refs))
        all_examples.append(example)
    
    print(f"  LinkedBook groups: kept {linkedbook_groups_kept}, filtered {linkedbook_groups_filtered}")
    if group_filter_reasons:
        for reason, count in group_filter_reasons.items():
            print(f"    - {reason}: {count}")
    
    # Process CEX samples
    print(f"Processing {len(cex_file_ids)} CEX file_ids...")
    for file_id in cex_file_ids:
        # Load parsed references from XML
        parsed_refs = load_cex_parsed_references(file_id, cex_xml_dir)
        if not parsed_refs:
            print(f"Warning: CEX {file_id} has no parsed references. Skipping.")
            continue
        
        if include_pdf:
            # Get PDF path and extract full text
            pdf_path = cex_pdf_dir / f"{file_id}.pdf"
            if not pdf_path.exists():
                print(f"Warning: CEX PDF {file_id} not found. Skipping.")
                continue
            
            # Extract PDF text
            pdf_text = extract_pdf_text(pdf_path)
            if not pdf_text:
                print(f"Warning: CEX {file_id} PDF text extraction failed. Skipping.")
                continue
            
            input_text = pdf_text
            # Use extraction+parsing variants for full PDF
            variant = random.choice(extraction_variant_list)
        else:
            # Use reference strings from the data
            if file_id not in cex_data:
                print(f"Warning: CEX {file_id} not in reference data. Skipping.")
                continue
            
            ref_strings = cex_data[file_id].get('references', [])
            if not ref_strings:
                print(f"Warning: CEX {file_id} has no reference strings. Skipping.")
                continue
            
            # Concatenate reference strings
            input_text = "\n".join(ref_strings)
            # Use parsing variants for reference strings only
            variant = random.choice(parsing_variant_list)
        
        output_json = {"references": parsed_refs}
        
        # Create metadata entry
        metadata = {
            'source': 'cex',
            'split': split,
            'mode': 'group',
            'file_id': file_id,
            'category': cex_data[file_id].get('category', 'UNKNOWN'),
            'ref_count': len(parsed_refs),
            'language': 'en',
            'reference_text': input_text if not include_pdf else '(PDF text)',
            'parsed_object': output_json,
            'uses_pdf': include_pdf
        }
        all_metadata.append(metadata)
        
        example = create_conversation(input_text, output_json, variant)
        example['source'] = 'cex'
        example['split'] = split
        example['mode'] = 'group'
        example['category'] = cex_data[file_id].get('category', 'UNKNOWN')
        example['file_id'] = file_id
        example['ref_count'] = len(parsed_refs)
        example['language'] = 'en'  # CEX is all English
        example['uses_pdf'] = include_pdf
        all_examples.append(example)
    
    # Process EXCITE samples
    print(f"Processing {len(excite_file_ids)} EXCITE file_ids...")
    for file_id in excite_file_ids:
        # Load parsed references from XML
        parsed_refs = load_excite_parsed_references(file_id, excite_xml_dir)
        if not parsed_refs:
            print(f"Warning: EXCITE {file_id} has no parsed references. Skipping.")
            continue
        
        if include_pdf:
            # Get PDF path and extract full text
            pdf_path = excite_pdf_dir / f"{file_id}.pdf"
            if not pdf_path.exists():
                print(f"Warning: EXCITE PDF {file_id} not found. Skipping.")
                continue
            
            # Extract PDF text
            pdf_text = extract_pdf_text(pdf_path)
            if not pdf_text:
                print(f"Warning: EXCITE {file_id} PDF text extraction failed. Skipping.")
                continue
            
            input_text = pdf_text
            # Use extraction+parsing variants for full PDF
            variant = random.choice(extraction_variant_list)
        else:
            # Use reference strings from the data
            if file_id not in excite_data:
                print(f"Warning: EXCITE {file_id} not in reference data. Skipping.")
                continue
            
            ref_strings = excite_data[file_id].get('references', [])
            if not ref_strings:
                print(f"Warning: EXCITE {file_id} has no reference strings. Skipping.")
                continue
            
            # Concatenate reference strings
            input_text = "\n".join(ref_strings)
            # Use parsing variants for reference strings only
            variant = random.choice(parsing_variant_list)
        
        output_json = {"references": parsed_refs}
        
        # Create metadata entry
        metadata = {
            'source': 'excite',
            'split': split,
            'mode': 'group',
            'file_id': file_id,
            'ref_count': len(parsed_refs),
            'language': excite_lang_lookup.get(str(file_id), 'UNKNOWN'),
            'reference_text': input_text if not include_pdf else '(PDF text)',
            'parsed_object': output_json,
            'uses_pdf': include_pdf
        }
        all_metadata.append(metadata)
        
        example = create_conversation(input_text, output_json, variant)
        example['source'] = 'excite'
        example['split'] = split
        example['mode'] = 'group'
        example['file_id'] = file_id
        example['ref_count'] = len(parsed_refs)
        example['language'] = excite_lang_lookup.get(str(file_id), 'UNKNOWN')
        example['uses_pdf'] = include_pdf
        all_examples.append(example)
    
    print(f"Processed {len(all_examples)} examples for {split} split (group mode)")
    print(f"  - Linkedbook groups: {sum(1 for ex in all_examples if ex['source'] == 'linkedbook')}")
    print(f"  - CEX: {sum(1 for ex in all_examples if ex['source'] == 'cex')}")
    print(f"  - EXCITE: {sum(1 for ex in all_examples if ex['source'] == 'excite')}")
    
    # Statistics on group sizes for linkedbook
    linkedbook_sizes = [ex['ref_count'] for ex in all_examples if ex['source'] == 'linkedbook']
    if linkedbook_sizes:
        print(f"LinkedBook group size statistics:")
        print(f"  - Min: {min(linkedbook_sizes)}")
        print(f"  - Max: {max(linkedbook_sizes)}")
        print(f"  - Mean: {sum(linkedbook_sizes) / len(linkedbook_sizes):.1f}")
        print(f"  - Median: {sorted(linkedbook_sizes)[len(linkedbook_sizes)//2]}")
    
    return all_examples, all_metadata


def process_unified_dataset(linkedbook_data: List[Dict],
                             cex_file_ids: List[str],
                             cex_data: Dict,
                             cex_xml_dir: Path,
                             excite_file_ids: List[str],
                             excite_data: Dict,
                             excite_xml_dir: Path,
                             excite_pdf_df: pd.DataFrame,
                             variants: Dict,
                             split: str):
    """Process all data and create a unified dataset mixing single and grouped examples.
    
    Returns:
        (training_examples, metadata): Lists of examples and metadata entries
    """
    all_examples = []
    all_metadata = []
    # Note: We use weighted selection via select_variant_weighted(), not uniform random choice
    
    # Create language lookup for EXCITE files
    excite_lang_lookup = dict(zip(
        excite_pdf_df['file_id'].astype(str),
        excite_pdf_df['lang']
    ))
    
    # PART 1: Process LinkedBook data (mix of single and grouped)
    print(f"\nProcessing {len(linkedbook_data)} LinkedBook examples...")
    
    # Shuffle and split into single vs group
    random.shuffle(linkedbook_data)
    n_single = int(len(linkedbook_data) * SINGLE_MODE_RATIO)
    linkedbook_single = linkedbook_data[:n_single]
    linkedbook_for_grouping = linkedbook_data[n_single:]
    
    linkedbook_filtered_single = 0
    linkedbook_kept_single = 0
    filter_reasons_single = defaultdict(int)
    
    # Process single references
    for item in linkedbook_single:
        input_text = item['reference']
        item['tags']['original_reference'] = input_text
        output_json, is_valid, reason = convert_linkedbook_tags_to_json(item['tags'], for_single=True)
        
        if not is_valid:
            linkedbook_filtered_single += 1
            filter_reasons_single[reason] += 1
            continue
        
        linkedbook_kept_single += 1
        
        metadata = {
            'source': 'linkedbook',
            'split': split,
            'mode': 'single',
            'language': item.get('language', 'UNKNOWN'),
            'reference_text': input_text,
            'parsed_object': output_json
        }
        all_metadata.append(metadata)
        
        variant = select_variant_weighted(variants)
        example = create_conversation(input_text, output_json, variant)
        example['source'] = 'linkedbook'
        example['split'] = split
        example['mode'] = 'single'
        example['category'] = None  # LinkedBook doesn't have categories
        example['file_id'] = None   # LinkedBook single refs don't have file_id
        example['ref_count'] = 1
        example['language'] = item.get('language', 'UNKNOWN')
        all_examples.append(example)
    
    print(f"  LinkedBook single: kept {linkedbook_kept_single}, filtered {linkedbook_filtered_single}")
    
    # Group remaining LinkedBook references
    linkedbook_groups = group_linkedbook_references(linkedbook_for_grouping)
    linkedbook_filtered_groups = 0
    linkedbook_kept_groups = 0
    filter_reasons_groups = defaultdict(int)
    
    for group in linkedbook_groups:
        refs = group['references']
        output_refs = []
        valid_ref_texts = []
        all_valid = True
        
        for ref in refs:
            ref['tags']['original_reference'] = ref['reference']
            output, is_valid, reason = convert_linkedbook_tags_to_json(ref['tags'], for_single=False)
            
            if not is_valid:
                all_valid = False
                filter_reasons_groups[reason] += 1
                break
            
            output_refs.append(output)
            valid_ref_texts.append(ref['reference'])
        
        if not all_valid:
            linkedbook_filtered_groups += 1
            continue
        
        linkedbook_kept_groups += 1
        input_text = "\n".join(valid_ref_texts)
        output_json = {"references": output_refs}
        
        metadata = {
            'source': 'linkedbook',
            'split': split,
            'mode': 'group',
            'ref_count': len(refs),
            'languages': list(set(ref.get('language', 'UNKNOWN') for ref in refs)),
            'reference_text': input_text,
            'parsed_object': output_json
        }
        all_metadata.append(metadata)
        
        variant = select_variant_weighted(variants)
        example = create_conversation(input_text, output_json, variant)
        example['source'] = 'linkedbook'
        example['split'] = split
        example['mode'] = 'group'
        example['category'] = None  # LinkedBook doesn't have categories
        example['file_id'] = None   # LinkedBook groups don't have file_id
        example['ref_count'] = len(refs)
        # Use single language field for consistency
        languages = list(set(ref.get('language', 'UNKNOWN') for ref in refs))
        example['language'] = languages[0] if len(languages) == 1 else 'mixed'
        all_examples.append(example)
    
    print(f"  LinkedBook groups: kept {linkedbook_kept_groups}, filtered {linkedbook_filtered_groups}")
    
    # PART 2: Process CEX samples (split into single and groups)
    print(f"\nProcessing {len(cex_file_ids)} CEX documents...")
    
    # Collect all CEX references
    cex_all_refs = []
    for file_id in cex_file_ids:
        parsed_refs = load_cex_parsed_references(file_id, cex_xml_dir)
        if not parsed_refs or len(parsed_refs) == 0:
            continue
        
        if file_id not in cex_data:
            continue
        ref_strings = cex_data[file_id].get('references', [])
        if len(ref_strings) != len(parsed_refs):
            print(f"  Warning: CEX {file_id} ref count mismatch, skipping")
            continue
        
        # Store each reference with metadata
        for ref_str, parsed_ref in zip(ref_strings, parsed_refs):
            cex_all_refs.append({
                'file_id': file_id,
                'category': cex_data[file_id].get('category', 'UNKNOWN'),
                'ref_string': ref_str,
                'parsed_ref': parsed_ref
            })
    
    # Shuffle and split into single vs group
    random.shuffle(cex_all_refs)
    n_cex_single = int(len(cex_all_refs) * SINGLE_MODE_RATIO)
    cex_single = cex_all_refs[:n_cex_single]
    cex_for_grouping = cex_all_refs[n_cex_single:]
    
    # Process single CEX references
    for item in cex_single:
        input_text = item['ref_string']
        output_json = {"references": [item['parsed_ref']]}
        
        metadata = {
            'source': 'cex',
            'split': split,
            'mode': 'single',
            'file_id': item['file_id'],
            'category': item['category'],
            'ref_count': 1,
            'language': 'en',
            'reference_text': input_text,
            'parsed_object': output_json
        }
        all_metadata.append(metadata)
        
        variant = select_variant_weighted(variants)
        example = create_conversation(input_text, output_json, variant)
        example['source'] = 'cex'
        example['split'] = split
        example['mode'] = 'single'
        example['category'] = item['category']
        example['file_id'] = str(item['file_id'])
        example['ref_count'] = 1
        example['language'] = 'en'
        all_examples.append(example)
    
    # Group remaining CEX references
    cex_groups = group_linkedbook_references([{'reference': item['ref_string'], 'tags': {}} for item in cex_for_grouping])
    for i, group in enumerate(cex_groups):
        refs = group['references']
        ref_strings = [ref['reference'] for ref in refs]
        
        # Map back to parsed refs (use same grouping order)
        group_parsed_refs = []
        for ref_str in ref_strings:
            # Find matching parsed ref
            for item in cex_for_grouping:
                if item['ref_string'] == ref_str:
                    group_parsed_refs.append(item['parsed_ref'])
                    break
        
        if not group_parsed_refs:
            continue
        
        input_text = "\n".join(ref_strings)
        output_json = {"references": group_parsed_refs}
        
        # Use first ref's metadata for the group
        first_item = next((item for item in cex_for_grouping if item['ref_string'] == ref_strings[0]), None)
        if not first_item:
            continue
        
        metadata = {
            'source': 'cex',
            'split': split,
            'mode': 'group',
            'file_id': first_item['file_id'],
            'category': first_item['category'],
            'ref_count': len(group_parsed_refs),
            'language': 'en',
            'reference_text': input_text,
            'parsed_object': output_json
        }
        all_metadata.append(metadata)
        
        variant = select_variant_weighted(variants)
        example = create_conversation(input_text, output_json, variant)
        example['source'] = 'cex'
        example['split'] = split
        example['mode'] = 'group'
        example['category'] = first_item['category']
        example['file_id'] = str(first_item['file_id'])
        example['ref_count'] = len(group_parsed_refs)
        example['language'] = 'en'
        all_examples.append(example)
    
    print(f"  CEX: {len(cex_single)} single, {len(cex_groups)} groups")
    
    # PART 3: Process EXCITE samples (split into single and groups)
    print(f"Processing {len(excite_file_ids)} EXCITE documents...")
    
    # Collect all EXCITE references
    excite_all_refs = []
    for file_id in excite_file_ids:
        parsed_refs = load_excite_parsed_references(file_id, excite_xml_dir)
        if not parsed_refs or len(parsed_refs) == 0:
            continue
        
        if file_id not in excite_data:
            continue
        ref_strings = excite_data[file_id].get('references', [])
        if len(ref_strings) != len(parsed_refs):
            print(f"  Warning: EXCITE {file_id} ref count mismatch, skipping")
            continue
        
        # Store each reference with metadata
        for ref_str, parsed_ref in zip(ref_strings, parsed_refs):
            excite_all_refs.append({
                'file_id': file_id,
                'ref_string': ref_str,
                'parsed_ref': parsed_ref,
                'language': excite_lang_lookup.get(str(file_id), 'UNKNOWN')
            })
    
    # Shuffle and split into single vs group
    random.shuffle(excite_all_refs)
    n_excite_single = int(len(excite_all_refs) * SINGLE_MODE_RATIO)
    excite_single = excite_all_refs[:n_excite_single]
    excite_for_grouping = excite_all_refs[n_excite_single:]
    
    # Process single EXCITE references
    for item in excite_single:
        input_text = item['ref_string']
        output_json = {"references": [item['parsed_ref']]}
        
        metadata = {
            'source': 'excite',
            'split': split,
            'mode': 'single',
            'file_id': item['file_id'],
            'ref_count': 1,
            'language': item['language'],
            'reference_text': input_text,
            'parsed_object': output_json
        }
        all_metadata.append(metadata)
        
        variant = select_variant_weighted(variants)
        example = create_conversation(input_text, output_json, variant)
        example['source'] = 'excite'
        example['split'] = split
        example['mode'] = 'single'
        example['category'] = None  # EXCITE doesn't have categories
        example['file_id'] = str(item['file_id'])
        example['ref_count'] = 1
        example['language'] = item['language']
        all_examples.append(example)
    
    # Group remaining EXCITE references
    excite_groups = group_linkedbook_references([{'reference': item['ref_string'], 'tags': {}} for item in excite_for_grouping])
    for i, group in enumerate(excite_groups):
        refs = group['references']
        ref_strings = [ref['reference'] for ref in refs]
        
        # Map back to parsed refs (use same grouping order)
        group_parsed_refs = []
        for ref_str in ref_strings:
            # Find matching parsed ref
            for item in excite_for_grouping:
                if item['ref_string'] == ref_str:
                    group_parsed_refs.append(item['parsed_ref'])
                    break
        
        if not group_parsed_refs:
            continue
        
        input_text = "\n".join(ref_strings)
        output_json = {"references": group_parsed_refs}
        
        # Use first ref's metadata for the group
        first_item = next((item for item in excite_for_grouping if item['ref_string'] == ref_strings[0]), None)
        if not first_item:
            continue
        
        metadata = {
            'source': 'excite',
            'split': split,
            'mode': 'group',
            'file_id': first_item['file_id'],
            'ref_count': len(group_parsed_refs),
            'language': first_item['language'],
            'reference_text': input_text,
            'parsed_object': output_json
        }
        all_metadata.append(metadata)
        
        variant = select_variant_weighted(variants)
        example = create_conversation(input_text, output_json, variant)
        example['source'] = 'excite'
        example['split'] = split
        example['mode'] = 'group'
        example['category'] = None  # EXCITE doesn't have categories
        example['file_id'] = str(first_item['file_id'])
        example['ref_count'] = len(group_parsed_refs)
        example['language'] = first_item['language']
        all_examples.append(example)
    
    print(f"  EXCITE: {len(excite_single)} single, {len(excite_groups)} groups")
    
    print(f"\nProcessed {len(all_examples)} total examples for {split} split")
    print(f"  - Single-ref examples: {sum(1 for ex in all_examples if ex['mode'] == 'single')}")
    print(f"  - Multi-ref examples: {sum(1 for ex in all_examples if ex['mode'] == 'group')}")
    print(f"  - By source: LinkedBook={sum(1 for ex in all_examples if ex['source'] == 'linkedbook')}, "
          f"CEX={sum(1 for ex in all_examples if ex['source'] == 'cex')}, "
          f"EXCITE={sum(1 for ex in all_examples if ex['source'] == 'excite')}")
    
    return all_examples, all_metadata


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Generate finetuning dataset for reference extraction and parsing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script generates a unified dataset mixing single and grouped reference examples.

Examples:
  # Generate standard dataset
  python generate_finetuning_dataset.py

  # Generate with custom output name
  python generate_finetuning_dataset.py --output-suffix "_v2"
        """
    )
    parser.add_argument('--output-suffix', type=str, default='',
                        help='Suffix to add to output filenames (e.g., "_v2")')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Define paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent  # benchmarks/ directory
    output_dir = script_dir  # finetune/ directory
    
    # Load data
    print("Loading data...")
    linkedbook_train_raw = load_linkedbook_jsonl(data_dir / 'linkedbook' / 'linkedbooks_train_references.jsonl')
    linkedbook_valid_raw = load_linkedbook_jsonl(data_dir / 'linkedbook' / 'linkedbooks_valid_references.jsonl')
    cex_data = load_json_dict(data_dir / 'cex' / 'all_references.json')
    excite_data = load_json_dict(data_dir / 'excite' / 'all_references.json')
    excite_pdf_df = pd.read_csv(data_dir / 'excite' / 'pdf_files_info.csv')
    
    print(f"Loaded {len(linkedbook_train_raw)} linkedbook train examples (raw)")
    print(f"Loaded {len(linkedbook_valid_raw)} linkedbook validation examples (raw)")
    print(f"Loaded {len(cex_data)} CEX entries")
    print(f"Loaded {len(excite_data)} EXCITE entries")
    print(f"Loaded {len(excite_pdf_df)} EXCITE PDF info entries")
    
    # Filter and sample LinkedBook data
    print("\nFiltering LinkedBook data...")
    max_train = None if LINKEDBOOK_USE_ALL else LINKEDBOOK_MAX_TRAIN
    max_valid = None if LINKEDBOOK_USE_ALL else LINKEDBOOK_MAX_VALID
    
    linkedbook_train, train_stats = filter_linkedbook_data(linkedbook_train_raw, max_count=max_train)
    linkedbook_valid, valid_stats = filter_linkedbook_data(linkedbook_valid_raw, max_count=max_valid)
    
    print(f"LinkedBook TRAIN: {train_stats['total']} → {train_stats['kept']} (filtered {train_stats['filtered']})")
    if train_stats.get('by_reason'):
        for reason, count in train_stats['by_reason'].items():
            print(f"  - {reason}: {count}")
    
    print(f"LinkedBook VALID: {valid_stats['total']} → {valid_stats['kept']} (filtered {valid_stats['filtered']})")
    if valid_stats.get('by_reason'):
        for reason, count in valid_stats['by_reason'].items():
            print(f"  - {reason}: {count}")
    
    # Define directories
    cex_pdf_dir = data_dir / 'cex' / 'all_pdfs'
    cex_xml_dir = data_dir / 'cex' / 'all_xmls'
    excite_pdf_dir = data_dir / 'excite' / 'all_pdfs'
    excite_xml_dir = data_dir / 'excite' / 'all_xml'
    
    # Load variants
    parsing_variants_path = data_dir.parent / 'prompts' / 'reference_parsing_variants.yaml'
    parsing_variants = load_variants(parsing_variants_path)
    print(f"Loaded {len(parsing_variants['variants'])} prompt variants")
    
    # Load or generate sampling IDs
    used_ids_file = output_dir / 'finetuning_used_ids.json'
    if used_ids_file.exists():
        print("\nLoading existing document IDs for consistency...")
        with open(used_ids_file, 'r', encoding='utf-8') as f:
            used_ids = json.load(f)
        cex_train_file_ids = used_ids['cex']['train']
        cex_valid_file_ids = used_ids['cex']['valid']
        excite_train_file_ids = used_ids['excite']['train']
        excite_valid_file_ids = used_ids['excite']['valid']
        print(f"Using {len(cex_train_file_ids)} CEX train, {len(cex_valid_file_ids)} CEX valid")
        print(f"Using {len(excite_train_file_ids)} EXCITE train, {len(excite_valid_file_ids)} EXCITE valid")
    else:
        print(f"\nSampling CEX data for training ({CEX_TRAIN_RATE*100:.0f}% of total)...")
        n_cex_train = max(CEX_TRAIN_MIN, int(len(cex_data) * CEX_TRAIN_RATE))
        cex_train_file_ids = stratified_sample_cex(cex_data, n_samples=n_cex_train)
        print(f"Sampled {len(cex_train_file_ids)} CEX train file_ids")
        
        # For validation, sample from remaining
        remaining_cex_data = {k: v for k, v in cex_data.items() if k not in cex_train_file_ids}
        n_cex_valid = random.randint(CEX_VALID_MIN, CEX_VALID_MAX)
        cex_valid_file_ids = stratified_sample_cex(remaining_cex_data, n_samples=n_cex_valid)
        print(f"Sampled {len(cex_valid_file_ids)} CEX validation file_ids")
        
        print(f"\nSampling EXCITE data for training ({EXCITE_TRAIN_RATE*100:.0f}% of total)...")
        n_excite_train = max(EXCITE_TRAIN_MIN, int(len(excite_data) * EXCITE_TRAIN_RATE))
        excite_train_file_ids = stratified_sample_excite(excite_pdf_df, excite_data, n_samples=n_excite_train)
        print(f"Sampled {len(excite_train_file_ids)} EXCITE train file_ids")
        
        # For validation, sample from remaining
        remaining_excite_df = excite_pdf_df[~excite_pdf_df['file_id'].astype(str).isin(excite_train_file_ids)]
        remaining_excite_data = {k: v for k, v in excite_data.items() if k not in excite_train_file_ids}
        n_excite_valid = random.randint(EXCITE_VALID_MIN, EXCITE_VALID_MAX)
        excite_valid_file_ids = stratified_sample_excite(remaining_excite_df, remaining_excite_data, n_samples=n_excite_valid)
        print(f"Sampled {len(excite_valid_file_ids)} EXCITE validation file_ids")
        
        # Save used IDs for consistency across runs
        used_ids = {
            "cex": {
                "train": cex_train_file_ids,
                "valid": cex_valid_file_ids,
                "all_used": list(set(cex_train_file_ids + cex_valid_file_ids))
            },
            "excite": {
                "train": excite_train_file_ids,
                "valid": excite_valid_file_ids,
                "all_used": list(set(excite_train_file_ids + excite_valid_file_ids))
            }
        }
        
        with open(used_ids_file, 'w', encoding='utf-8') as f:
            json.dump(used_ids, f, indent=2, ensure_ascii=False)
        print(f"\nSaved used IDs to {used_ids_file}")
        print(f"  - CEX: {len(used_ids['cex']['all_used'])} documents")
        print(f"  - EXCITE: {len(used_ids['excite']['all_used'])} documents")
    
    # Collect all examples and metadata
    all_training_examples = []
    all_metadata = []
    
    # Generate unified datasets (mixing single and grouped)
    print("\n" + "="*60)
    print("GENERATING UNIFIED DATASET")
    print("="*60)
    
    # Create training dataset
    print("\n### TRAINING SET ###")
    train_examples, train_metadata = process_unified_dataset(
        linkedbook_train,
        cex_train_file_ids,
        cex_data,
        cex_xml_dir,
        excite_train_file_ids,
        excite_data,
        excite_xml_dir,
        excite_pdf_df,
        parsing_variants,
        'train'
    )
    all_training_examples.extend(train_examples)
    all_metadata.extend(train_metadata)
    
    # Create validation dataset
    print("\n### VALIDATION SET ###")
    valid_examples, valid_metadata = process_unified_dataset(
        linkedbook_valid,
        cex_valid_file_ids,
        cex_data,
        cex_xml_dir,
        excite_valid_file_ids,
        excite_data,
        excite_xml_dir,
        excite_pdf_df,
        parsing_variants,
        'valid'
    )
    all_training_examples.extend(valid_examples)
    all_metadata.extend(valid_metadata)
    
    # Write output files
    print("\n" + "="*60)
    print("WRITING OUTPUT FILES")
    print("="*60)
    
    # 1. Write metadata file (JSONL)
    metadata_file = output_dir / f'finetuning_references_metadata{args.output_suffix}.jsonl'
    print(f"\nWriting metadata to {metadata_file.name}...")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        for entry in all_metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Saved {len(all_metadata)} metadata entries")
    
    # 2. Write training data file (JSONL)
    # Shuffle examples for better training
    random.shuffle(all_training_examples)
    
    data_file = output_dir / f'finetuning_data{args.output_suffix}.jsonl'
    print(f"\nWriting training data to {data_file.name}...")
    with open(data_file, 'w', encoding='utf-8') as f:
        for example in all_training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"Saved {len(all_training_examples)} training examples")
    
    # Print statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total examples: {len(all_training_examples)}")
    train_count = sum(1 for ex in all_training_examples if ex['split'] == 'train')
    valid_count = sum(1 for ex in all_training_examples if ex['split'] == 'valid')
    single_count = sum(1 for ex in all_training_examples if ex['mode'] == 'single')
    group_count = sum(1 for ex in all_training_examples if ex['mode'] == 'group')
    
    print(f"  By split:")
    print(f"    - train: {train_count} ({train_count/len(all_training_examples)*100:.1f}%)")
    print(f"    - valid: {valid_count} ({valid_count/len(all_training_examples)*100:.1f}%)")
    print(f"  By mode:")
    print(f"    - single: {single_count} ({single_count/len(all_training_examples)*100:.1f}%)")
    print(f"    - group:  {group_count} ({group_count/len(all_training_examples)*100:.1f}%)")
    print(f"  By source:")
    print(f"    - linkedbook: {sum(1 for ex in all_training_examples if ex['source'] == 'linkedbook')}")
    print(f"    - cex: {sum(1 for ex in all_training_examples if ex['source'] == 'cex')}")
    print(f"    - excite: {sum(1 for ex in all_training_examples if ex['source'] == 'excite')}")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()

