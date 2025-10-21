#!/usr/bin/env python3
import json
import random
import yaml
import pandas as pd
import pymupdf
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any


def load_linkedbook_jsonl(file_path: Path) -> List[Dict]:
    """Load linkedbook JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_json_dict(file_path: Path) -> Dict:
    """Load JSON file with dictionary structure."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_variants(file_path: Path) -> Dict:
    """Load prompt variants from YAML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_pdf_text(pdf_path: Path, max_chars: int = 100000) -> str:
    """Extract text from PDF using PyMuPDF."""
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


def convert_linkedbook_tags_to_json(tags: Dict) -> Dict:
    """Convert linkedbook tags to the expected JSON format."""
    from citation_index.utils import parse_author_high_precision
    
    # Use high-precision author parser
    author_str = tags.get('author', '')
    authors = parse_author_high_precision(author_str)
    
    reference = {
        "authors": authors,
        "full_title": tags.get('title', '').strip(),
        "journal_title": tags.get('journal', tags.get('publicationnumber', '')).strip(),
        "volume": tags.get('volume', '').strip(),
        "issue": tags.get('issue', '').strip(),
        "pages": tags.get('pagination', '').strip(),
        "publication_date": tags.get('year', '').strip(),
        "publisher": tags.get('publisher', '').strip(),
        "publication_place": tags.get('publicationplace', '').strip()
    }
    
    return {"reference": reference}


def group_linkedbook_references(data: List[Dict]) -> List[Dict]:
    """
    Group linkedbook references into batches.
    Most groups: 10-50 references
    Edge cases: 1-2 references, or 100+ references
    """
    grouped = []
    i = 0
    
    while i < len(data):
        remaining = len(data) - i
        
        # Determine group size with weighted probabilities
        rand = random.random()
        if rand < 0.05:  # 5% chance of tiny group (1-2)
            group_size = random.randint(1, min(2, remaining))
        elif rand < 0.10 and remaining >= 100:  # 5% chance of very large group (100+) only if enough data
            group_size = random.randint(100, min(200, remaining))
        elif remaining >= 10:  # 90% chance of normal group (10-50) if enough data
            group_size = random.randint(10, min(50, remaining))
        else:  # Less than 10 remaining, take all
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


def stratified_sample_cex(cex_data: Dict, n_samples: int = None, sample_rate: float = 0.1) -> List[str]:
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
                             n_samples: int = None, sample_rate: float = 0.1) -> List[str]:
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


def create_conversation(input_text: str, output_json: Dict, variant: Dict) -> Dict:
    """Create a conversation-style training example."""
    system_prompt = variant['system'].strip()
    user_prompt = variant['user'].replace('[[input_text]]', input_text).strip()
    assistant_response = json.dumps(output_json, ensure_ascii=False)
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def process_dataset(linkedbook_groups: List[Dict],
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
                    output_file: Path,
                    include_pdf: bool = True):
    """Process all data and create conversation-style training examples.
    
    Args:
        include_pdf: If True, include CEX and EXCITE PDF-based examples.
                     If False, only include linkedbook reference-only examples.
    """
    
    all_examples = []
    parsing_variant_list = list(parsing_variants['variants'].values())
    extraction_variant_list = list(extraction_variants['variants'].values())
    
    # Create language lookup for EXCITE files
    excite_lang_lookup = dict(zip(
        excite_pdf_df['file_id'].astype(str),
        excite_pdf_df['lang']
    ))
    
    # Process linkedbook groups
    print(f"Processing {len(linkedbook_groups)} linkedbook groups...")
    for group in linkedbook_groups:
        refs = group['references']
        
        # Concatenate reference strings
        input_text = "\n".join(ref['reference'] for ref in refs)
        
        # Convert all references to JSON format
        output_refs = [convert_linkedbook_tags_to_json(ref['tags']) for ref in refs]
        output_json = {"references": output_refs}
        
        # Use parsing variants for linkedbook (references already extracted)
        variant = random.choice(parsing_variant_list)
        example = create_conversation(input_text, output_json, variant)
        example['source'] = 'linkedbook'
        example['ref_count'] = len(refs)
        example['languages'] = list(set(ref.get('language', 'UNKNOWN') for ref in refs))
        all_examples.append(example)
    
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
        example = create_conversation(input_text, output_json, variant)
        example['source'] = 'cex'
        example['category'] = cex_data[file_id].get('category', 'UNKNOWN')
        example['file_id'] = file_id
        example['ref_count'] = len(parsed_refs)
        example['language'] = 'en'  # CEX is all English
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
        example = create_conversation(input_text, output_json, variant)
        example['source'] = 'excite'
        example['file_id'] = file_id
        example['ref_count'] = len(parsed_refs)
        example['language'] = excite_lang_lookup.get(str(file_id), 'UNKNOWN')
        all_examples.append(example)
    
    # Shuffle examples
    random.shuffle(all_examples)
    
    # Write to output file
    print(f"Writing {len(all_examples)} examples to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset saved to {output_file}")
    print(f"Total examples: {len(all_examples)}")
    print(f"  - Linkedbook groups: {sum(1 for ex in all_examples if ex['source'] == 'linkedbook')}")
    print(f"  - CEX PDFs: {sum(1 for ex in all_examples if ex['source'] == 'cex')}")
    print(f"  - EXCITE PDFs: {sum(1 for ex in all_examples if ex['source'] == 'excite')}")
    
    # Statistics on group sizes for linkedbook
    linkedbook_sizes = [ex['ref_count'] for ex in all_examples if ex['source'] == 'linkedbook']
    if linkedbook_sizes:
        print(f"\nLinkedBook group size statistics:")
        print(f"  - Min: {min(linkedbook_sizes)}")
        print(f"  - Max: {max(linkedbook_sizes)}")
        print(f"  - Mean: {sum(linkedbook_sizes) / len(linkedbook_sizes):.1f}")
        print(f"  - Median: {sorted(linkedbook_sizes)[len(linkedbook_sizes)//2]}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate finetuning dataset with grouped references')
    parser.add_argument('--include-pdf', action='store_true', default=True,
                        help='Include PDF-based examples (CEX and EXCITE). Default: True')
    parser.add_argument('--no-pdf', dest='include_pdf', action='store_false',
                        help='Exclude PDF-based examples, only use linkedbook references')
    parser.add_argument('--output-suffix', type=str, default='',
                        help='Suffix to add to output filenames (e.g., "_nopdf")')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent  # benchmarks/ directory
    output_dir = script_dir  # finetune/ directory
    
    # Load data
    print("Loading data...")
    linkedbook_train = load_linkedbook_jsonl(data_dir / 'linkedbook' / 'linkedbooks_train_references.jsonl')
    linkedbook_valid = load_linkedbook_jsonl(data_dir / 'linkedbook' / 'linkedbooks_valid_references.jsonl')
    cex_data = load_json_dict(data_dir / 'cex' / 'all_references.json')
    excite_data = load_json_dict(data_dir / 'excite' / 'all_references.json')
    excite_pdf_df = pd.read_csv(data_dir / 'excite' / 'pdf_files_info.csv')
    
    print(f"Loaded {len(linkedbook_train)} linkedbook train examples")
    print(f"Loaded {len(linkedbook_valid)} linkedbook validation examples")
    print(f"Loaded {len(cex_data)} CEX entries")
    print(f"Loaded {len(excite_data)} EXCITE entries")
    print(f"Loaded {len(excite_pdf_df)} EXCITE PDF info entries")
    
    # Group linkedbook references
    print("\nGrouping linkedbook references...")
    linkedbook_train_groups = group_linkedbook_references(linkedbook_train)
    linkedbook_valid_groups = group_linkedbook_references(linkedbook_valid)
    print(f"Created {len(linkedbook_train_groups)} train groups")
    print(f"Created {len(linkedbook_valid_groups)} validation groups")
    
    # Define directories
    cex_pdf_dir = data_dir / 'cex' / 'all_pdfs'
    cex_xml_dir = data_dir / 'cex' / 'all_xmls'
    excite_pdf_dir = data_dir / 'excite' / 'all_pdfs'
    excite_xml_dir = data_dir / 'excite' / 'all_xml'
    
    # Load variants
    parsing_variants_path = data_dir.parent / 'prompts' / 'reference_parsing_variants.yaml'
    extraction_variants_path = data_dir.parent / 'prompts' / 'reference_extraction_and_parsing_variants.yaml'
    parsing_variants = load_variants(parsing_variants_path)
    extraction_variants = load_variants(extraction_variants_path)
    print(f"Loaded {len(parsing_variants['variants'])} parsing prompt variants")
    print(f"Loaded {len(extraction_variants['variants'])} extraction+parsing prompt variants")
    
    # Sample CEX and EXCITE for training (10% of total combined)
    print("\nSampling CEX data for training (10% of 112 = ~11 documents)...")
    n_cex_train = max(10, int(len(cex_data) * 0.1))
    cex_train_file_ids = stratified_sample_cex(cex_data, n_samples=n_cex_train)
    print(f"Sampled {len(cex_train_file_ids)} CEX train file_ids")
    
    # For validation, sample 2-5 from remaining
    remaining_cex_data = {k: v for k, v in cex_data.items() if k not in cex_train_file_ids}
    n_cex_valid = random.randint(2, 5)
    cex_valid_file_ids = stratified_sample_cex(remaining_cex_data, n_samples=n_cex_valid)
    print(f"Sampled {len(cex_valid_file_ids)} CEX validation file_ids")
    
    print("\nSampling EXCITE data for training (10% of 351 = ~35 documents)...")
    n_excite_train = max(30, int(len(excite_data) * 0.1))
    excite_train_file_ids = stratified_sample_excite(excite_pdf_df, excite_data, n_samples=n_excite_train)
    print(f"Sampled {len(excite_train_file_ids)} EXCITE train file_ids")
    
    # For validation, sample 2-5 from remaining
    remaining_excite_df = excite_pdf_df[~excite_pdf_df['file_id'].astype(str).isin(excite_train_file_ids)]
    remaining_excite_data = {k: v for k, v in excite_data.items() if k not in excite_train_file_ids}
    n_excite_valid = random.randint(2, 5)
    excite_valid_file_ids = stratified_sample_excite(remaining_excite_df, remaining_excite_data, n_samples=n_excite_valid)
    print(f"Sampled {len(excite_valid_file_ids)} EXCITE validation file_ids")
    
    # Save used IDs for exclusion in future test sets
    used_ids = {
        "cex": {
            "train": cex_train_file_ids,
            "valid": cex_valid_file_ids,
            "all_used": cex_train_file_ids + cex_valid_file_ids
        },
        "excite": {
            "train": excite_train_file_ids,
            "valid": excite_valid_file_ids,
            "all_used": excite_train_file_ids + excite_valid_file_ids
        }
    }
    
    used_ids_file = output_dir / 'finetuning_used_ids.json'
    with open(used_ids_file, 'w', encoding='utf-8') as f:
        json.dump(used_ids, f, indent=2, ensure_ascii=False)
    print(f"\nSaved used IDs to {used_ids_file}")
    print(f"  - CEX: {len(used_ids['cex']['all_used'])} documents used (exclude from test)")
    print(f"  - EXCITE: {len(used_ids['excite']['all_used'])} documents used (exclude from test)")
    
    # Create training dataset
    print("\n" + "="*60)
    print(f"Creating TRAINING dataset (include_pdf={args.include_pdf})...")
    print("="*60)
    train_output_file = output_dir / f'finetuning_train_group{args.output_suffix}.json'
    process_dataset(
        linkedbook_train_groups,
        cex_train_file_ids,
        cex_data,
        cex_pdf_dir,
        cex_xml_dir,
        excite_train_file_ids,
        excite_data,
        excite_pdf_dir,
        excite_xml_dir,
        excite_pdf_df,
        parsing_variants,
        extraction_variants,
        train_output_file,
        include_pdf=args.include_pdf
    )
    
    # Create validation dataset
    print("\n" + "="*60)
    print(f"Creating VALIDATION dataset (include_pdf={args.include_pdf})...")
    print("="*60)
    valid_output_file = output_dir / f'finetuning_valid_group{args.output_suffix}.json'
    process_dataset(
        linkedbook_valid_groups,
        cex_valid_file_ids,
        cex_data,
        cex_pdf_dir,
        cex_xml_dir,
        excite_valid_file_ids,
        excite_data,
        excite_pdf_dir,
        excite_xml_dir,
        excite_pdf_df,
        parsing_variants,
        extraction_variants,
        valid_output_file,
        include_pdf=args.include_pdf
    )
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()

