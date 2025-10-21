#!/usr/bin/env python3

import json
import random
import yaml
import pandas as pd
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
    
    return {"references": [{"reference": reference}]}


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


def stratified_sample_cex(cex_data: Dict, sample_rate: float = 0.1) -> List[str]:
    """Sample CEX data stratified by category. Returns list of file_ids."""
    # Group by category
    by_category = defaultdict(list)
    for file_id, entry in cex_data.items():
        category = entry.get('category', 'UNKNOWN')
        by_category[category].append(file_id)
    
    # Sample from each category
    sampled = []
    for category, file_ids in by_category.items():
        n_samples = max(1, int(len(file_ids) * sample_rate))
        sampled.extend(random.sample(file_ids, min(n_samples, len(file_ids))))
    
    return sampled


def stratified_sample_excite(excite_pdf_df: pd.DataFrame, excite_data: Dict, sample_rate: float = 0.1) -> List[str]:
    """Sample EXCITE data stratified by class. Returns list of file_ids that have references."""
    # First, filter to only files that have references in excite_data
    file_ids_with_refs = [fid for fid, entry in excite_data.items() 
                          if entry.get('references') and len(entry['references']) > 0]
    
    # Convert PDF file_ids to strings for comparison
    excite_pdf_df = excite_pdf_df.copy()
    excite_pdf_df['file_id'] = excite_pdf_df['file_id'].astype(str)
    
    excite_df_filtered = excite_pdf_df[excite_pdf_df['file_id'].isin(file_ids_with_refs)]
    
    # Group by class
    sampled = []
    for class_num in excite_df_filtered['class'].unique():
        class_df = excite_df_filtered[excite_df_filtered['class'] == class_num]
        if len(class_df) == 0:
            continue
        n_samples = max(1, int(len(class_df) * sample_rate))
        sampled_ids = class_df.sample(n=min(n_samples, len(class_df)), random_state=42)['file_id'].tolist()
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


def process_dataset(linkedbook_data: List[Dict], 
                    cex_file_ids: List[str],
                    cex_data: Dict,
                    cex_xml_dir: Path,
                    excite_file_ids: List[str],
                    excite_data: Dict,
                    excite_xml_dir: Path,
                    excite_pdf_df: pd.DataFrame,
                    variants: Dict,
                    output_file: Path):
    """Process all data and create conversation-style training examples."""
    
    all_examples = []
    variant_list = list(variants['variants'].values())
    
    # Create language lookup for EXCITE files
    excite_lang_lookup = dict(zip(
        excite_pdf_df['file_id'].astype(str),
        excite_pdf_df['lang']
    ))
    
    # Process linkedbook data
    print(f"Processing {len(linkedbook_data)} linkedbook examples...")
    for item in linkedbook_data:
        input_text = item['reference']
        output_json = convert_linkedbook_tags_to_json(item['tags'])
        
        # Randomly select a variant
        variant = random.choice(variant_list)
        example = create_conversation(input_text, output_json, variant)
        example['source'] = 'linkedbook'
        example['language'] = item.get('language', 'UNKNOWN')
        all_examples.append(example)
    
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
            variant = random.choice(variant_list)
            example = create_conversation(raw_ref, output_json, variant)
            example['source'] = 'cex'
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
            variant = random.choice(variant_list)
            example = create_conversation(raw_ref, output_json, variant)
            example['source'] = 'excite'
            example['file_id'] = file_id
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
    print(f"  - Linkedbook: {sum(1 for ex in all_examples if ex['source'] == 'linkedbook')}")
    print(f"  - CEX: {sum(1 for ex in all_examples if ex['source'] == 'cex')}")
    print(f"  - EXCITE: {sum(1 for ex in all_examples if ex['source'] == 'excite')}")


def main():
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
    
    # Define XML directories
    cex_xml_dir = data_dir / 'cex' / 'all_xmls'
    excite_xml_dir = data_dir / 'excite' / 'all_xml'
    
    # Load variants
    variants_path = data_dir.parent / 'prompts' / 'reference_parsing_variants.yaml'
    variants = load_variants(variants_path)
    print(f"Loaded {len(variants['variants'])} prompt variants")
    
    # Load group dataset IDs to use the same documents for consistency
    group_ids_file = output_dir / 'finetuning_used_ids.json'
    if group_ids_file.exists():
        print("\nLoading document IDs from group dataset for consistency...")
        with open(group_ids_file, 'r', encoding='utf-8') as f:
            ids = json.load(f)
        cex_train_file_ids = ids['cex']['train']
        cex_valid_file_ids = ids['cex']['valid']
        excite_train_file_ids = ids['excite']['train']
        excite_valid_file_ids = ids['excite']['valid']
        print(f"Using IDs: {len(cex_train_file_ids)} CEX train, {len(cex_valid_file_ids)} CEX valid")
        print(f"Using IDs: {len(excite_train_file_ids)} EXCITE train, {len(excite_valid_file_ids)} EXCITE valid")
    else:
        # Fallback to sampling if V2 IDs not available
        print("\nV2 IDs not found, using reduced sampling...")
        print("Sampling CEX data...")
        n_cex_train = max(10, int(len(cex_data) * 0.1))
        cex_train_file_ids = stratified_sample_cex(cex_data, sample_rate=0.1)[:n_cex_train]
        
        remaining_cex_data = {k: v for k, v in cex_data.items() if k not in cex_train_file_ids}
        n_cex_valid = random.randint(2, 5)
        cex_valid_file_ids = list(remaining_cex_data.keys())[:n_cex_valid]
        print(f"Sampled {len(cex_train_file_ids)} CEX train, {len(cex_valid_file_ids)} CEX valid")
        
        print("\nSampling EXCITE data...")
        excite_train_file_ids = stratified_sample_excite(excite_pdf_df, excite_data, sample_rate=0.1)[:35]
        
        remaining_excite_df = excite_pdf_df[~excite_pdf_df['file_id'].astype(str).isin(excite_train_file_ids)]
        remaining_excite_data = {k: v for k, v in excite_data.items() if k not in excite_train_file_ids}
        excite_valid_file_ids = stratified_sample_excite(remaining_excite_df, remaining_excite_data, sample_rate=0.1)[:5]
        print(f"Sampled {len(excite_train_file_ids)} EXCITE train, {len(excite_valid_file_ids)} EXCITE valid")
    
    # Save used IDs (same as V2 for consistency)
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
    
    used_ids_file = output_dir / 'finetuning_used_ids.json'
    with open(used_ids_file, 'w', encoding='utf-8') as f:
        json.dump(used_ids, f, indent=2, ensure_ascii=False)
    print(f"\nSaved used IDs to {used_ids_file} (shared with group dataset)")
    print(f"  - CEX: {len(used_ids['cex']['all_used'])} documents")
    print(f"  - EXCITE: {len(used_ids['excite']['all_used'])} documents")
    
    # Create training dataset
    print("\n" + "="*60)
    print("Creating TRAINING dataset...")
    print("="*60)
    process_dataset(
        linkedbook_train,
        cex_train_file_ids,
        cex_data,
        cex_xml_dir,
        excite_train_file_ids,
        excite_data,
        excite_xml_dir,
        excite_pdf_df,
        variants,
        output_dir / 'finetuning_train_single.json'
    )
    
    # Create validation dataset
    print("\n" + "="*60)
    print("Creating VALIDATION dataset...")
    print("="*60)
    process_dataset(
        linkedbook_valid,
        cex_valid_file_ids,
        cex_data,
        cex_xml_dir,
        excite_valid_file_ids,
        excite_data,
        excite_xml_dir,
        excite_pdf_df,
        variants,
        output_dir / 'finetuning_valid_single.json'
    )
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()

