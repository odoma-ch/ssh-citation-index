import json
import re
import random
from collections import defaultdict, Counter
from lingua import Language, LanguageDetectorBuilder
from pathlib import Path

# Input and output paths
input_files = {
    "train": "clean_train.txt",
    "valid": "clean_valid.txt", 
    "test": "clean_test.txt"
}

output_dir = Path("")
output_dir.mkdir(exist_ok=True)

# Language detector setup
languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH, 
             Language.ITALIAN, Language.PORTUGUESE, Language.DUTCH, Language.RUSSIAN]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

def detokenize(tokens):
    """Make a clean reference string from token list."""
    if not tokens:
        return ""
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)        # no space before punctuation
    text = re.sub(r"\s+([\)\]\}»'\"])", r"\1", text)       # no space before closing brackets/quotes
    text = re.sub(r"([\(\[\{«'\"])", r"\1", text)       # no space after opening brackets/quotes
    text = re.sub(r"\s{2,}", " ", text)                   # collapse multiple spaces
    return text.strip()

def detect_language(text):
    """Detect language of the given text."""
    try:
        language = detector.detect_language_of(text)
        return language.iso_code_639_1.name if language else "UNKNOWN"
    except:
        return "UNKNOWN"

def process_file(file_path, dataset_name):
    """Process a single file and extract references with language detection."""
    refs = []
    vocab = set()
    vocab_freq = Counter()
    language_stats = Counter()
    
    print(f"\nProcessing {dataset_name} dataset from {file_path}...")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            in_ref = False
            current_tokens = []
            per_tag_tokens = defaultdict(list)

            for line_num, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 4:
                    continue

                token, t1, t2, t3 = parts[0], parts[1], parts[2], parts[3]

                # Skip DOCSTART
                if token == "-DOCSTART-":
                    continue
                
                # skip if t2 is '*-primary' tag
                if "primary" in t2.lower():
                    continue

                # Build Task1 vocabulary from the whole file
                if t1 not in {"-X-", "o"}:
                    vocab.add(t1)
                    vocab_freq[t1] += 1
                
                # Start of a reference span
                if t3.lower() == "b-r":
                    in_ref = True
                    current_tokens = []
                    per_tag_tokens = defaultdict(list)

                # Inside a reference span
                if in_ref and t3.lower() in {"b-r", "i-r", "e-r"}:
                    current_tokens.append(token)
                    if t1 not in {"-X-", "o"}:
                        per_tag_tokens[t1].append(token)

                # End of a reference span
                if in_ref and t3.lower() == "e-r":
                    ref_str = detokenize(current_tokens)
                    tags = {tag: detokenize(toks) for tag, toks in per_tag_tokens.items()}
                    
                    # Detect language
                    lang = detect_language(ref_str)
                    language_stats[lang] += 1
                    
                    refs.append({
                        "reference": ref_str, 
                        "tags": tags,
                        "language": lang,
                        "dataset": dataset_name
                    })
                    
                    in_ref = False
                    current_tokens = []
                    per_tag_tokens = defaultdict(list)
                    
                # Progress indicator for large files
                if line_num % 100000 == 0:
                    print(f"  Processed {line_num:,} lines...")
                    
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found, skipping...")
        return [], set(), Counter(), Counter()
    
    return refs, vocab, vocab_freq, language_stats

def save_outputs(refs, vocab, dataset_name):
    """Save outputs for a specific dataset."""
    # Save references
    refs_path = output_dir / f"linkedbooks_{dataset_name}_references.jsonl"
    with open(refs_path, "w", encoding="utf-8") as out_f:
        for r in refs:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    return refs_path

def save_vocabulary(all_vocab, dataset_name):
    """Save vocabulary to a single text file."""
    vocab_path = output_dir / f"tag_vocabulary_{dataset_name}.txt"
    with open(vocab_path, "w", encoding="utf-8") as out_v:
        for item in sorted(all_vocab):
            out_v.write(item + "\n")
    return vocab_path

def print_statistics(dataset_name, refs, vocab, vocab_freq, language_stats):
    """Print comprehensive statistics for a dataset."""
    print(f"\n{'='*60}")
    print(f"STATISTICS FOR {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    print(f"Total references extracted: {len(refs):,}")
    print(f"Task1 vocabulary size: {len(vocab):,}")
    
    if refs:
        # Language distribution
        print(f"\nLanguage distribution:")
        for lang, count in language_stats.most_common():
            percentage = (count / len(refs)) * 100
            print(f"  {lang}: {count:,} ({percentage:.1f}%)")
        
        # Tag statistics
        tag_counts = Counter()
        for ref in refs:
            for tag in ref['tags']:
                tag_counts[tag] += 1
        
        print(f"\nTag distribution:")
        for tag, count in tag_counts.most_common():
            percentage = (count / len(refs)) * 100
            print(f"  {tag}: {count:,} ({percentage:.1f}%)")
        
        # Reference length statistics
        ref_lengths = [len(ref['reference'].split()) for ref in refs]
        avg_length = sum(ref_lengths) / len(ref_lengths)
        print(f"\nReference length statistics:")
        print(f"  Average words per reference: {avg_length:.1f}")
        print(f"  Shortest reference: {min(ref_lengths)} words")
        print(f"  Longest reference: {max(ref_lengths)} words")
    
    # Top vocabulary items
    print(f"\nTop 10 most frequent vocabulary items:")
    for item, freq in vocab_freq.most_common(10):
        print(f"  {item}: {freq:,}")

def generate_grouped_set_by_lang(refs, min_refs=10, max_refs=100):
    """
    Generate grouped version of the test set where references are grouped by language.
    Each group contains 10-100 references in the same language.
    
    Args:
        refs: List of reference dictionaries from the test set
        min_refs: Minimum number of references per group (default: 10)
        max_refs: Maximum number of references per group (default: 100)
    
    Returns:
        List of grouped reference objects
    """
    print(f"\nGenerating grouped test set...")
    
    # Group references by language
    refs_by_language = defaultdict(list)
    for ref in refs:
        if ref['dataset'] == 'test':  # Only process test set references
            refs_by_language[ref['language']].append(ref)
    
    grouped_refs = []
    group_id = 1
    
    for language, lang_refs in refs_by_language.items():
        if len(lang_refs) < min_refs:
            # if smaller than minimum, put them all in one group
            print(f"  Language '{language}' has only {len(lang_refs)} references, grouping them together.")
            refs_strings = [ref['reference'] for ref in lang_refs]
            ground_truth = [ref['tags'] for ref in lang_refs]
            
            grouped_refs.append({
                "id": group_id,
                "refs": refs_strings,
                "ground_truth": ground_truth,
                "lang": language
            })
            group_id += 1
            continue
        print(f"  Processing language '{language}' with {len(lang_refs)} references...")
        
        # Shuffle references to create random groups
        shuffled_refs = lang_refs.copy()
        random.shuffle(shuffled_refs)
        
        # Create groups of references
        i = 0
        while i < len(shuffled_refs):
            # Determine group size (between min_refs and max_refs)
            remaining_refs = len(shuffled_refs) - i
            if remaining_refs < min_refs:
                # If remaining references are less than minimum, merge with previous group
                if grouped_refs and grouped_refs[-1]['lang'] == language:
                    remaining_refs_strings = [ref['reference'] for ref in shuffled_refs[i:]]
                    remaining_ground_truth = [ref['tags'] for ref in shuffled_refs[i:]]
                    grouped_refs[-1]['refs'].extend(remaining_refs_strings)
                    grouped_refs[-1]['ground_truth'].extend(remaining_ground_truth)
                break
            
            # Calculate group size
            if remaining_refs <= max_refs:
                group_size = remaining_refs
            else:
                # Random size between min_refs and max_refs, but ensure we can form complete groups
                max_possible = min(max_refs, remaining_refs - min_refs + 1)
                group_size = random.randint(min_refs, max_possible)
            
            # Create group
            group_refs = shuffled_refs[i:i + group_size]
            
            # Create grouped reference object with the required format
            refs_strings = [ref['reference'] for ref in group_refs]
            ground_truth = [ref['tags'] for ref in group_refs]
            
            grouped_ref = {
                "id": group_id,
                "refs": refs_strings,
                "ground_truth": ground_truth,
                "lang": language
            }
            
            grouped_refs.append(grouped_ref)
            group_id += 1
            i += group_size
    
    print(f"  Created {len(grouped_refs)} groups from {sum(len(g['refs']) for g in grouped_refs)} references")
    
    return grouped_refs

def save_grouped_outputs(grouped_refs):
    """Save grouped test set outputs."""
    # Save grouped references
    grouped_path = output_dir / "linkedbooks_test_grouped_references.jsonl"
    with open(grouped_path, "w", encoding="utf-8") as out_f:
        for group in grouped_refs:
            out_f.write(json.dumps(group, ensure_ascii=False) + "\n")
    
    return grouped_path

def print_grouped_statistics(grouped_refs):
    """Print statistics for grouped test set."""
    if not grouped_refs:
        return
    
    print(f"\n{'='*60}")
    print("GROUPED TEST SET STATISTICS")
    print(f"{'='*60}")
    print(f"Total groups created: {len(grouped_refs):,}")
    
    # Group size distribution
    group_sizes = [len(group['refs']) for group in grouped_refs]
    print(f"Group size statistics:")
    print(f"  Average references per group: {sum(group_sizes) / len(group_sizes):.1f}")
    print(f"  Smallest group: {min(group_sizes)} references")
    print(f"  Largest group: {max(group_sizes)} references")
    
    # Language distribution
    lang_counts = Counter()
    for group in grouped_refs:
        lang_counts[group['lang']] += 1
    
    print(f"\nGroups by language:")
    for lang, count in lang_counts.most_common():
        total_refs = sum(len(g['refs']) for g in grouped_refs if g['lang'] == lang)
        print(f"  {lang}: {count} groups ({total_refs} total references)")
    
    # Tag distribution across groups
    all_group_tags = []
    for group in grouped_refs:
        for gt in group['ground_truth']:
            all_group_tags.extend(gt.keys())
    
    tag_counts = Counter(all_group_tags)
    print(f"\nMost common tags across groups:")
    for tag, count in tag_counts.most_common(10):
        print(f"  {tag}: appears {count} times across all groups")

def main():
    """Main processing function."""
    print("LinkedBook Reference Extraction and Analysis")
    print("=" * 50)
    
    all_refs = []
    all_vocab = set()
    all_vocab_freq = Counter()
    all_language_stats = Counter()
    test_refs = []  # Store test references separately for grouping
    
    # Process each dataset
    for dataset_name, file_path in input_files.items():
        refs, vocab, vocab_freq, language_stats = process_file(file_path, dataset_name)
        
        if refs:  # Only process if we found references
            # Save outputs for this dataset
            refs_path = save_outputs(refs, vocab, dataset_name)
            print(f"  Saved {len(refs):,} references to {refs_path}")
            
            # Print statistics
            print_statistics(dataset_name, refs, vocab, vocab_freq, language_stats)

            # save vocabulary to a file
            vocab_path = save_vocabulary(vocab, dataset_name)
            print(f"  Saved vocabulary to {vocab_path}")
            
            # Store test references for grouping
            if dataset_name == "test":
                test_refs = refs
            
            # Accumulate for overall statistics
            all_refs.extend(refs)
            all_vocab.update(vocab)
            all_vocab_freq.update(vocab_freq)
            all_language_stats.update(language_stats)
    
    # Generate grouped version of test set
    if test_refs:
        grouped_test_refs = generate_grouped_set_by_lang(test_refs)
        if grouped_test_refs:
            grouped_path = save_grouped_outputs(grouped_test_refs)
            print(f"  Saved {len(grouped_test_refs)} grouped references to {grouped_path}")
            print_grouped_statistics(grouped_test_refs)
    
    # Save vocabulary to single text file
    vocab_path = save_vocabulary(all_vocab, "all")
    print(f"\nSaved vocabulary to {vocab_path}")
    
    # Overall statistics
    if all_refs:
        print(f"\n{'='*60}")
        print("OVERALL STATISTICS (ALL DATASETS COMBINED)")
        print(f"{'='*60}")
        print(f"Total references across all datasets: {len(all_refs):,}")
        print(f"Total unique vocabulary items: {len(all_vocab):,}")
        
        # Overall language distribution
        print(f"\nOverall language distribution:")
        for lang, count in all_language_stats.most_common():
            percentage = (count / len(all_refs)) * 100
            print(f"  {lang}: {count:,} ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
