import json
import re
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

                # Build Task1 vocabulary from the whole file
                if t1 not in {"-X-", "o"}:
                    vocab.add(t1)
                    vocab_freq[t1] += 1

                # Skip DOCSTART
                if token == "-DOCSTART-":
                    continue

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

def save_vocabulary(all_vocab):
    """Save vocabulary to a single text file."""
    vocab_path = output_dir / "tag_vocabulary.txt"
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

def main():
    """Main processing function."""
    print("LinkedBook Reference Extraction and Analysis")
    print("=" * 50)
    
    all_refs = []
    all_vocab = set()
    all_vocab_freq = Counter()
    all_language_stats = Counter()
    
    # Process each dataset
    for dataset_name, file_path in input_files.items():
        refs, vocab, vocab_freq, language_stats = process_file(file_path, dataset_name)
        
        if refs:  # Only process if we found references
            # Save outputs for this dataset
            refs_path = save_outputs(refs, vocab, dataset_name)
            print(f"  Saved {len(refs):,} references to {refs_path}")
            
            # Print statistics
            print_statistics(dataset_name, refs, vocab, vocab_freq, language_stats)
            
            # Accumulate for overall statistics
            all_refs.extend(refs)
            all_vocab.update(vocab)
            all_vocab_freq.update(vocab_freq)
            all_language_stats.update(language_stats)
    
    # Save vocabulary to single text file
    vocab_path = save_vocabulary(all_vocab)
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
    
    print(f"\nProcessing complete! Check the 'outputs' directory for results.")

if __name__ == "__main__":
    main()
