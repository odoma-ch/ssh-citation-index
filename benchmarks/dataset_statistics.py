#!/usr/bin/env python3
"""
Dataset Statistics Generator
Generates comprehensive statistics for CEX, EXCITE, and LinkedBook datasets
"""

import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_cex_data():
    """Load CEX dataset"""
    cex_path = "cex"
    pdf_df = pd.read_csv(os.path.join(cex_path, "pdf_files_info.csv"))
    
    with open(os.path.join(cex_path, "all_references.json"), "r", encoding="utf-8") as f:
        references_data = json.load(f)
    
    return pdf_df, references_data

def load_excite_data():
    """Load EXCITE dataset"""
    excite_path = "excite"
    pdf_df = pd.read_csv(os.path.join(excite_path, "pdf_files_info.csv"))
    
    with open(os.path.join(excite_path, "all_references.json"), "r", encoding="utf-8") as f:
        references_data = json.load(f)
    
    return pdf_df, references_data

def load_linkedbook_data():
    """Load LinkedBook test dataset"""
    linkedbook_path = "linkedbook"
    
    # Load test references - each line is a single reference
    references = []
    with open(os.path.join(linkedbook_path, "linkedbooks_test_references.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            references.append(data)
    
    # Create a single "document" containing all references for LinkedBook
    # This is because LinkedBook is structured as individual references, not documents
    references_data = {
        '0': {
            'references': [ref.get('reference', '') for ref in references],
            'total_references': len(references),
            'languages': list(set(ref.get('language', 'Unknown') for ref in references))
        }
    }
    
    # Calculate some basic statistics
    ref_lengths = [len(ref.get('reference', '')) for ref in references]
    ref_words = [len(ref.get('reference', '').split()) for ref in references]
    
    # Create a basic dataframe for LinkedBook
    pdf_data = [{
        'doc_id': '0',
        'reference_count': len(references),
        'avg_ref_length': sum(ref_lengths) / len(ref_lengths) if ref_lengths else 0,
        'avg_ref_words': sum(ref_words) / len(ref_words) if ref_words else 0,
        'total_characters': sum(ref_lengths),
        'total_words': sum(ref_words)
    }]
    
    pdf_df = pd.DataFrame(pdf_data)
    
    return pdf_df, references_data

def calculate_basic_stats(pdf_df, references_data, dataset_name):
    """Calculate basic statistics for a dataset"""
    print(f"\n{'='*50}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*50}")
    
    # Basic counts
    total_docs = len(pdf_df)
    total_refs = sum(len(data.get('references', [])) for data in references_data.values())
    
    print(f"Total documents: {total_docs:,}")
    print(f"Total references: {total_refs:,}")
    
    # Reference statistics
    ref_counts = [len(data.get('references', [])) for data in references_data.values()]
    if ref_counts:
        print(f"Average references per document: {np.mean(ref_counts):.2f}")
        print(f"Median references per document: {np.median(ref_counts):.2f}")
        print(f"Min references per document: {min(ref_counts)}")
        print(f"Max references per document: {max(ref_counts)}")
        print(f"Std deviation of references: {np.std(ref_counts):.2f}")
    
    # Document length statistics (if available)
    if 'page_count' in pdf_df.columns:
        print(f"Total pages: {pdf_df['page_count'].sum():,}")
        print(f"Average pages per document: {pdf_df['page_count'].mean():.2f}")
        print(f"Median pages per document: {pdf_df['page_count'].median():.2f}")
        print(f"Min pages per document: {pdf_df['page_count'].min()}")
        print(f"Max pages per document: {pdf_df['page_count'].max()}")
        print(f"Std deviation of pages: {pdf_df['page_count'].std():.2f}")
    
    if 'text_length' in pdf_df.columns:
        print(f"Average text length (characters): {pdf_df['text_length'].mean():.0f}")
        print(f"Median text length (characters): {pdf_df['text_length'].median():.0f}")
    
    if 'word_count' in pdf_df.columns:
        print(f"Average word count: {pdf_df['word_count'].mean():.0f}")
        print(f"Median word count: {pdf_df['word_count'].median():.0f}")
    
    # Category information (if available)
    if 'category' in pdf_df.columns:
        categories = pdf_df['category'].unique()
        print(f"Number of categories: {len(categories)}")
        print("Category distribution:")
        for cat in sorted(categories):
            count = len(pdf_df[pdf_df['category'] == cat])
            print(f"  {cat}: {count} documents ({count/total_docs*100:.1f}%)")
    
    # Language information (if available)
    if 'lang' in pdf_df.columns:
        languages = pdf_df['lang'].value_counts()
        print("Language distribution:")
        for lang, count in languages.items():
            print(f"  {lang}: {count} documents ({count/total_docs*100:.1f}%)")
    
    # Special handling for LinkedBook dataset
    if dataset_name.upper() == 'LINKEDBOOK':
        print("Note: LinkedBook contains individual reference entries, not full documents")
        
        # Get language info from references_data and count by language
        linkedbook_path = "linkedbook"
        lang_counts = {}
        with open(os.path.join(linkedbook_path, "linkedbooks_test_references.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                lang = data.get('language', 'Unknown')
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        print("Language distribution in references:")
        total_refs = sum(lang_counts.values())
        for lang, count in sorted(lang_counts.items()):
            percentage = (count / total_refs) * 100
            print(f"  {lang}: {count} references ({percentage:.1f}%)")
        
        # Additional LinkedBook-specific stats
        if 'avg_ref_length' in pdf_df.columns:
            print(f"Average reference length (characters): {pdf_df['avg_ref_length'].iloc[0]:.0f}")
            print(f"Average reference length (words): {pdf_df['avg_ref_words'].iloc[0]:.1f}")
            print(f"Total characters in all references: {pdf_df['total_characters'].iloc[0]:,}")
            print(f"Total words in all references: {pdf_df['total_words'].iloc[0]:,}")
    
    return {
        'dataset': dataset_name,
        'total_docs': total_docs,
        'total_refs': total_refs,
        'avg_refs_per_doc': np.mean(ref_counts) if ref_counts else 0,
        'median_refs_per_doc': np.median(ref_counts) if ref_counts else 0,
        'min_refs': min(ref_counts) if ref_counts else 0,
        'max_refs': max(ref_counts) if ref_counts else 0,
        'std_refs': np.std(ref_counts) if ref_counts else 0,
        'avg_pages': pdf_df['page_count'].mean() if 'page_count' in pdf_df.columns else None,
        'total_pages': pdf_df['page_count'].sum() if 'page_count' in pdf_df.columns else None,
        'avg_text_length': pdf_df['text_length'].mean() if 'text_length' in pdf_df.columns else None,
        'avg_word_count': pdf_df['word_count'].mean() if 'word_count' in pdf_df.columns else None
    }

def generate_visualizations(stats_list):
    """Generate comparison visualizations"""
    print(f"\n{'='*50}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*50}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Dataset Comparison Statistics', fontsize=16)
    
    datasets = [s['dataset'] for s in stats_list]
    
    # Total documents
    total_docs = [s['total_docs'] for s in stats_list]
    axes[0, 0].bar(datasets, total_docs, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[0, 0].set_title('Total Documents')
    axes[0, 0].set_ylabel('Number of Documents')
    for i, v in enumerate(total_docs):
        axes[0, 0].text(i, v + max(total_docs)*0.01, str(v), ha='center')
    
    # Total references
    total_refs = [s['total_refs'] for s in stats_list]
    axes[0, 1].bar(datasets, total_refs, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[0, 1].set_title('Total References')
    axes[0, 1].set_ylabel('Number of References')
    for i, v in enumerate(total_refs):
        axes[0, 1].text(i, v + max(total_refs)*0.01, str(v), ha='center')
    
    # Average references per document (exclude LinkedBook since it has no documents)
    doc_datasets = [s['dataset'] for s in stats_list if s['dataset'].upper() != 'LINKEDBOOK']
    doc_avg_refs = [s['avg_refs_per_doc'] for s in stats_list if s['dataset'].upper() != 'LINKEDBOOK']
    
    if doc_avg_refs:
        axes[1, 0].bar(doc_datasets, doc_avg_refs, color=['skyblue', 'lightgreen'][:len(doc_datasets)])
        axes[1, 0].set_title('Average References per Document\n(Document-based datasets only)')
        axes[1, 0].set_ylabel('References per Document')
        for i, v in enumerate(doc_avg_refs):
            axes[1, 0].text(i, v + max(doc_avg_refs)*0.01, f'{v:.1f}', ha='center')
    else:
        axes[1, 0].text(0.5, 0.5, 'No document-based datasets', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Average References per Document')
    
    # Average pages per document (where available)
    avg_pages = [s['avg_pages'] if s['avg_pages'] is not None else 0 for s in stats_list]
    valid_datasets = [d for d, p in zip(datasets, avg_pages) if p > 0]
    valid_pages = [p for p in avg_pages if p > 0]
    
    if valid_pages:
        axes[1, 1].bar(valid_datasets, valid_pages, color=['skyblue', 'lightgreen'][:len(valid_pages)])
        axes[1, 1].set_title('Average Pages per Document')
        axes[1, 1].set_ylabel('Pages per Document')
        for i, v in enumerate(valid_pages):
            axes[1, 1].text(i, v + max(valid_pages)*0.01, f'{v:.1f}', ha='center')
    else:
        axes[1, 1].text(0.5, 0.5, 'No page data available', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Average Pages per Document')
    
    plt.tight_layout()
    plt.savefig('dataset_statistics.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'dataset_statistics.png'")
    plt.show()

def generate_summary_table(stats_list):
    """Generate a summary table"""
    print(f"\n{'='*50}")
    print("SUMMARY TABLE")
    print(f"{'='*50}")
    
    # Create summary dataframe
    summary_data = []
    for stats in stats_list:
        summary_data.append({
            'Dataset': stats['dataset'],
            'Total Docs': f"{stats['total_docs']:,}",
            'Total Refs': f"{stats['total_refs']:,}",
            'Avg Refs/Doc': f"{stats['avg_refs_per_doc']:.2f}",
            'Median Refs/Doc': f"{stats['median_refs_per_doc']:.2f}",
            'Min Refs': f"{stats['min_refs']}",
            'Max Refs': f"{stats['max_refs']}",
            'Avg Pages/Doc': f"{stats['avg_pages']:.2f}" if stats['avg_pages'] is not None else "N/A",
            'Total Pages': f"{stats['total_pages']:,}" if stats['total_pages'] is not None else "N/A"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save to CSV
    summary_df.to_csv('dataset_summary.csv', index=False)
    print(f"\nSummary table saved as 'dataset_summary.csv'")
    
    return summary_df

def main():
    """Main function to generate all statistics"""
    print("Dataset Statistics Generator")
    print("="*50)
    
    stats_list = []
    
    # Process CEX dataset
    try:
        print("Loading CEX dataset...")
        cex_pdf_df, cex_references = load_cex_data()
        cex_stats = calculate_basic_stats(cex_pdf_df, cex_references, "CEX")
        stats_list.append(cex_stats)
    except Exception as e:
        print(f"Error loading CEX dataset: {e}")
    
    # Process EXCITE dataset
    try:
        print("\nLoading EXCITE dataset...")
        excite_pdf_df, excite_references = load_excite_data()
        excite_stats = calculate_basic_stats(excite_pdf_df, excite_references, "EXCITE")
        stats_list.append(excite_stats)
    except Exception as e:
        print(f"Error loading EXCITE dataset: {e}")
    
    # Process LinkedBook dataset
    try:
        print("\nLoading LinkedBook test dataset...")
        linkedbook_pdf_df, linkedbook_references = load_linkedbook_data()
        linkedbook_stats = calculate_basic_stats(linkedbook_pdf_df, linkedbook_references, "LinkedBook")
        stats_list.append(linkedbook_stats)
    except Exception as e:
        print(f"Error loading LinkedBook dataset: {e}")
    
    # Generate summary and visualizations
    if stats_list:
        summary_df = generate_summary_table(stats_list)
        generate_visualizations(stats_list)
        
        print(f"\n{'='*50}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*50}")
        print("Files generated:")
        print("- dataset_summary.csv: Summary statistics table")
        print("- dataset_statistics.png: Comparison visualizations")
    else:
        print("No datasets could be loaded successfully.")

if __name__ == "__main__":
    main()
