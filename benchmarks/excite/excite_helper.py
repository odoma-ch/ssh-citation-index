import os
import pandas as pd
import shutil
import json
from pathlib import Path
from typing import List, Dict, Tuple
import PyPDF2
import fitz  # PyMuPDF as fallback

def get_pdf_page_count(pdf_path):
    """
    Get the number of pages in a PDF file.
    Uses PyPDF2 first, falls back to PyMuPDF if that fails.
    """
    try:
        # Try PyPDF2 first
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return len(pdf_reader.pages)
    except Exception as e1:
        try:
            # Fallback to PyMuPDF
            doc = fitz.open(pdf_path)
            page_count = doc.page_count
            doc.close()
            return page_count
        except Exception as e2:
            print(f"Error reading PDF {pdf_path}: PyPDF2 error: {e1}, PyMuPDF error: {e2}")
            return None

def process_papers_folders():
    """
    Process all German and English papers folders to:
    1. Create a DataFrame with PDF file information (filename, class, language, file_path, page_count)
    2. Extract all references from CSV/TXT files as {file_id: xxx, references: [list of str]}
    3. Consolidate PDFs into single folder with original names
    4. Consolidate XML files from EXRefSegmentation into a single folder
    """
    
    # Base path
    goldstandard_path = Path("EXgoldstandard/Goldstandard_EXparser")
    output_path = Path("benchmarks/excite")
    
    # Initialize lists to store data
    pdf_data = []
    references_data = {}
    
    # Define folder mappings
    folder_mappings = {
        "1-German_papers": {
            "lang": "de",
            "subfolders": [
                ("1-German_papers(with_reference_section_at_end_of_paper)", 1),
                ("2-German_papers(with_reference_in_footnote)", 2),
                ("3-German_papers(with_reference_in_footnote_and_end_of_paper)", 3)
            ]
        },
        "2-English_papers": {
            "lang": "en",
            "subfolders": [
                ("1-English_papers(with_reference_section_at_end_of_paper)", 1)
            ]
        }
    }
    
    # Create consolidated folders
    consolidated_pdfs_path = output_path / "all_pdfs"
    consolidated_xml_path = output_path / "all_xml"
    consolidated_pdfs_path.mkdir(exist_ok=True)
    consolidated_xml_path.mkdir(exist_ok=True)
    
    # Process each main folder (German/English)
    for main_folder, config in folder_mappings.items():
        lang = config["lang"]
        main_folder_path = goldstandard_path / main_folder
        
        if not main_folder_path.exists():
            print(f"Warning: {main_folder_path} does not exist")
            continue
            
        # Process each subfolder (class 1, 2, 3)
        for subfolder_name, class_num in config["subfolders"]:
            subfolder_path = main_folder_path / subfolder_name
            
            if not subfolder_path.exists():
                print(f"Warning: {subfolder_path} does not exist")
                continue
                
            print(f"Processing {subfolder_path}")
            
            # Process PDFs
            pdfs_path = subfolder_path / "1-pdfs"
            if pdfs_path.exists():
                for pdf_file in pdfs_path.glob("*.pdf"):
                    # Skip system files
                    if pdf_file.name.startswith('.') or pdf_file.name == "Thumbs.db":
                        continue
                        
                    file_id = pdf_file.stem  # filename without extension
                    
                    # Get page count
                    # print(f"  Processing {pdf_file.name}...")
                    page_count = get_pdf_page_count(pdf_file)
                    
                    pdf_data.append({
                        "file_id": file_id,
                        "filename": pdf_file.name,
                        "class": class_num,
                        "lang": lang,
                        "file_path": str(pdf_file),
                        "page_count": page_count
                    })
                    
                    # Copy to consolidated folder with original name
                    try:
                        shutil.copy2(pdf_file, consolidated_pdfs_path / pdf_file.name)
                    except Exception as e:
                        print(f"Error copying {pdf_file}: {e}")
            
            # Process References (CSV/TXT files)
            refs_path = subfolder_path / "4-References_extracted_from_Layout"
            if refs_path.exists():
                for ref_file in refs_path.glob("*"):
                    # Skip directories and system files
                    if ref_file.is_dir() or ref_file.name.startswith('.'):
                        continue
                        
                    try:
                        # Read the file content (treating as text, ignoring CSV structure)
                        with open(ref_file, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                        
                        file_id = ref_file.stem  # filename without extension
                        references = []
                        
                        # Process each line as a reference
                        for line in lines:
                            line = line.strip()
                            if line:  # Skip empty lines
                                references.append(line)
                        
                        # Store in the format {file_id: xxx, references: [list of str]}
                        if references:  # Only add if there are references
                            references_data[file_id] = {
                                "file_id": file_id,
                                "references": references
                            }
                        
                    except Exception as e:
                        print(f"Error processing {ref_file}: {e}")
            
            # Process XML files from EXRefSegmentation
            xml_path = subfolder_path / "5-References_segmented_by_EXRefSegmentation"
            if xml_path.exists():
                for xml_file in xml_path.glob("*.xml"):
                    # Skip system files
                    if xml_file.name.startswith('.') or xml_file.name == "Thumbs.db":
                        continue
                    
                    # Copy to consolidated folder with original name
                    try:
                        shutil.copy2(xml_file, consolidated_xml_path / xml_file.name)
                    except Exception as e:
                        print(f"Error copying {xml_file}: {e}")
    
    # Create DataFrame for PDFs
    pdf_df = pd.DataFrame(pdf_data)
    
    # Calculate total reference count
    total_references = sum(len(data["references"]) for data in references_data.values())
    
    # Save results under Goldstandard_EXparser
    pdf_df.to_csv(output_path / "pdf_files_info.csv", index=False)
    
    # Save references as JSON
    with open(output_path / "all_references.json", "w", encoding="utf-8") as f:
        json.dump(references_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Total PDFs processed: {len(pdf_df)}")
    print(f"Total papers with references: {len(references_data)}")
    print(f"Total individual references: {total_references}")
    print(f"\nPDFs by language and class:")
    if not pdf_df.empty:
        summary = pdf_df.groupby(['lang', 'class']).size().reset_index(name='count')
        print(summary.to_string(index=False))
    
    # Print page count statistics
    if not pdf_df.empty and 'page_count' in pdf_df.columns:
        valid_pages = pdf_df[pdf_df['page_count'].notna()]
        if not valid_pages.empty:
            print(f"\nPage count statistics:")
            print(f"  Mean pages: {valid_pages['page_count'].mean():.1f}")
            print(f"  Median pages: {valid_pages['page_count'].median():.1f}")
            print(f"  Min pages: {valid_pages['page_count'].min()}")
            print(f"  Max pages: {valid_pages['page_count'].max()}")
            print(f"  PDFs with page count errors: {len(pdf_df) - len(valid_pages)}")
    
    print(f"\nFiles saved:")
    print(f"- {output_path}/pdf_files_info.csv")
    print(f"- {output_path}/all_references.json")
    print(f"- {output_path}/all_pdfs/")
    print(f"- {output_path}/all_xml/")
    
    return pdf_df, references_data

def get_sample_data(pdf_df, references_data, n_samples=5):
    """Display sample data for verification"""
    print(f"\n=== SAMPLE PDF DATA ===")
    if not pdf_df.empty:
        print(pdf_df.head(n_samples).to_string(index=False))
    
    print(f"\n=== SAMPLE REFERENCES DATA ===")
    if references_data:
        sample_keys = list(references_data.keys())[:n_samples]
        for i, file_id in enumerate(sample_keys):
            ref_data = references_data[file_id]
            print(f"Paper {i+1} (ID: {file_id}):")
            print(f"  Number of references: {len(ref_data['references'])}")
            print(f"  First few references:")
            for j, ref in enumerate(ref_data['references'][:3]):
                print(f"    {j+1}. {ref}")
            if len(ref_data['references']) > 3:
                print(f"    ... and {len(ref_data['references']) - 3} more")
            print()

def evaluate_whole_dataset(pred_pkl_path, xml_dir, mode='exact',fuzzy_threshold=90,focus_fields=None):
    """
    Evaluate the whole dataset using predictions from a pickle file and ground truth from XML files.
    Args:
        pred_pkl_path: Path to the pickle file with predictions (list of dicts with 'id' and 'references').
        xml_dir: Directory containing ground truth XML files (named <file_id>.xml).
    Returns:
        metrics: dict with overall evaluation metrics.
        per_doc_df: DataFrame with per-document metrics.
    """
    import pickle
    import os
    import pandas as pd
    from citation_index.core.models import References
    from citation_index.evaluation.ref_metrics import RefEvaluator

    # Load predictions
    with open(pred_pkl_path, 'rb') as f:
        pred_list = pickle.load(f)

    # Build a dict for fast lookup
    pred_dict = {str(item['id']): item['references']['references'] for item in pred_list}

    # For each file in xml_dir, load ground truth and prediction
    gt_refs_list = []
    pred_refs_list = []
    file_ids = []
    missing_preds = 0
    missing_gts = 0
    per_doc_metrics = []
    evaluator = RefEvaluator(mode=mode, fuzzy_threshold=fuzzy_threshold)

    for fname in os.listdir(xml_dir):
        if not fname.endswith('.xml'):
            continue
        file_id = fname[:-4]
        gt_path = os.path.join(xml_dir, fname)
        print(f"Processing {file_id}")
        try:
            gt_refs = References.from_excite_xml(gt_path)
        except Exception as e:
            print(f"Error loading GT for {file_id}: {e}")
            missing_gts += 1
            continue
        # Find prediction
        pred_refs_raw = pred_dict.get(file_id)
        if pred_refs_raw is None or len(pred_refs_raw) == 0:
            print(f"No prediction for {file_id}")
            missing_preds += 1
            pred_refs = References(references=[])
        else:
            # Convert list of dicts to References object
            reference_dicts = []
            for ref in pred_refs_raw:
                ref_data = ref["reference"].copy()
                if "title" in ref_data:
                    ref_data["full_title"] = ref_data.pop("title")
                reference_dicts.append(ref_data)
            pred_refs = References.from_dict(reference_dicts)
        gt_refs_list.append(gt_refs)
        pred_refs_list.append(pred_refs)
        file_ids.append(file_id)

        # --- Per-document metrics ---
        metrics = evaluator.evaluate([pred_refs], [gt_refs], focus_fields=focus_fields)
        # Flatten the metrics dict for this doc
        flat_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        flat_metrics['file_id'] = file_id
        per_doc_metrics.append(flat_metrics)

    print(f"Total files: {len(gt_refs_list)}")
    print(f"Missing predictions: {missing_preds}")
    print(f"Missing GTs: {missing_gts}")

    # Overall metrics
    overall_metrics = evaluator.evaluate(pred_refs_list, gt_refs_list, focus_fields=focus_fields)
    print("Reference eval (exact):")
    for k, v in overall_metrics.items():
        print(f"{k}: {v}")

    # Per-document DataFrame
    per_doc_df = pd.DataFrame(per_doc_metrics)
    # print(per_doc_df.head())

    return overall_metrics, per_doc_df

if __name__ == "__main__":
    # Process all folders
    # pdf_df, references_data = process_papers_folders()
    
    # Show sample data
    # get_sample_data(pdf_df, references_data)
    
    # Example usage:
    pred_pkl_path = "benchmarks/benchmarking/references_ref_extparsing_deepseek_pymupdf.pkl"
    xml_dir = "benchmarks/excite/all_xml"
    overall_metrics, per_doc_df = evaluate_whole_dataset(pred_pkl_path, xml_dir)
    per_doc_df.to_csv("per_document_metrics.csv", index=False)


def load_excite_data() -> Tuple[pd.DataFrame, Dict]:
    """
    Load the EXCITE dataset information.
    
    Returns:
        Tuple containing:
        - pdf_df: DataFrame with PDF file information
        - references_data: Dictionary with ground truth references
    """
    base_path = Path("benchmarks/excite")
    pdf_info_path = base_path / "pdf_files_info.csv"
    references_path = base_path / "all_references.json"

    if not pdf_info_path.exists() or not references_path.exists():
        print("Data files not found. Running pre-processing step...")
        process_papers_folders()

    pdf_df = pd.read_csv(pdf_info_path)
    with open(references_path, "r", encoding="utf-8") as f:
        references_data = json.load(f)

    print(f"Loaded {len(pdf_df)} PDF records and {len(references_data)} papers with references.")
    return pdf_df, references_data
