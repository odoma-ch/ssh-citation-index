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
    base_path = Path("EXgoldstandard/Goldstandard_EXparser")
    
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
    consolidated_pdfs_path = base_path / "all_pdfs"
    consolidated_xml_path = base_path / "all_xml"
    consolidated_pdfs_path.mkdir(exist_ok=True)
    consolidated_xml_path.mkdir(exist_ok=True)
    
    # Process each main folder (German/English)
    for main_folder, config in folder_mappings.items():
        lang = config["lang"]
        main_folder_path = base_path / main_folder
        
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
    pdf_df.to_csv(base_path / "pdf_files_info.csv", index=False)
    
    # Save references as JSON
    with open(base_path / "all_references.json", "w", encoding="utf-8") as f:
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
    print(f"- {base_path}/pdf_files_info.csv")
    print(f"- {base_path}/all_references.json")
    print(f"- {base_path}/all_pdfs/")
    print(f"- {base_path}/all_xml/")
    
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

if __name__ == "__main__":
    # Process all folders
    pdf_df, references_data = process_papers_folders()
    
    # Show sample data
    get_sample_data(pdf_df, references_data) 