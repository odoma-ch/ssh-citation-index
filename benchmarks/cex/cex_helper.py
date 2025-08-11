import os
import pandas as pd
import shutil
import json
from pathlib import Path
from typing import List, Dict, Tuple
import PyPDF2
import fitz  # PyMuPDF as fallback
import re

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

def parse_goldstandard_txt():
    """
    Parse the GS_info.txt file to extract paper information (titles, categories, etc.).
    Returns a dictionary mapping file_id to paper info.
    """
    goldstandard_path = Path("benchmarks/cex/GS_info.txt")
    
    if not goldstandard_path.exists():
        print(f"Warning: {goldstandard_path} does not exist")
        return {}
    
    papers_data = {}
    current_category = None
    current_file_id = None
    current_title = None
    
    with open(goldstandard_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a category header (e.g., "AGR-BIO-SCI - AGRICULTURAL")
        if " - " in line and not line.startswith("-"):
            current_category = line.split(" - ")[0]
            continue
            
        # Check if this is a paper entry (e.g., "- AGR-BIO-SCI_1: ...")
        if line.startswith("- ") and "_" in line:
            # Extract file_id and title
            parts = line[2:].split(": ", 1)  # Remove "- " and split on first ": "
            if len(parts) == 2:
                current_file_id = parts[0]
                current_title = parts[1]
                
                # Parse the title to extract basic info
                paper_info = parse_title_string(current_title)
                paper_info['category'] = current_category
                paper_info['file_id'] = current_file_id
                
                papers_data[current_file_id] = paper_info
    
    return papers_data

def parse_title_string(title_string):
    """
    Parse a title string to extract basic information.
    Simply save the title as a string without parsing.
    """
    # Extract DOI if present - look for both "doi:" and "https://doi.org/" patterns
    doi = None
    doi_match = re.search(r'doi:([^\s\)]+)', title_string, re.IGNORECASE)
    if doi_match:
        doi = doi_match.group(1)
    else:
        # Try to find https://doi.org/ pattern
        doi_match = re.search(r'https://doi\.org/([^\s\)]+)', title_string, re.IGNORECASE)
        if doi_match:
            doi = doi_match.group(1)
    
    # Extract year if present
    year_match = re.search(r'(\d{4})', title_string)
    year = year_match.group(1) if year_match else None
    
    return {
        'title': title_string,  # Save the full title as a string
        'doi': doi,
        'year': year
    }

def extract_xml_info(file_id, xml_dir):
    """
    Extract information from TEI XML file header for a given file_id.
    Returns a dictionary with extracted information.
    """
    xml_path = xml_dir / f"{file_id}.xml"
    if not xml_path.exists():
        return {}
    
    try:
        import xml.etree.ElementTree as ET
        
        # Parse XML with namespace
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Define namespace
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        
        xml_info = {}
        
        # Extract title from titleStmt
        title_stmt = root.find('.//tei:titleStmt', ns)
        if title_stmt is not None:
            title_elem = title_stmt.find('.//tei:title[@level="a"]', ns)
            if title_elem is not None:
                xml_info['xml_title'] = title_elem.text
        
        # Extract publication information
        pub_stmt = root.find('.//tei:publicationStmt', ns)
        if pub_stmt is not None:
            publisher = pub_stmt.find('.//tei:publisher', ns)
            if publisher is not None:
                xml_info['publisher'] = publisher.text
            
            date_elem = pub_stmt.find('.//tei:date[@type="published"]', ns)
            if date_elem is not None:
                xml_info['publication_date'] = date_elem.text
                # Extract year from date
                when_attr = date_elem.get('when')
                if when_attr:
                    year_match = re.search(r'(\d{4})', when_attr)
                    if year_match:
                        xml_info['publication_year'] = year_match.group(1)
        
        # Extract DOI and other identifiers
        source_desc = root.find('.//tei:sourceDesc//tei:biblStruct', ns)
        if source_desc is not None:
            idno_elems = source_desc.findall('.//tei:idno', ns)
            for idno in idno_elems:
                id_type = idno.get('type')
                if id_type == 'DOI':
                    xml_info['xml_doi'] = idno.text
                elif id_type == 'PMID':
                    xml_info['pmid'] = idno.text
                elif id_type == 'PMCID':
                    xml_info['pmcid'] = idno.text
        
        # Extract journal information
        monogr = source_desc.find('.//tei:monogr', ns) if source_desc is not None else None
        if monogr is not None:
            journal_title = monogr.find('.//tei:title[@level="j"]', ns)
            if journal_title is not None:
                xml_info['journal_title'] = journal_title.text
            
            imprint = monogr.find('.//tei:imprint', ns)
            if imprint is not None:
                volume = imprint.find('.//tei:biblScope[@unit="volume"]', ns)
                if volume is not None:
                    xml_info['volume'] = volume.text
                
                pages = imprint.find('.//tei:biblScope[@unit="page"]', ns)
                if pages is not None:
                    from_page = pages.get('from')
                    to_page = pages.get('to')
                    if from_page and to_page:
                        xml_info['pages'] = f"{from_page}-{to_page}"
                    elif from_page:
                        xml_info['pages'] = from_page
        
        return xml_info
        
    except Exception as e:
        print(f"Error extracting XML info for {file_id}: {e}")
        return {}

def load_references_from_xml(file_id, xml_dir):
    """
    Load references from TEI XML file for a given file_id.
    Returns a list of reference strings.
    """
    xml_path = xml_dir / f"{file_id}.xml"
    if not xml_path.exists():
        return []
    
    try:
        from citation_index.core.models import References
        references = References.from_xml(file_path=xml_path)
        # Convert structured references back to strings for compatibility
        reference_strings = []
        for ref in references:
            # Create a simple string representation of the reference
            ref_parts = []
            if ref.authors:
                author_names = []
                for author in ref.authors:
                    # Build full name from all available name components
                    name_parts = []
                    
                    # Add role_name (titles/credentials) first if present
                    if hasattr(author, 'role_name') and author.role_name:
                        name_parts.append(author.role_name)
                    
                    # Add first name
                    if hasattr(author, 'first_name') and author.first_name:
                        name_parts.append(author.first_name)
                    
                    # Add middle name
                    if hasattr(author, 'middle_name') and author.middle_name:
                        name_parts.append(author.middle_name)
                    
                    # Add name_link (connecting phrases like "van der")
                    if hasattr(author, 'name_link') and author.name_link:
                        name_parts.append(author.name_link)
                    
                    # Add surname
                    if hasattr(author, 'surname') and author.surname:
                        name_parts.append(author.surname)
                    elif hasattr(author, 'name') and author.name:
                        name_parts.append(author.name)
                    
                    if name_parts:
                        author_names.append(" ".join(name_parts))
                
                if author_names:
                    ref_parts.append(", ".join(author_names))
            
            if ref.full_title:
                ref_parts.append(ref.full_title)
            
            if ref.journal_title:
                ref_parts.append(ref.journal_title)
            
            if ref.publication_date:
                ref_parts.append(ref.publication_date)
            
            if ref.volume:
                ref_parts.append(f"Vol. {ref.volume}")
            
            if ref.pages:
                ref_parts.append(f"pp. {ref.pages}")
            
            if ref_parts:
                reference_strings.append(". ".join(ref_parts))
        
        return reference_strings
    except Exception as e:
        print(f"Error loading references for {file_id}: {e}")
        return []

def process_cex_data():
    """
    Process CEX dataset to:
    1. Process all PDFs in the all_pdfs folder
    2. Extract paper information from GS_info.txt
    3. Load actual references from TEI XML files
    4. Match PDFs with paper information and references
    """
    
    # Base paths
    output_path = Path("benchmarks/cex")
    
    # Initialize lists to store data
    pdf_data = []
    papers_data = parse_goldstandard_txt()
    
    # Process all PDFs in the all_pdfs folder
    all_pdfs_path = output_path / "all_pdfs"
    all_xmls_path = output_path / "all_xmls"
    
    if all_pdfs_path.exists():
        for pdf_file in all_pdfs_path.glob("*.pdf"):
            # Skip system files and test files
            if (pdf_file.name.startswith('.') or 
                pdf_file.name == "Thumbs.db" or 
                pdf_file.name.startswith('z_notes_test')):
                continue
                
            file_id = pdf_file.stem  # filename without extension
            
            # Get page count
            page_count = get_pdf_page_count(pdf_file)
            
            # Get paper info from parsed data
            paper_info = papers_data.get(file_id, {})
            category = paper_info.get('category', 'Unknown')
            
            # Load actual references from XML file
            references = load_references_from_xml(file_id, all_xmls_path)
            
            # Extract additional information from XML file
            xml_info = extract_xml_info(file_id, all_xmls_path)
            
            # Merge XML information with paper info
            paper_info.update(xml_info)
            paper_info['references'] = references
            
            pdf_data.append({
                "file_id": file_id,
                "filename": pdf_file.name,
                "category": category,
                "file_path": str(pdf_file),
                "page_count": page_count,
                "title": paper_info.get('title', ''),
                "publication_year": paper_info.get('publication_year', ''),
                "doi": paper_info.get('doi', ''),
                "xml_title": paper_info.get('xml_title', ''),
                "journal": paper_info.get('journal_title', ''),
                "publisher": paper_info.get('publisher', '')
            })
    
    # Create DataFrame for PDFs
    pdf_df = pd.DataFrame(pdf_data)
    
    # Save results
    pdf_df.to_csv(output_path / "pdf_files_info.csv", index=False)
    
    # Save papers data as JSON (for compatibility with excite format)
    with open(output_path / "all_references.json", "w", encoding="utf-8") as f:
        json.dump(papers_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n=== CEX PROCESSING SUMMARY ===")
    print(f"Total PDFs processed: {len(pdf_df)}")
    print(f"Total papers with references: {len(papers_data)}")
    print(f"Total individual references: {sum(len(data.get('references', [])) for data in papers_data.values())}")
    
    # Check for missing papers
    gs_papers = set(papers_data.keys())
    pdf_papers = set(pdf_df['file_id'].tolist())
    missing_papers = gs_papers - pdf_papers
    extra_papers = pdf_papers - gs_papers
    
    if missing_papers:
        print(f"\nPapers in GS_info.txt but missing PDFs: {len(missing_papers)}")
        for paper in sorted(missing_papers):
            print(f"  - {paper}")
    
    if extra_papers:
        print(f"\nPDFs without GS_info.txt entries: {len(extra_papers)}")
        for paper in sorted(extra_papers):
            print(f"  - {paper}")
    
    
    print(f"\nFiles updated:")
    print(f"- {output_path}/pdf_files_info.csv")
    print(f"- {output_path}/all_references.json")
    
    return pdf_df, papers_data

def get_sample_data(pdf_df, papers_data, n_samples=5):
    """Display sample data for verification"""
    print(f"\n=== SAMPLE PDF DATA ===")
    if not pdf_df.empty:
        print(pdf_df.head(n_samples).to_string(index=False))
    
    print(f"\n=== SAMPLE PAPERS DATA ===")
    if papers_data:
        # Show first few samples
        sample_keys = list(papers_data.keys())[:n_samples]
        for i, file_id in enumerate(sample_keys):
            paper_data = papers_data[file_id]
            print(f"Paper {i+1} (ID: {file_id}):")
            print(f"  Category: {paper_data.get('category', 'Unknown')}")
            print(f"  Title: {paper_data.get('title', 'Unknown')}")
            print(f"  Year: {paper_data.get('year', 'Unknown')}")
            print(f"  DOI: {paper_data.get('doi', 'Unknown')}")
            print(f"  XML Title: {paper_data.get('xml_title', 'Unknown')}")
            print(f"  Journal: {paper_data.get('journal_title', 'Unknown')}")
            print(f"  Publisher: {paper_data.get('publisher', 'Unknown')}")
            print(f"  Publication Year: {paper_data.get('publication_year', 'Unknown')}")
            print(f"  Number of References: {len(paper_data.get('references', []))}")
            print()
        
        # Also show the file with XML header if it exists
   

def evaluate_whole_dataset(pred_pkl_path, xml_dir, mode='exact', fuzzy_threshold=90, focus_fields=None):
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
            gt_refs = References.from_xml(file_path=gt_path)
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

    return overall_metrics, per_doc_df

def load_cex_data() -> Tuple[pd.DataFrame, Dict]:
    """
    Load the CEX dataset information.
    
    Returns:
        Tuple containing:
        - pdf_df: DataFrame with PDF file information
        - papers_data: Dictionary with ground truth references
    """
    base_path = Path("benchmarks/cex")
    pdf_info_path = base_path / "pdf_files_info.csv"
    references_path = base_path / "all_references.json"

    if not pdf_info_path.exists() or not references_path.exists():
        print("Data files not found. Running pre-processing step...")
        process_cex_data()

    pdf_df = pd.read_csv(pdf_info_path)
    with open(references_path, "r", encoding="utf-8") as f:
        papers_data = json.load(f)

    print(f"Loaded {len(pdf_df)} PDF records and {len(papers_data)} papers with references.")
    return pdf_df, papers_data

if __name__ == "__main__":
    # Process all folders
    pdf_df, papers_data = process_cex_data()
    
    # Show sample data
    get_sample_data(pdf_df, papers_data) 