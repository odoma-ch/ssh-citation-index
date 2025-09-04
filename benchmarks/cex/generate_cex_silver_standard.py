#!/usr/bin/env python3
"""
Generate Silver Standard for CEX Dataset

This script runs reference extraction on the CEX dataset and generates:
1. A silver standard file EXACTLY like all_references.json but with extracted references
2. Statistics on over/under/exact extraction patterns
3. Detailed results saved to tmp.txt

Usage:
    python generate_cex_silver_standard.py [options]
"""

import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import logging
import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.contrib.logging import logging_redirect_tqdm

from citation_index.core.extractors import ExtractorFactory
from citation_index.llm.client import LLMClient
from citation_index.pipelines.reference_extraction import extract_text_references
from cex_helper import load_cex_data


class CEXSilverStandardGenerator:
    """Generates silver standard from CEX dataset using reference extraction."""

    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store extraction results
        self.extraction_results = []
        self.reference_counts = []
        
        # Statistics counters
        self.over_extracted = 0
        self.under_extracted = 0
        self.exact_match = 0
        self.total_processed = 0
        
        # Detailed results for tmp.txt
        self.detailed_results = []
        
        tqdm.write(f"Silver standard will be saved to: {self.output_dir}")

    def run(self):
        """Main entry point for generating silver standard."""
        logging.debug("Starting CEX silver standard generation with the following arguments:")
        logging.debug(json.dumps(vars(self.args), indent=2))
        
        # Load CEX dataset
        pdf_df, papers_data = load_cex_data()
        
        # Keep a copy for later per-category summaries
        self.pdf_df = pdf_df
        
        # Initialize extractor and LLM client
        extractor = ExtractorFactory.create(self.args.extractor)
        llm_client = LLMClient(
            endpoint=self.args.api_base,
            model=self.args.model_name,
            api_key=self.args.api_key,
            timeout=1000,
            first_token_timeout=120,
            max_retries=3,
        )
        
        # Create all_markdown directory if it doesn't exist
        markdown_dir = Path("benchmarks/cex/all_markdown")
        markdown_dir.mkdir(parents=True, exist_ok=True)
        
        if self.args.limit:
            pdf_df = pdf_df.head(self.args.limit)
        
        # Prepare all inputs for LLM calls
        llm_tasks = []
        for _, row in tqdm(pdf_df.iterrows(), total=len(pdf_df), desc="Preparing tasks"):
            file_id = str(row["file_id"])
            file_path = row["file_path"]
            logging.debug(f"Preparing task for file_id: {file_id}")
            
            gt_references = papers_data.get(file_id, {}).get("references", [])
            if not gt_references:
                logging.warning(f"No ground truth references found for {file_id}. Skipping.")
                continue

            try:
                # Check if markdown file exists for reuse
                markdown_path = markdown_dir / f"{file_id}_{self.args.extractor}.md"
                if markdown_path.exists():
                    with open(markdown_path, "r", encoding="utf-8") as md_file:
                        input_text = md_file.read()
                else:
                    # Extract text from PDF
                    extracted_text_result = extractor.extract(file_path)
                    if not extracted_text_result.text.strip():
                        logging.warning(f"No text extracted from {file_id}.pdf. Skipping.")
                        continue
                    input_text = extracted_text_result.text
                    # Save extracted text to markdown file for reuse
                    with open(markdown_path, "w", encoding="utf-8") as md_file:
                        md_file.write(input_text)
                
                task_info = {
                    "file_id": file_id,
                    "input_text": input_text,
                    "gt_references": gt_references,
                    "paper_info": papers_data.get(file_id, {})
                }
                llm_tasks.append(task_info)
            except Exception as e:
                logging.error(f"Error preparing task for document {file_id}: {e}", exc_info=True)

        # Execute LLM calls concurrently
        tqdm.write(f"Submitting {len(llm_tasks)} tasks to LLM with {self.args.max_workers} workers.")
        start_time = time.time()
        
        llm_responses = []
        with ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
            future_to_task = {
                executor.submit(self._execute_extraction, llm_client, task): task
                for task in llm_tasks
            }
            
            for future in tqdm(as_completed(future_to_task), total=len(llm_tasks), desc="Executing reference extraction"):
                task = future_to_task[future]
                file_id = task["file_id"]
                logging.debug(f"Processing response for file_id: {file_id}")
                try:
                    response_str = future.result()
                    if response_str:
                        logging.debug(f"Received response for file_id: {file_id}")
                        llm_responses.append({
                            "file_id": task["file_id"],
                            "response": response_str,
                            "gt_references": task["gt_references"],
                            "paper_info": task["paper_info"]
                        })
                    else:
                        logging.warning(f"No response received for file_id: {file_id}")
                except Exception as exc:
                    logging.error(f'Task for {file_id} generated an exception: {exc}', exc_info=True)
        
        extraction_time = time.time() - start_time
        tqdm.write(f"Reference extraction completed in {extraction_time:.2f} seconds.")

        # Process all responses and generate silver standard
        tqdm.write(f"Processing {len(llm_responses)} responses to generate silver standard.")
        for response_data in tqdm(llm_responses, desc="Processing responses"):
            try:
                self._process_extraction_response(response_data)
            except Exception as e:
                logging.error(f"Error processing response for document {response_data['file_id']}: {e}", exc_info=True)

        # Generate and save results
        self._generate_silver_standard()
        self._save_detailed_results()
        self._print_summary()

    def _execute_extraction(self, llm_client, task_info):
        """Execute reference extraction for a single document."""
        file_id = task_info["file_id"]
        input_text = task_info["input_text"]
        prompt_path = Path("prompts") / self.args.prompt_name
        logging.debug(f"Executing reference extraction for file_id: {file_id}")

        # Extract reference strings
        reference_lines = extract_text_references(
            text=input_text,
            llm_client=llm_client,
            prompt_name=str(prompt_path),
        )
        
        return "\n".join(reference_lines)

    def _process_extraction_response(self, response_data):
        """Process a single extraction response and update statistics."""
        file_id = response_data["file_id"]
        llm_response = response_data["response"]
        gt_references = response_data["gt_references"]
        paper_info = response_data["paper_info"]
        
        # Parse extracted references
        pred_references = [ref.strip() for ref in llm_response.split("\n") if ref.strip()]
        
        # Calculate statistics
        gt_count = len(gt_references)
        pred_count = len(pred_references)
        difference = pred_count - gt_count
        
        # Update counters
        if difference > 0:
            self.over_extracted += 1
        elif difference < 0:
            self.under_extracted += 1
        else:
            self.exact_match += 1
        
        self.total_processed += 1
        
        # Store reference counts for analysis
        self.reference_counts.append({
            "file_id": file_id,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "difference": difference,
            "category": paper_info.get("category", "Unknown")
        })
        
        # Store extraction results for silver standard
        self.extraction_results.append({
            "file_id": file_id,
            "extracted_references": pred_references,
            "gt_references": gt_references,
            "paper_info": paper_info
        })
        
        # Store detailed results for tmp.txt
        percentage_diff = (difference / gt_count * 100) if gt_count > 0 else 0
        self.detailed_results.append({
            "file_id": file_id,
            "category": paper_info.get("category", "Unknown"),
            "gt_count": gt_count,
            "pred_count": pred_count,
            "difference": difference,
            "percentage_difference": percentage_diff,
            "extraction_type": "over" if difference > 0 else "under" if difference < 0 else "exact"
        })

    def _generate_silver_standard(self):
        """Generate silver standard file EXACTLY like all_references.json but with extracted references."""
        silver_standard = {}
        
        for result in self.extraction_results:
            file_id = result["file_id"]
            paper_info = result["paper_info"].copy()
            
            # ONLY replace the references field with extracted references
            # Keep everything else exactly the same
            paper_info["references"] = result["extracted_references"]
            
            silver_standard[file_id] = paper_info
        
        # Save silver standard
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_slug = self.args.model_name.replace('/', '_').replace('-', '_')
        silver_standard_path = self.output_dir / f"silver_standard_{model_name_slug}_{self.args.extractor}_{timestamp}.json"
        
        with open(silver_standard_path, "w", encoding="utf-8") as f:
            json.dump(silver_standard, f, indent=2, ensure_ascii=False)
        
        tqdm.write(f"Silver standard saved to: {silver_standard_path}")
        
        # # Also save a copy with generic name for easy access
        # generic_path = self.output_dir / "silver_standard_latest.json"
        # with open(generic_path, "w", encoding="utf-8") as f:
        #     json.dump(silver_standard, f, indent=2, ensure_ascii=False)
        
        # tqdm.write(f"Latest silver standard also saved to: {generic_path}")

    def _save_detailed_results(self):
        """Save detailed extraction results to tmp.txt."""
        tmp_path = self.output_dir / "tmp.txt"
        
        with open(tmp_path, "w", encoding="utf-8") as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write("CEX SILVER STANDARD GENERATION RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.args.model_name}\n")
            f.write(f"Extractor: {self.args.extractor}\n")
            f.write(f"Prompt: {self.args.prompt_name}\n")
            f.write(f"Total documents processed: {self.total_processed}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write summary statistics
            f.write("EXTRACTION SUMMARY STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Over-extracted: {self.over_extracted}\n")
            f.write(f"Under-extracted: {self.under_extracted}\n")
            f.write(f"Exact matches: {self.exact_match}\n")
            f.write(f"Total: {self.total_processed}\n\n")
            
            # Calculate percentages
            if self.total_processed > 0:
                f.write("PERCENTAGES\n")
                f.write("-" * 20 + "\n")
                f.write(f"Over-extracted: {self.over_extracted/self.total_processed*100:.1f}%\n")
                f.write(f"Under-extracted: {self.under_extracted/self.total_processed*100:.1f}%\n")
                f.write(f"Exact matches: {self.exact_match/self.total_processed*100:.1f}%\n\n")
            
            # Write detailed results
            f.write("DETAILED EXTRACTION RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write("Format: file_id | category | GT_count | Pred_count | Difference | %Diff | Type\n")
            f.write("-" * 80 + "\n")
            
            for result in sorted(self.detailed_results, key=lambda x: x["file_id"]):
                f.write(f"{result['file_id']} | {result['category']} | {result['gt_count']} | {result['pred_count']} | {result['difference']:+d} | {result['percentage_difference']:+.1f}% | {result['extraction_type']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF RESULTS\n")
            f.write("=" * 80 + "\n")
        
        tqdm.write(f"Detailed results saved to: {tmp_path}")

    def _print_summary(self):
        """Print a summary of the extraction results."""
        tqdm.write("\n" + "=" * 60)
        tqdm.write("CEX SILVER STANDARD GENERATION SUMMARY")
        tqdm.write("=" * 60)
        
        tqdm.write(f"Total documents processed: {self.total_processed}")
        tqdm.write(f"Over-extracted: {self.over_extracted}")
        tqdm.write(f"Under-extracted: {self.under_extracted}")
        tqdm.write(f"Exact matches: {self.exact_match}")
        
        if self.total_processed > 0:
            tqdm.write(f"\nPercentages:")
            tqdm.write(f"  Over-extracted: {self.over_extracted/self.total_processed*100:.1f}%")
            tqdm.write(f"  Under-extracted: {self.under_extracted/self.total_processed*100:.1f}%")
            tqdm.write(f"  Exact matches: {self.exact_match/self.total_processed*100:.1f}%")
        
        # Per-category breakdown if requested
        if self.args.per_category and self.reference_counts:
            tqdm.write(f"\nPer-category breakdown:")
            ref_counts_df = pd.DataFrame(self.reference_counts)
            grouped = ref_counts_df.groupby("category").agg({
                "gt_count": ["mean", "std"],
                "pred_count": ["mean", "std"],
                "difference": ["mean", "std"]
            }).round(2)
            
            for category in grouped.index:
                stats = grouped.loc[category]
                tqdm.write(f"  {category}:")
                tqdm.write(f"    GT: {stats[('gt_count', 'mean')]}±{stats[('gt_count', 'std')]}")
                tqdm.write(f"    Pred: {stats[('pred_count', 'mean')]}±{stats[('pred_count', 'std')]}")
                tqdm.write(f"    Diff: {stats[('difference', 'mean')]}±{stats[('difference', 'std')]}")
        
        tqdm.write(f"\nResults saved to: {self.output_dir}")
        tqdm.write("=" * 60)


def main():
    """Main function to parse arguments and run the silver standard generation."""
    parser = argparse.ArgumentParser(
        description="Generate silver standard from CEX dataset using reference extraction."
    )

    # LLM Configuration
    parser.add_argument("--model_name", type=str, default="google/gemma-3-27b-it", 
                       help="Name of the LLM model to use.")
    parser.add_argument("--api_key", type=str, default=os.environ.get("DEEPSEEK_API_KEY"), 
                       help="API key for the LLM endpoint. Defaults to DEEPSEEK_API_KEY env var.")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1", 
                       help="Base URL for the LLM API endpoint.")

    # Prompt Configuration
    parser.add_argument("--prompt_name", type=str, default="reference_extraction.md", 
                       help="Name of the prompt file in the 'prompts/' directory.")

    # Extractor Configuration
    parser.add_argument("--extractor", type=str, default="marker", 
                       choices=["pymupdf", "marker", "mineru"], 
                       help="The PDF text extractor to use.")

    # Execution Configuration
    parser.add_argument("--output_path", type=str, default="benchmarks/cex/silver_standard", 
                       help="Directory to save the silver standard results.")
    parser.add_argument("--limit", type=int, default=None, 
                       help="Limit the number of documents to process for a quick test run.")
    parser.add_argument("--max_workers", type=int, default=25, 
                       help="Maximum number of concurrent requests to the LLM.")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Enable verbose logging for debugging.")

    # Output Configuration
    parser.add_argument("--per_category", action="store_true", 
                       help="If set, additionally display extraction statistics broken down by document category.")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not args.api_key:
        raise ValueError("API key must be provided via --api_key or DEEPSEEK_API_KEY environment variable.")

    generator = CEXSilverStandardGenerator(args)
    with logging_redirect_tqdm():
        generator.run()


if __name__ == "__main__":
    main()
