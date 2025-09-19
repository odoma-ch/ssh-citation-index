import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import pickle
import logging
import datetime
import time
import re  # Added for regex operations when cleaning model responses
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import List

from citation_index.core.extractors import ExtractorFactory
from citation_index.llm.client import LLMClient, DeepSeekClient
from citation_index.pipelines.reference_extraction import (
    extract_text_references, extract_text_references_semantic_sections, extract_text_references_by_page
)
from citation_index.pipelines.reference_extraction_and_parsing import run_pdf_one_step
from citation_index.pipelines.reference_parsing import parse_reference_strings
from citation_index.pipelines.text_extraction import split_pages, extract_text
from citation_index.evaluation.ref_metrics import string_reference_eval, RefEvaluator
from excite_helper import load_excite_data
from citation_index.core.models import References


class BenchmarkRunner:
    """Orchestrates the benchmarking process for citation extraction and parsing."""

    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.metrics = []
        # Will store the duration (in seconds) of the LLM call phase (Step 2)
        self.llm_duration = 0.0
        # Counters for error tracking
        self.parse_errors = 0  # JSON or format errors while reading model output
        self.eval_errors = 0   # Any failure during evaluation phase
        
        # Initialize chunker once for methods that need it
        self.chunker = None
        method = getattr(self.args, 'method', 1)
        if (method == 2 and self.args.task == "extraction") or (method == 3 and self.args.task == "extraction_and_parsing"):  # Methods that use semantic section detection
            try:
                from chonkie import LateChunker
                # Create chunker once for pre-chunking texts
                self.chunker = LateChunker.from_recipe("markdown", lang="en")
                tqdm.write(f"Initialized chunker for method {method}")
            except ImportError as e:
                raise ImportError(f"Method {method} requires chonkie package but it's not available. "
                                f"Please install chonkie: pip install chonkie") from e
        
        tqdm.write(f"Results will be saved to: {self.output_dir}")

    def run(self):
        """Main entry point for running the benchmark."""
        logging.debug("Starting benchmark run with the following arguments:")
        logging.debug(json.dumps(vars(self.args), indent=2))
        
        pdf_df, references_data = load_excite_data()

        # Keep a copy for later per-class summaries
        self.pdf_df = pdf_df
        
        extractor = ExtractorFactory.create(self.args.extractor)
        
        # Use DeepSeekClient for DeepSeek models, LLMClient for others
        if "deepseek" in self.args.model_name.lower() or "api.deepseek.com" in self.args.api_base:
            llm_client = DeepSeekClient(api_key=self.args.api_key, 
                                        endpoint=self.args.api_base,
                                        model=self.args.model_name)
            llm_client.timeout = 800
            llm_client.first_token_timeout = 120
            llm_client.max_retries = 3
        else:
            llm_client = LLMClient(
                endpoint=self.args.api_base,
                model=self.args.model_name,
                api_key=self.args.api_key,
                timeout=800,
                first_token_timeout=120,
                max_retries=3,
            )
        
        # ---------------------------------------------------------------
        # Optional fast-path: load previously saved LLM responses and run
        # only the evaluation phase. This allows users to reuse costly
        # generations without repeating them.
        # ---------------------------------------------------------------
        if self.args.responses_path:
            with open(self.args.responses_path, "rb") as f:
                llm_responses = pickle.load(f)
            tqdm.write(
                f"Loaded {len(llm_responses)} LLM responses from {self.args.responses_path}. Skipping LLM calls."
            )
        else:
            # Create all_markdown directory if it doesn't exist
            markdown_dir = Path("benchmarks/excite/all_markdown")
            markdown_dir.mkdir(parents=True, exist_ok=True)
            
            if self.args.limit:
                pdf_df = pdf_df.head(self.args.limit)
            
            # 1. Prepare all inputs for LLM calls
            llm_tasks = []
            for _, row in tqdm(pdf_df.iterrows(), total=len(pdf_df), desc="Preparing tasks"):
                file_id = str(row["file_id"])
                file_path = row["file_path"]
                logging.debug(f"Preparing task for file_id: {file_id}")
                
                gt_references = references_data.get(file_id, {}).get("references", [])
                if not gt_references:
                    logging.warning(f"No ground truth references found for {file_id}. Skipping.")
                    continue

                try:
                    input_text = None
                    if self.args.task != "parsing":
                        # Special case for Grobid - don't extract text, just pass the file path
                        if self.args.extractor == "grobid":
                            input_text = None  # Not needed for Grobid
                        else:
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
                    else: # parsing task
                        input_text = "\n".join(gt_references)
                    
                    # Pre-chunk text for methods that need it (2, 3)
                    chunks = None
                    method = getattr(self.args, 'method', 1)
                    if method in [2, 3] and self.chunker is not None and self.args.task != "parsing":
                        from citation_index.core.segmenters.semantic_reference_locator import pre_chunk_text
                        chunks = pre_chunk_text(input_text, self.chunker)
                    
                    task_info = {
                        "file_id": file_id,
                        "input_text": input_text,
                        "gt_references": gt_references,
                        "file_path": file_path,  # Add file path for Grobid
                        "chunks": chunks,  # Pre-computed chunks for semantic methods
                    }
                    llm_tasks.append(task_info)
                except Exception as e:
                    logging.error(f"Error preparing task for document {file_id}: {e}", exc_info=True)

            # 2. Concurrently process all tasks
            # Methods 4, 5 handle parallelization internally
            # For Grobid, we're not using LLM but still use parallel processing
            method = getattr(self.args, 'method', 1)
            effective_max_workers = 1 if method in [4, 5] else self.args.max_workers
            
            if self.args.extractor == "grobid":
                tqdm.write(f"Processing {len(llm_tasks)} tasks with Grobid extractor using {effective_max_workers} workers.")
            else:
                tqdm.write(f"Submitting {len(llm_tasks)} tasks to LLM with {effective_max_workers} workers (method {getattr(self.args, 'method', 1)}).")
            # Start timer – we only want to measure the duration of the actual
            # LLM requests (Step 2).
            start_llm_timer = time.time()
            llm_responses = []
            with ThreadPoolExecutor(max_workers=effective_max_workers) as executor:
                future_to_task = {
                    executor.submit(self._execute_llm_call, llm_client, task): task
                    for task in llm_tasks
                }
                
                progress_desc = "Processing with Grobid" if self.args.extractor == "grobid" else "Executing LLM calls"
                for future in tqdm(as_completed(future_to_task), total=len(llm_tasks), desc=progress_desc):
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
                                "gt_references": task["gt_references"]
                            })
                        else:
                            logging.warning(f"No response received for file_id: {file_id}")
                    except Exception as exc:
                        logging.error(f'Task for {file_id} generated an exception: {exc}', exc_info=True)
                
                # Persist raw LLM responses so they can be reused later without
                # re-issuing expensive model calls.
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name_slug = self.args.model_name.replace('/', '_').replace('-', '_')
                method = getattr(self.args, 'method', 1)
                run_name = f"{self.args.task}_m{method}_{model_name_slug}_{self.args.extractor}_{timestamp}"
                responses_path = self.output_dir / f"{run_name}_responses.pkl"
                with open(responses_path, "wb") as f:
                    pickle.dump(llm_responses, f)
                tqdm.write(f"Saved LLM responses to {responses_path}")
            # End timer for LLM calls
            self.llm_duration = time.time() - start_llm_timer

        # 3. Process all responses and evaluate
        tqdm.write(f"Evaluating {len(llm_responses)} responses.")
        for response_data in tqdm(llm_responses, desc="Evaluating responses"):
            try:
                if self.args.task == "extraction":
                    self._evaluate_extraction_task(response_data)
                elif self.args.task in ["extraction_and_parsing", "parsing"]:
                    self._evaluate_structured_task(response_data)
            except Exception as e:
                logging.error(f"Error evaluating response for document {response_data['file_id']}: {e}", exc_info=True)
                self.eval_errors += 1

        self._save_results()
        self._summarize_results()

    def _execute_llm_call(self, llm_client, task_info):
        """Execute a single LLM task using pipeline functions."""
        file_id = task_info["file_id"]
        input_text = task_info["input_text"]
        prompt_path = Path("prompts") / self.args.prompt_name
        method = getattr(self.args, 'method', 1)
        logging.debug(f"Executing LLM call for file_id: {file_id} with task: {self.args.task}, method: {method}")

        if self.args.task == "extraction":
            lines = self._execute_extraction_method(
                task_info=task_info,
                llm_client=llm_client,
                prompt_path=str(prompt_path),
            )
            return "\n".join(lines)

        elif self.args.task == "extraction_and_parsing":
            # Special case for Grobid extractor - bypass LLM and use Grobid directly
            if self.args.extractor == "grobid":
                refs = self._execute_grobid_extraction(task_info)
                return json.dumps(refs.model_dump())
            else:
                include_schema = "pydantic" in self.args.prompt_name
                refs = self._execute_extraction_and_parsing_method(
                    task_info=task_info,
                    llm_client=llm_client,
                    prompt_path=str(prompt_path),
                    include_schema=include_schema,
                )
                return json.dumps(refs.model_dump())

        elif self.args.task == "parsing":
            include_schema = "pydantic" in self.args.prompt_name
            reference_lines = [ln for ln in str(input_text).splitlines() if ln.strip()]
            refs = parse_reference_strings(
                reference_lines=reference_lines,
                llm_client=llm_client,
                prompt_name=str(prompt_path),
                include_schema=include_schema,
            )
            return json.dumps(refs.model_dump())

        return None

    def _execute_extraction_method(
        self, task_info: dict, llm_client, prompt_path: str
    ) -> List[str]:
        """Execute the appropriate extraction method based on self.args.method."""
        input_text = task_info["input_text"]
        file_id = task_info["file_id"]
        method = getattr(self.args, 'method', 1)  # Default to method 1 for backwards compatibility
        
        if method == 1:
            # Method 1: Standard LLM-based extraction on full text
            return extract_text_references(
                text=input_text,
                llm_client=llm_client,
                prompt_name=prompt_path,
            )
        
        elif method == 2:
            # Method 2: Semantic section detection with LLM
            chunks = task_info.get("chunks")  # Use pre-computed chunks
            return extract_text_references_semantic_sections(
                text_or_pdf=input_text,
                llm_client=llm_client,
                chunker=self.chunker,
                chunks=chunks,
                extractor=None,
                prompt_name=prompt_path,
                fast_path=True,  # Try regex first for efficiency
            )
            
        
        elif method == 3:
            # Method 3: Page-wise extraction with LLM
            # Calculate optimal max_workers: min(benchmark max_workers, number of pages)
            pages = split_pages(input_text, extractor_type=self.args.extractor)
            optimal_workers = min(self.args.max_workers, len(pages))
            logging.debug(f"Method 3 for {file_id}: {len(pages)} pages, using {optimal_workers} workers")
            
            return extract_text_references_by_page(
                text_or_pdf=input_text,
                llm_client=llm_client,
                extractor=None,
                prompt_name=prompt_path,
                max_workers=optimal_workers,
            )
        
        else:
            raise ValueError(f"Unsupported extraction method: {method}. Must be 1, 2, or 3.")

    def _execute_extraction_and_parsing_method(
        self, task_info: dict, llm_client, prompt_path: str, include_schema: bool
    ) -> References:
        """Execute the appropriate extraction and parsing method based on configuration."""
        input_text = task_info["input_text"]
        file_id = task_info["file_id"]
        method = getattr(self.args, 'method', 1)
        
        if method == 1:
            # Method 1: One-step extraction+parsing on full text
            return run_pdf_one_step(
                text_or_pdf=input_text,
                llm_client=llm_client,
                extractor=None,
                prompt_name=prompt_path,
                include_schema=include_schema,
            )
        
        elif method == 2:
            # Method 2: Two-step – extract reference strings, then parse to structured refs
            from citation_index.pipelines.reference_extraction_and_parsing import run_pdf_two_step
            return run_pdf_two_step(
                text_or_pdf=input_text,
                llm_client=llm_client,
                extractor=None,
                include_schema=include_schema,
            )
        
        elif method == 3:
            # Method 3: Semantic section detection + one-step extraction and parsing
            from citation_index.pipelines.reference_extraction_and_parsing import run_pdf_semantic_one_step
            chunks = task_info.get("chunks")  # Use pre-computed chunks
            return run_pdf_semantic_one_step(
                text_or_pdf=input_text,
                llm_client=llm_client,
                chunker=self.chunker,
                chunks=chunks,
                extractor=None,
                prompt_name=prompt_path,
                include_schema=include_schema,
                fast_path=True,
            )
        
        elif method == 4:
            # Method 4: Page-wise one-step extraction+parsing, then aggregate
            from citation_index.pipelines.reference_extraction_and_parsing import run_pdf_one_step_by_page
            pages = split_pages(input_text, extractor_type=self.args.extractor)
            optimal_workers = min(self.args.max_workers, len(pages))
            logging.debug(f"Method 4 for {file_id}: {len(pages)} pages, using {optimal_workers} workers")
            
            return run_pdf_one_step_by_page(
                text_or_pdf=input_text,
                llm_client=llm_client,
                extractor=None,
                prompt_name=prompt_path,
                include_schema=include_schema,
                max_workers=optimal_workers,
            )
        
        elif method == 5:
            # Method 5: Page-wise extraction of strings, concatenate, then parse once
            from citation_index.pipelines.reference_extraction_and_parsing import run_pdf_two_step_by_page
            pages = split_pages(input_text, extractor_type=self.args.extractor)
            optimal_workers = min(self.args.max_workers, len(pages))
            logging.debug(f"Method 5 for {file_id}: {len(pages)} pages, using {optimal_workers} workers")
            
            return run_pdf_two_step_by_page(
                text_or_pdf=input_text,
                llm_client=llm_client,
                extractor=None,
                include_schema=include_schema,
                max_workers=optimal_workers,
            )
        
        else:
            raise ValueError(f"Unsupported method: {method}. Must be 1, 2, 3, 4, or 5.")

    def _execute_grobid_extraction(self, task_info) -> References:
        """
        Execute Grobid extraction directly from PDF file.
        
        This method bypasses LLM processing entirely and uses Grobid service
        to extract structured references directly from the PDF.
        
        Args:
            task_info: Dictionary containing file_id and file_path
            
        Returns:
            References object containing extracted references
        """
        from citation_index.core.extractors.grobid import GrobidExtractor
        
        file_id = task_info["file_id"]
        pdf_path = Path(task_info["file_path"])
        
        if not pdf_path.exists():
            logging.error(f"PDF file not found: {pdf_path}")
            return References(references=[])
        
        # Initialize Grobid extractor with specified endpoint
        grobid_extractor = GrobidExtractor(endpoint=self.args.grobid_endpoint)
        
        try:
            # Set up XML save directory if flag is enabled
            save_dir = None
            if getattr(self.args, 'save_grobid_xml', False):
                save_dir = Path("benchmarks/excite/all_grobid_xml")
                save_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract references using Grobid
            result = grobid_extractor.extract(
                filepath=str(pdf_path),
                save_dir=str(save_dir) if save_dir else None,
                mode="references",
                parse_references=True,
                consolidate_references=True,
                include_raw_references=True
            )
            
            # Parse XML content to References object
            if result.text:
                references = References.from_xml(xml_str=result.text)
                logging.info(f"Grobid extracted {len(references)} references from {file_id}")
                return references
            else:
                logging.warning(f"Grobid returned empty XML for {file_id}")
                return References(references=[])
                
        except Exception as e:
            logging.error(f"Grobid extraction failed for {file_id}: {e}", exc_info=True)
            return References(references=[])

    def _evaluate_extraction_task(self, response_data):
        """Evaluate a plain text extraction response."""
        file_id = response_data["file_id"]
        logging.debug(f"Evaluating extraction for {file_id}")
        llm_response = response_data["response"]
        gt_references = response_data["gt_references"]
        
        pred_references = [ref.strip() for ref in llm_response.split("\n") if ref.strip()]
        
        self.results.append({"id": file_id, "response": llm_response})
        
        metric = string_reference_eval(gt_references, pred_references)
        metric["file_id"] = file_id
        self.metrics.append(metric)

    def _evaluate_structured_task(self, response_data):
        """Evaluate a structured extraction and/or parsing response."""
        file_id = response_data["file_id"]
        logging.debug(f"Evaluating structured task for {file_id}")
        llm_response_str = response_data["response"]

        json_match = {} # Initialize to an empty dict
        try:
            # Remove common wrapping markers the LLM might include
            llm_response_str = re.sub(r"```(?:json)?\s*", "", llm_response_str, flags=re.IGNORECASE)  # opening fence
            llm_response_str = re.sub(r"\s*```", "", llm_response_str)  # closing fence

            # Remove custom <start> / <end> style tags (case-insensitive)
            llm_response_str = re.sub(r"<\/?\s*start\s*>", "", llm_response_str, flags=re.IGNORECASE)
            llm_response_str = re.sub(r"<\/?\s*end\s*>", "", llm_response_str, flags=re.IGNORECASE)

            llm_response_str = llm_response_str.strip()

            # Parse JSON response using safe parser (same max_attempts as pipeline functions)
            from citation_index.utils.json_helper import safe_json_parse
            json_match = safe_json_parse(llm_response_str)
            
            # Check if parsing failed
            if json_match is None:
                raise json.JSONDecodeError("Failed to parse JSON after all attempts", llm_response_str, 0)

            # Flexible handling: the LLM may return a list of references directly
            # or a dict without the expected top-level key.
            refs_raw = None

            if isinstance(json_match, list):
                # Already a list of reference dicts
                refs_raw = json_match
            elif isinstance(json_match, dict):
                # Try common keys
                for key in ("references", "parsed_references", "refs"):
                    if key in json_match:
                        refs_raw = json_match[key]
                        break
                # Fallback: maybe the whole dict is actually a single reference
                if refs_raw is None and all(k in json_match for k in ("title", "full_title", "authors")):
                    refs_raw = [json_match]

            if refs_raw:
                pred_references = References.from_dict(refs_raw)
            else:
                logging.warning(
                    f"No reference list found in JSON for {file_id}; treating as empty prediction."
                )
                self.parse_errors += 1
                pred_references = References(references=[])
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Error parsing JSON for {file_id}: {e}", exc_info=True)
            self.parse_errors += 1
            pred_references = References(references=[])

        self.results.append({"id": file_id, "references": json_match})

        gt_references_xml_path = f"benchmarks/excite/all_xml/{file_id}.xml"
        if os.path.exists(gt_references_xml_path):
            gt_struct_references = References.from_excite_xml(gt_references_xml_path)
            evaluator = RefEvaluator(mode=self.args.mode, fuzzy_threshold=self.args.fuzzy_threshold)
            metric = evaluator.evaluate(
                [pred_references],
                [gt_struct_references],
                focus_fields=self.args.focus_fields,
            )
            metric["file_id"] = file_id
            self.metrics.append(metric)
        else:
            logging.warning(f"Warning: No XML ground truth for {file_id}, skipping structured evaluation.")

    def _save_results(self):
        """Save the raw results and metrics to files."""

        # Respect --skip_save flag
        if getattr(self.args, "skip_save", False):
            tqdm.write("Skipping saving results & metrics (--skip_save enabled).")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_slug = self.args.model_name.replace('/', '_').replace('-', '_')
        method = getattr(self.args, 'method', 1)
        run_name = f"{self.args.task}_m{method}_{model_name_slug}_{self.args.extractor}_{timestamp}"
        
        results_path = self.output_dir / f"{run_name}_results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(self.results, f)
        tqdm.write(f"Saved raw results to {results_path}")

        if not self.metrics:
            logging.warning("No metrics to save.")
            return

        metrics_df = pd.DataFrame(self.metrics)
        metrics_path = self.output_dir / f"{run_name}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        tqdm.write(f"Saved metrics to {metrics_path}")

    def _summarize_results(self):
        """Print a summary of the benchmark metrics and optionally save to file."""
        if not self.metrics:
            logging.warning("No metrics were generated.")
            return

        metrics_df = pd.DataFrame(self.metrics)
        metrics_df["file_id"] = metrics_df["file_id"].astype(str)
        if hasattr(self, "pdf_df"):
            self.pdf_df["file_id"] = self.pdf_df["file_id"].astype(str)

        output_lines = []
        output_lines.append("\n--- EXCITE Benchmark Summary ---")
        
        # Always show overall metrics first
        if self.args.task == "extraction":
            overall_summary = metrics_df[['precision', 'recall', 'f1_score', 'avg_similarity']].mean().to_dict()
        else:
            overall_summary = metrics_df[['overall_precision', 'overall_recall', 'overall_micro_f1', 'overall_macro_f1']].mean().to_dict()
        
        output_lines.append("Overall metrics (all fields):")
        overall_summary_json = json.dumps(overall_summary, indent=2)
        output_lines.append(overall_summary_json)
        
        # Show focused metrics if focus_fields were used
        if self.args.task != "extraction" and self.args.focus_fields and 'focused_precision' in metrics_df.columns:
            focused_summary = metrics_df[['focused_precision', 'focused_recall', 'focused_micro_f1', 'focused_macro_f1']].mean().to_dict()
            output_lines.append("Focused metrics (focus fields only):")
            focused_summary_json = json.dumps(focused_summary, indent=2)
            output_lines.append(focused_summary_json)
        
        output_lines.append("-------------------------\n")

        tqdm.write("\n--- EXCITE Benchmark Summary ---")
        tqdm.write("Overall metrics (all fields):")
        tqdm.write(overall_summary_json)
        if self.args.task != "extraction" and self.args.focus_fields and 'focused_precision' in metrics_df.columns:
            tqdm.write("Focused metrics (focus fields only):")
            tqdm.write(focused_summary_json)
        tqdm.write("-------------------------\n")

        if (self.args.task in ["extraction_and_parsing", "parsing"] and \
            self.args.focus_fields and 'per_field_metrics' in metrics_df.columns):
            
            # Collect all per_field_metrics dictionaries from all documents
            all_per_field_metrics = []
            for _, row in metrics_df.iterrows():
                if isinstance(row['per_field_metrics'], dict):
                    all_per_field_metrics.append(row['per_field_metrics'])
            
            if all_per_field_metrics:
                # Calculate average metrics across all documents for each field
                field_metrics = {}
                for field in self.args.focus_fields:
                    field_precisions = [doc_metrics.get(field, {}).get('precision', 0.0) for doc_metrics in all_per_field_metrics if field in doc_metrics]
                    field_recalls = [doc_metrics.get(field, {}).get('recall', 0.0) for doc_metrics in all_per_field_metrics if field in doc_metrics]
                    field_f1s = [doc_metrics.get(field, {}).get('f1', 0.0) for doc_metrics in all_per_field_metrics if field in doc_metrics]
                    
                    if field_precisions:
                        field_metrics[field] = {
                            'precision': sum(field_precisions) / len(field_precisions),
                            'recall': sum(field_recalls) / len(field_recalls),
                            'f1': sum(field_f1s) / len(field_f1s)
                        }
                    else:
                        field_metrics[field] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                
                output_lines.append("Per-field metrics:")
                tqdm.write("Per-field metrics:")
                for field, metrics in field_metrics.items():
                    line = f"  {field}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}"
                    output_lines.append(line)
                    tqdm.write(line)
                output_lines.append("-------------------------\n")
                tqdm.write("-------------------------\n")

        output_lines.append("Error statistics:")
        tqdm.write("Error statistics:")
        error_stats_line1 = f"  Parsing errors:   {self.parse_errors}"
        error_stats_line2 = f"  Evaluation errors: {self.eval_errors}"
        output_lines.append(error_stats_line1)
        output_lines.append(error_stats_line2)
        tqdm.write(error_stats_line1)
        tqdm.write(error_stats_line2)

        if self.args.per_class and hasattr(self, "pdf_df"):
            joined = metrics_df.merge(
                self.pdf_df[["file_id", "class", "lang"]], on="file_id", how="left"
            )
            metric_cols = (
                ["precision", "recall", "f1_score", "avg_similarity"]
                if self.args.task == "extraction"
                else ["overall_precision", "overall_recall", "overall_micro_f1", "overall_macro_f1"]
            )
            grouped = (
                joined.groupby(["class", "lang"], dropna=True)[metric_cols]
                .mean()
                .reset_index()
            )
            output_lines.append("Per-class / language metrics:")
            tqdm.write("Per-class / language metrics:")
            for _, row in grouped.iterrows():
                cls, lang = int(row["class"]), row["lang"]
                metrics_str = ", ".join(
                    f"{col}: {row[col]:.4f}" for col in metric_cols if col in row
                )
                line = f"  class {cls}, lang {lang}: {metrics_str}"
                output_lines.append(line)
                tqdm.write(line)

        if self.args.save_scores:
            try:
                scores_path = Path(self.args.save_scores)
                scores_path.parent.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                run_info = [
                    f"\n{'='*80}",
                    f"EXCITE Benchmark Run - {timestamp}",
                    f"{'='*80}",
                    f"Task: {self.args.task}",
                    f"Method: {getattr(self.args, 'method', 1)}",
                    f"Model: {self.args.model_name}",
                    f"Extractor: {self.args.extractor}",
                    f"Prompt: {self.args.prompt_name}",
                    f"Mode: {self.args.mode}",
                    f"Fuzzy Threshold: {self.args.fuzzy_threshold}",
                    f"Focus Fields: {', '.join(self.args.focus_fields) if self.args.focus_fields else 'None'}",
                    f"Documents Processed: {len(self.metrics)}",
                    f"LLM Duration: {self.llm_duration:.2f}s" if self.llm_duration else "LLM Duration: N/A (loaded from file)",
                    f"{'='*80}\n"
                ]
                mode = 'a' if scores_path.exists() else 'w'
                with open(scores_path, mode, encoding='utf-8') as f:
                    f.write('\n'.join(run_info + output_lines))
                tqdm.write(f"Benchmark scores summary appended to: {scores_path}")
            except Exception as e:
                logging.error(f"Failed to save scores to {self.args.save_scores}: {e}")
                tqdm.write(f"Warning: Failed to save scores to {self.args.save_scores}: {e}")


def main():
    """Main function to parse arguments and run the benchmark."""
    parser = argparse.ArgumentParser(
        description="Run citation extraction and parsing benchmarks."
    )

    # Task Configuration
    parser.add_argument("--task", type=str, required=True, choices=["extraction", "extraction_and_parsing", "parsing"], help="The evaluation task to run.")
    parser.add_argument(
        "--method", 
        type=int, 
        default=1, 
        choices=[1, 2, 3, 4, 5], 
        help="Reference extraction/parsing method: "
             "For extraction task: 1=Standard LLM-based extraction on full text (default), "
             "2=Semantic section detection with LLM, 3=Page-wise extraction with LLM. "
             "For extraction_and_parsing task: 1=One-step full text, 2=Two-step full text, "
             "3=Semantic section detection + parsing, 4=Page-wise one-step, 5=Page-wise two-step. "
             "Methods 4, 5 use max_workers=1 due to internal parallelization."
    )

    # LLM Configuration
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-Small-3.2-24B-Instruct-2506", help="Name of the LLM model to use.")
    parser.add_argument("--api_key", type=str, default=os.environ.get("DEEPSEEK_API_KEY"), help="API key for the LLM endpoint. Defaults to DEEPSEEK_API_KEY env var.")
    parser.add_argument("--api_base", type=str, default="http://localhost:8001/v1", help="Base URL for the LLM API endpoint.")

    # Prompt Configuration
    parser.add_argument("--prompt_name", type=str, default="reference_extraction.md", help="Name of the prompt file in the 'prompts/' directory.")

    # Extractor Configuration
    parser.add_argument("--extractor", type=str, default="marker", choices=["pymupdf", "marker", "mineru", "grobid"], help="The PDF text extractor to use. Note: 'grobid' is only available for extraction_and_parsing task.")
    parser.add_argument(
        "--grobid_endpoint",
        type=str,
        default="https://grobid-graphia-app1-staging.apps.bst2.paas.psnc.pl",
        help="Grobid service endpoint URL. Only used when --extractor grobid is specified."
    )

    # Evaluation Configuration
    parser.add_argument("--fuzzy_threshold", type=float, default=90, help="Fuzzy threshold for reference parsing. If ≤1, treated as proportion (e.g., 0.95 → 95).")
    parser.add_argument("--mode", type=str, default="soft_fuzzy", choices=["exact", "fuzzy", "soft_fuzzy"], help="Mode for reference parsing.")
    parser.add_argument(
        "--focus_fields",
        type=str,
        default="authors,full_title,publication_date",
        help="Comma-separated list of fields to evaluate (default: authors,full_title,publication_date).",
    )

    # I/O and Execution Configuration
    parser.add_argument("--output_path", type=str, default="benchmarks/excite/outputs", help="Directory to save the evaluation results.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of documents to process for a quick test run.")
    parser.add_argument("--max_workers", type=int, default=25, help="Maximum number of concurrent requests to the LLM.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging for debugging.")
    parser.add_argument("--responses_path", type=str, default=None, help="Path to a pre-saved pickle file containing LLM responses to load for evaluation-only runs.")

    # Output Configuration
    parser.add_argument(
        "--per_class",
        action="store_true",
        help=(
            "If set, additionally display evaluation metrics broken down by document "
            "class (1, 2, 3) and language (de/en)."
        ),
    )

    # Persistence Configuration
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="If set, do NOT write raw results or metrics to disk.",
    )
    parser.add_argument(
        "--save_scores",
        type=str,
        default=None,
        help="Path to save the final benchmark scores summary. If provided, the summary will be saved to this file in addition to being printed.",
    )
    parser.add_argument(
        "--save_grobid_xml",
        action="store_true",
        help="Save Grobid XML outputs to 'all_grobid_xml' folder. Only applies when using --extractor grobid.",
    )

    args = parser.parse_args()

    # Parse and normalize focus_fields -> List[str] or None
    if args.focus_fields:
        if isinstance(args.focus_fields, str):
            args.focus_fields = [f.strip() for f in args.focus_fields.split(",") if f.strip()]
        elif not isinstance(args.focus_fields, list):
            # Unexpected type; fallback to None
            args.focus_fields = None

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not args.api_key:
        raise ValueError("API key must be provided via --api_key or DEEPSEEK_API_KEY environment variable.")

    # Validate grobid extractor is only used with extraction_and_parsing task
    if args.extractor == "grobid" and args.task != "extraction_and_parsing":
        raise ValueError("The 'grobid' extractor can only be used with the 'extraction_and_parsing' task.")

    runner = BenchmarkRunner(args)
    with logging_redirect_tqdm():
        runner.run()
    
    # Report only the duration spent on LLM calls (Step 2)
    if runner.llm_duration:
        print(f"LLM call execution time: {runner.llm_duration:.2f} seconds")
    else:
        print("LLM calls were skipped (responses loaded from file).")


if __name__ == "__main__":
    main()

