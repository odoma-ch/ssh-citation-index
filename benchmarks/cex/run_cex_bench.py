"""
CEX Benchmark Runner for Citation Extraction and Parsing

This module provides a comprehensive benchmarking framework for evaluating citation
extraction and parsing methods on the CEX (Citation Extraction) dataset. It supports
multiple extraction methods including LLM-based approaches and Grobid.

Key Features:
- Multiple extraction tasks: extraction, extraction_and_parsing, parsing
- Support for different extraction methods (1-5) with various strategies
- Grobid integration for direct PDF reference extraction
- Parallel processing for efficient batch evaluation
- Comprehensive metrics and error analysis
- Results persistence and reusability
"""

# Standard library imports
import argparse
import datetime
import json
import logging
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Third-party imports
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Local imports - citation_index modules
from citation_index.core.extractors import ExtractorFactory
from citation_index.core.models import References
from citation_index.evaluation.ref_metrics import RefEvaluator, string_reference_eval
from citation_index.llm.client import LLMClient, DeepSeekClient
from citation_index.pipelines.reference_extraction import extract_text_references, extract_text_references_semantic_sections
from citation_index.pipelines.reference_extraction_and_parsing import (
    run_pdf_one_step,
    run_pdf_one_step_by_page,
    run_pdf_semantic_one_step,
    run_pdf_two_step,
    run_pdf_two_step_by_page,
)
from citation_index.pipelines.reference_parsing import parse_reference_strings
from citation_index.pipelines.text_extraction import split_pages

# Local imports - benchmark specific
from cex_helper import load_cex_data


class CEXBenchmarkRunner:
    """
    Orchestrates the benchmarking process for citation extraction and parsing on CEX dataset.
    
    This class manages the complete benchmark workflow including:
    - Task preparation and data loading
    - Parallel execution of extraction/parsing methods
    - Evaluation and metrics calculation
    - Results persistence and summarization
    """

    def __init__(self, args):
        """
        Initialize the benchmark runner with configuration arguments.
        
        Args:
            args: Parsed command line arguments containing all configuration
        """
        self.args = args
        self.output_dir = Path(args.output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = []
        self.metrics = []
        
        # Performance tracking
        self.llm_duration = 0.0  # Duration of LLM call phase in seconds
        
        # Error counters
        self.parse_errors = 0   # JSON or format errors while reading model output
        self.eval_errors = 0    # Any failure during evaluation phase
        
        # Initialize chunker once for methods that need it
        self.chunker = None
        method = getattr(self.args, 'method', 1)
        if method in [2, 3]:  # Methods that use semantic section detection
            try:
                from chonkie import LateChunker
                # Create chunker once - we'll handle threading issues by using max_workers=1
                self.chunker = LateChunker.from_recipe("markdown", lang="en")
                tqdm.write(f"Initialized chunker for method {method}")
            except ImportError as e:
                raise ImportError(f"Method {method} requires chonkie package but it's not available. "
                                f"Please install chonkie: pip install chonkie") from e
        
        tqdm.write(f"Results will be saved to: {self.output_dir}")

    def run(self):
        """
        Main entry point for running the benchmark.
        
        This method orchestrates the complete benchmark workflow:
        1. Load CEX dataset and prepare tasks
        2. Execute extraction/parsing (LLM or Grobid)
        3. Evaluate results against ground truth
        4. Save results and generate summary
        """
        logging.debug("Starting CEX benchmark run with the following arguments:")
        logging.debug(json.dumps(vars(self.args), indent=2))
        
        # Load CEX dataset
        pdf_df, papers_data = load_cex_data()
        self.pdf_df = pdf_df  # Keep for per-category summaries
        
        # Initialize extractors and clients
        extractor = ExtractorFactory.create(self.args.extractor)
        
        # Use DeepSeekClient for DeepSeek models, LLMClient for others
        if "deepseek" in self.args.model_name.lower() or "api.deepseek.com" in self.args.api_base:
            llm_client = DeepSeekClient(api_key=self.args.api_key, 
                                        endpoint=self.args.api_base,
                                        model=self.args.model_name)
            llm_client.timeout = 1200
            llm_client.first_token_timeout = 180
            llm_client.max_retries = 2
        else:
            llm_client = LLMClient(
                endpoint=self.args.api_base,
                model=self.args.model_name,
                api_key=self.args.api_key,
                timeout=1200,
                first_token_timeout=180,
                max_retries=2,
            )
        
        # Check for pre-saved responses (fast-path for evaluation-only runs)
        if self.args.responses_path:
            with open(self.args.responses_path, "rb") as f:
                llm_responses = pickle.load(f)
            tqdm.write(
                f"Loaded {len(llm_responses)} responses from {self.args.responses_path}. "
                "Skipping extraction phase."
            )
        else:
            llm_responses = self._execute_extraction_phase(
                pdf_df, papers_data, extractor, llm_client
            )

        # Evaluation phase
        self._execute_evaluation_phase(llm_responses)
        
        # Results processing
        self._save_results()
        self._summarize_results()

    def _execute_extraction_phase(self, pdf_df, papers_data, extractor, llm_client):
        """
        Execute the extraction/parsing phase for all documents.
        
        Args:
            pdf_df: DataFrame containing PDF file information
            papers_data: Dictionary containing ground truth references
            extractor: Text extractor instance
            llm_client: LLM client for API calls
            
        Returns:
            List of response dictionaries containing extraction results
        """
        # Create markdown cache directory
        markdown_dir = Path("benchmarks/cex/all_markdown")
        markdown_dir.mkdir(parents=True, exist_ok=True)
        
        # Limit documents if specified
        if self.args.limit:
            pdf_df = pdf_df.head(self.args.limit)
        
        # Prepare tasks for parallel execution
        llm_tasks = self._prepare_tasks(pdf_df, papers_data, extractor, markdown_dir)
        
        # Execute tasks in parallel
        return self._execute_parallel_tasks(llm_tasks, llm_client)

    def _prepare_tasks(self, pdf_df, papers_data, extractor, markdown_dir):
        """
        Prepare all tasks for parallel execution.
        
        Args:
            pdf_df: DataFrame containing PDF file information
            papers_data: Dictionary containing ground truth references
            extractor: Text extractor instance
            markdown_dir: Directory for caching extracted text
            
        Returns:
            List of task dictionaries ready for parallel execution
        """
        llm_tasks = []
        
        for _, row in tqdm(pdf_df.iterrows(), total=len(pdf_df), desc="Preparing tasks"):
            file_id = str(row["file_id"])
            file_path = row["file_path"]
            logging.debug(f"Preparing task for file_id: {file_id}")
            
            # Check for ground truth references
            gt_references = papers_data.get(file_id, {}).get("references", [])
            if not gt_references:
                logging.warning(f"No ground truth references found for {file_id}. Skipping.")
                continue

            try:
                input_text = self._prepare_input_text(
                    file_id, file_path, extractor, markdown_dir
                )
                
                task_info = {
                    "file_id": file_id,
                    "input_text": input_text,
                    "gt_references": gt_references,
                    "file_path": file_path,  # Required for Grobid
                }
                llm_tasks.append(task_info)
                
            except Exception as e:
                logging.error(f"Error preparing task for document {file_id}: {e}", exc_info=True)

        return llm_tasks

    def _prepare_input_text(self, file_id, file_path, extractor, markdown_dir):
        """
        Prepare input text for a single document based on task type and extractor.
        
        Args:
            file_id: Unique identifier for the document
            file_path: Path to the PDF file
            extractor: Text extractor instance
            markdown_dir: Directory for caching extracted text
            
        Returns:
            Input text string or None for Grobid
        """
        if self.args.task == "parsing":
            # For parsing task, input is ground truth references
            # This will be handled separately in the calling function
            return None
        
        # Special handling for Grobid - no text extraction needed
        if self.args.extractor == "grobid":
            return None
        
        # Check for cached extracted text
        markdown_path = markdown_dir / f"{file_id}_{self.args.extractor}.md"
        if markdown_path.exists():
            with open(markdown_path, "r", encoding="utf-8") as md_file:
                return md_file.read()
        
        # Extract text from PDF
        extracted_text_result = extractor.extract(file_path)
        if not extracted_text_result.text.strip():
            raise ValueError(f"No text extracted from {file_id}.pdf")
        
        # Cache extracted text
        with open(markdown_path, "w", encoding="utf-8") as md_file:
            md_file.write(extracted_text_result.text)
        
        return extracted_text_result.text

    def _execute_parallel_tasks(self, llm_tasks, llm_client):
        """
        Execute all tasks in parallel using ThreadPoolExecutor.
        
        Args:
            llm_tasks: List of prepared task dictionaries
            llm_client: LLM client for API calls
            
        Returns:
            List of response dictionaries containing results
        """
        # Adjust worker count for different methods
        # Methods 2, 3 use shared chunker instance, so need max_workers=1 to avoid threading issues
        # Methods 4, 5 handle parallelization internally
        method = getattr(self.args, 'method', 1)
        effective_max_workers = (
            1 if method in [2, 3, 4, 5] else self.args.max_workers
        )
        
        # Display appropriate progress message
        if self.args.extractor == "grobid":
            tqdm.write(
                f"Processing {len(llm_tasks)} tasks with Grobid extractor "
                f"using {effective_max_workers} workers."
            )
        else:
            tqdm.write(
                f"Submitting {len(llm_tasks)} tasks to LLM with {effective_max_workers} "
                f"workers (method {self.args.method})."
            )

        # Execute tasks with timing
        start_time = time.time()
        llm_responses = []
        
        with ThreadPoolExecutor(max_workers=effective_max_workers) as executor:
            future_to_task = {
                executor.submit(self._execute_single_task, llm_client, task): task
                for task in llm_tasks
            }
            
            progress_desc = (
                "Processing with Grobid" if self.args.extractor == "grobid" 
                else "Executing LLM calls"
            )
            
            for future in tqdm(
                as_completed(future_to_task), total=len(llm_tasks), desc=progress_desc
            ):
                task = future_to_task[future]
                file_id = task["file_id"]
                
                try:
                    response_str = future.result()
                    if response_str:
                        llm_responses.append({
                            "file_id": file_id,
                            "response": response_str,
                            "gt_references": task["gt_references"]
                        })
                    else:
                        logging.warning(f"No response received for file_id: {file_id}")
                        
                except Exception as exc:
                    logging.error(f'Task for {file_id} generated an exception: {exc}', exc_info=True)
            
            # Save responses for future reuse
            self._save_responses(llm_responses)
        
        self.llm_duration = time.time() - start_time
        return llm_responses

    def _execute_single_task(self, llm_client, task_info):
        """
        Execute a single extraction/parsing task.
        
        Args:
            llm_client: LLM client for API calls
            task_info: Dictionary containing task information
            
        Returns:
            String representation of the extraction result
        """
        file_id = task_info["file_id"]
        input_text = task_info["input_text"]
        prompt_path = Path("prompts") / self.args.prompt_name
        
        logging.debug(
            f"Executing task for file_id: {file_id} with task: {self.args.task}, "
            f"method: {self.args.method}"
        )

        # Route to appropriate processing method based on task type
        if self.args.task == "extraction":
            return self._execute_extraction_task(input_text, llm_client, prompt_path)
        
        elif self.args.task == "extraction_and_parsing":
            return self._execute_extraction_and_parsing_task(
                task_info, llm_client, prompt_path
            )
        
        elif self.args.task == "parsing":
            return self._execute_parsing_task(task_info, llm_client, prompt_path)
        
        return None

    def _execute_extraction_task(self, input_text, llm_client, prompt_path):
        """Execute plain text extraction task using the specified method."""
        method = getattr(self.args, 'method', 1)
        
        if method == 1:
            # Method 1: Standard LLM-based extraction on full text
            lines = extract_text_references(
                text=input_text,
                llm_client=llm_client,
                prompt_name=str(prompt_path),
            )
            return "\n".join(lines)
        
        elif method == 2:
            # Method 2: Semantic section detection with LLM
            lines = extract_text_references_semantic_sections(
                text_or_pdf=input_text,
                llm_client=llm_client,
                chunker=self.chunker,
                extractor=None,
                prompt_name=str(prompt_path),
                fast_path=True,
            )
            return "\n".join(lines)
            
        
        elif method == 3:
            # Method 3: Page-wise extraction with LLM
            from citation_index.pipelines.reference_extraction import extract_text_references_by_page
            pages = split_pages(input_text, extractor_type=self.args.extractor)
            optimal_workers = min(self.args.max_workers, len(pages))
            logging.debug(f"Method 3 extraction: {len(pages)} pages, using {optimal_workers} workers")
            
            lines = extract_text_references_by_page(
                text_or_pdf=input_text,
                llm_client=llm_client,
                extractor=None,
                prompt_name=str(prompt_path),
                max_workers=optimal_workers,
            )
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported extraction method: {method}. Must be 1, 2, or 3.")

    def _execute_extraction_and_parsing_task(self, task_info, llm_client, prompt_path):
        """Execute extraction and parsing task (structured output)."""
        # Special case for Grobid - bypass LLM entirely
        if self.args.extractor == "grobid":
            refs = self._execute_grobid_extraction(task_info)
            return json.dumps(refs.model_dump())
        
        # Regular LLM-based extraction and parsing
        include_schema = "pydantic" in self.args.prompt_name
        refs = self._execute_extraction_and_parsing_method(
            input_text=task_info["input_text"],
            llm_client=llm_client,
            prompt_path=str(prompt_path),
            include_schema=include_schema,
            file_id=task_info["file_id"],
        )
        return json.dumps(refs.model_dump())

    def _execute_parsing_task(self, task_info, llm_client, prompt_path):
        """Execute parsing-only task on ground truth reference strings."""
        include_schema = "pydantic" in self.args.prompt_name
        gt_references = task_info["gt_references"]
        reference_lines = [ln.strip() for ln in gt_references if ln.strip()]
        
        refs = parse_reference_strings(
            reference_lines=reference_lines,
            llm_client=llm_client,
            prompt_name=str(prompt_path),
            include_schema=include_schema,
        )
        return json.dumps(refs.model_dump())

    def _execute_extraction_and_parsing_method(
        self, input_text: str, llm_client, prompt_path: str, include_schema: bool, file_id: str
    ) -> References:
        """
        Execute the appropriate extraction and parsing method based on configuration.
        
        Args:
            input_text: Extracted text from PDF
            llm_client: LLM client for API calls
            prompt_path: Path to prompt file
            include_schema: Whether to include Pydantic schema in prompt
            file_id: Document identifier for logging
            
        Returns:
            References object containing extracted references
        """
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
            return run_pdf_two_step(
                text_or_pdf=input_text,
                llm_client=llm_client,
                extractor=None,
                include_schema=include_schema,
            )
        
        elif method == 3:
            # Method 3: Semantic section detection + one-step extraction and parsing
            return run_pdf_semantic_one_step(
                text_or_pdf=input_text,
                llm_client=llm_client,
                chunker=self.chunker,
                extractor=None,
                include_schema=include_schema,
            )
            
        
        elif method == 4:
            # Method 4: Page-wise one-step extraction+parsing, then aggregate
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
                save_dir = Path("benchmarks/cex/all_grobid_xml")
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

    def _save_responses(self, llm_responses):
        """Save LLM responses to pickle file for future reuse."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_slug = self.args.model_name.replace('/', '_').replace('-', '_')
        method = getattr(self.args, 'method', 1)
        run_name = f"{self.args.task}_m{method}_{model_name_slug}_{self.args.extractor}_{timestamp}"
        
        responses_path = self.output_dir / f"{run_name}_responses.pkl"
        with open(responses_path, "wb") as f:
            pickle.dump(llm_responses, f)
        tqdm.write(f"Saved responses to {responses_path}")

    def _execute_evaluation_phase(self, llm_responses):
        """
        Execute the evaluation phase for all responses.
        
        Args:
            llm_responses: List of response dictionaries from extraction phase
        """
        tqdm.write(f"Evaluating {len(llm_responses)} responses.")
        
        for response_data in tqdm(llm_responses, desc="Evaluating responses"):
            try:
                if self.args.task == "extraction":
                    self._evaluate_extraction_task(response_data)
                elif self.args.task in ["extraction_and_parsing", "parsing"]:
                    self._evaluate_structured_task(response_data)
            except Exception as e:
                logging.error(
                    f"Error evaluating response for document {response_data['file_id']}: {e}", 
                    exc_info=True
                )
                self.eval_errors += 1

    def _evaluate_extraction_task(self, response_data):
        """
        Evaluate a plain text extraction response.
        
        For extraction tasks, we calculate precision, recall, and F1 metrics
        using fuzzy string matching to compare predicted vs ground truth references.
        
        Args:
            response_data: Dictionary containing response and ground truth
        """
        file_id = response_data["file_id"]
        llm_response = response_data["response"]
        gt_references = response_data["gt_references"]
        
        # Parse predicted references
        pred_references = [ref.strip() for ref in llm_response.split("\n") if ref.strip()]
        
        # Calculate evaluation metrics using string-based comparison
        try:
            metric = string_reference_eval(
                references_data=gt_references,
                response_list=pred_references,
                similarity_mode='fuzzy',
                similarity_threshold=0.8
            )
            metric["file_id"] = file_id
            self.metrics.append(metric)
            logging.debug(f"Extraction metrics for {file_id}: {metric}")
        except Exception as e:
            logging.error(f"Error calculating metrics for {file_id}: {e}", exc_info=True)
            self.eval_errors += 1
        
        self.results.append({"id": file_id, "response": llm_response})

    def _evaluate_structured_task(self, response_data):
        """
        Evaluate a structured extraction and/or parsing response.
        
        Args:
            response_data: Dictionary containing response and ground truth
        """
        file_id = response_data["file_id"]
        llm_response_str = response_data["response"]

        # Parse and clean LLM response
        try:
            pred_references = self._parse_llm_response(llm_response_str, file_id)
        except Exception as e:
            logging.error(f"Error parsing response for {file_id}: {e}", exc_info=True)
            self.parse_errors += 1
            pred_references = References(references=[])

        self.results.append({"id": file_id, "references": pred_references.model_dump()})

        # Evaluate against ground truth if available
        gt_references_xml_path = f"benchmarks/cex/all_xmls/{file_id}.xml"
        if os.path.exists(gt_references_xml_path):
            try:
                gt_struct_references = References.from_xml(file_path=gt_references_xml_path)
                evaluator = RefEvaluator(
                    mode=self.args.mode, 
                    fuzzy_threshold=self.args.fuzzy_threshold
                )
                metric = evaluator.evaluate(
                    [pred_references],
                    [gt_struct_references],
                    focus_fields=self.args.focus_fields,
                )
                metric["file_id"] = file_id
                self.metrics.append(metric)
            except Exception as e:
                logging.error(f"Error evaluating {file_id}: {e}", exc_info=True)
                self.eval_errors += 1
        else:
            logging.warning(f"No XML ground truth for {file_id}, skipping structured evaluation.")

    def _parse_llm_response(self, llm_response_str, file_id):
        """
        Parse and clean LLM response string into References object.
        
        Args:
            llm_response_str: Raw response string from LLM
            file_id: Document identifier for logging
            
        Returns:
            References object containing parsed references
        """
        # Clean common LLM response artifacts
        llm_response_str = re.sub(r"```(?:json)?\s*", "", llm_response_str, flags=re.IGNORECASE)
        llm_response_str = re.sub(r"\s*```", "", llm_response_str)
        llm_response_str = re.sub(r"<\/?\s*start\s*>", "", llm_response_str, flags=re.IGNORECASE)
        llm_response_str = re.sub(r"<\/?\s*end\s*>", "", llm_response_str, flags=re.IGNORECASE)
        llm_response_str = llm_response_str.strip()

        # Parse JSON response using safe parser
        from citation_index.utils.json_helper import safe_json_parse
        json_match = safe_json_parse(llm_response_str)
        
        # Check if parsing failed
        if json_match is None:
            raise json.JSONDecodeError("Failed to parse JSON after all attempts", llm_response_str, 0)

        # Handle different response formats
        refs_raw = None
        if isinstance(json_match, list):
            refs_raw = json_match
        elif isinstance(json_match, dict):
            # Try common keys
            for key in ("references", "parsed_references", "refs"):
                if key in json_match:
                    refs_raw = json_match[key]
                    break
            # Fallback: single reference object
            if refs_raw is None and all(k in json_match for k in ("title", "full_title", "authors")):
                refs_raw = [json_match]

        if refs_raw:
            return References.from_dict(refs_raw)
        else:
            logging.warning(f"No reference list found in JSON for {file_id}")
            return References(references=[])

    def _save_results(self):
        """Save the raw results and metrics to files."""
        if getattr(self.args, "skip_save", False):
            tqdm.write("Skipping saving results & metrics (--skip_save enabled).")
            return

        # Generate run identifier
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_slug = self.args.model_name.replace('/', '_').replace('-', '_')
        method = getattr(self.args, 'method', 1)
        run_name = f"{self.args.task}_m{method}_{model_name_slug}_{self.args.extractor}_{timestamp}"
        
        # Save raw results
        results_path = self.output_dir / f"{run_name}_results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(self.results, f)
        tqdm.write(f"Saved raw results to {results_path}")

        # Save metrics if available
        if self.metrics:
            metrics_df = pd.DataFrame(self.metrics)
            metrics_path = self.output_dir / f"{run_name}_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            tqdm.write(f"Saved metrics to {metrics_path}")
        else:
            logging.warning("No metrics to save.")

    def _summarize_results(self):
        """Print and optionally save a comprehensive summary of benchmark results."""
        # Validate we have results to summarize
        if not self.metrics:
            logging.warning("No metrics were generated.")
            return

        output_lines = []
        
        # Generate metrics summary for all tasks
        self._generate_metrics_summary(output_lines)
        
        
        # Generate per-field metrics if applicable
        if (self.args.task in ["extraction_and_parsing", "parsing"] and 
            self.args.focus_fields and self.metrics):
            self._generate_per_field_metrics(output_lines)
        
        # Generate error statistics
        self._generate_error_statistics(output_lines)
        
        # Generate per-category breakdown if requested
        if self.args.per_category and hasattr(self, "pdf_df"):
            self._generate_per_category_breakdown(output_lines)
        
        # Save summary to file if requested
        if self.args.save_scores:
            self._save_scores_summary(output_lines)

    def _generate_metrics_summary(self, output_lines):
        """Generate overall metrics summary for all task types."""
        metrics_df = pd.DataFrame(self.metrics)
        metrics_df["file_id"] = metrics_df["file_id"].astype(str)
        if hasattr(self, "pdf_df"):
            self.pdf_df["file_id"] = self.pdf_df["file_id"].astype(str)
        
        output_lines.append("\n--- CEX Benchmark Summary ---")
        
        # Determine which metrics to use based on task type
        if self.args.task == "extraction":
            # For extraction task, use string-based metrics
            metric_cols = ['precision', 'recall', 'f1_score', 'avg_similarity']
            overall_summary = metrics_df[metric_cols].mean().to_dict()
            output_lines.append("Overall extraction metrics:")
            overall_summary_json = json.dumps(overall_summary, indent=2)
            output_lines.append(overall_summary_json)
        else:
            # For structured tasks, use structured metrics
            metric_cols = ['overall_precision', 'overall_recall', 'overall_micro_f1', 'overall_macro_f1']
            overall_summary = metrics_df[metric_cols].mean().to_dict()
            output_lines.append("Overall metrics (all fields):")
            overall_summary_json = json.dumps(overall_summary, indent=2)
            output_lines.append(overall_summary_json)
            
            # Focused metrics if available for structured tasks
            if self.args.focus_fields and 'focused_precision' in metrics_df.columns:
                focused_summary = metrics_df[
                    ['focused_precision', 'focused_recall', 'focused_micro_f1', 'focused_macro_f1']
                ].mean().to_dict()
                output_lines.append("Focused metrics (focus fields only):")
                focused_summary_json = json.dumps(focused_summary, indent=2)
                output_lines.append(focused_summary_json)
        
        output_lines.append("-------------------------\n")
        
        # Print to console
        tqdm.write("\n--- CEX Benchmark Summary ---")
        if self.args.task == "extraction":
            tqdm.write("Overall extraction metrics:")
        else:
            tqdm.write("Overall metrics (all fields):")
        tqdm.write(overall_summary_json)
        
        if (self.args.task != "extraction" and 
            self.args.focus_fields and 'focused_precision' in metrics_df.columns):
            tqdm.write("Focused metrics (focus fields only):")
            tqdm.write(focused_summary_json)
        tqdm.write("-------------------------\n")

    def _generate_per_field_metrics(self, output_lines):
        """Generate per-field metrics analysis."""
        metrics_df = pd.DataFrame(self.metrics)
        
        if 'per_field_metrics' not in metrics_df.columns:
            return
        
        # Collect all per-field metrics
        all_per_field_metrics = []
        for _, row in metrics_df.iterrows():
            if isinstance(row['per_field_metrics'], dict):
                all_per_field_metrics.append(row['per_field_metrics'])
        
        if not all_per_field_metrics:
            return
        
        # Calculate average metrics for each field
        field_metrics = {}
        for field in self.args.focus_fields:
            field_precisions = [
                doc_metrics.get(field, {}).get('precision', 0.0) 
                for doc_metrics in all_per_field_metrics if field in doc_metrics
            ]
            field_recalls = [
                doc_metrics.get(field, {}).get('recall', 0.0) 
                for doc_metrics in all_per_field_metrics if field in doc_metrics
            ]
            field_f1s = [
                doc_metrics.get(field, {}).get('f1', 0.0) 
                for doc_metrics in all_per_field_metrics if field in doc_metrics
            ]
            
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
            field_line = (
                f"  {field}: P={metrics['precision']:.4f}, "
                f"R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}"
            )
            output_lines.append(field_line)
            tqdm.write(field_line)
        output_lines.append("-------------------------\n")
        tqdm.write("-------------------------\n")

    def _generate_error_statistics(self, output_lines):
        """Generate error statistics summary."""
        output_lines.append("Error statistics:")
        tqdm.write("Error statistics:")
        error_stats_line1 = f"  Parsing errors:   {self.parse_errors}"
        error_stats_line2 = f"  Evaluation errors: {self.eval_errors}"
        output_lines.extend([error_stats_line1, error_stats_line2])
        tqdm.write(error_stats_line1)
        tqdm.write(error_stats_line2)

    def _generate_per_category_breakdown(self, output_lines):
        """Generate per-category performance breakdown."""
        if self.metrics:
            metrics_df = pd.DataFrame(self.metrics)
            joined = metrics_df.merge(
                self.pdf_df[["file_id", "category"]], on="file_id", how="left"
            )

            # Choose appropriate metric columns based on task type
            if self.args.task == "extraction":
                metric_cols = ["precision", "recall", "f1_score", "avg_similarity"]
                output_lines.append("Per-category extraction metrics:")
                tqdm.write("Per-category extraction metrics:")
            else:
                metric_cols = ["overall_precision", "overall_recall", "overall_micro_f1", "overall_macro_f1"]
                output_lines.append("Per-category metrics:")
                tqdm.write("Per-category metrics:")
            
            grouped = (
                joined.groupby(["category"], dropna=True)[metric_cols]
                .mean()
                .reset_index()
            )

            for _, row in grouped.iterrows():
                category = row["category"]
                metrics_str = ", ".join(
                    f"{col}: {row[col]:.4f}" for col in metric_cols if col in row
                )
                category_line = f"  {category}: {metrics_str}"
                output_lines.append(category_line)
                tqdm.write(category_line)

    def _save_scores_summary(self, output_lines):
        """Save benchmark scores summary to file."""
        try:
            scores_path = Path(self.args.save_scores)
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare run information header
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            documents_processed = len(self.metrics)
            
            run_info = [
                f"\n{'='*80}",
                f"CEX Benchmark Run - {timestamp}",
                f"{'='*80}",
                f"Task: {self.args.task}",
                f"Method: {getattr(self.args, 'method', 1)}",
                f"Model: {self.args.model_name}",
                f"Extractor: {self.args.extractor}",
                f"Prompt: {self.args.prompt_name}",
                f"Mode: {self.args.mode}",
                f"Fuzzy Threshold: {self.args.fuzzy_threshold}",
                f"Focus Fields: {', '.join(self.args.focus_fields) if self.args.focus_fields else 'None'}",
                f"Documents Processed: {documents_processed}",
                f"LLM Duration: {self.llm_duration:.2f}s" if self.llm_duration else "LLM Duration: N/A (loaded from file)",
                f"{'='*80}\n"
            ]
            
            # Append to existing file or create new one
            mode = 'a' if scores_path.exists() else 'w'
            with open(scores_path, mode, encoding='utf-8') as f:
                f.write('\n'.join(run_info + output_lines))
            
            tqdm.write(f"Benchmark scores summary appended to: {scores_path}")
            
        except Exception as e:
            logging.error(f"Failed to save scores to {self.args.save_scores}: {e}")
            tqdm.write(f"Warning: Failed to save scores to {self.args.save_scores}: {e}")


def main():
    """
    Main function to parse arguments and run the benchmark.
    
    This function handles argument parsing, validation, and orchestrates
    the complete benchmark execution workflow.
    """
    parser = argparse.ArgumentParser(
        description="Run citation extraction and parsing benchmarks on CEX dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run LLM-based extraction and parsing
  python run_cex_bench.py --task extraction_and_parsing --method 1 --model_name gpt-4 --api_key YOUR_KEY

  # Run Grobid-based extraction and parsing with XML saving
  python run_cex_bench.py --task extraction_and_parsing --extractor grobid --save_grobid_xml --model_name dummy --api_key dummy

  # Run with limited documents for testing
  python run_cex_bench.py --task extraction --limit 10 --model_name gpt-3.5-turbo --api_key YOUR_KEY
        """
    )

    # Task Configuration
    parser.add_argument(
        "--task", 
        type=str, 
        required=True, 
        choices=["extraction", "extraction_and_parsing", "parsing"], 
        help="The evaluation task to run."
    )
    parser.add_argument(
        "--method", 
        type=int, 
        default=1, 
        choices=[1, 2, 3, 4, 5], 
        help="Reference extraction/parsing method: "
             "For extraction task: 1=Standard LLM-based extraction on full text (default), "
             "2=Semantic section detection with LLM, 3=Page-wise extraction with LLM. "
             "For extraction_and_parsing task: 1=One-step full text (default), "
             "2=Two-step full text, 3=Semantic section detection + parsing, "
             "4=Page-wise one-step, 5=Page-wise two-step. "
             "Methods 2, 3, 4, 5 use max_workers=1 (methods 2&3 due to shared chunker, methods 4&5 due to internal parallelization)."
    )

    # LLM Configuration
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="google/gemma-3-27b-it", 
        help="Name of the LLM model to use."
    )
    parser.add_argument(
        "--api_key", 
        type=str, 
        default=os.environ.get("DEEPSEEK_API_KEY"), 
        help="API key for the LLM endpoint. Defaults to DEEPSEEK_API_KEY env var."
    )
    parser.add_argument(
        "--api_base", 
        type=str, 
        default="http://localhost:8000/v1", 
        help="Base URL for the LLM API endpoint."
    )

    # Prompt Configuration
    parser.add_argument(
        "--prompt_name", 
        type=str, 
        default="reference_extraction.md", 
        help="Name of the prompt file in the 'prompts/' directory."
    )

    # Extractor Configuration
    parser.add_argument(
        "--extractor", 
        type=str, 
        default="marker", 
        choices=["pymupdf", "marker", "mineru", "grobid"], 
        help="The PDF text extractor to use. Note: 'grobid' is only available for extraction_and_parsing task."
    )
    parser.add_argument(
        "--grobid_endpoint",
        type=str,
        default="https://grobid-graphia-app1-staging.apps.bst2.paas.psnc.pl",
        help="Grobid service endpoint URL. Only used when --extractor grobid is specified."
    )

    # Evaluation Configuration
    parser.add_argument(
        "--fuzzy_threshold", 
        type=float, 
        default=90, 
        help="Fuzzy threshold for reference parsing. If ≤1, treated as proportion (e.g., 0.95 → 95)."
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="soft_fuzzy", 
        choices=["exact", "fuzzy", "soft_fuzzy"], 
        help="Mode for reference parsing."
    )
    parser.add_argument(
        "--focus_fields",
        type=str,
        default="authors,full_title,publication_date",
        help="Comma-separated list of fields to evaluate (default: authors,full_title,publication_date).",
    )

    # I/O and Execution Configuration
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="benchmarks/cex/outputs", 
        help="Directory to save the evaluation results."
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Limit the number of documents to process for a quick test run."
    )
    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=25, 
        help="Maximum number of concurrent requests to the LLM."
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose logging for debugging."
    )
    parser.add_argument(
        "--responses_path", 
        type=str, 
        default=None, 
        help="Path to a pre-saved pickle file containing LLM responses to load for evaluation-only runs."
    )

    # Output Configuration
    parser.add_argument(
        "--per_category",
        action="store_true",
        help="If set, additionally display evaluation metrics broken down by document category.",
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

    # Parse and normalize focus_fields
    if args.focus_fields:
        if isinstance(args.focus_fields, str):
            args.focus_fields = [f.strip() for f in args.focus_fields.split(",") if f.strip()]
        elif not isinstance(args.focus_fields, list):
            args.focus_fields = None

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Validate required arguments
    if not args.api_key:
        raise ValueError("API key must be provided via --api_key or DEEPSEEK_API_KEY environment variable.")

    # Validate grobid extractor usage
    if args.extractor == "grobid" and args.task != "extraction_and_parsing":
        raise ValueError("The 'grobid' extractor can only be used with the 'extraction_and_parsing' task.")

    # Run benchmark
    runner = CEXBenchmarkRunner(args)
    with logging_redirect_tqdm():
        runner.run()
    
    # Report execution time
    if runner.llm_duration:
        print(f"Processing execution time: {runner.llm_duration:.2f} seconds")
    else:
        print("Processing was skipped (responses loaded from file).")


if __name__ == "__main__":
    main()