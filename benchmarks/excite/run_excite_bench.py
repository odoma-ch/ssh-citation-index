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

from citation_index.core.extractors import ExtractorFactory
from citation_index.llm.client import LLMClient
from citation_index.llm.prompt_loader import (
    ReferenceExtractionPrompt,
    ReferenceExtractionAndParsingPrompt,
    ReferenceParsingPrompt,
)
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
        tqdm.write(f"Results will be saved to: {self.output_dir}")

    def run(self):
        """Main entry point for running the benchmark."""
        logging.debug("Starting benchmark run with the following arguments:")
        logging.debug(json.dumps(vars(self.args), indent=2))
        
        pdf_df, references_data = load_excite_data()

        # Keep a copy for later per-class summaries
        self.pdf_df = pdf_df
        
        extractor = ExtractorFactory.create(self.args.extractor)
        llm_client = LLMClient(
            endpoint=self.args.api_base,
            model=self.args.model_name,
            api_key=self.args.api_key,
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
                    
                    task_info = {
                        "file_id": file_id,
                        "input_text": input_text,
                        "gt_references": gt_references,
                    }
                    llm_tasks.append(task_info)
                except Exception as e:
                    logging.error(f"Error preparing task for document {file_id}: {e}", exc_info=True)

            # 2. Concurrently call LLM for all tasks
            tqdm.write(f"Submitting {len(llm_tasks)} tasks to LLM with {self.args.max_workers} workers.")
            # Start timer – we only want to measure the duration of the actual
            # LLM requests (Step 2).
            start_llm_timer = time.time()
            llm_responses = []
            with ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
                future_to_task = {
                    executor.submit(self._execute_llm_call, llm_client, task): task
                    for task in llm_tasks
                }
                
                for future in tqdm(as_completed(future_to_task), total=len(llm_tasks), desc="Executing LLM calls"):
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
                run_name = f"{self.args.task}_{model_name_slug}_{self.args.extractor}_{timestamp}"
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
        """Prepare prompt and execute a single LLM call."""
        file_id = task_info["file_id"]
        input_text = task_info["input_text"]
        prompt_path = Path("prompts") / self.args.prompt_name
        logging.debug(f"Executing LLM call for file_id: {file_id} with task: {self.args.task}")
        
        if self.args.task == "extraction":
            prompt = ReferenceExtractionPrompt(prompt=str(prompt_path), input_text=input_text).prompt
            return llm_client.call(prompt, json_output=False)
        
        elif self.args.task == "extraction_and_parsing":
            include_schema = "pydantic" in self.args.prompt_name
            prompt_obj = ReferenceExtractionAndParsingPrompt(
                prompt=str(prompt_path),
                input_text=input_text,
                include_json_schema=include_schema
            )
            return llm_client.call(prompt_obj.prompt, json_output=True, json_schema=prompt_obj.json_schema)

        elif self.args.task == "parsing":
            include_schema = "pydantic" in self.args.prompt_name
            prompt_obj = ReferenceParsingPrompt(
                prompt=str(prompt_path),
                input_text=input_text,
                include_json_schema=include_schema
            )
            return llm_client.call(prompt_obj.prompt, json_output=True, json_schema=prompt_obj.json_schema)
        return None

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

            json_match = json.loads(llm_response_str)

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
        run_name = f"{self.args.task}_{model_name_slug}_{self.args.extractor}_{timestamp}"
        
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
        """Print a summary of the benchmark metrics."""
        if not self.metrics:
            logging.warning("No metrics were generated.")
            return

        metrics_df = pd.DataFrame(self.metrics)
        # Ensure file_id is str for merging
        metrics_df["file_id"] = metrics_df["file_id"].astype(str)
        if hasattr(self, "pdf_df"):
            self.pdf_df["file_id"] = self.pdf_df["file_id"].astype(str)
        tqdm.write("\n--- Benchmark Summary ---")
        if self.args.task == "extraction":
            summary = metrics_df[['precision', 'recall', 'f1_score', 'avg_similarity']].mean().to_dict()
        else: # extraction_and_parsing
            summary = metrics_df[['precision', 'recall', 'micro_f1', 'macro_f1']].mean().to_dict()
        
        tqdm.write(json.dumps(summary, indent=2))
        tqdm.write("-------------------------\n")

        # ----------------------------------------------------------
        # Error statistics
        # ----------------------------------------------------------
        tqdm.write("Error statistics:")
        tqdm.write(f"  Parsing errors:   {self.parse_errors}")
        tqdm.write(f"  Evaluation errors: {self.eval_errors}")

        # ----------------------------------------------------------
        # Optional per-class / language breakdown
        # ----------------------------------------------------------
        if self.args.per_class and hasattr(self, "pdf_df"):
            joined = metrics_df.merge(
                self.pdf_df[["file_id", "class", "lang"]], on="file_id", how="left"
            )

            metric_cols = (
                ["precision", "recall", "f1_score", "avg_similarity"]
                if self.args.task == "extraction"
                else ["precision", "recall", "micro_f1", "macro_f1"]
            )

            grouped = (
                joined.groupby(["class", "lang"], dropna=True)[metric_cols]
                .mean()
                .reset_index()
            )

            tqdm.write("Per-class / language metrics:")
            for _, row in grouped.iterrows():
                cls, lang = int(row["class"]), row["lang"]
                metrics_str = ", ".join(
                    f"{col}: {row[col]:.4f}" for col in metric_cols if col in row
                )
                tqdm.write(f"  class {cls}, lang {lang}: {metrics_str}")


def main():
    """Main function to parse arguments and run the benchmark."""
    parser = argparse.ArgumentParser(
        description="Run citation extraction and parsing benchmarks."
    )

    # Task Configuration
    parser.add_argument("--task", type=str, required=True, choices=["extraction", "extraction_and_parsing", "parsing"], help="The evaluation task to run.")

    # LLM Configuration
    parser.add_argument("--model_name", type=str, default="google/gemma-3-27b-it", help="Name of the LLM model to use.")
    parser.add_argument("--api_key", type=str, default=os.environ.get("DEEPSEEK_API_KEY"), help="API key for the LLM endpoint. Defaults to DEEPSEEK_API_KEY env var.")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1", help="Base URL for the LLM API endpoint.")

    # Prompt Configuration
    parser.add_argument("--prompt_name", type=str, default="reference_extraction.md", help="Name of the prompt file in the 'prompts/' directory.")

    # Extractor Configuration
    parser.add_argument("--extractor", type=str, default="pymupdf", choices=["pymupdf", "marker", "mineru"], help="The PDF text extractor to use.")

    # Evaluation Configuration
    parser.add_argument("--fuzzy_threshold", type=float, default=90, help="Fuzzy threshold for reference parsing. If ≤1, treated as proportion (e.g., 0.95 → 95).")
    parser.add_argument("--mode", type=str, default="exact", choices=["exact", "fuzzy", "soft_fuzzy"], help="Mode for reference parsing.")
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

    args = parser.parse_args()

    # Parse and normalize focus_fields -> List[str] or None
    if args.focus_fields:
        if isinstance(args.focus_fields, str):
            args.focus_fields = [f.strip() for f in args.focus_fields.split(",") if f.strip()]
        elif not isinstance(args.focus_fields, list):
            # Unexpected type; fallback to None
            args.focus_fields = None

    # Configure logging
    log_level = logging.DEBUG if args.verbose else tqdm.write
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not args.api_key:
        raise ValueError("API key must be provided via --api_key or DEEPSEEK_API_KEY environment variable.")

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

