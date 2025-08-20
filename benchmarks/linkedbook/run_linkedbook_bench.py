"""
LinkedBook Reference Parsing Benchmark

This script runs reference parsing benchmarks on the LinkedBook test dataset.
It produces a clean JSON output format containing:

- reference_string: Original reference text to parse
- llm_response: Raw response from the LLM 
- parsed_result: Structured parsing result
- ground_truth: Expected ground truth data

The script supports two modes:
- single: Parse each reference individually 
- grouped: Parse references in random groups (10-50 refs per group)

Results can be saved and reloaded to re-evaluate metrics without re-running LLM calls.
"""

import argparse
import os
import json
import random
import logging
import datetime
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from citation_index.llm.client import LLMClient
from citation_index.pipelines.reference_parsing import parse_reference_strings
from citation_index.core.models import References, Reference
from citation_index.evaluation.ref_metrics import RefEvaluator


LINKEDBOOK_TEST_PATH = Path("benchmarks/linkedbook/linkedbooks_test_references.jsonl")
LINKEDBOOK_GROUPED_PATH = Path("benchmarks/linkedbook/linkedbooks_test_grouped_references.jsonl")


class LinkedbookBenchmarkRunner:
    """Run reference parsing benchmarks on the LinkedBook test set.

    This class supports two parsing modes:
    - single: Parse each reference string individually (one LLM call per reference)
    - grouped: Randomly group references (10-50 by default) and parse per group

    The benchmark produces a simple JSON output format with:
    - reference_string: The original reference text
    - llm_response: Raw LLM response  
    - parsed_result: Structured parsing result
    - ground_truth: Expected ground truth data

    Results can be saved and reloaded to re-evaluate metrics without re-running LLM calls.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(args.output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []
        self.llm_duration: float = 0.0
        tqdm.write(f"Results will be saved to: {self.output_dir}")

    def run(self):
        """Main execution method for the benchmark."""
        logging.debug("Starting LinkedBook benchmark run with the following arguments:")
        logging.debug(json.dumps(vars(self.args), indent=2))

        # Load test references (raw items + strings)
        raw_items, references = self._load_test_references()
        if self.args.limit and self.args.mode == "single":
            raw_items = raw_items[: self.args.limit]
            references = references[: self.args.limit]
        
        if self.args.mode == "single":
            tqdm.write(f"Loaded {len(references)} reference strings from LinkedBook test set.")
        else:
            tqdm.write(f"Using pre-generated groups with {len(references)} total references.")

        llm_client = LLMClient(
            endpoint=self.args.api_base,
            model=self.args.model_name,
            api_key=self.args.api_key,
            timeout = 300,  
            first_token_timeout = 60,
            max_retries = 3,
        )

        # Optionally load previously saved results
        if self.args.results_path:
            results_data = self._load_results(self.args.results_path)
            # Count parsing errors from loaded results
            parsing_errors = sum(1 for r in results_data if r.get("parsing_error", False))
            tqdm.write(
                f"Loaded {len(results_data)} results from {self.args.results_path}. Skipping LLM calls."
            )
            if parsing_errors > 0:
                tqdm.write(f"Found {parsing_errors} parsing errors in loaded results.")
            
            # Set run name for detailed analysis based on loaded file
            loaded_path = Path(self.args.results_path)
            if loaded_path.stem.endswith("_results"):
                self._run_name = loaded_path.stem[:-8]  # Remove "_results" suffix
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name_slug = self.args.model_name.replace('/', '_').replace('-', '_')
                self._run_name = f"linkedbook_{self.args.mode}_{model_name_slug}_{timestamp}_reeval"
        else:
            # Prepare tasks according to mode
            if self.args.mode == "single":
                tasks = self._prepare_single_tasks(references)
            elif self.args.mode == "grouped":
                tasks = self._prepare_grouped_tasks(references)
            else:
                raise ValueError("Unsupported mode. Use 'single' or 'grouped'.")

            # Execute LLM calls concurrently
            tqdm.write(f"Submitting {len(tasks)} tasks to LLM with {self.args.max_workers} workers (mode: {self.args.mode}).")
            start_llm_timer = time.time()
            llm_responses = []
            with ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
                future_to_task = {
                    executor.submit(self._execute_llm_call, llm_client, task): task for task in tasks
                }
                for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Executing LLM calls"):
                    task = future_to_task[future]
                    try:
                        resp_str = future.result()
                        llm_responses.append({
                            "task": task,
                            "response": resp_str,
                        })
                    except Exception as exc:
                        logging.error(f"Task failed for task_id={task.get('task_id')}: {exc}", exc_info=True)

            self.llm_duration = time.time() - start_llm_timer
            
            # Convert LLM responses to the final results format
            results_data, parsing_errors = self._convert_llm_responses_to_results(llm_responses, raw_items)

        # Save results and prepare for evaluation
        pred_batches, gt_batches = self._save_results_and_prepare_evaluation(results_data)
        
        # Evaluate fields and authors using RefEvaluator
        evaluator = RefEvaluator(
            mode=self.args.eval_mode,
            fuzzy_threshold=self.args.fuzzy_threshold if hasattr(self.args, 'fuzzy_threshold') else 80
        )
        
        # Evaluate with focus fields if specified
        metrics = evaluator.evaluate(
            predictions=pred_batches,
            labels=gt_batches,
            focus_fields=getattr(self.args, 'focus_fields', None)
        )
        
        # Add parsing error statistics
        total_tasks = len(results_data) if self.args.mode == "single" else len(set(r.get("task_id") for r in results_data))
        metrics["parsing_errors"] = {
            "total_errors": parsing_errors,
            "total_tasks": total_tasks,
            "error_rate": round(parsing_errors / total_tasks * 100, 2) if total_tasks > 0 else 0.0
        }
        per_lang_metrics = None
        if self.args.per_category:
            per_lang_metrics = self._compute_per_language_metrics(results_data)
        self._summarize(references, results_data, metrics, per_lang_metrics)
        
        # Save detailed analysis JSON
        self._save_detailed_analysis(results_data, metrics, per_lang_metrics)

    # ============================================================================
    # Data Loading Methods
    # ============================================================================

    def _load_test_references(self) -> tuple[List[Dict[str, Any]], List[str]]:
        """Load test references based on mode (single or grouped)."""
        if self.args.mode == "grouped":
            return self._load_grouped_references()
        else:
            return self._load_single_references()

    def _load_single_references(self) -> tuple[List[Dict[str, Any]], List[str]]:
        """Load individual references from the main test file."""
        if not LINKEDBOOK_TEST_PATH.exists():
            raise FileNotFoundError(
                f"LinkedBook test file not found at {LINKEDBOOK_TEST_PATH}. Make sure it exists."
            )
        raw_items: List[Dict[str, Any]] = []
        refs: List[str] = []
        
        with LINKEDBOOK_TEST_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    raw_items.append(obj)
                    ref = obj.get("reference")
                    if isinstance(ref, str) and ref.strip():
                        refs.append(ref.strip())
                except json.JSONDecodeError:
                    continue
        
        return raw_items, refs

    def _load_grouped_references(self) -> tuple[List[Dict[str, Any]], List[str]]:
        """Load pre-grouped references from the grouped test file."""
        if not LINKEDBOOK_GROUPED_PATH.exists():
            raise FileNotFoundError(
                f"LinkedBook grouped test file not found at {LINKEDBOOK_GROUPED_PATH}. Make sure it exists."
            )
        
        raw_items: List[Dict[str, Any]] = []
        refs: List[str] = []
        total_groups = 0
        
        with LINKEDBOOK_GROUPED_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    group = json.loads(line)
                    total_groups += 1
                    
                    group_refs = group.get("refs", [])
                    group_gt = group.get("ground_truth", [])
                    
                    # Add references and ground truth
                    refs.extend(group_refs)
                    raw_items.extend(group_gt)
                    
                except json.JSONDecodeError:
                    continue
        
        tqdm.write(f"Loaded {len(refs)} references from {total_groups} pre-generated groups.")
        return raw_items, refs

    def _load_results(self, results_path: str) -> List[Dict[str, Any]]:
        """Load results from JSON file."""
        path = Path(results_path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        with path.open("r", encoding="utf-8") as f:
            results_data = json.load(f)
        
        return results_data

    # ============================================================================
    # Task Preparation Methods
    # ============================================================================

    def _prepare_single_tasks(self, references: List[str]) -> List[Dict[str, Any]]:
        tasks = []
        for idx, ref in enumerate(references):
            tasks.append({
                "task_id": f"single_{idx}",
                "mode": "single",
                "indices": [idx],
                "references": [ref],
            })
        return tasks

    def _prepare_grouped_tasks(self, references: List[str]) -> List[Dict[str, Any]]:
        """Prepare tasks using pre-generated groups from the grouped file."""
        if not LINKEDBOOK_GROUPED_PATH.exists():
            raise FileNotFoundError(
                f"LinkedBook grouped test file not found at {LINKEDBOOK_GROUPED_PATH}. Make sure it exists."
            )
        
        tasks = []
        processed_refs = 0
        
        with LINKEDBOOK_GROUPED_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    group = json.loads(line)
                    group_id = group.get("id")
                    group_refs = group.get("refs", [])
                    group_gt = group.get("ground_truth", [])
                    
                    indices = list(range(processed_refs, processed_refs + len(group_refs)))
                    
                    # Apply limit if specified
                    if self.args.limit and processed_refs >= self.args.limit:
                        break
                    
                    if self.args.limit:
                        # Trim group if it would exceed limit
                        remaining = self.args.limit - processed_refs
                        if remaining < len(group_refs):
                            group_refs = group_refs[:remaining]
                            indices = indices[:remaining]
                    
                    if group_refs:  # Only add non-empty groups
                        tasks.append({
                            "task_id": f"group_{group_id}",
                            "mode": "grouped",
                            "indices": indices,
                            "references": group_refs,
                            "original_group_id": group_id
                        })
                    
                    processed_refs += len(group_refs)
                    
                    if self.args.limit and processed_refs >= self.args.limit:
                        break
                        
                except json.JSONDecodeError:
                    continue
        
        return tasks

    # ============================================================================
    # LLM Execution Methods  
    # ============================================================================

    def _execute_llm_call(self, llm_client: LLMClient, task: Dict[str, Any]) -> str:
        prompt_path = Path("prompts") / self.args.prompt_name
        include_schema = "pydantic" in self.args.prompt_name
        refs_model: References = parse_reference_strings(
            reference_lines=task["references"],
            llm_client=llm_client,
            prompt_name=str(prompt_path),
            include_schema=include_schema,
        )
        return json.dumps(refs_model.model_dump())

    def _convert_llm_responses_to_results(self, llm_responses: List[Dict[str, Any]], raw_items: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
        """Convert LLM responses to final results format with reference string, LLM response, parsed result, and ground truth."""
        results_data = []
        parsing_errors = 0
        
        for item in llm_responses:
            task = item.get("task", {})
            response_str = item.get("response", "")
            indices = task.get("indices", [])
            task_id = task.get("task_id", "unknown")
            
            # Parse the LLM response and track parsing errors
            parsing_error = False
            try:
                parsed_response = json.loads(response_str) if isinstance(response_str, str) and response_str else {}
                if not response_str.strip():
                    parsing_error = True
                    parsing_errors += 1
                    tqdm.write(f"Warning: Empty response for task {task_id}")
            except json.JSONDecodeError as e:
                parsed_response = {}
                parsing_error = True
                parsing_errors += 1
                tqdm.write(f"Warning: JSON parsing error for task {task_id}: {str(e)}")
            
            # Extract parsed references and check for structural issues
            parsed_references = []
            if isinstance(parsed_response, dict) and "references" in parsed_response:
                parsed_references = parsed_response["references"]
                if not isinstance(parsed_references, list):
                    parsing_error = True
                    parsing_errors += 1
                    tqdm.write(f"Warning: Task {task_id} has non-list 'references' field: {type(parsed_references)}")
                    parsed_references = []
            elif isinstance(parsed_response, list):
                parsed_references = parsed_response
            elif parsed_response and not parsing_error:
                # Non-empty response but wrong structure
                parsing_error = True
                parsing_errors += 1
                tqdm.write(f"Warning: Task {task_id} has unexpected response structure: {type(parsed_response)}")
                
            # Create results for each reference in this task
            task_references = task.get("references", [])
            
            # Validate alignment between expected and parsed references
            if len(parsed_references) != len(task_references):
                tqdm.write(f"Warning: Task {task_id} expected {len(task_references)} refs but got {len(parsed_references)} parsed refs")
            
            for i, ref_idx in enumerate(indices):
                if i < len(task_references) and ref_idx < len(raw_items):
                    reference_string = task_references[i]
                    ground_truth = raw_items[ref_idx]
                    
                    # Get corresponding parsed result - handle mismatched counts more robustly
                    if i < len(parsed_references):
                        parsed_result = parsed_references[i]
                    else:
                        # If LLM returned fewer references than expected, use empty dict
                        parsed_result = {}
                        tqdm.write(f"Warning: Missing parsed result for reference {i} in task {task_id}")
                    
                    results_data.append({
                        "reference_string": reference_string,
                        "llm_response": response_str,
                        "parsed_result": parsed_result,
                        "ground_truth": ground_truth,
                        "task_id": task_id,
                        "parsing_error": parsing_error
                    })
                else:
                    tqdm.write(f"Warning: Index mismatch in task {task_id}: ref_idx={ref_idx}, i={i}, len(raw_items)={len(raw_items)}, len(task_references)={len(task_references)}")
        
        # Log parsing error summary
        if parsing_errors > 0:
            tqdm.write(f"Total parsing errors: {parsing_errors} out of {len(llm_responses)} tasks ({parsing_errors/len(llm_responses)*100:.1f}%)")
        
        return results_data, parsing_errors

    # ============================================================================
    # Results Processing Methods
    # ============================================================================

    def _save_results_and_prepare_evaluation(self, results_data: List[Dict[str, Any]]):
        """Save results to JSON file and prepare data for evaluation."""
        if not getattr(self.args, "skip_save", False):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name_slug = self.args.model_name.replace('/', '_').replace('-', '_')
            run_name = f"linkedbook_{self.args.mode}_{model_name_slug}_{timestamp}"
            results_path = self.output_dir / f"{run_name}_results.json"
            
            with results_path.open("w", encoding="utf-8") as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)
            tqdm.write(f"Saved results to {results_path}")
            
            # Store paths for detailed analysis later
            self._results_path = results_path
            self._run_name = run_name
        else:
            # If skipping save, still set run_name for potential detailed analysis
            if not hasattr(self, '_run_name'):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name_slug = self.args.model_name.replace('/', '_').replace('-', '_')
                self._run_name = f"linkedbook_{self.args.mode}_{model_name_slug}_{timestamp}"
        
        # Prepare batches for evaluation
        pred_batches: List[References] = []
        gt_batches: List[References] = []
        
        # Group results by task (for grouped mode)
        if self.args.mode == "grouped":
            # For grouped mode, reconstruct original tasks using task_id
            task_groups = {}
            for result in results_data:
                task_id = result.get("task_id", "unknown")
                if task_id not in task_groups:
                    task_groups[task_id] = []
                task_groups[task_id].append(result)
            
            # Sort task groups by task_id to ensure consistent ordering
            sorted_task_ids = sorted(task_groups.keys())
            for task_id in sorted_task_ids:
                group = task_groups[task_id]
                # Extract parsed results for this group
                parsed_refs = [r["parsed_result"] for r in group]
                refs_obj = References.from_dict(parsed_refs)
                pred_batches.append(refs_obj)
                
                # Extract ground truth for this group and convert to References
                # Handle both formats: full LinkedBook format with 'tags' and flattened format
                gt_batch = []
                for r in group:
                    gt_item = r["ground_truth"]
                    if "tags" in gt_item:
                        # Full LinkedBook format
                        gt_batch.append(gt_item)
                    else:
                        # Flattened format - wrap in tags structure
                        wrapped_gt = {"tags": gt_item, "language": "", "dataset": "test"}
                        gt_batch.append(wrapped_gt)
                
                gt_refs_obj = References.from_linkedbook(gt_batch)
                gt_batches.append(gt_refs_obj)
        else:
            # For single mode, each result is its own batch
            for result in results_data:
                parsed_result = result["parsed_result"]
                refs_obj = References.from_dict([parsed_result] if parsed_result else [])
                pred_batches.append(refs_obj)
                
                # Convert ground truth to References
                # Handle both formats: full LinkedBook format with 'tags' and flattened format
                gt_item = result["ground_truth"]
                if "tags" in gt_item:
                    # Full LinkedBook format
                    gt_refs_obj = References.from_linkedbook([gt_item])
                else:
                    # Flattened format - wrap in tags structure
                    wrapped_gt = {"tags": gt_item, "language": "", "dataset": "test"}
                    gt_refs_obj = References.from_linkedbook([wrapped_gt])
                gt_batches.append(gt_refs_obj)
        
        return pred_batches, gt_batches

    def _compute_per_language_metrics(self, results_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Efficiently compute per-language metrics by filtering results_data."""
        # Group results by language
        lang_results = {}
        for result in results_data:
            gt = result.get("ground_truth", {})
            lang = (gt.get('language') or '').strip()
            if lang:
                if lang not in lang_results:
                    lang_results[lang] = []
                lang_results[lang].append(result)
        
        # Compute metrics for each language
        per_lang_metrics = {}
        for lang, lang_data in lang_results.items():
            # Convert filtered results back to the format expected by evaluation
            filtered_pred_batches = []
            filtered_gt_batches = []
            
            # Group by original batch structure if needed, or treat each as individual batch
            if self.args.mode == "grouped":
                # For grouped mode, reconstruct batches using task_id
                batch_groups = {}
                for result in lang_data:
                    task_id = result.get("task_id", "unknown")
                    if task_id not in batch_groups:
                        batch_groups[task_id] = []
                    batch_groups[task_id].append(result)
                
                # Sort by task_id for consistent ordering
                sorted_task_ids = sorted(batch_groups.keys())
                for task_id in sorted_task_ids:
                    group = batch_groups[task_id]
                    parsed_refs = [r["parsed_result"] for r in group]
                    # Handle both formats for ground truth
                    gt_items = []
                    for r in group:
                        gt_item = r["ground_truth"]
                        if "tags" in gt_item:
                            gt_items.append(gt_item)
                        else:
                            wrapped_gt = {"tags": gt_item, "language": "", "dataset": "test"}
                            gt_items.append(wrapped_gt)
                    
                    refs_obj = References.from_dict(parsed_refs)
                    gt_refs_obj = References.from_linkedbook(gt_items)
                    filtered_pred_batches.append(refs_obj)
                    filtered_gt_batches.append(gt_refs_obj)
            else:
                # For single mode, each result is its own batch
                for result in lang_data:
                    parsed_result = result["parsed_result"]
                    refs_obj = References.from_dict([parsed_result] if parsed_result else [])
                    
                    # Handle both formats for ground truth
                    gt_item = result["ground_truth"]
                    if "tags" in gt_item:
                        gt_refs_obj = References.from_linkedbook([gt_item])
                    else:
                        wrapped_gt = {"tags": gt_item, "language": "", "dataset": "test"}
                        gt_refs_obj = References.from_linkedbook([wrapped_gt])
                    
                    filtered_pred_batches.append(refs_obj)
                    filtered_gt_batches.append(gt_refs_obj)
            
            # Evaluate this language subset using RefEvaluator
            lang_evaluator = RefEvaluator(
                mode=self.args.eval_mode,
                fuzzy_threshold=self.args.fuzzy_threshold if hasattr(self.args, 'fuzzy_threshold') else 80
            )
            lang_metrics = lang_evaluator.evaluate(
                predictions=filtered_pred_batches,
                labels=filtered_gt_batches,
                focus_fields=getattr(self.args, 'focus_fields', None)
            )
            
            # Add parsing error statistics for this language
            lang_parsing_errors = sum(1 for r in lang_data if r.get("parsing_error", False))
            lang_total_tasks = len(lang_data) if self.args.mode == "single" else len(set(r.get("task_id") for r in lang_data))
            lang_metrics["parsing_errors"] = {
                "total_errors": lang_parsing_errors,
                "total_tasks": lang_total_tasks,
                "error_rate": round(lang_parsing_errors / lang_total_tasks * 100, 2) if lang_total_tasks > 0 else 0.0
            }
            
            per_lang_metrics[lang] = lang_metrics
        
        return per_lang_metrics

    # ============================================================================
    # Summary and Reporting Methods
    # ============================================================================

    def _save_detailed_analysis(self, results_data: List[Dict[str, Any]], metrics: Dict[str, Any], per_lang_metrics: Dict[str, Any] | None = None):
        """Save detailed analysis JSON with GT, parsed results, LLM responses, and scores."""
        if getattr(self.args, "skip_save", False) or not getattr(self.args, "detailed_analysis", False):
            return
            
        if not hasattr(self, '_run_name'):
            # Fallback if no run name was set
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name_slug = self.args.model_name.replace('/', '_').replace('-', '_')
            self._run_name = f"linkedbook_{self.args.mode}_{model_name_slug}_{timestamp}"
        
        analysis_path = self.output_dir / f"{self._run_name}_detailed_analysis.json"
        
        # Create detailed analysis structure
        analysis = {
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "mode": self.args.mode,
                "model_name": self.args.model_name,
                "prompt_name": self.args.prompt_name,
                "fuzzy_threshold": self.args.fuzzy_threshold,
                "eval_mode": self.args.eval_mode,
                "focus_fields": getattr(self.args, 'focus_fields', None),
                "total_references": len(results_data),
                "llm_duration_sec": round(self.llm_duration, 2) if self.llm_duration else None,
            },
            "overall_metrics": metrics,
            "per_language_metrics": per_lang_metrics,
            "detailed_results": []
        }
        
        # Group results by task_id for better organization
        if self.args.mode == "grouped":
            # Group by task_id to show all references from the same group together
            task_groups = {}
            for i, result in enumerate(results_data):
                task_id = result.get("task_id", "unknown")
                if task_id not in task_groups:
                    task_groups[task_id] = {
                        "task_id": task_id,
                        "references": [],
                        "llm_response": result.get("llm_response", ""),  # Same for all refs in group
                        "parsing_error": result.get("parsing_error", False),  # Same for all refs in group
                    }
                
                # Add this reference to the group
                task_groups[task_id]["references"].append({
                    "index": i,
                    "reference_string": result.get("reference_string", ""),
                    "ground_truth": result.get("ground_truth", {}),
                    "parsed_result": result.get("parsed_result", {}),
                    "language": result.get("ground_truth", {}).get("language", ""),
                })
            
            # Convert to list and add individual scores for each reference in the group
            for task_id in sorted(task_groups.keys()):
                group_data = task_groups[task_id]
                
                # Add individual scores for each reference in this group
                for ref_data in group_data["references"]:
                    gt = ref_data["ground_truth"]
                    tags = gt.get("tags", {})
                    parsed = ref_data["parsed_result"]
                    
                    # Create a mini-evaluator to use the same logic as the main evaluation
                    from citation_index.evaluation.ref_metrics import RefEvaluator
                    mini_evaluator = RefEvaluator(
                        mode=self.args.eval_mode,
                        fuzzy_threshold=self.args.fuzzy_threshold
                    )
                    
                    # Simple field-by-field comparison for this individual reference
                    field_scores = {}
                    
                    # Field scoring
                    gt_title = tags.get("title", "").strip()
                    gt_place = tags.get("publicationplace", "").strip() 
                    gt_year = tags.get("year", "") or tags.get("publicationnumber-year", "")
                    
                    parsed_title = parsed.get("full_title", "")
                    parsed_place = parsed.get("publication_place", "")
                    parsed_year = parsed.get("publication_date", "")
                    
                    # Use the same evaluation logic as the main RefEvaluator
                    field_scores["title_match_score"] = float(mini_evaluator._is_match(gt_title, parsed_title)) if gt_title and parsed_title else 0.0
                    field_scores["place_match_score"] = float(mini_evaluator._is_match(gt_place, parsed_place)) if gt_place and parsed_place else 0.0
                    field_scores["year_match_score"] = float(mini_evaluator._is_match(gt_year, parsed_year)) if gt_year and parsed_year else 0.0
                    
                    # Author scoring using RefEvaluator's field matching logic
                    gt_author_text = tags.get("author", "")
                    parsed_authors_raw = parsed.get("authors", [])
                    
                    # Convert raw author dictionaries to Person objects
                    parsed_authors = []
                    if parsed_authors_raw:
                        from citation_index.core.models.person import Person
                        for author_data in parsed_authors_raw:
                            if isinstance(author_data, dict):
                                try:
                                    person = Person(**author_data)
                                    parsed_authors.append(person)
                                except Exception:
                                    # Fallback to string representation if Person creation fails
                                    parsed_authors.append(str(author_data))
                            else:
                                parsed_authors.append(author_data)
                    
                    # Simple author comparison metadata
                    field_scores["has_gt_author"] = bool(gt_author_text.strip())
                    field_scores["has_parsed_authors"] = bool(parsed_authors)
                    
                    # Convert ground truth authors
                    gt_authors = Reference._split_gt_authors(gt_author_text) if gt_author_text else []
                    
                    # Use RefEvaluator's _field_match method for proper author comparison
                    matches, n_pred, n_label = mini_evaluator._field_match(parsed_authors, gt_authors)
                    
                    # Calculate author-specific metrics
                    field_scores["author_count_match"] = n_pred == n_label
                    if n_label > 0:
                        field_scores["author_similarity"] = float(matches) / float(n_label)
                    else:
                        field_scores["author_similarity"] = 0.0
                    
                    field_scores["author_match_count"] = int(matches) if mini_evaluator.mode != 'soft_fuzzy' else round(float(matches), 2)
                    
                    # Overall field match score including author similarity
                    field_scores["field_match_score_continuous"] = (
                        field_scores["title_match_score"] + 
                        field_scores["place_match_score"] + 
                        field_scores["year_match_score"] +
                        field_scores["author_similarity"]
                    ) / 4
                    
                    ref_data["individual_scores"] = field_scores
                
                analysis["detailed_results"].append(group_data)
        else:
            # For single mode, keep the original behavior
            for i, result in enumerate(results_data):
                detailed_result = {
                    "index": i,
                    "task_id": result.get("task_id", "unknown"),
                    "reference_string": result.get("reference_string", ""),
                    "ground_truth": result.get("ground_truth", {}),
                    "llm_response": result.get("llm_response", ""),
                    "parsed_result": result.get("parsed_result", {}),
                    "parsing_error": result.get("parsing_error", False),
                    "language": result.get("ground_truth", {}).get("language", ""),
                }
                
                # Add individual scoring if possible
                gt = result.get("ground_truth", {})
                tags = gt.get("tags", {})
                parsed = result.get("parsed_result", {})
                
                # Create a mini-evaluator to use the same logic as the main evaluation
                from citation_index.evaluation.ref_metrics import RefEvaluator
                mini_evaluator = RefEvaluator(
                    mode=self.args.eval_mode,
                    fuzzy_threshold=self.args.fuzzy_threshold
                )
                
                # Simple field-by-field comparison for this individual reference
                field_scores = {}
                
                # Field scoring
                gt_title = tags.get("title", "").strip()
                gt_place = tags.get("publicationplace", "").strip() 
                gt_year = tags.get("year", "") or tags.get("publicationnumber-year", "")
                
                parsed_title = parsed.get("full_title", "")
                parsed_place = parsed.get("publication_place", "")
                parsed_year = parsed.get("publication_date", "")
                
                # Use the same evaluation logic as the main RefEvaluator
                field_scores["title_match_score"] = float(mini_evaluator._is_match(gt_title, parsed_title)) if gt_title and parsed_title else 0.0
                field_scores["place_match_score"] = float(mini_evaluator._is_match(gt_place, parsed_place)) if gt_place and parsed_place else 0.0
                field_scores["year_match_score"] = float(mini_evaluator._is_match(gt_year, parsed_year)) if gt_year and parsed_year else 0.0
                
                # Author scoring using RefEvaluator's field matching logic
                gt_author_text = tags.get("author", "")
                parsed_authors_raw = parsed.get("authors", [])
                
                # Convert raw author dictionaries to Person objects
                parsed_authors = []
                if parsed_authors_raw:
                    from citation_index.core.models.person import Person
                    for author_data in parsed_authors_raw:
                        if isinstance(author_data, dict):
                            try:
                                person = Person(**author_data)
                                parsed_authors.append(person)
                            except Exception:
                                # Fallback to string representation if Person creation fails
                                parsed_authors.append(str(author_data))
                        else:
                            parsed_authors.append(author_data)
                
                # Simple author comparison metadata
                field_scores["has_gt_author"] = bool(gt_author_text.strip())
                field_scores["has_parsed_authors"] = bool(parsed_authors)
                
                # Convert ground truth authors
                gt_authors = Reference._split_gt_authors(gt_author_text) if gt_author_text else []
                
                # Use RefEvaluator's _field_match method for proper author comparison
                matches, n_pred, n_label = mini_evaluator._field_match(parsed_authors, gt_authors)
                
                # Calculate author-specific metrics
                field_scores["author_count_match"] = n_pred == n_label
                if n_label > 0:
                    field_scores["author_similarity"] = float(matches) / float(n_label)
                else:
                    field_scores["author_similarity"] = 0.0
                
                field_scores["author_match_count"] = int(matches) if mini_evaluator.mode != 'soft_fuzzy' else round(float(matches), 2)
                
                # Overall field match score including author similarity
                field_scores["field_match_score_continuous"] = (
                    field_scores["title_match_score"] + 
                    field_scores["place_match_score"] + 
                    field_scores["year_match_score"] +
                    field_scores["author_similarity"]
                ) / 4
                
                detailed_result["individual_scores"] = field_scores
                analysis["detailed_results"].append(detailed_result)
        
        # Save the detailed analysis
        with analysis_path.open("w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        tqdm.write(f"Saved detailed analysis to {analysis_path}")

    def _summarize(self, references: List[str], results_data: List[Dict[str, Any]], metrics: Dict[str, Any], per_lang_metrics: Dict[str, Any] | None = None):
        total_inputs = len(references)
        total_tasks = len(results_data)
        lines = []
        lines.append("\n--- LinkedBook Parsing Benchmark Summary ---")
        lines.append(json.dumps({
            "mode": self.args.mode,
            "total_inputs": total_inputs,
            "total_tasks": total_tasks,
            "llm_duration_sec": round(self.llm_duration, 2) if self.llm_duration else None,
            "model": self.args.model_name,
            "prompt": self.args.prompt_name,
        }, indent=2))
        lines.append("--------------------------------------------\n")

        # Print structured summary
        tqdm.write("\n--- LinkedBook Parsing Benchmark Summary ---")
        tqdm.write(lines[1])
        
        # Use new unified metrics structure
        f_p = float(metrics.get('overall_precision', 0.0))
        f_r = float(metrics.get('overall_recall', 0.0))
        f_micro = float(metrics.get('overall_micro_f1', 0.0))
        f_macro = float(metrics.get('overall_macro_f1', 0.0))
        
        # Extract per-field metrics
        per_field_map = metrics.get('per_field_metrics', {}) or {}
        per_field_precision = {}
        per_field_recall = {}
        if per_field_map:
            for field_name, field_metrics in per_field_map.items():
                per_field_precision[field_name] = field_metrics.get('precision', 0.0)
                per_field_recall[field_name] = field_metrics.get('recall', 0.0)

        # Overall metrics
        tqdm.write(f"Overall: P={f_p:.4f}, R={f_r:.4f}, micro-F1={f_micro:.4f}, macro-F1={f_macro:.4f}")
        
        # Focused metrics (if focus_fields were used)
        if 'focused_precision' in metrics:
            f_focused_p = float(metrics.get('focused_precision', 0.0))
            f_focused_r = float(metrics.get('focused_recall', 0.0))
            f_focused_micro = float(metrics.get('focused_micro_f1', 0.0))
            f_focused_macro = float(metrics.get('focused_macro_f1', 0.0))
            focus_fields_str = ', '.join(getattr(self.args, 'focus_fields', []))
            tqdm.write(f"Focused ({focus_fields_str}): P={f_focused_p:.4f}, R={f_focused_r:.4f}, micro-F1={f_focused_micro:.4f}, macro-F1={f_focused_macro:.4f}")
        
        # Per-field breakdown
        if per_field_map:
            for field_name in sorted(per_field_map.keys()):
                field_p = float(per_field_precision.get(field_name, 0.0))
                field_r = float(per_field_recall.get(field_name, 0.0))
                field_f1 = float(per_field_map.get(field_name, {}).get('f1', 0.0))
                tqdm.write(f"  {field_name}: P={field_p:.4f}, R={field_r:.4f}, micro-F1={field_f1:.4f}")
        else:
            tqdm.write(f"  overall: P={f_p:.4f}, R={f_r:.4f}, micro-F1={f_micro:.4f}")
        
        # Parsing errors
        parsing_errors = metrics.get('parsing_errors', {})
        if parsing_errors:
            error_count = parsing_errors.get('total_errors', 0)
            total_tasks = parsing_errors.get('total_tasks', 0)
            error_rate = parsing_errors.get('error_rate', 0.0)
            tqdm.write(f"Parsing Errors: {error_count}/{total_tasks} tasks ({error_rate}%)")

        # Per-language metrics
        if per_lang_metrics:
            tqdm.write("Per-language:")
            for lang, m in sorted(per_lang_metrics.items()):
                # Use new unified metrics structure
                lp = float(m.get('overall_precision', 0.0))
                lr = float(m.get('overall_recall', 0.0))
                lf1 = float(m.get('overall_micro_f1', 0.0))
                pe = (m.get('parsing_errors') or {})
                
                lang_errors = pe.get('total_errors', 0)
                lang_tasks = pe.get('total_tasks', 0)
                lang_error_rate = pe.get('error_rate', 0.0)
                
                # Main language metrics
                tqdm.write(f"  {lang}: P={lp:.4f}, R={lr:.4f}, F1={lf1:.4f}, Errors={lang_errors}/{lang_tasks}({lang_error_rate}%)")
                
                # Show focused metrics if available
                if 'focused_precision' in m:
                    l_focused_p = float(m.get('focused_precision', 0.0))
                    l_focused_r = float(m.get('focused_recall', 0.0))
                    l_focused_f1 = float(m.get('focused_micro_f1', 0.0))
                    focus_fields_str = ', '.join(getattr(self.args, 'focus_fields', []))
                    tqdm.write(f"    Focused ({focus_fields_str}): P={l_focused_p:.4f}, R={l_focused_r:.4f}, F1={l_focused_f1:.4f}")
        tqdm.write("--------------------------------------------\n")

        # Optionally append to a scores file
        if self.args.save_scores:
            try:
                scores_path = Path(self.args.save_scores)
                scores_path.parent.mkdir(parents=True, exist_ok=True)
                header = [
                    f"\n{'='*80}",
                    f"LinkedBook Benchmark Run - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"{'='*80}",
                    f"Mode: {self.args.mode}",
                    f"Model: {self.args.model_name}",
                    f"Prompt: {self.args.prompt_name}",
                    f"Inputs: {total_inputs}",
                    f"Tasks: {total_tasks}",
                    f"LLM Duration: {round(self.llm_duration, 2) if self.llm_duration else 'N/A (loaded)'}s",
                    f"{'='*80}\n",
                ]
                mode = 'a' if scores_path.exists() else 'w'
                with scores_path.open(mode, encoding='utf-8') as f:
                    f.write('\n'.join(header + lines))
                    f.write("\n")
                    # Write structured output to file as well
                    f.write(f"Overall: P={f_p:.4f}, R={f_r:.4f}, micro-F1={f_micro:.4f}, macro-F1={f_macro:.4f}\n")
                    
                    # Write focused metrics if available
                    if 'focused_precision' in metrics:
                        f_focused_p = float(metrics.get('focused_precision', 0.0))
                        f_focused_r = float(metrics.get('focused_recall', 0.0))
                        f_focused_micro = float(metrics.get('focused_micro_f1', 0.0))
                        f_focused_macro = float(metrics.get('focused_macro_f1', 0.0))
                        focus_fields_str = ', '.join(getattr(self.args, 'focus_fields', []))
                        f.write(f"Focused ({focus_fields_str}): P={f_focused_p:.4f}, R={f_focused_r:.4f}, micro-F1={f_focused_micro:.4f}, macro-F1={f_focused_macro:.4f}\n")
                    
                    if per_field_map:
                        for field_name in sorted(per_field_map.keys()):
                            field_p = float(per_field_precision.get(field_name, 0.0))
                            field_r = float(per_field_recall.get(field_name, 0.0))
                            field_f1 = float(per_field_map.get(field_name, {}).get('f1', 0.0))
                            f.write(f"  {field_name}: P={field_p:.4f}, R={field_r:.4f}, micro-F1={field_f1:.4f}\n")
                    else:
                        f.write(f"  overall: P={f_p:.4f}, R={f_r:.4f}, micro-F1={f_micro:.4f}\n")
                    if per_lang_metrics:
                        f.write("Per-language:\n")
                        for lang, m in sorted(per_lang_metrics.items()):
                            # Use new unified metrics structure
                            lp = float(m.get('overall_precision', 0.0))
                            lr = float(m.get('overall_recall', 0.0))
                            lf1 = float(m.get('overall_micro_f1', 0.0))
                            
                            f.write(f"  {lang}: P={lp:.4f}, R={lr:.4f}, F1={lf1:.4f}\n")
                            
                            # Write focused metrics for this language if available
                            if 'focused_precision' in m:
                                l_focused_p = float(m.get('focused_precision', 0.0))
                                l_focused_r = float(m.get('focused_recall', 0.0))
                                l_focused_f1 = float(m.get('focused_micro_f1', 0.0))
                                focus_fields_str = ', '.join(getattr(self.args, 'focus_fields', []))
                                f.write(f"    Focused ({focus_fields_str}): P={l_focused_p:.4f}, R={l_focused_r:.4f}, F1={l_focused_f1:.4f}\n")
                tqdm.write(f"Benchmark summary appended to: {scores_path}")
            except Exception as e:
                logging.error(f"Failed to save summary to {self.args.save_scores}: {e}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the LinkedBook benchmark script."""
    parser = argparse.ArgumentParser(
        description="Run LinkedBook reference parsing benchmark on test set. "
                   "Produces JSON output with reference strings, LLM responses, parsed results, and ground truth."
    )

    # Core configuration
    parser.add_argument("--mode", type=str, choices=["single", "grouped"], default="single", 
                       help="Parsing mode: single line per call or grouped batches.")
    parser.add_argument("--prompt_name", type=str, default="reference_parsing.md", 
                       help="Name of the prompt file in the 'prompts/' directory.")

    # LLM configuration
    parser.add_argument("--model_name", type=str, default="google/gemma-3-27b-it", 
                       help="LLM model name")
    parser.add_argument("--api_key", type=str, default=os.environ.get("DEEPSEEK_API_KEY"), 
                       help="API key for the LLM endpoint. Defaults to DEEPSEEK_API_KEY env var.")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1", 
                       help="Base URL for the LLM API endpoint.")

    # Execution configuration
    parser.add_argument("--output_path", type=str, default="benchmarks/linkedbook/outputs", 
                       help="Directory to save outputs")
    parser.add_argument("--limit", type=int, default=None, 
                       help="Limit number of references for a quick run")
    parser.add_argument("--max_workers", type=int, default=25, 
                       help="Max concurrent LLM requests")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Enable verbose logging")
    parser.add_argument("--results_path", type=str, default=None, 
                       help="Load previously saved results JSON file to skip LLM generation and re-evaluate")
    parser.add_argument("--skip_save", action="store_true", 
                       help="Do not write results to disk")
    parser.add_argument("--detailed_analysis", action="store_true",
                       help="Save detailed analysis JSON with GT, parsed results, LLM responses, and individual scores")
    parser.add_argument("--save_scores", type=str, default=None, 
                       help="Path to append benchmark summary")

    
    # Evaluation configuration
    parser.add_argument("--fuzzy_threshold", type=float, default=85.0, 
                       help="Fuzzy ratio threshold (0-100) for author matching")
    parser.add_argument("--eval_mode", type=str, default="soft_fuzzy", 
                       choices=["exact", "fuzzy", "soft_fuzzy"], help="Field evaluation mode")
    parser.add_argument("--focus_fields", nargs="*", 
                       default=["full_title", "authors", "publication_date"],
                       help="List of fields to focus evaluation on. Available fields: full_title, publication_place, publication_date, publisher, volume, journal_title, pages")
    parser.add_argument("--per_category", action="store_true",
                       help="If set, display metrics per language (category).")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not args.api_key:
        raise ValueError("API key must be provided via --api_key or DEEPSEEK_API_KEY environment variable.")

    # Basic validation - removed grouping checks since using pre-generated groups

    runner = LinkedbookBenchmarkRunner(args)
    with logging_redirect_tqdm():
        runner.run()

    if runner.llm_duration:
        print(f"LLM call execution time: {runner.llm_duration:.2f} seconds")
    else:
        print("LLM calls were skipped (responses loaded from file).")


if __name__ == "__main__":
    main()


