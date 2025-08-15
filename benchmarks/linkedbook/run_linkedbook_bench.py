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
from citation_index.core.models import References
from citation_index.evaluation.ref_metrics import evaluate_linkedbook_fields_and_authors


LINKEDBOOK_TEST_PATH = Path("benchmarks/linkedbook/linkedbooks_test_references.jsonl")


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
        if self.args.limit:
            raw_items = raw_items[: self.args.limit]
            references = references[: self.args.limit]
        tqdm.write(f"Loaded {len(references)} reference strings from LinkedBook test set.")

        llm_client = LLMClient(
            endpoint=self.args.api_base,
            model=self.args.model_name,
            api_key=self.args.api_key,
            timeout = 180,
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
        
        # Evaluate fields and authors
        metrics = evaluate_linkedbook_fields_and_authors(
            pred_batches=pred_batches,
            gt_batches=gt_batches,
            focus_fields=["full_title", "publication_place", "publication_date"],
            author_threshold=self.args.author_threshold,
            mode=self.args.eval_mode,
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

    def _load_test_references(self) -> (List[Dict[str, Any]], List[str]):
        if not LINKEDBOOK_TEST_PATH.exists():
            raise FileNotFoundError(
                f"LinkedBook test file not found at {LINKEDBOOK_TEST_PATH}. Make sure it exists."
            )
        raw_items: List[Dict[str, Any]] = []
        refs: List[str] = []
        total_loaded = 0
        excluded_count = 0
        
        with LINKEDBOOK_TEST_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    total_loaded += 1
                    
                    # Check if we should exclude non-reference entries
                    if getattr(self.args, "exclude_non_ref", False):
                        # Check if this record has a title in the tags field
                        tags = obj.get("tags", {})
                        has_title = bool(tags.get("title"))
                        if not has_title:
                            excluded_count += 1
                            continue
                    
                    raw_items.append(obj)
                    ref = obj.get("reference")
                    if isinstance(ref, str) and ref.strip():
                        refs.append(ref.strip())
                except json.JSONDecodeError:
                    continue
        
        if getattr(self.args, "exclude_non_ref", False) and excluded_count > 0:
            tqdm.write(f"Excluded {excluded_count} non-reference entries (no title in GT tags). "
                      f"Processing {len(refs)} references out of {total_loaded} total records.")
        
        return raw_items, refs

    def _load_results(self, results_path: str) -> List[Dict[str, Any]]:
        """Load results from JSON file and apply filtering if needed."""
        path = Path(results_path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        with path.open("r", encoding="utf-8") as f:
            results_data = json.load(f)
        
        # Apply the same filtering as when loading fresh data
        if getattr(self.args, "exclude_non_ref", False):
            original_count = len(results_data)
            filtered_results = []
            excluded_count = 0
            
            for result in results_data:
                gt = result.get("ground_truth", {})
                tags = gt.get("tags", {})
                has_title = bool(tags.get("title"))
                if has_title:
                    filtered_results.append(result)
                else:
                    excluded_count += 1
            
            if excluded_count > 0:
                tqdm.write(f"Excluded {excluded_count} non-reference entries from loaded results. "
                          f"Processing {len(filtered_results)} references out of {original_count} total records.")
            
            return filtered_results
        
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
        rng = random.Random(self.args.seed)
        tasks = []
        i = 0
        task_id = 0
        n = len(references)
        min_g, max_g = self.args.min_group, self.args.max_group
        while i < n:
            group_size = rng.randint(min_g, max_g)
            indices = list(range(i, min(i + group_size, n)))
            lines = [references[j] for j in indices]
            tasks.append({
                "task_id": f"group_{task_id}",
                "mode": "grouped",
                "indices": indices,
                "references": lines,
            })
            task_id += 1
            i += group_size
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

    def _convert_llm_responses_to_results(self, llm_responses: List[Dict[str, Any]], raw_items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
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
        gt_batches: List[List[Dict[str, Any]]] = []
        
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
                
                # Extract ground truth for this group
                gt_batch = [r["ground_truth"] for r in group]
                gt_batches.append(gt_batch)
        else:
            # For single mode, each result is its own batch
            for result in results_data:
                parsed_result = result["parsed_result"]
                refs_obj = References.from_dict([parsed_result] if parsed_result else [])
                pred_batches.append(refs_obj)
                
                gt_batch = [result["ground_truth"]]
                gt_batches.append(gt_batch)
        
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
                    gt_items = [r["ground_truth"] for r in group]
                    refs_obj = References.from_dict(parsed_refs)
                    filtered_pred_batches.append(refs_obj)
                    filtered_gt_batches.append(gt_items)
            else:
                # For single mode, each result is its own batch
                for result in lang_data:
                    parsed_result = result["parsed_result"]
                    refs_obj = References.from_dict([parsed_result] if parsed_result else [])
                    filtered_pred_batches.append(refs_obj)
                    filtered_gt_batches.append([result["ground_truth"]])
            
            # Evaluate this language subset
            lang_metrics = evaluate_linkedbook_fields_and_authors(
                pred_batches=filtered_pred_batches,
                gt_batches=filtered_gt_batches,
                focus_fields=["full_title", "publication_place", "publication_date"],
                author_threshold=self.args.author_threshold,
                mode=self.args.eval_mode,
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
                "author_threshold": self.args.author_threshold,
                "eval_mode": self.args.eval_mode,
                "total_references": len(results_data),
                "llm_duration_sec": round(self.llm_duration, 2) if self.llm_duration else None,
            },
            "overall_metrics": metrics,
            "per_language_metrics": per_lang_metrics,
            "detailed_results": []
        }
        
        # Add detailed results for each reference
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
            
            # Simple field-by-field comparison for this individual reference
            field_scores = {}
            
            # Field scoring
            gt_title = tags.get("title", "").strip()
            gt_place = tags.get("publicationplace", "").strip() 
            gt_year = tags.get("year", "") or tags.get("publicationnumber-year", "")
            
            parsed_title = parsed.get("full_title", "")
            parsed_place = parsed.get("publication_place", "")
            parsed_year = parsed.get("publication_date", "")
            
            # Simple exact match scoring for individual reference
            field_scores["title_exact_match"] = bool(gt_title and parsed_title and gt_title.lower().strip() == str(parsed_title).lower().strip())
            field_scores["place_exact_match"] = bool(gt_place and parsed_place and gt_place.lower().strip() == str(parsed_place).lower().strip())
            field_scores["year_exact_match"] = bool(gt_year and parsed_year and str(gt_year).strip() == str(parsed_year).strip())
            
            # Author scoring
            gt_author_text = tags.get("author", "")
            parsed_authors = parsed.get("authors", [])
            
            # Simple author comparison
            field_scores["has_gt_author"] = bool(gt_author_text.strip())
            field_scores["has_parsed_authors"] = bool(parsed_authors)
            field_scores["author_count_match"] = False
            field_scores["author_similarity"] = 0.0
            
            if gt_author_text.strip() and parsed_authors:
                # Use the same canonicalization logic from evaluation
                from citation_index.evaluation.ref_metrics import _split_gt_authors, _canonicalize_author_token, _person_to_canonical
                
                gt_authors = [_canonicalize_author_token(a) for a in _split_gt_authors(gt_author_text)]
                parsed_author_tokens = []
                
                for author in parsed_authors:
                    if isinstance(author, dict):
                        # Handle case where authors might be dictionaries
                        canonical = _person_to_canonical(author) if hasattr(author, 'model_fields') else str(author)
                    else:
                        canonical = str(author)
                    if canonical:
                        parsed_author_tokens.append(_canonicalize_author_token(canonical))
                
                field_scores["author_count_match"] = len(gt_authors) == len(parsed_author_tokens)
                
                # Simple author similarity (exact matches)
                if gt_authors and parsed_author_tokens:
                    matches = sum(1 for gt_a in gt_authors if any(gt_a == p_a for p_a in parsed_author_tokens))
                    field_scores["author_similarity"] = matches / max(len(gt_authors), len(parsed_author_tokens))
            
            # Overall field match score
            field_matches = sum([
                field_scores["title_exact_match"],
                field_scores["place_exact_match"], 
                field_scores["year_exact_match"]
            ])
            field_scores["field_match_score"] = field_matches / 3
            
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
        
        field = metrics.get('field_metrics', {}) or {}
        author = metrics.get('author_metrics', {}) or {}
        f_p = float(field.get('precision', 0.0))
        f_r = float(field.get('recall', 0.0))
        f_micro = float(field.get('micro_f1', 0.0))
        f_macro = float(field.get('macro_f1', 0.0))
        # Determine number of evaluated fields for weighting
        per_field_map = (field.get('per_class_f1') or {})
        per_field_precision = (field.get('per_class_precision') or {})
        per_field_recall = (field.get('per_class_recall') or {})
        n_fields = len(per_field_map) if per_field_map else 3
        a_p = float(author.get('precision', 0.0))
        a_r = float(author.get('recall', 0.0))
        a_micro = float(author.get('micro_f1', 0.0))
        a_macro = float(author.get('macro_f1', 0.0))
        # Treat authors as one additional field among the focus fields
        c_p = ((n_fields * f_p) + a_p) / (n_fields + 1) if (f_p or a_p) else 0.0
        c_r = ((n_fields * f_r) + a_r) / (n_fields + 1) if (f_r or a_r) else 0.0
        c_micro = (2 * c_p * c_r / (c_p + c_r)) if (c_p + c_r) else 0.0
        c_macro = ((n_fields * f_macro) + a_macro) / (n_fields + 1) if (f_macro or a_macro) else 0.0

        # Overall metrics (including authors)
        tqdm.write(f"Overall: P={c_p:.4f}, R={c_r:.4f}, micro-F1={c_micro:.4f}, macro-F1={c_macro:.4f}")
        
        # Fields-only metrics (excluding authors)
        tqdm.write(f"Fields-only: P={f_p:.4f}, R={f_r:.4f}, micro-F1={f_micro:.4f}, macro-F1={f_macro:.4f}")
        
        # Per-field breakdown
        tqdm.write("Fields:")
        if per_field_map:
            for field_name in sorted(per_field_map.keys()):
                field_p = float(per_field_precision.get(field_name, 0.0))
                field_r = float(per_field_recall.get(field_name, 0.0))
                field_f1 = float(per_field_map.get(field_name, 0.0))
                tqdm.write(f"  {field_name}: P={field_p:.4f}, R={field_r:.4f}, micro-F1={field_f1:.4f}")
        else:
            tqdm.write(f"  overall: P={f_p:.4f}, R={f_r:.4f}, micro-F1={f_micro:.4f}")
        
        # Authors metrics
        tqdm.write(f"Authors: P={a_p:.4f}, R={a_r:.4f}, micro-F1={a_micro:.4f}, macro-F1={a_macro:.4f}")
        
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
                fm = (m.get('field_metrics') or {})
                am = (m.get('author_metrics') or {})
                pe = (m.get('parsing_errors') or {})
                lp = float(fm.get('precision', 0.0)); ap = float(am.get('precision', 0.0))
                lr = float(fm.get('recall', 0.0));    ar = float(am.get('recall', 0.0))
                pf_map = (fm.get('per_class_f1') or {})
                nf = len(pf_map) if pf_map else 3
                lcp = ((nf * lp) + ap) / (nf + 1) if (lp or ap) else 0.0
                lcr = ((nf * lr) + ar) / (nf + 1) if (lr or ar) else 0.0
                lcf1 = (2 * lcp * lcr / (lcp + lcr)) if (lcp + lcr) else 0.0
                lang_errors = pe.get('total_errors', 0)
                lang_tasks = pe.get('total_tasks', 0)
                lang_error_rate = pe.get('error_rate', 0.0)
                tqdm.write(f"  {lang}: P={lcp:.4f}, R={lcr:.4f}, F1={lcf1:.4f}, Errors={lang_errors}/{lang_tasks}({lang_error_rate}%)")
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
                    f.write(f"Overall: P={c_p:.4f}, R={c_r:.4f}, micro-F1={c_micro:.4f}, macro-F1={c_macro:.4f}\n")
                    f.write(f"Fields-only: P={f_p:.4f}, R={f_r:.4f}, micro-F1={f_micro:.4f}, macro-F1={f_macro:.4f}\n")
                    f.write("Fields:\n")
                    if per_field_map:
                        for field_name in sorted(per_field_map.keys()):
                            field_p = float(per_field_precision.get(field_name, 0.0))
                            field_r = float(per_field_recall.get(field_name, 0.0))
                            field_f1 = float(per_field_map.get(field_name, 0.0))
                            f.write(f"  {field_name}: P={field_p:.4f}, R={field_r:.4f}, micro-F1={field_f1:.4f}\n")
                    else:
                        f.write(f"  overall: P={f_p:.4f}, R={f_r:.4f}, micro-F1={f_micro:.4f}\n")
                    f.write(f"Authors: P={a_p:.4f}, R={a_r:.4f}, micro-F1={a_micro:.4f}, macro-F1={a_macro:.4f}\n")
                    if per_lang_metrics:
                        f.write("Per-language:\n")
                        for lang, m in sorted(per_lang_metrics.items()):
                            fm = (m.get('field_metrics') or {})
                            am = (m.get('author_metrics') or {})
                            lp = float(fm.get('precision', 0.0)); ap = float(am.get('precision', 0.0))
                            lr = float(fm.get('recall', 0.0));    ar = float(am.get('recall', 0.0))
                            pf_map = (fm.get('per_class_f1') or {})
                            nf = len(pf_map) if pf_map else 3
                            lcp = ((nf * lp) + ap) / (nf + 1) if (lp or ap) else 0.0
                            lcr = ((nf * lr) + ar) / (nf + 1) if (lr or ar) else 0.0
                            lcf1 = (2 * lcp * lcr / (lcp + lcr)) if (lcp + lcr) else 0.0
                            f.write(f"  {lang}: P={lcp:.4f}, R={lcr:.4f}, F1={lcf1:.4f}\n")
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
    parser.add_argument("--prompt_name", type=str, default="reference_parsing_zeroshot.md", 
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

    # Grouping configuration (for grouped mode)
    parser.add_argument("--min_group", type=int, default=10, 
                       help="Minimum group size (grouped mode)")
    parser.add_argument("--max_group", type=int, default=50, 
                       help="Maximum group size (grouped mode)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for grouping")
    
    # Evaluation configuration
    parser.add_argument("--author_threshold", type=float, default=85.0, 
                       help="Fuzzy ratio threshold (0-100) for author matching")
    parser.add_argument("--eval_mode", type=str, default="soft_fuzzy", 
                       choices=["exact", "fuzzy", "soft_fuzzy"], help="Field evaluation mode")
    parser.add_argument("--per_category", action="store_true",
                       help="If set, display metrics per language (category).")
    parser.add_argument("--exclude-non-ref", action="store_true", dest="exclude_non_ref",
                       help="Exclude records that don't have a title in ground truth tags (non-reference entries).")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not args.api_key:
        raise ValueError("API key must be provided via --api_key or DEEPSEEK_API_KEY environment variable.")

    # Basic validation
    if args.mode == "grouped" and args.min_group > args.max_group:
        raise ValueError("--min_group cannot be greater than --max_group")

    runner = LinkedbookBenchmarkRunner(args)
    with logging_redirect_tqdm():
        runner.run()

    if runner.llm_duration:
        print(f"LLM call execution time: {runner.llm_duration:.2f} seconds")
    else:
        print("LLM calls were skipped (responses loaded from file).")


if __name__ == "__main__":
    main()


