"""
Main error analyzer using benchmark adapters.

This class orchestrates the error analysis process using a benchmark-specific
adapter to handle benchmark-specific logic while keeping the core analysis
logic generic and reusable.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from tqdm import tqdm

from .base_adapter import BenchmarkAdapter
from .failure_classifier import FailureClassifier

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """Main error analyzer using benchmark adapters."""
    
    def __init__(
        self,
        adapter: BenchmarkAdapter,
        model_pattern: str,
        output_name: str,
        method: str = "m1"
    ):
        """
        Initialize the error analyzer.
        
        Args:
            adapter: Benchmark-specific adapter
            model_pattern: Pattern to match in model filenames
            output_name: Name for output directory
            method: Method identifier (default: "m1")
        """
        self.adapter = adapter
        self.model_pattern = model_pattern
        self.output_name = output_name
        self.method = method
        self.classifier = FailureClassifier()
        
        # Setup output directory
        benchmark_name = adapter.get_benchmark_name()
        self.output_dir = Path("benchmarks/error_analysis/outputs") / benchmark_name / output_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.detailed_results = []
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        logger.info(
            f"Starting error analysis for {self.adapter.get_benchmark_name()} benchmark, "
            f"model pattern: {self.model_pattern}"
        )
        
        # Find and analyze all task files
        task_files = self.adapter.find_response_files(self.model_pattern, self.method)
        
        if not task_files:
            logger.error(f"No response files found for pattern: {self.model_pattern}")
            return
        
        for task, file_path in task_files.items():
            self.analyze_task(task, file_path)
        
        # Generate reports
        self.export_csv_reports()
        
        logger.info(f"Analysis complete. Results saved to {self.output_dir}")
    
    def analyze_task(self, task: str, file_path: Path):
        """Analyze all responses for a given task."""
        logger.info(f"Analyzing {task} task from {file_path.name}")
        
        responses = self.adapter.load_responses(file_path)
        
        for item in tqdm(responses, desc=f"Analyzing {task}"):
            file_id = str(item['file_id'])
            response = item['response']
            gt_references = item['gt_references']
            
            # Extract metadata using adapter
            metadata = self.adapter.extract_metadata(file_id)
            
            # Classify the response
            classification = self.classifier.classify_response(
                response, gt_references, task
            )
            
            # Store detailed result
            result = {
                'file_id': file_id,
                'task': task,
                **metadata,  # Add benchmark-specific metadata
                **classification
            }
            
            self.detailed_results.append(result)
    
    def export_csv_reports(self):
        """Export analysis results to CSV files."""
        df = pd.DataFrame(self.detailed_results)
        
        if df.empty:
            logger.warning("No results to export")
            return
        
        # 1. Detailed failures CSV
        detailed_path = self.output_dir / "detailed_failures.csv"
        df.to_csv(detailed_path, index=False)
        logger.info(f"Exported detailed failures to {detailed_path}")
        
        # 2. Summary by task
        summary_by_task = self._create_summary_by_task(df)
        task_path = self.output_dir / "summary_by_task.csv"
        summary_by_task.to_csv(task_path, index=False)
        logger.info(f"Exported task summary to {task_path}")
        
        # 3. Summary by category (if benchmark has categories)
        if self.adapter.has_document_categories and 'document_category' in df.columns:
            summary_by_category = self._create_summary_by_category(df)
            category_path = self.output_dir / "summary_by_category.csv"
            summary_by_category.to_csv(category_path, index=False)
            logger.info(f"Exported category summary to {category_path}")
        
        # 4. Overall summary
        summary_overall = self._create_overall_summary(df)
        overall_path = self.output_dir / "summary_overall.csv"
        summary_overall.to_csv(overall_path, index=False)
        logger.info(f"Exported overall summary to {overall_path}")
    
    def _create_summary_by_task(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics by task type."""
        summary_rows = []
        
        for task in df['task'].unique():
            task_df = df[df['task'] == task]
            
            total = len(task_df)
            structural_errors = task_df['has_structural_error'].sum()
            refusals = task_df['has_refusal'].sum()
            factual_errors = task_df['has_factual_error'].sum()
            successes = total - structural_errors - refusals - factual_errors
            
            # Factual error breakdown
            total_minor_errors = task_df['minor_error_count'].sum()
            total_major_errors = task_df['major_error_count'].sum()
            total_correct = task_df['correct_count'].sum()
            
            summary_rows.append({
                'task': task,
                'total_documents': total,
                'success_count': successes,
                'success_rate': successes / total if total > 0 else 0,
                'structural_error_count': structural_errors,
                'structural_error_rate': structural_errors / total if total > 0 else 0,
                'refusal_count': refusals,
                'refusal_rate': refusals / total if total > 0 else 0,
                'factual_error_count': factual_errors,
                'factual_error_rate': factual_errors / total if total > 0 else 0,
                'total_minor_errors': total_minor_errors,
                'total_major_errors': total_major_errors,
                'total_correct_refs': total_correct,
                'avg_gt_ref_count': task_df['gt_ref_count'].mean(),
                'avg_extracted_ref_count': task_df['extracted_ref_count'].mean(),
            })
        
        return pd.DataFrame(summary_rows)
    
    def _create_summary_by_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics by document category."""
        if 'document_category' not in df.columns:
            return pd.DataFrame()
        
        summary_rows = []
        
        for category in sorted(df['document_category'].unique()):
            cat_df = df[df['document_category'] == category]
            
            total = len(cat_df)
            structural_errors = cat_df['has_structural_error'].sum()
            refusals = cat_df['has_refusal'].sum()
            factual_errors = cat_df['has_factual_error'].sum()
            successes = total - structural_errors - refusals - factual_errors
            
            summary_rows.append({
                'document_category': category,
                'total_documents': total,
                'success_count': successes,
                'success_rate': successes / total if total > 0 else 0,
                'structural_error_count': structural_errors,
                'refusal_count': refusals,
                'factual_error_count': factual_errors,
                'avg_gt_ref_count': cat_df['gt_ref_count'].mean(),
            })
        
        return pd.DataFrame(summary_rows)
    
    def _create_overall_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create overall summary statistics."""
        total = len(df)
        structural_errors = df['has_structural_error'].sum()
        refusals = df['has_refusal'].sum()
        factual_errors = df['has_factual_error'].sum()
        successes = total - structural_errors - refusals - factual_errors
        
        summary = {
            'metric': [
                'total_documents',
                'success_count',
                'success_rate',
                'structural_error_count',
                'structural_error_rate',
                'refusal_count',
                'refusal_rate',
                'factual_error_count',
                'factual_error_rate',
                'total_minor_errors',
                'total_major_errors',
                'total_correct_refs',
                'avg_gt_ref_count',
                'avg_extracted_ref_count',
            ],
            'value': [
                total,
                successes,
                successes / total if total > 0 else 0,
                structural_errors,
                structural_errors / total if total > 0 else 0,
                refusals,
                refusals / total if total > 0 else 0,
                factual_errors,
                factual_errors / total if total > 0 else 0,
                df['minor_error_count'].sum(),
                df['major_error_count'].sum(),
                df['correct_count'].sum(),
                df['gt_ref_count'].mean(),
                df['extracted_ref_count'].mean(),
            ]
        }
        
        return pd.DataFrame(summary)
