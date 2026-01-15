"""
LinkedBook Benchmark Adapter

Handles LinkedBook-specific file loading and metadata extraction.
LinkedBook uses JSON format and is parsing-only.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_adapter import BenchmarkAdapter

logger = logging.getLogger(__name__)


class LinkedBookAdapter(BenchmarkAdapter):
    """Adapter for LinkedBook benchmark."""
    
    def find_response_files(self, model_pattern: str, method: str = "m1") -> Dict[str, Path]:
        """
        Find all response files matching the model pattern.
        
        LinkedBook files are named: linkedbook_grouped_<model>_<timestamp>_results.json
        Method parameter is ignored since LinkedBook doesn't use method variants.
        """
        outputs_dir = self.get_outputs_base_path()
        pattern = f"linkedbook_grouped_*{model_pattern}*_results.json"
        files = list(outputs_dir.glob(pattern))
        
        logger.info(f"Found {len(files)} LinkedBook response files matching pattern: {pattern}")
        
        # LinkedBook only has parsing task
        task_files = {}
        if files:
            # Use the most recent file
            most_recent = max(files, key=lambda p: p.stat().st_mtime)
            task_files['parsing'] = most_recent
            logger.info(f"Selected LinkedBook file: {most_recent.name}")
        
        return task_files
    
    def load_responses(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load responses from JSON file and normalize to standard format.
        
        LinkedBook format:
        {
            'reference_string': str,
            'llm_response': str,
            'parsed_result': dict,
            'ground_truth': dict,
            'task_id': str,
            ...
        }
        
        Normalized to:
        {
            'file_id': task_id,
            'response': llm_response,
            'gt_references': [reference_string],
        }
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Normalize to standard format
        normalized = []
        for item in data:
            normalized.append({
                'file_id': item.get('task_id', 'unknown'),
                'response': item.get('llm_response', ''),
                # LinkedBook has single reference per item
                'gt_references': [item.get('reference_string', '')],
                # Keep original data for reference
                '_original': item
            })
        
        return normalized
    
    def extract_metadata(self, file_id: str) -> Dict[str, Any]:
        """
        Extract metadata from LinkedBook file_id.
        
        LinkedBook file IDs are group IDs (e.g., "group_22").
        Since LinkedBook doesn't have document categories,
        we return a generic category.
        """
        return {
            'document_category': 'general'
        }
    
    def get_benchmark_name(self) -> str:
        """Return benchmark name."""
        return 'linkedbook'
    
    @property
    def supported_tasks(self) -> List[str]:
        """LinkedBook only supports parsing."""
        return ['parsing']
    
    @property
    def has_document_categories(self) -> bool:
        """LinkedBook doesn't have meaningful document categories."""
        return False
