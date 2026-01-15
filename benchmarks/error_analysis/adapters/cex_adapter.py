"""
CEX Benchmark Adapter

Handles CEX-specific file loading and metadata extraction.
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_adapter import BenchmarkAdapter

logger = logging.getLogger(__name__)


class CEXAdapter(BenchmarkAdapter):
    """Adapter for CEX benchmark."""
    
    def find_response_files(self, model_pattern: str, method: str = "m1") -> Dict[str, Path]:
        """Find all response files matching the model pattern and method."""
        outputs_dir = self.get_outputs_base_path()
        pattern = f"*_{method}_*{model_pattern}*_responses.pkl"
        files = list(outputs_dir.glob(pattern))
        
        logger.info(f"Found {len(files)} CEX response files matching pattern: {pattern}")
        
        # Organize by task type
        task_files = {}
        for file_path in files:
            filename = file_path.name
            
            if filename.startswith('extraction_and_parsing_'):
                task = 'extraction_and_parsing'
            elif filename.startswith('extraction_'):
                task = 'extraction'
            elif filename.startswith('parsing_'):
                task = 'parsing'
            else:
                logger.warning(f"Could not determine task for file: {filename}")
                continue
            
            # Keep only the most recent file for each task
            if task not in task_files or file_path.stat().st_mtime > task_files[task].stat().st_mtime:
                task_files[task] = file_path
        
        logger.info(f"Selected CEX files by task: {list(task_files.keys())}")
        return task_files
    
    def load_responses(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load responses from pickle file."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def extract_metadata(self, file_id: str) -> Dict[str, Any]:
        """
        Extract document category from CEX file_id.
        
        CEX file IDs have format: CATEGORY_NUMBER
        Example: "AGR-BIO-SCI_1" → category is "AGR-BIO-SCI"
        """
        # Split on last underscore to separate category from number
        parts = file_id.rsplit('_', 1)
        category = parts[0] if len(parts) > 1 else file_id
        
        return {
            'document_category': category
        }
    
    def get_benchmark_name(self) -> str:
        """Return benchmark name."""
        return 'cex'
    
    @property
    def supported_tasks(self) -> List[str]:
        """CEX supports all three task types."""
        return ['extraction', 'parsing', 'extraction_and_parsing']
    
    @property
    def has_document_categories(self) -> bool:
        """CEX has document categories."""
        return True
