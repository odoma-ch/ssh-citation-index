"""
EXCITE Benchmark Adapter

Handles EXCITE-specific file loading and metadata extraction.
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


class EXCITEAdapter(BenchmarkAdapter):
    """Adapter for EXCITE benchmark."""
    
    def find_response_files(self, model_pattern: str, method: str = "m1") -> Dict[str, Path]:
        """Find all response files matching the model pattern and method."""
        outputs_dir = self.get_outputs_base_path()
        pattern = f"*_{method}_*{model_pattern}*_responses.pkl"
        files = list(outputs_dir.glob(pattern))
        
        logger.info(f"Found {len(files)} EXCITE response files matching pattern: {pattern}")
        
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
        
        logger.info(f"Selected EXCITE files by task: {list(task_files.keys())}")
        return task_files
    
    def load_responses(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load responses from pickle file."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def extract_metadata(self, file_id: str) -> Dict[str, Any]:
        """
        Extract metadata from EXCITE file_id.
        
        EXCITE file IDs are numeric (e.g., "44404").
        We would need to load the EXCITE dataset info to get language and class.
        For now, we'll just use a simple category based on numeric ID.
        """
        # Simplified approach: just use the file_id as category
        # In a more complete implementation, we could load the EXCITE metadata
        # from pdf_files_info.csv to get language (de/en) and class (1/2/3)
        
        return {
            'document_category': f"doc_{file_id}"
        }
    
    def get_benchmark_name(self) -> str:
        """Return benchmark name."""
        return 'excite'
    
    @property
    def supported_tasks(self) -> List[str]:
        """EXCITE supports all three task types."""
        return ['extraction', 'parsing', 'extraction_and_parsing']
    
    @property
    def has_document_categories(self) -> bool:
        """EXCITE has document categories (language + class)."""
        return True
