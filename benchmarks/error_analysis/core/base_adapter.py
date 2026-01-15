"""
Abstract base class for benchmark-specific adapters.

Each benchmark (CEX, EXCITE, LinkedBook) implements this interface
to provide benchmark-specific logic while maintaining a common API.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any


class BenchmarkAdapter(ABC):
    """Abstract base class for benchmark-specific adapters."""
    
    @abstractmethod
    def find_response_files(self, model_pattern: str, method: str = "m1") -> Dict[str, Path]:
        """
        Find response files for the given model pattern and method.
        
        Args:
            model_pattern: Pattern to match in model filenames (e.g., "mistralai", "Qwen")
            method: Method identifier (e.g., "m1", "m2", "m3"). Default is "m1".
        
        Returns:
            Dictionary mapping task names to file paths
            Example: {'extraction': Path(...), 'parsing': Path(...)}
        """
        pass
    
    @abstractmethod
    def load_responses(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load and normalize responses from a file.
        
        The returned list should have a consistent format across all benchmarks:
        [
            {
                'file_id': str,           # Document/task identifier
                'response': str,          # LLM response string
                'gt_references': List[str],  # Ground truth references
                ... # Additional benchmark-specific fields
            },
            ...
        ]
        
        Args:
            file_path: Path to the response file
        
        Returns:
            List of response dictionaries in normalized format
        """
        pass
    
    @abstractmethod
    def extract_metadata(self, file_id: str) -> Dict[str, Any]:
        """
        Extract metadata from file_id.
        
        This can include document category, language, class, or any other
        benchmark-specific categorization that can be derived from the file_id.
        
        Args:
            file_id: Document/task identifier
        
        Returns:
            Dictionary with metadata fields
            Example: {'document_category': 'AGR-BIO-SCI', 'language': 'en'}
        """
        pass
    
    @abstractmethod
    def get_benchmark_name(self) -> str:
        """
        Return the benchmark name for output paths.
        
        Returns:
            Benchmark name (e.g., 'cex', 'excite', 'linkedbook')
        """
        pass
    
    @property
    @abstractmethod
    def supported_tasks(self) -> List[str]:
        """
        Return list of supported task types for this benchmark.
        
        Returns:
            List of task names (e.g., ['extraction', 'parsing', 'extraction_and_parsing'])
        """
        pass
    
    @property
    def has_document_categories(self) -> bool:
        """
        Whether this benchmark has document categories for grouping.
        
        Can be overridden by subclasses. Default is True.
        
        Returns:
            True if benchmark has categories, False otherwise
        """
        return True
    
    def get_outputs_base_path(self) -> Path:
        """
        Get the base path for benchmark outputs.
        
        Can be overridden by subclasses for custom paths.
        
        Returns:
            Path to benchmark outputs directory
        """
        return Path("benchmarks") / self.get_benchmark_name() / "outputs"
