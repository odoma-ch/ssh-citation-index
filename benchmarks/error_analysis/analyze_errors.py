"""
Multi-Benchmark Error Analysis Tool

Analyzes LLM responses from benchmark results to identify and categorize failures:
1. Structural Errors (unreadable results)
2. Factual Errors (readable but incorrect - minor vs major errors)
3. LLM Refusal

Supports multiple benchmarks: CEX, EXCITE, LinkedBook

Usage:
    # CEX benchmark
    python analyze_errors.py --benchmark cex --model-pattern "mistralai" --output-name "mistral_small"
    
    # EXCITE benchmark
    python analyze_errors.py --benchmark excite --model-pattern "Qwen" --output-name "qwen_32b"
    
    # LinkedBook benchmark
    python analyze_errors.py --benchmark linkedbook --model-pattern "Qwen" --output-name "qwen_8b"
"""

import argparse
import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from adapters.cex_adapter import CEXAdapter
from adapters.excite_adapter import EXCITEAdapter
from adapters.linkedbook_adapter import LinkedBookAdapter
from core.error_analyzer import ErrorAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_adapter(benchmark_name: str):
    """
    Get the appropriate adapter for the benchmark.
    
    Args:
        benchmark_name: Name of the benchmark (cex, excite, linkedbook)
    
    Returns:
        BenchmarkAdapter instance
    
    Raises:
        ValueError: If benchmark name is not supported
    """
    adapters = {
        'cex': CEXAdapter,
        'excite': EXCITEAdapter,
        'linkedbook': LinkedBookAdapter,
    }
    
    if benchmark_name.lower() not in adapters:
        raise ValueError(
            f"Unsupported benchmark: {benchmark_name}. "
            f"Supported benchmarks: {', '.join(adapters.keys())}"
        )
    
    return adapters[benchmark_name.lower()]()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LLM errors in benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze CEX benchmark with Mistral-Small
  python analyze_errors.py --benchmark cex --model-pattern "mistralai" --output-name "mistral_small"
  
  # Analyze EXCITE benchmark with Qwen
  python analyze_errors.py --benchmark excite --model-pattern "Qwen" --output-name "qwen_32b"
  
  # Analyze LinkedBook benchmark
  python analyze_errors.py --benchmark linkedbook --model-pattern "deepseek" --output-name "deepseek"
  
  # Use specific method (for CEX/EXCITE only)
  python analyze_errors.py --benchmark cex --model-pattern "mistralai" --output-name "mistral_m2" --method m2
        """
    )
    
    parser.add_argument(
        '--benchmark',
        type=str,
        required=True,
        choices=['cex', 'excite', 'linkedbook'],
        help='Benchmark name (cex, excite, or linkedbook)'
    )
    parser.add_argument(
        '--model-pattern',
        type=str,
        required=True,
        help='Pattern to match model name in filenames (e.g., "mistralai", "Qwen", "deepseek")'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        required=True,
        help='Name for output directory (e.g., "mistral_small", "qwen_32b")'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='m1',
        help='Method identifier (default: m1). Only applies to CEX and EXCITE benchmarks.'
    )
    
    args = parser.parse_args()
    
    try:
        # Get appropriate adapter
        adapter = get_adapter(args.benchmark)
        
        logger.info(f"Using {args.benchmark.upper()} benchmark adapter")
        logger.info(f"Model pattern: {args.model_pattern}")
        logger.info(f"Method: {args.method}")
        logger.info(f"Output name: {args.output_name}")
        
        # Create and run analyzer
        analyzer = ErrorAnalyzer(
            adapter=adapter,
            model_pattern=args.model_pattern,
            output_name=args.output_name,
            method=args.method
        )
        
        analyzer.run_analysis()
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
