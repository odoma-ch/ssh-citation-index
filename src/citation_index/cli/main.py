"""Main CLI entry point for citation index."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .. import ExtractorFactory, TeiBiblParser, __version__


def extract_command(args):
    """Extract citations from a PDF file."""
    try:
        # Create extractor
        extractor = ExtractorFactory.create(args.extractor)
        
        # Extract text from PDF
        result = extractor.extract(args.input, save_dir=args.output_dir)
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.text)
            print(f"Extracted text saved to: {output_path}")
        else:
            print(result.text)
            
    except Exception as e:
        print(f"Error extracting from {args.input}: {e}", file=sys.stderr)
        sys.exit(1)


def parse_command(args):
    """Parse citations from text or XML."""
    try:
        parser = TeiBiblParser()
        
        if args.input.suffix.lower() == '.xml':
            # Parse XML file
            references_lists = parser.from_xml(file_path=args.input)
            references = [ref for refs in references_lists for ref in refs]
        else:
            # TODO: Implement text parsing with LLM
            print("Text parsing not yet implemented", file=sys.stderr)
            sys.exit(1)
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if args.format == 'xml':
                from ..core.models import References
                refs = References(references)
                xml_output = refs.to_xml(pretty_print=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(xml_output)
            elif args.format == 'json':
                import json
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump([ref.model_dump() for ref in references], f, indent=2)
            
            print(f"Parsed references saved to: {output_path}")
        else:
            for ref in references:
                print(ref)
                
    except Exception as e:
        print(f"Error parsing {args.input}: {e}", file=sys.stderr)
        sys.exit(1)


def benchmark_command(args):
    """Run benchmarking evaluation."""
    print("Benchmarking functionality not yet implemented", file=sys.stderr)
    sys.exit(1)


def main(argv: Optional[list] = None):
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="citation-index",
        description="Citation extraction and parsing system for academic documents"
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"citation-index {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract text from PDF")
    extract_parser.add_argument("input", type=Path, help="Input PDF file")
    extract_parser.add_argument(
        "--extractor", 
        choices=ExtractorFactory.get_available_extractors(),
        default="pymupdf",
        help="PDF extractor to use"
    )
    extract_parser.add_argument("--output", type=Path, help="Output text file")
    extract_parser.add_argument("--output-dir", type=Path, help="Output directory for intermediate files")
    extract_parser.set_defaults(func=extract_command)
    
    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse citations from text or XML")
    parse_parser.add_argument("input", type=Path, help="Input file")
    parse_parser.add_argument("--output", type=Path, help="Output file")
    parse_parser.add_argument(
        "--format", 
        choices=["xml", "json"],
        default="xml",
        help="Output format"
    )
    parse_parser.set_defaults(func=parse_command)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run evaluation benchmarks")
    benchmark_parser.add_argument("--dataset", type=Path, help="Dataset directory")
    benchmark_parser.add_argument("--output", type=Path, help="Results output file")
    benchmark_parser.set_defaults(func=benchmark_command)
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main() 