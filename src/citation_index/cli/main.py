"""Main CLI entry point for citation index."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .. import ExtractorFactory, TeiBiblParser, __version__
from ..core.models import References
from ..pipelines.reference_parsing import parse_reference_file
from ..pipelines.reference_extraction_and_parsing import run_pdf_extract_and_parse
from ..llm.client import LLMClient


def extract_command(args):
    """Extract citations from a PDF file."""
    try:
        # Prepare extractor kwargs for GROBID
        extractor_kwargs = {}
        if args.extractor == 'grobid':
            if hasattr(args, 'grobid_endpoint') and args.grobid_endpoint:
                extractor_kwargs['grobid_endpoint'] = args.grobid_endpoint
            if hasattr(args, 'grobid_timeout') and args.grobid_timeout:
                extractor_kwargs['grobid_timeout'] = args.grobid_timeout
        
        # Create extractor
        extractor = ExtractorFactory.create(args.extractor, **extractor_kwargs)
        
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


def parse_xml_command(args):
    """Parse citations from a TEI/XML file and output JSON or XML."""
    try:
        refs = References.from_xml(file_path=args.input)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if args.format == 'xml':
                xml_output = refs.to_xml(pretty_print=True)
                output_path.write_text(xml_output, encoding='utf-8')
            else:
                import json
                output_path.write_text(
                    json.dumps([r.model_dump() for r in refs], indent=2),
                    encoding='utf-8',
                )
            print(f"Parsed references saved to: {output_path}")
        else:
            for r in refs:
                print(r)
    except Exception as e:
        print(f"Error parsing XML {args.input}: {e}", file=sys.stderr)
        sys.exit(1)


def parse_text_command(args):
    """Parse citations from a raw text/markdown file using LLM."""
    try:
        api_key = None
        if args.api_key_env:
            import os
            api_key = os.environ.get(args.api_key_env)
        client = LLMClient(endpoint=args.api_base, model=args.model, api_key=api_key)
        refs = parse_reference_file(
            args.input, llm_client=client, prompt_name=str(args.prompt), temperature=args.temperature
        )
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if args.format == 'xml':
                xml_output = refs.to_xml(pretty_print=True)
                output_path.write_text(xml_output, encoding='utf-8')
            else:
                import json
                output_path.write_text(
                    json.dumps([r.model_dump() for r in refs], indent=2),
                    encoding='utf-8',
                )
            print(f"Parsed references saved to: {output_path}")
        else:
            for r in refs:
                print(r)
    except Exception as e:
        print(f"Error parsing text {args.input}: {e}", file=sys.stderr)
        sys.exit(1)


def run_command(args):
    """End-to-end: PDF -> text -> references (LLM)."""
    try:
        # Handle GROBID-only extraction (no LLM needed)
        if args.extractor == 'grobid':
            extractor_kwargs = {}
            if hasattr(args, 'grobid_endpoint') and args.grobid_endpoint:
                extractor_kwargs['grobid_endpoint'] = args.grobid_endpoint
            if hasattr(args, 'grobid_timeout') and args.grobid_timeout:
                extractor_kwargs['grobid_timeout'] = args.grobid_timeout
            
            extractor = ExtractorFactory.create(args.extractor, **extractor_kwargs)
            
            # For GROBID, extract references directly
            refs = extractor.extract_references_only(args.input, save_dir=None)
            
            # Convert to References object
            from ..core.models import References
            refs_obj = References(references=refs)
        else:
            # Use LLM-based pipeline for other extractors
            # Validate required LLM parameters
            if not args.api_base:
                print("Error: --api-base is required for non-GROBID extractors", file=sys.stderr)
                sys.exit(1)
            if not args.model:
                print("Error: --model is required for non-GROBID extractors", file=sys.stderr)
                sys.exit(1)
            
            api_key = None
            if args.api_key_env:
                import os
                api_key = os.environ.get(args.api_key_env)
            client = LLMClient(endpoint=args.api_base, model=args.model, api_key=api_key)
            refs_obj = run_pdf_extract_and_parse(
                args.input,
                llm_client=client,
                extractor=args.extractor,
                temperature=args.temperature,
            )
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if args.format == 'xml':
                xml_output = refs_obj.to_xml(pretty_print=True)
                output_path.write_text(xml_output, encoding='utf-8')
            else:
                import json
                output_path.write_text(
                    json.dumps([r.model_dump() for r in refs_obj], indent=2),
                    encoding='utf-8',
                )
            print(f"Parsed references saved to: {output_path}")
        else:
            for r in refs_obj:
                print(r)
    except Exception as e:
        print(f"Error running pipeline for {args.input}: {e}", file=sys.stderr)
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
    
    # GROBID-specific options
    extract_parser.add_argument(
        "--grobid-endpoint", 
        type=str, 
        default="http://localhost:8070",
        help="GROBID server endpoint (only used with --extractor grobid)"
    )
    extract_parser.add_argument(
        "--grobid-timeout", 
        type=float, 
        default=180.0,
        help="GROBID request timeout in seconds (only used with --extractor grobid)"
    )
    
    extract_parser.set_defaults(func=extract_command)
    
    # Parse XML command
    parse_xml_parser = subparsers.add_parser("parse-xml", help="Parse citations from TEI/XML file")
    parse_xml_parser.add_argument("input", type=Path, help="Input XML file")
    parse_xml_parser.add_argument("--output", type=Path, help="Output file")
    parse_xml_parser.add_argument(
        "--format",
        choices=["xml", "json"],
        default="xml",
        help="Output format",
    )
    parse_xml_parser.set_defaults(func=parse_xml_command)

    # Parse text via LLM
    parse_text_parser = subparsers.add_parser("parse-text", help="Parse citations from text using LLM")
    parse_text_parser.add_argument("input", type=Path, help="Input text/markdown file")
    parse_text_parser.add_argument("--output", type=Path, help="Output file")
    parse_text_parser.add_argument("--prompt", type=Path, default=Path("prompts/reference_extraction_and_parsing_pydantic.md"))
    parse_text_parser.add_argument("--api-base", type=str, required=True)
    parse_text_parser.add_argument("--model", type=str, required=True)
    parse_text_parser.add_argument("--api-key-env", type=str, default=None, help="Env var that holds API key (optional)")
    parse_text_parser.add_argument("--temperature", type=float, default=0.0)
    parse_text_parser.add_argument(
        "--format",
        choices=["xml", "json"],
        default="json",
        help="Output format",
    )
    parse_text_parser.set_defaults(func=parse_text_command)

    # End-to-end run
    run_parser = subparsers.add_parser("run", help="Run end-to-end PDF → parsed references")
    run_parser.add_argument("input", type=Path, help="Input PDF file")
    run_parser.add_argument(
        "--extractor",
        choices=ExtractorFactory.get_available_extractors(),
        default="pymupdf",
        help="PDF extractor to use",
    )
    run_parser.add_argument("--output", type=Path, help="Output file")
    run_parser.add_argument("--api-base", type=str, help="LLM API endpoint (not required for GROBID extractor)")
    run_parser.add_argument("--model", type=str, help="LLM model name (not required for GROBID extractor)")
    run_parser.add_argument("--api-key-env", type=str, default=None, help="Environment variable containing API key")
    run_parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    run_parser.add_argument(
        "--format",
        choices=["xml", "json"],
        default="json",
        help="Output format",
    )
    
    # GROBID-specific options
    run_parser.add_argument(
        "--grobid-endpoint", 
        type=str, 
        default="http://localhost:8070",
        help="GROBID server endpoint (only used with --extractor grobid)"
    )
    run_parser.add_argument(
        "--grobid-timeout", 
        type=float, 
        default=180.0,
        help="GROBID request timeout in seconds (only used with --extractor grobid)"
    )
    
    run_parser.set_defaults(func=run_command)
    
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