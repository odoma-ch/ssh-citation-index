"""Heuristic utilities to locate reference sections in extracted text."""

from typing import Iterable, List, Tuple, Optional, Dict
import re
from markdown_it import MarkdownIt

# ---------- Utilities ----------
DEFAULT_SECTION_ALIASES = [
    "references", "reference",
    "bibliography",
    "works cited", "literature cited",
    "citations", "cited references",
]

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower().rstrip(":"))

def _match_target(title: str, targets: Iterable[str]) -> bool:
    nt = _norm(title)
    for t in targets:
        if nt == _norm(t):
            return True
        # soft fuzzy: allow partial match if title starts with the target or vice versa
        if nt.startswith(_norm(t)) or _norm(t).startswith(nt):
            return True
    return False

# ---------- Token-based section iterator (preferred path) ----------
def iter_sections_markdownit(md_text: str) -> Iterable[Dict]:
    """
    Yield sections as dicts: {title, level, start_line, end_line, text}
    Uses markdown-it-py tokens + line maps to slice original text.
    """
    md = MarkdownIt()
    tokens = md.parse(md_text)
    lines = md_text.splitlines()

    headings: List[Tuple[int, str, int, int]] = []  # (level, title, token_index, start_line)
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open" and tok.tag and tok.tag.startswith("h"):
            level = int(tok.tag[1])
            # The inline token right after heading_open holds the title and map
            inline = tokens[i+1] if i + 1 < len(tokens) and tokens[i+1].type == "inline" else None
            title = inline.content.strip() if inline else ""
            # Prefer the inline token's map for the line number; fallback to heading_open.map
            if inline and inline.map:
                start_line = inline.map[0]
            elif tok.map:
                start_line = tok.map[0]
            else:
                # conservative fallback: derive by counting lines up to this point
                start_line = 0
            headings.append((level, title, i, start_line))

    # If no headings, nothing to iterate
    if not headings:
        return

    # Determine section spans by next heading start
    for idx, (level, title, token_idx, start_line) in enumerate(headings):
        next_start = headings[idx + 1][3] if idx + 1 < len(headings) else len(lines)
        # Slice text (inclusive of the heading line)
        section_text = "\n".join(lines[start_line:next_start])
        yield {
            "title": title,
            "level": level,
            "start_line": start_line,
            "end_line": next_start,
            "text": section_text,
        }

# ---------- Heuristic fallback when headings are missing ----------
_HEADING_LIKE = re.compile(
    r"""
    ^                           # start of line
    (?:
        \#{1,6}\s+.+            # ATX: #, ##, ...
      | [A-Z][A-Z0-9\s\-,:/&]+$ # ALLCAPS-ish line
      | .+:\s*$                 # Ends with colon
      | \*\s*[A-Za-z][A-Za-z0-9\s\-,:/&]*\s*\*\s*$  # heading with * around it
      | \*\*\s*[A-Za-z][A-Za-z0-9\s\-,:/&]*\s*\*\*\s*$  # heading with ** around it (bold)
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

def _looks_like_heading(line: str) -> bool:
    if not line.strip():
        return False
    # Avoid counting list items or reference entries as headings
    if re.match(r"^\s*([\*\-\+]|(\d+|\[\d+\])\.)\s+", line):
        return False
    return bool(_HEADING_LIKE.match(line.strip()))

def _clean_heading_text(line: str) -> str:
    line = re.sub(r"^\s*\#{1,6}\s*", "", line)  # strip leading # marks
    line = re.sub(r"^\s*\*\*\s*", "", line)  # strip leading double asterisks
    line = re.sub(r"\s*\*\*\s*$", "", line)  # strip trailing double asterisks
    line = re.sub(r"^\s*\*\s*", "", line)  # strip leading asterisk
    line = re.sub(r"\s*\*\s*$", "", line)  # strip trailing asterisk
    return line.strip().rstrip(":").strip()

def iter_sections_heuristic(md_text: str) -> Iterable[Dict]:
    """
    Heuristic section splitter for dirty PDF-derived Markdown with missing '#'.
    Treat heading-like lines as section starts.
    """
    lines = md_text.splitlines()
    indices = []
    for i, ln in enumerate(lines):
        if _looks_like_heading(ln):
            indices.append(i)
    if not indices:
        return
    indices.append(len(lines))
    for i in range(len(indices)-1):
        start = indices[i]
        end = indices[i+1]
        title = _clean_heading_text(lines[start])
        section_text = "\n".join(lines[start:end])
        # Heuristic level: try to infer from number of leading #'s; otherwise default to 2
        m = re.match(r"^\s*(\#{1,6})\s+", lines[start])
        level = len(m.group(1)) if m else 2
        yield {
            "title": title,
            "level": level,
            "start_line": start,
            "end_line": end,
            "text": section_text,
        }

# ---------- Core API ----------
def extract_all_sections(
    md_text: str,
    prefer_tokens: bool = True,
) -> List[Dict]:
    """
    Extract ALL sections from the text (general purpose).
    
    Returns:
        List of section dicts: {title, level, start_line, end_line, text, method}
    """
    # First try token-based
    if prefer_tokens:
        try:
            sections = list(iter_sections_markdownit(md_text))
            if sections:
                for sec in sections:
                    sec["method"] = "markdown-it tokens"
                return sections
        except Exception:
            # fall through to heuristic
            pass

    # Heuristic fallback
    sections = list(iter_sections_heuristic(md_text))
    for sec in sections:
        sec["method"] = "heuristic"
    return sections


def extract_all_reference_sections(
    md_text: str,
    target_names: Iterable[str] = DEFAULT_SECTION_ALIASES,
    prefer_tokens: bool = True,
) -> List[Dict]:
    """
    Find ALL reference sections in the text by first extracting all sections,
    then filtering for reference-like titles.
    
    Returns:
        List of section dicts: {title, level, start_line, end_line, text, method}
    """
    all_sections = extract_all_sections(md_text, prefer_tokens)
    reference_sections = []
    
    for sec in all_sections:
        if _match_target(sec["title"], target_names):
            reference_sections.append(sec)

    return reference_sections


# def find_references_section(
#     md_text: str,
#     target_names: Iterable[str] = DEFAULT_SECTION_ALIASES,
#     prefer_tokens: bool = True,
# ) -> Tuple[str, Tuple[int, int]]:
#     """
#     Find the FIRST reference section in the text.
    
#     Returns:
#         Tuple of (section_text, (start_position, end_position))
#     """
#     sections = extract_all_reference_sections(md_text, target_names, prefer_tokens)
#     if not sections:
#         return "", (0, 0)
    
#     # Return first reference section
#     section = sections[0]
#     lines = md_text.splitlines()
#     start_line = section.get("start_line", 0)
#     end_line = section.get("end_line", len(lines))
    
#     # Convert line numbers to character positions
#     start_pos = sum(len(line) + 1 for line in lines[:start_line])  # +1 for newline
#     end_pos = sum(len(line) + 1 for line in lines[:end_line])
    
#     # Adjust for the last line (no trailing newline)
#     if end_line >= len(lines):
#         end_pos = len(md_text)
    
#     section_text = section.get("text", "")
#     return section_text, (start_pos, end_pos)


if __name__ == "__main__":
    # Test with the actual benchmark file
    benchmark_file = "benchmarks/cex/all_markdown/PHY-AST_96_pymupdf.md"
    
    print(f"=== TESTING ON BENCHMARK FILE: {benchmark_file} ===")
    
    try:
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"File length: {len(content)} characters")
        
        # Test general section detection with both methods
        print("\n=== ALL SECTIONS (Token-based) ===")
        all_sections_tokens = extract_all_sections(content, prefer_tokens=True)
        print(f"Found {len(all_sections_tokens)} total sections:")
        for i, section in enumerate(all_sections_tokens):
            title_preview = section['title'][:100] + "..." if len(section['title']) > 100 else section['title']
            print(f"{i+1:2}. '{title_preview}' (level {section['level']}) via {section['method']}")
        
        print("\n=== ALL SECTIONS (Heuristic) ===")
        all_sections_heuristic = extract_all_sections(content, prefer_tokens=False)
        print(f"Found {len(all_sections_heuristic)} total sections:")
        for i, section in enumerate(all_sections_heuristic):
            title_preview = section['title'][:80] + "..." if len(section['title']) > 80 else section['title']
            print(f"{i+1:2}. '{title_preview}' (level {section['level']}) via {section['method']}")
        
        # Test reference section detection with heuristic
        print(f"\n=== REFERENCE SECTIONS (Heuristic) ===")
        reference_sections = extract_all_reference_sections(content, prefer_tokens=False)
        print(f"Found {len(reference_sections)} reference sections:")
        for i, section in enumerate(reference_sections):
            print(f"{i+1}. '{section['title']}' (level {section['level']}) - lines {section['start_line']}-{section['end_line']}")
            # Show preview of section content
            preview = section['text'][:200].replace('\n', ' ')
            print(f"   Preview: {preview}...")
        
            
    except FileNotFoundError:
        print(f"Benchmark file not found: {benchmark_file}")
        print("Testing with simple example instead...")
        
        # Fallback test
        simple_test = """# Introduction
This is a paper.

**References**

1. Author et al. (2020). Some paper.
2. Another author (2021). Another paper.

## Conclusion
The end."""
        
        all_sections = extract_all_sections(simple_test)
        print(f"Found {len(all_sections)} sections in simple test")
        
        ref_sections = extract_all_reference_sections(simple_test)
        print(f"Found {len(ref_sections)} reference sections in simple test")