#!/usr/bin/env python3
"""
High-precision author name parser for LinkedBook dataset.
Only parses patterns with >95% confidence, leaves rest unparsed.
"""

import re
from typing import List, Dict


def parse_author_high_precision(author_string: str) -> List:
    """
    Parse author string with high precision approach.
    
    Returns:
    - If parsed: List of author dicts [{"first_name": "...", "middle_name": "...", "surname": "..."}]
    - If not parsed: List of strings ["original string"]
    
    Args:
        author_string: Raw author string from linkedbook dataset
        
    Returns:
        List of either author dicts or strings
    """
    if not author_string:
        return [""]
    
    # Only strip whitespace and trailing commas, NOT periods (needed for initial detection)
    author_str = author_string.strip().rstrip(',')
    
    # SKIP CONDITIONS - Return unparsed
    skip_patterns = [
        ' and ',        # English connector
        ' - ',          # Dash separator (with spaces)
        'ed.',          # Editor marker
        'Ed.',          # Editor marker (capital)
        '(',            # Parenthetical content
        'et al.',       # Et alii
        'et al',        # Et alii (no period)
    ]
    
    for pattern in skip_patterns:
        if pattern in author_str:
            return [author_string]
    
    # Skip all caps (likely OCR issue), but allow if it looks like "Initial. SURNAME"
    if author_str.isupper() and len(author_str) > 5:
        # Check if it has pattern "I. SURNAME" which is parseable despite caps
        if not re.match(r'^[A-Z]\.\s+[A-Z]+$', author_str):
            return [author_string]
    
    # Skip special markers
    special_markers = ['AA. VV', 'AA. VV.', 'Ead.', 'Ibid.', 'AA.VV', 'AA.VV.']
    if author_str in special_markers:
        return [author_string]
    
    # Pre-process: Normalize Italian/Latin connectors to commas
    normalized = re.sub(r'\s+e\s+', ', ', author_str)  # Italian "e" = and
    normalized = re.sub(r'\s+et\s+', ', ', normalized)  # Latin "et" = and
    
    # TRY TIER 1.5a: Multi-author patterns
    # Pattern 1: "Surname, Initial., Surname, Initial., ..."
    multi_init_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s+[A-Z]\.(?:\s+[A-Z]\.)*,?\s*){2,}$'
    if re.match(multi_init_pattern, normalized):
        result = parse_multi_author_initials(normalized)
        if result:
            return result
    
    # Pattern 2: "Surname Initial., Initial. Surname, ..." (mixed format)
    # Example: "Barrucand M., A. Bednorz," = "Surname Initial., Initial. Surname"
    mixed_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+[A-Z]\.),\s+([A-Z]\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),?\s*$'
    match = re.match(mixed_pattern, normalized)
    if match:
        # Parse first author: "Surname Initial."
        first_part = match.group(1)
        first_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+([A-Z]\.)$', first_part)
        if first_match:
            surname1 = first_match.group(1)
            initial1 = first_match.group(2)
            
            # Parse second author: "Initial. Surname"
            second_part = match.group(2)
            second_match = re.match(r'^([A-Z]\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)$', second_part)
            if second_match:
                initial2 = second_match.group(1)
                surname2 = second_match.group(2)
                
                return [
                    {"first_name": initial1, "middle_name": "", "surname": surname1},
                    {"first_name": initial2, "middle_name": "", "surname": surname2}
                ]
    
    # SKIP TIER 1.5b for now: "Two-Word, Two-Word" is ambiguous
    # Could be "Firstname Surname" or "Surname Firstname" - can't tell reliably
    # Examples:
    #   "Federico Berchet, Agostino Sagredo" = Firstname Surname (EN style)
    #   "Cessi Roberto, Bennato Fanny" = Surname Firstname (IT style)
    # Better to keep unparsed than risk wrong order
    
    # TRY TIER 1.3: "Firstname Surname" (no comma = standard bibliographic format)
    # Pattern: Two capitalized words, no comma, no periods (not initials)
    # Assumption: No comma means Firstname Surname order (standard format)
    # With comma would be "Surname, Firstname" (handled by TIER 1.2)
    if ',' not in author_str and '.' not in author_str:
        # Check if it's exactly 2 or 3 capitalized words (Firstname [Middle] Surname)
        words = author_str.split()
        if 2 <= len(words) <= 3:
            # Check all words are capitalized properly (not all caps, not lowercase)
            all_proper = all(
                word[0].isupper() and (len(word) == 1 or word[1:].islower() or word[1:].istitle())
                for word in words
            )
            if all_proper:
                if len(words) == 2:
                    # "Paolo Camerini" = Firstname Surname
                    return [{
                        "first_name": words[0],
                        "middle_name": "",
                        "surname": words[1]
                    }]
                elif len(words) == 3:
                    # "John Paul Smith" = Firstname Middle Surname
                    return [{
                        "first_name": words[0],
                        "middle_name": words[1],
                        "surname": words[2]
                    }]
    
    # Check for name prefixes (da, di, del, van, von, etc.)
    name_prefixes = ['da', 'di', 'del', 'della', "dell'", 'degli', 
                     'van', 'von', 'de', 'le', 'la']
    prefix_pattern = r'^(' + '|'.join(name_prefixes) + r')\s+'
    prefix_match = re.match(prefix_pattern, author_str, re.IGNORECASE)
    
    # TRY TIER 1.1: "Surname Initial(s)." or "Prefix Surname Initial(s)."
    # Pattern: Capitalized word(s) followed by capital letter(s) with periods
    if prefix_match:
        # Has prefix, adjust the pattern to capture it as part of surname
        pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+([A-Z]\.(?:\s+[A-Z]\.)*)$'
    else:
        pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+([A-Z]\.(?:\s+[A-Z]\.)*)$'
    
    match = re.match(pattern, author_str)
    if match:
        surname = match.group(1)
        initials = match.group(2).split()
        first = initials[0] if len(initials) > 0 else ""
        middle = " ".join(initials[1:]) if len(initials) > 1 else ""
        return [{
            "first_name": first,
            "middle_name": middle,
            "surname": surname
        }]
    
    # TRY TIER 1.4: "Initial(s). Surname" (reverse order)
    # Pattern: Capital letter(s) with periods followed by capitalized word(s)
    # Also handle all-caps surnames like "N. COLAK"
    pattern_normal = r'^([A-Z]\.(?:\s+[A-Z]\.)*)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)$'
    pattern_allcaps = r'^([A-Z]\.(?:\s+[A-Z]\.)*)\s+([A-Z]+)$'
    
    match = re.match(pattern_normal, author_str) or re.match(pattern_allcaps, author_str)
    if match:
        initials = match.group(1).split()
        surname = match.group(2)
        first = initials[0] if len(initials) > 0 else ""
        middle = " ".join(initials[1:]) if len(initials) > 1 else ""
        return [{
            "first_name": first,
            "middle_name": middle,
            "surname": surname
        }]
    
    # TRY TIER 1.2: "Surname, Firstname [Middlename]" or "Surname, Initial(s)." 
    # Pattern: Surname (possibly multi-word with prefix), comma, then firstname(s) or initial(s)
    # BUT: Must validate it's not actually multi-author in disguise
    if prefix_match:
        # With prefix, surname can be multi-word
        pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+),\s+([A-Z](?:[a-z]+|\.(?:\s+[A-Z]\.)*))'
    else:
        # Capture: Surname (1-2 words), comma, then Firstname OR Initial(s).
        # Firstname: [A-Z][a-z]+ (e.g., "George")
        # Initial(s): [A-Z]\. or [A-Z]\.\s+[A-Z]\. (e.g., "G." or "G. F.")
        pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s+([A-Z](?:[a-z]+|\.(?:\s+[A-Z]\.)*))'
    
    match = re.match(pattern, author_str)
    if match:
        surname = match.group(1)
        firstname_part = match.group(2)
        
        # Check if there's additional content after firstname (middlename)
        remaining = author_str[match.end():].strip()
        
        # Parse firstname and middlename
        if '.' in firstname_part:
            # It's initials: split them
            initials = firstname_part.replace('.', '. ').split()
            initials = [i for i in initials if i]  # Remove empty strings
            firstname = initials[0] if len(initials) > 0 else firstname_part
            middlename = " ".join(initials[1:]) if len(initials) > 1 else ""
        else:
            # It's a full name
            firstname = firstname_part
            middlename = ""
        
        # Check remaining content for middlename
        if remaining:
            # IMPORTANT: Check if this might actually be multi-author
            # If remaining text looks like another name (no periods, full words),
            # it's likely a second author - skip to maintain precision
            if '.' not in remaining and len(remaining.split()) >= 1:
                # Too ambiguous - could be second author
                return [author_string]
            
            # If it has periods, treat as middle initials
            if middlename:
                middlename = middlename + " " + remaining.strip()
            else:
                middlename = remaining.strip()
        
        return [{
            "first_name": firstname,
            "middle_name": middlename,
            "surname": surname
        }]
    
    # DEFAULT: Keep original unparsed
    return [author_string]


def parse_multi_author_initials(author_str: str) -> List[Dict[str, str]]:
    """
    Parse multi-author format: "Surname, Initial., Surname, Initial., ..."
    
    Args:
        author_str: Normalized author string
        
    Returns:
        List of parsed authors, or empty list if parsing fails
    """
    authors = []
    remaining = author_str.strip().rstrip(',')
    
    # Strategy: Iteratively match "Surname, Initial(s)." segments
    while remaining:
        # Match one segment: Surname (possibly multi-word), comma, Initial(s).
        pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s+([A-Z]\.(?:\s+[A-Z]\.)*)'
        match = re.match(pattern, remaining)
        
        if not match:
            # If we already found at least one author, consider it success
            if len(authors) >= 1:
                break
            return []  # Failed to parse
        
        surname = match.group(1)
        initials_str = match.group(2)
        initials = initials_str.split()
        
        authors.append({
            "first_name": initials[0] if len(initials) > 0 else "",
            "middle_name": " ".join(initials[1:]) if len(initials) > 1 else "",
            "surname": surname
        })
        
        # Move to next segment
        remaining = remaining[match.end():].strip().lstrip(',').strip()
    
    return authors if len(authors) >= 2 else []
