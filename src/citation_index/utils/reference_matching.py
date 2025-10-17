"""
Utility functions for reference matching and normalization.

Extracted from various matching tools to provide common functionality
for fuzzy matching, title normalization, and score calculation.
"""

import re
from typing import Optional, Dict, Any
from fuzzywuzzy import fuzz


def normalize_title(title: str) -> str:
    """Normalize title for more robust matching.
    
    This function:
    - Converts to lowercase
    - Removes trailing punctuation
    - Replaces Greek letters with Latin equivalents
    - Normalizes special characters and quotes
    - Removes non-alphanumeric characters (except spaces and hyphens)
    - Normalizes whitespace
    
    Args:
        title: Title string to normalize
        
    Returns:
        Normalized title string
    """
    if not title:
        return ""
    
    # Convert to lowercase
    text = title.lower()
    
    # Remove trailing punctuation
    text = text.rstrip('.!?')
    
    # Replace Greek letters and special characters
    # Note: Be careful with single letter replacements - they can be too aggressive
    replacements = {
        'ω': 'omega',
        'α': 'alpha',
        'β': 'beta',
        'γ': 'gamma',
        'δ': 'delta',
        '(1)h': '1h',
        'ω3': 'omega3',
        'ω-3': 'omega-3',
        ''': "'",
        ''': "'",
        '"': '"',
        '"': '"',
        '–': '-',
        '—': '-',
        # Note: Removed 'l': 'i' mapping as it's too aggressive and causes false matches
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove non-alphanumeric characters except spaces and hyphens
    text = re.sub(r'[^a-z0-9\s\-]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


def extract_family_name(author_name: str) -> Optional[str]:
    """Extract family name (last name) from author string.
    
    Handles multiple author name formats commonly found in academic databases:
    - "LastName, FirstName" (e.g., "Parmesan, Camille")
    - "FirstName LastName" (e.g., "Camille Parmesan")
    - "LastName, FirstName MiddleInitial" (e.g., "Smith, John A.")
    - "FirstName MiddleName LastName" (e.g., "John Allen Smith")
    
    The function prioritizes comma-separated formats (common in citations)
    and extracts the part before the comma as the family name.
    
    Args:
        author_name: Author name string in various formats
        
    Returns:
        Family name (last name) if found, None otherwise
        
    Examples:
        >>> extract_family_name("Parmesan, Camille")
        'Parmesan'
        >>> extract_family_name("John Smith")
        'Smith'
        >>> extract_family_name("von Neumann, John")
        'von Neumann'
    """
    if not author_name or not isinstance(author_name, str):
        return None
    
    author_str = author_name.strip()
    if not author_str:
        return None
    
    # Handle "LastName, FirstName" format (most common in citations)
    if ',' in author_str:
        # Take everything before the first comma as family name
        family_name = author_str.split(',')[0].strip()
        return family_name if family_name else None
    
    # Handle "FirstName LastName" format
    # Take the last word as family name
    parts = author_str.split()
    if parts:
        return parts[-1]
    
    return None


def extract_year(year_str: str) -> Optional[int]:
    """Extract year from various date string formats.
    
    Handles:
    - Plain year integers (e.g., "2020")
    - Date strings with year (e.g., "2020-03-15")
    - Month names (returns None)
    - Various separators (-, /, space)
    
    Args:
        year_str: String potentially containing a year
        
    Returns:
        Integer year if found, None otherwise
    """
    if not year_str:
        return None

    months = {
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'may': '05', 'june': '06', 'july': '07', 'august': '08',
        'september': '09', 'october': '10', 'november': '11', 'december': '12',
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'jun': '06',
        'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11',
        'dec': '12'
    }
    
    try:
        return int(year_str)
    except ValueError:
        year_str = year_str.lower().strip()
        
        # Try to find a 4-digit year pattern
        year_match = re.search(r'\b(19|20)\d{2}\b', year_str)
        if year_match:
            return int(year_match.group())
        
        # Return None if it's just a month name
        if year_str in months:
            return None
        
        # Try splitting by common separators
        parts = re.split(r'[-/\s]', year_str)
        for part in parts:
            if part.isdigit() and len(part) == 4:
                year = int(part)
                if 1700 <= year <= 2025:
                    return year
        
        return None


def calculate_title_similarity(title1: str, title2: str, normalize: bool = True) -> float:
    """Calculate similarity score between two titles using multiple fuzzy matching algorithms.
    
    Args:
        title1: First title
        title2: Second title
        normalize: Whether to normalize titles before comparison (default: True)
        
    Returns:
        Similarity score between 0 and 100
    """
    if not title1 or not title2:
        return 0.0
    
    if normalize:
        title1 = normalize_title(title1)
        title2 = normalize_title(title2)
    
    # Calculate various fuzzy matching scores
    exact_match = 100.0 if title1 == title2 else 0.0
    ratio = fuzz.ratio(title1, title2)
    partial_ratio = fuzz.partial_ratio(title1, title2)
    token_sort_ratio = fuzz.token_sort_ratio(title1, title2)
    token_set_ratio = fuzz.token_set_ratio(title1, title2)
    
    # Return the best score
    return max(exact_match, ratio, partial_ratio, token_sort_ratio, token_set_ratio)


def calculate_matching_score(
    reference_data: Dict[str, Any],
    candidate_data: Dict[str, Any],
    weights: Optional[Dict[str, int]] = None
) -> int:
    """Calculate matching score between reference and candidate using flexible criteria.
    
    This function compares multiple fields and returns a weighted score. The default
    weights are optimized for bibliographic matching.
    
    Default weights:
    - Year exact match: 20 points
    - Year within 1 year: 15 points
    - Title similarity: 0-50 points (scaled by similarity percentage)
    - Volume match: 15 points
    - Page match: 15 points
    
    Args:
        reference_data: Dictionary with reference metadata
            Expected keys: year, title/article_title/volume_title/journal_title,
                          volume, first_page
        candidate_data: Dictionary with candidate metadata
            Expected keys: year/pub_date, title, volume/volume_num, 
                          first_page/start_page
        weights: Optional custom weights for different matching criteria
        
    Returns:
        Integer score representing match quality (0-100)
    """
    if weights is None:
        weights = {
            'year_exact': 20,
            'year_close': 15,
            'title_perfect': 50,
            'title_excellent': 45,
            'title_very_good': 40,
            'title_good': 35,
            'title_decent': 30,
            'title_fair': 25,
            'volume': 15,
            'page': 15
        }
    
    score = 0
    
    # Year matching
    ref_year = reference_data.get('year')
    cand_year = candidate_data.get('year') or candidate_data.get('pub_date')
    
    if ref_year and cand_year:
        try:
            ref_year_int = extract_year(str(ref_year))
            cand_year_int = extract_year(str(cand_year)) if isinstance(cand_year, str) else int(str(cand_year)[:4])
            
            if ref_year_int and cand_year_int:
                if ref_year_int == cand_year_int:
                    score += weights['year_exact']
                elif abs(ref_year_int - cand_year_int) == 1:
                    score += weights['year_close']
        except (ValueError, TypeError):
            pass
    
    # Title matching - try multiple title fields
    cand_title = candidate_data.get('title', '')
    
    # Collect all possible title variants from reference
    ref_titles = [
        reference_data.get('title'),
        reference_data.get('article_title'),
        reference_data.get('volume_title'),
        reference_data.get('journal_title'),
        reference_data.get('full_title')
    ]
    
    best_title_score = 0.0
    for ref_title in ref_titles:
        if ref_title:
            title_similarity = calculate_title_similarity(ref_title, cand_title)
            best_title_score = max(best_title_score, title_similarity)
    
    # Map title similarity to score
    if best_title_score == 100:
        score += weights['title_perfect']
    elif best_title_score > 95:
        score += weights['title_excellent']
    elif best_title_score > 90:
        score += weights['title_very_good']
    elif best_title_score > 85:
        score += weights['title_good']
    elif best_title_score > 80:
        score += weights['title_decent']
    elif best_title_score > 75:
        score += weights['title_fair']
    
    # Volume matching
    ref_volume = reference_data.get('volume')
    cand_volume = candidate_data.get('volume') or candidate_data.get('volume_num')
    
    if ref_volume and cand_volume:
        if str(ref_volume) == str(cand_volume):
            score += weights['volume']
    
    # Page matching
    ref_page = reference_data.get('first_page') or reference_data.get('pages', '').split('-')[0]
    cand_page = candidate_data.get('first_page') or candidate_data.get('start_page') or candidate_data.get('pages', '').split('-')[0]
    
    if ref_page and cand_page:
        # Strip leading zeros for comparison
        ref_page_clean = str(ref_page).lstrip('0') or '0'
        cand_page_clean = str(cand_page).lstrip('0') or '0'
        
        if ref_page_clean == cand_page_clean:
            score += weights['page']
    
    return score

