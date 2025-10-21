"""Utility functions for citation index."""

from .reference_matching import (
    normalize_title,
    extract_year,
    calculate_title_similarity,
    calculate_matching_score
)
from .author_parser import parse_author_high_precision

__all__ = [
    'normalize_title',
    'extract_year', 
    'calculate_title_similarity',
    'calculate_matching_score',
    'parse_author_high_precision'
] 