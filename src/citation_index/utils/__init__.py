"""Utility functions for citation index."""

from .reference_matching import (
    normalize_title,
    extract_year,
    calculate_title_similarity,
    calculate_matching_score
)

__all__ = [
    'normalize_title',
    'extract_year', 
    'calculate_title_similarity',
    'calculate_matching_score'
] 