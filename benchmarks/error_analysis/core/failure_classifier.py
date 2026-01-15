"""
Hierarchical failure classifier for LLM responses.

Classifies responses into three main categories:
1. Structural Errors (unreadable results)
2. Factual Errors (readable but incorrect - minor vs major errors)
3. LLM Refusal

Error Types:
- Minor Error: Medium similarity (0.30-0.70) - partially correct references
- Major Error: Low similarity (<0.30) - severely incorrect or unmatched references
"""

import json
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional, Any


class FailureClassifier:
    """Hierarchical failure classifier for LLM responses."""
    
    # Refusal patterns
    REFUSAL_PATTERNS = [
        r"i cannot",
        r"i'm not able",
        r"i am not able",
        r"as an ai",
        r"i don't have access",
        r"i apologize",
        r"i'm unable",
        r"i am unable",
        r"cannot provide",
        r"not possible for me",
    ]
    
    # Similarity thresholds for classification
    HIGH_SIMILARITY_THRESHOLD = 0.70  # >= 0.70: Correct match
    LOW_SIMILARITY_THRESHOLD = 0.30   # 0.30-0.70: Minor error, <0.30: Major error
    
    def __init__(self):
        self.refusal_regex = re.compile(
            '|'.join(self.REFUSAL_PATTERNS),
            re.IGNORECASE
        )
    
    def classify_response(
        self, 
        response: str, 
        gt_references: List[str],
        task: str
    ) -> Dict[str, Any]:
        """
        Hierarchically classify a single response.
        
        Args:
            response: LLM response string
            gt_references: Ground truth reference strings
            task: Task type (extraction, parsing, extraction_and_parsing)
        
        Returns:
            Dictionary with classification results
        """
        result = {
            'has_structural_error': False,
            'has_refusal': False,
            'has_factual_error': False,
            'failure_category': 'success',
            'structural_error_type': None,
            'minor_error_count': 0,  # Renamed from misclassification_count
            'major_error_count': 0,  # Renamed from hallucination_count
            'correct_count': 0,
            'extracted_ref_count': 0,
            'gt_ref_count': len(gt_references),
            'error_details': '',
            'response_length': len(response) if response else 0,
        }
        
        # Step 1: Check for structural errors
        structural_error = self._check_structural_error(response, task)
        if structural_error:
            result['has_structural_error'] = True
            result['failure_category'] = 'structural_error'
            result['structural_error_type'] = structural_error
            result['error_details'] = structural_error
            return result
        
        # Step 2: Check for refusal
        if self._check_refusal(response):
            result['has_refusal'] = True
            result['failure_category'] = 'refusal'
            result['error_details'] = 'LLM refused to complete the task'
            return result
        
        # Step 3: Analyze factual correctness
        factual_result = self._analyze_factual_correctness(
            response, gt_references, task
        )
        
        result.update(factual_result)
        
        # Determine if there's a factual error
        if factual_result['minor_error_count'] > 0 or factual_result['major_error_count'] > 0:
            result['has_factual_error'] = True
            result['failure_category'] = 'factual_error'
        
        return result
    
    def _check_structural_error(self, response: str, task: str) -> Optional[str]:
        """Check for structural errors that make the response unreadable."""
        # Empty response
        if not response or not response.strip():
            return "empty_response"
        
        # For tasks that expect JSON output
        if task in ['parsing', 'extraction_and_parsing']:
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError as e:
                return f"invalid_json: {str(e)}"
            
            # Check for missing 'references' key
            if 'references' not in parsed:
                return "missing_references_key"
            
            # Check if references is empty when it shouldn't be
            if not isinstance(parsed['references'], list):
                return "references_not_list"
            
            if len(parsed['references']) == 0:
                return "empty_references_list"
        
        # For extraction task (plain text output)
        elif task == 'extraction':
            # Check if response is just whitespace or very short
            if len(response.strip()) < 10:
                return "response_too_short"
        
        return None
    
    def _check_refusal(self, response: str) -> bool:
        """Check if the response contains refusal language."""
        return bool(self.refusal_regex.search(response))
    
    def _analyze_factual_correctness(
        self, 
        response: str, 
        gt_references: List[str],
        task: str
    ) -> Dict[str, Any]:
        """
        Analyze factual correctness by comparing with ground truth.
        
        Returns counts of minor errors, major errors, and correct extractions.
        """
        result = {
            'minor_error_count': 0,  # Renamed from misclassification_count
            'major_error_count': 0,  # Renamed from hallucination_count
            'correct_count': 0,
            'extracted_ref_count': 0,
            'error_details': '',
        }
        
        # Extract references from response
        extracted_refs = self._extract_references(response, task)
        result['extracted_ref_count'] = len(extracted_refs)
        
        if not extracted_refs:
            result['error_details'] = 'No references extracted'
            return result
        
        # Match extracted references against ground truth
        matched, minor_errors, major_errors = self._match_references(
            extracted_refs, gt_references
        )
        
        result['correct_count'] = matched
        result['minor_error_count'] = minor_errors
        result['major_error_count'] = major_errors
        
        # Build error details
        error_parts = []
        if minor_errors > 0:
            error_parts.append(f"{minor_errors} minor errors")
        if major_errors > 0:
            error_parts.append(f"{major_errors} major errors")
        if matched > 0:
            error_parts.append(f"{matched} correct")
        
        result['error_details'] = ', '.join(error_parts) if error_parts else 'All correct'
        
        return result
    
    def _extract_references(self, response: str, task: str) -> List[str]:
        """Extract reference strings from response based on task type."""
        refs = []
        
        if task in ['parsing', 'extraction_and_parsing']:
            try:
                parsed = json.loads(response)
                references = parsed.get('references', [])
                
                # Convert structured references to strings for comparison
                for ref in references:
                    if isinstance(ref, dict):
                        # Create a string representation of the reference
                        ref_str = self._reference_dict_to_string(ref)
                        refs.append(ref_str)
                    elif isinstance(ref, str):
                        refs.append(ref)
            except (json.JSONDecodeError, AttributeError):
                pass
        
        elif task == 'extraction':
            # Split by newlines for extraction task
            refs = [line.strip() for line in response.split('\n') if line.strip()]
        
        return refs
    
    def _reference_dict_to_string(self, ref_dict: Dict) -> str:
        """Convert a structured reference dictionary to a string for comparison."""
        parts = []
        
        # Authors
        if 'authors' in ref_dict and ref_dict['authors']:
            author_strs = []
            for author in ref_dict['authors']:
                if isinstance(author, dict):
                    name_parts = [
                        author.get('first_name', ''),
                        author.get('middle_name', ''),
                        author.get('surname', '')
                    ]
                    author_str = ' '.join([p for p in name_parts if p])
                    author_strs.append(author_str)
                elif isinstance(author, str):
                    author_strs.append(author)
            if author_strs:
                parts.append(', '.join(author_strs))
        
        # Title
        if 'full_title' in ref_dict and ref_dict['full_title']:
            parts.append(ref_dict['full_title'])
        
        # Journal/Publisher
        if 'journal_title' in ref_dict and ref_dict['journal_title']:
            parts.append(ref_dict['journal_title'])
        elif 'publisher' in ref_dict and ref_dict['publisher']:
            parts.append(ref_dict['publisher'])
        
        # Year
        if 'publication_date' in ref_dict and ref_dict['publication_date']:
            parts.append(str(ref_dict['publication_date']))
        
        # Volume, Issue, Pages
        if 'volume' in ref_dict and ref_dict['volume']:
            parts.append(f"Vol. {ref_dict['volume']}")
        if 'pages' in ref_dict and ref_dict['pages']:
            parts.append(f"pp. {ref_dict['pages']}")
        
        return '. '.join(parts) if parts else ''
    
    def _match_references(
        self, 
        extracted_refs: List[str], 
        gt_refs: List[str]
    ) -> Tuple[int, int, int]:
        """
        Match extracted references against ground truth.
        
        Returns:
            (matched_count, minor_error_count, major_error_count)
        """
        matched = 0
        minor_errors = 0  # Similarity 0.30-0.70 (medium similarity)
        major_errors = 0  # Similarity < 0.30 (low similarity)
        
        # Track which ground truth refs have been matched
        gt_matched = set()
        
        for ext_ref in extracted_refs:
            best_similarity = 0.0
            best_gt_idx = -1
            
            # Find best match in ground truth
            for idx, gt_ref in enumerate(gt_refs):
                if idx in gt_matched:
                    continue
                
                similarity = self._string_similarity(ext_ref, gt_ref)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_gt_idx = idx
            
            # Classify based on similarity
            if best_similarity >= self.HIGH_SIMILARITY_THRESHOLD:
                matched += 1
                if best_gt_idx >= 0:
                    gt_matched.add(best_gt_idx)
            elif best_similarity >= self.LOW_SIMILARITY_THRESHOLD:
                # Medium similarity - minor error
                minor_errors += 1
            else:
                # Low similarity - major error
                major_errors += 1
        
        return matched, minor_errors, major_errors
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using SequenceMatcher."""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
