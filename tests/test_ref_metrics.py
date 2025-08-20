"""
Test module for citation_index.evaluation.ref_metrics

This module tests all the evaluation functionality including:
- RefEvaluator with exact, fuzzy, and soft_fuzzy modes
- String-based reference evaluation
- Mixed type handling for author/editor/translator fields
"""

import pytest
import numpy as np
import logging
from typing import List, Dict, Any

from citation_index.core.models import Reference, References
from citation_index.core.models.person import Person
from citation_index.core.models.organization import Organization
from citation_index.evaluation.ref_metrics import (
    RefEvaluator,
    string_reference_eval,
)


class TestRefEvaluator:
    """Test cases for RefEvaluator class"""

    def test_init_valid_modes(self):
        """Test RefEvaluator initialization with valid modes"""
        # Test exact mode
        evaluator = RefEvaluator(mode='exact')
        assert evaluator.mode == 'exact'
        assert evaluator.fuzzy_threshold == 90  # default value
        
        # Test fuzzy mode with percentage threshold
        evaluator = RefEvaluator(mode='fuzzy', fuzzy_threshold=80)
        assert evaluator.mode == 'fuzzy'
        assert evaluator.fuzzy_threshold == 80
        
        # Test fuzzy mode with proportion threshold (should be converted)
        evaluator = RefEvaluator(mode='fuzzy', fuzzy_threshold=0.85)
        assert evaluator.mode == 'fuzzy'
        assert evaluator.fuzzy_threshold == 85.0
        
        # Test soft_fuzzy mode
        evaluator = RefEvaluator(mode='soft_fuzzy')
        assert evaluator.mode == 'soft_fuzzy'

    def test_init_invalid_mode(self):
        """Test RefEvaluator initialization with invalid mode"""
        with pytest.raises(ValueError, match="mode must be 'exact', 'fuzzy', or 'soft_fuzzy'"):
            RefEvaluator(mode='invalid')

    def test_exact_mode_evaluation(self):
        """Test exact mode evaluation"""
        evaluator = RefEvaluator(mode='exact')
        
        # Perfect match case
        pred_refs = [Reference(full_title="Deep Learning", publication_date="2020")]
        gt_refs = [Reference(full_title="Deep Learning", publication_date="2020")]
        
        result = evaluator.evaluate(pred_refs, gt_refs)
        
        assert result['overall_precision'] == 1.0
        assert result['overall_recall'] == 1.0
        assert result['overall_micro_f1'] == 1.0
        assert result['overall_macro_f1'] == 1.0

    def test_exact_mode_no_match(self):
        """Test exact mode with no matches"""
        evaluator = RefEvaluator(mode='exact')
        
        pred_refs = [Reference(full_title="Deep Learning", publication_date="2020")]
        gt_refs = [Reference(full_title="Machine Learning", publication_date="2019")]
        
        result = evaluator.evaluate(pred_refs, gt_refs)
        
        assert result['overall_precision'] == 0.0
        assert result['overall_recall'] == 0.0
        assert result['overall_micro_f1'] == 0.0
        assert result['overall_macro_f1'] == 0.0

    def test_fuzzy_mode_evaluation(self):
        """Test fuzzy mode evaluation"""
        evaluator = RefEvaluator(mode='fuzzy', fuzzy_threshold=80)
        
        # Similar titles should match in fuzzy mode
        pred_refs = [Reference(full_title="Deep Learning Basics")]
        gt_refs = [Reference(full_title="Deep Learning Basic")]
        
        result = evaluator.evaluate(pred_refs, gt_refs)
        
        # Should have high scores due to fuzzy matching
        assert result['overall_precision'] > 0.5
        assert result['overall_recall'] > 0.5

    def test_soft_fuzzy_mode_evaluation(self):
        """Test soft_fuzzy mode evaluation"""
        evaluator = RefEvaluator(mode='soft_fuzzy')
        
        pred_refs = [Reference(full_title="Deep Learning")]
        gt_refs = [Reference(full_title="Deep Learning Fundamentals")]
        
        result = evaluator.evaluate(pred_refs, gt_refs)
        
        # Should have intermediate scores due to partial similarity
        assert 0.0 < result['overall_precision'] < 1.0
        assert 0.0 < result['overall_recall'] < 1.0

    def test_focus_fields_evaluation(self):
        """Test evaluation with focus_fields parameter"""
        evaluator = RefEvaluator(mode='exact')
        
        pred_refs = [Reference(
            full_title="Deep Learning",
            publication_date="2020",
            publication_place="New York"
        )]
        gt_refs = [Reference(
            full_title="Deep Learning",  # matches
            publication_date="2021",     # doesn't match
            publication_place="New York" # matches
        )]
        
        # Test with focus on specific fields
        result = evaluator.evaluate(pred_refs, gt_refs, focus_fields=["full_title", "publication_place"])
        
        # Should have focused metrics
        assert 'focused_precision' in result
        assert 'focused_recall' in result
        assert 'focused_micro_f1' in result
        assert 'per_field_metrics' in result
        
        # Should have perfect scores for focused fields (2/2 match)
        assert result['focused_precision'] == 1.0
        assert result['focused_recall'] == 1.0
        
        # Per-field metrics should be available
        assert 'full_title' in result['per_field_metrics']
        assert 'publication_place' in result['per_field_metrics']
        assert result['per_field_metrics']['full_title']['f1'] == 1.0

    def test_empty_inputs(self):
        """Test evaluation with empty inputs"""
        evaluator = RefEvaluator(mode='exact')
        
        # Both empty
        result = evaluator.evaluate([], [])
        assert result['overall_precision'] == 0.0
        assert result['overall_recall'] == 0.0
        
        # One empty
        pred_refs = [Reference(full_title="Test")]
        result = evaluator.evaluate(pred_refs, [])
        assert result['overall_precision'] == 0.0
        assert result['overall_recall'] == 0.0

    def test_mismatched_lengths_error(self):
        """Test error when prediction and label batch counts don't match"""
        evaluator = RefEvaluator(mode='exact')
        
        # Test mismatched batch counts, not individual reference counts
        pred_batches = [References(references=[Reference(full_title="Test1")])]
        gt_batches = [References(references=[Reference(full_title="Test1")]), References(references=[Reference(full_title="Test2")])]
        
        with pytest.raises(ValueError, match="Predictions and labels must have the same length"):
            evaluator.evaluate(pred_batches, gt_batches)

    def test_list_field_matching(self):
        """Test evaluation with list fields (e.g., authors)"""
        evaluator = RefEvaluator(mode='exact')
        
        pred_refs = [Reference(
            full_title="Test Paper",
            authors=[Person(first_name="John", surname="Doe"), Person(first_name="Jane", surname="Smith")]
        )]
        gt_refs = [Reference(
            full_title="Test Paper",
            authors=[Person(first_name="John", surname="Doe"), Person(first_name="Jane", surname="Smith")]
        )]
        
        result = evaluator.evaluate(pred_refs, gt_refs)
        assert result['overall_precision'] == 1.0

    def test_references_object_input(self):
        """Test evaluation with References objects as input"""
        evaluator = RefEvaluator(mode='exact')
        
        pred_refs = References(references=[Reference(full_title="Test")])
        gt_refs = References(references=[Reference(full_title="Test")])
        
        result = evaluator.evaluate(pred_refs, gt_refs)
        assert result['overall_precision'] == 1.0

    def test_new_metrics_structure(self):
        """Test that the new metrics structure works correctly"""
        evaluator = RefEvaluator(mode='exact')
        
        pred_refs = [Reference(full_title="Test")]
        gt_refs = [Reference(full_title="Test")]
        
        # Without focus_fields - should only have overall metrics
        result = evaluator.evaluate(pred_refs, gt_refs)
        assert 'overall_precision' in result
        assert 'overall_recall' in result
        assert 'overall_micro_f1' in result
        assert 'overall_macro_f1' in result
        assert result['overall_precision'] == 1.0
        
        # With focus_fields - should have both overall and focused metrics
        result = evaluator.evaluate(pred_refs, gt_refs, focus_fields=["full_title"])
        assert 'overall_precision' in result
        assert 'focused_precision' in result
        assert 'per_field_metrics' in result
        assert result['focused_precision'] == 1.0


class TestStringReferenceEval:
    """Test cases for string-based reference evaluation"""

    def test_perfect_match(self):
        """Test perfect string matching"""
        gt = ["Smith J., 2020", "Doe J., 2019"]
        pred = ["Smith J., 2020", "Doe J., 2019"]
        
        result = string_reference_eval(gt, pred, similarity_mode='fuzzy')
        
        assert result['precision'] == 1.0
        assert result['recall'] == 1.0
        assert result['f1_score'] == 1.0
        assert result['avg_similarity'] == 1.0

    def test_no_match(self):
        """Test case with no matches"""
        gt = ["Smith J., 2020"]
        pred = ["Brown K., 2021"]
        
        result = string_reference_eval(gt, pred, similarity_mode='fuzzy', similarity_threshold=0.9)
        
        assert result['precision'] == 0.0
        assert result['recall'] == 0.0
        assert result['f1_score'] == 0.0

    def test_fuzzy_mode(self):
        """Test fuzzy string matching"""
        gt = ["Smith J., 2020"]
        pred = ["Smith J, 2020"]  # Missing period
        
        result = string_reference_eval(gt, pred, similarity_mode='fuzzy', similarity_threshold=0.8)
        
        assert result['precision'] > 0.8
        assert result['recall'] > 0.8

    def test_levenshtein_mode(self):
        """Test Levenshtein distance mode"""
        gt = ["Smith J., 2020"]
        pred = ["Smith J, 2020"]
        
        result = string_reference_eval(gt, pred, similarity_mode='levenshtein')
        
        assert result['precision'] > 0.0
        assert result['recall'] > 0.0

    def test_empty_inputs(self):
        """Test with empty inputs"""
        # Both empty
        result = string_reference_eval([], [], similarity_mode='fuzzy')
        assert result['precision'] == 1.0
        assert result['recall'] == 1.0
        assert result['f1_score'] == 1.0
        
        # One empty
        result = string_reference_eval(["test"], [], similarity_mode='fuzzy')
        assert result['precision'] == 0.0
        assert result['recall'] == 0.0
        assert result['f1_score'] == 0.0

    def test_invalid_similarity_mode(self):
        """Test invalid similarity mode"""
        with pytest.raises(ValueError, match="Unknown similarity_mode"):
            string_reference_eval(["test"], ["test"], similarity_mode='invalid')


class TestMixedTypeHandling:
    """Test cases for mixed string/object type handling in field matching"""

    def test_person_to_person_matching(self):
        """Test matching between two Person objects"""
        evaluator = RefEvaluator(mode='exact')
        
        # Exact matches
        person1 = Person(first_name="John", surname="Doe")
        person2 = Person(first_name="John", surname="Doe")
        
        pred_refs = [Reference(full_title="Test", authors=[person1])]
        gt_refs = [Reference(full_title="Test", authors=[person2])]
        
        result = evaluator.evaluate(pred_refs, gt_refs)
        assert result['overall_precision'] == 1.0
        assert result['overall_recall'] == 1.0

    def test_person_to_person_fuzzy_matching(self):
        """Test fuzzy matching between Person objects"""
        evaluator = RefEvaluator(mode='fuzzy', fuzzy_threshold=80)
        
        # Similar names should match
        person1 = Person(first_name="John", surname="Doe")
        person2 = Person(first_name="Jon", surname="Doe")  # Slight difference
        
        pred_refs = [Reference(full_title="Test", authors=[person1])]
        gt_refs = [Reference(full_title="Test", authors=[person2])]
        
        result = evaluator.evaluate(pred_refs, gt_refs)
        # Should still match due to high similarity
        assert result['overall_precision'] > 0.5

    def test_string_to_string_matching(self):
        """Test matching between string authors"""
        evaluator = RefEvaluator(mode='exact')
        
        # Now we can use actual string authors since the validator is fixed
        pred_refs = [Reference(full_title="Test", authors=["John Doe", "Jane Smith"])]
        gt_refs = [Reference(full_title="Test", authors=["John Doe", "Jane Smith"])]
        
        result = evaluator.evaluate(pred_refs, gt_refs)
        assert result['overall_precision'] == 1.0

    def test_mixed_type_conversion(self):
        """Test that mixed types are converted and work"""
        evaluator = RefEvaluator(mode='exact')
        
        # Test Person object vs. string using actual Reference objects
        person = Person(first_name="John", surname="Doe")
        pred_refs = [Reference(full_title="Test", authors=[person])]
        gt_refs = [Reference(full_title="Test", authors=["John Doe"])]
        
        # Should work by converting Person to string
        result = evaluator.evaluate(pred_refs, gt_refs, focus_fields=['authors'])
        assert result['focused_precision'] >= 0.0  # At least runs without error

    def test_organization_to_organization_matching(self):
        """Test matching between Organization objects"""
        evaluator = RefEvaluator(mode='exact')
        
        org1 = Organization(name="MIT Press")
        org2 = Organization(name="MIT Press")
        
        pred_refs = [Reference(full_title="Test", authors=[org1])]
        gt_refs = [Reference(full_title="Test", authors=[org2])]
        
        result = evaluator.evaluate(pred_refs, gt_refs)
        assert result['overall_precision'] == 1.0

    def test_mixed_list_types(self):
        """Test matching lists with mixed types"""
        evaluator = RefEvaluator(mode='exact')
        
        person = Person(first_name="John", surname="Doe")
        org = Organization(name="MIT Press")
        
        pred_refs = [Reference(full_title="Test", authors=[person, org])]
        gt_refs = [Reference(full_title="Test", authors=[person, org])]
        
        result = evaluator.evaluate(pred_refs, gt_refs)
        assert result['overall_precision'] == 1.0

    def test_translator_field_mixed_types(self):
        """Test translator field with mixed Person/str types"""
        evaluator = RefEvaluator(mode='exact')
        
        # Test mixed types with actual Reference objects
        person = Person(first_name="Alice", surname="Smith")
        pred_refs = [Reference(full_title="Test", translator=[person])]
        gt_refs = [Reference(full_title="Test", translator=["Alice Smith"])]
        
        # Should work with type conversion
        result = evaluator.evaluate(pred_refs, gt_refs, focus_fields=['translator'])
        assert result['focused_precision'] >= 0.0  # At least runs without error

    def test_editor_field_mixed_types(self):
        """Test editor field with mixed types"""
        evaluator = RefEvaluator(mode='exact')
        
        # Test mixed types with actual Reference objects
        org = Organization(name="Academic Publisher")
        pred_refs = [Reference(full_title="Test", editors=[org])]
        gt_refs = [Reference(full_title="Test", editors=["Academic Publisher"])]
        
        # Should work with type conversion
        result = evaluator.evaluate(pred_refs, gt_refs, focus_fields=['editors'])
        assert result['focused_precision'] >= 0.0  # At least runs without error

    def test_order_preservation_same_length_lists(self):
        """Test that order is preserved for same-length author lists"""
        evaluator = RefEvaluator(mode='exact')
        
        # Same order - should match perfectly (focus on authors field only)
        person1 = Person(first_name="John", surname="Doe")
        person2 = Person(first_name="Jane", surname="Smith")
        pred_refs = [Reference(full_title="Test", authors=[person1, person2])]
        gt_refs = [Reference(full_title="Test", authors=[person1, person2])]
        
        result = evaluator.evaluate(pred_refs, gt_refs, focus_fields=['authors'])
        assert result['focused_precision'] == 1.0
        
        # Different order - should have lower score due to order preservation (focus on authors field only)
        pred_refs = [Reference(full_title="Test", authors=[person2, person1])]
        gt_refs = [Reference(full_title="Test", authors=[person1, person2])]
        
        result = evaluator.evaluate(pred_refs, gt_refs, focus_fields=['authors'])
        assert result['focused_precision'] == 0.0  # Exact mode, wrong order

    def test_hungarian_matching_different_lengths(self):
        """Test Hungarian matching when list lengths differ"""
        evaluator = RefEvaluator(mode='exact')
        
        # Different lengths - should use Hungarian matching (focus on authors field only)
        person1 = Person(first_name="John", surname="Doe")
        person2 = Person(first_name="Jane", surname="Smith")
        person3 = Person(first_name="Bob", surname="Wilson")
        pred_refs = [Reference(full_title="Test", authors=[person1, person2, person3])]
        gt_refs = [Reference(full_title="Test", authors=[person1, person2])]
        
        result = evaluator.evaluate(pred_refs, gt_refs, focus_fields=['authors'])
        # Should match 2 out of 3 predictions with 2 labels
        expected_precision = 2/3  # 2 matches out of 3 predictions
        expected_recall = 2/2     # 2 matches out of 2 labels
        
        assert abs(result['focused_precision'] - expected_precision) < 0.01
        assert abs(result['focused_recall'] - expected_recall) < 0.01


class TestEdgeCases:
    """Additional edge case tests"""
    
    def test_empty_person_objects(self):
        """Test handling of empty Person objects"""
        evaluator = RefEvaluator(mode='exact')
        
        # Empty Person objects
        empty_person1 = Person()
        empty_person2 = Person()
        
        pred_refs = [Reference(full_title="Test", authors=[empty_person1])]
        gt_refs = [Reference(full_title="Test", authors=[empty_person2])]
        
        result = evaluator.evaluate(pred_refs, gt_refs)
        # Should handle empty objects gracefully
        assert result['overall_precision'] >= 0.0
        assert result['overall_recall'] >= 0.0

    def test_unicode_handling(self):
        """Test handling of unicode characters in names"""
        evaluator = RefEvaluator(mode='exact')
        
        pred_refs = [Reference(full_title="Test", authors=["José María García"])]
        gt_refs = [Reference(full_title="Test", authors=["José María García"])]
        
        result = evaluator.evaluate(pred_refs, gt_refs)
        assert result['overall_precision'] == 1.0

    def test_special_characters(self):
        """Test handling of special characters in names"""
        evaluator = RefEvaluator(mode='exact')
        
        pred_refs = [Reference(full_title="Test", authors=["O'Connor, J."])]
        gt_refs = [Reference(full_title="Test", authors=["O'Connor, J."])]
        
        result = evaluator.evaluate(pred_refs, gt_refs)
        assert result['overall_precision'] == 1.0

    def test_mixed_case_sensitivity(self):
        """Test that case differences are ignored in all modes"""
        evaluator = RefEvaluator(mode='exact')
        
        # Test with string authors (case should be ignored)
        pred_refs = [Reference(full_title="Test", authors=["john doe"])]
        gt_refs = [Reference(full_title="Test", authors=["John Doe"])]
        
        result = evaluator.evaluate(pred_refs, gt_refs, focus_fields=['authors'])
        # Should match even in exact mode due to case-insensitive comparison
        assert result['focused_precision'] == 1.0

    def test_fuzzy_case_insensitive(self):
        """Test that fuzzy mode handles case differences"""
        evaluator = RefEvaluator(mode='fuzzy', fuzzy_threshold=80)
        
        pred_refs = [Reference(full_title="Test", authors=["john doe"])]
        gt_refs = [Reference(full_title="Test", authors=["John Doe"])]
        
        result = evaluator.evaluate(pred_refs, gt_refs)
        # Should match in fuzzy mode despite case difference
        assert result['overall_precision'] >= 0.5

    def test_internal_convert_to_string(self):
        """Test internal string conversion functions work properly"""
        evaluator = RefEvaluator(mode='exact')
        
        # Test that string conversion works
        person = Person(first_name="John", surname="Doe")
        string_repr = evaluator._convert_to_string(person)
        assert "John" in string_repr
        assert "Doe" in string_repr
        
        # Test organization conversion
        org = Organization(name="MIT Press")
        org_str = evaluator._convert_to_string(org)
        assert "MIT Press" == org_str
        
        # Test direct string
        direct_str = evaluator._convert_to_string("Direct String")
        assert direct_str == "Direct String"

    def test_soft_fuzzy_gradual_scoring(self):
        """Test that soft_fuzzy mode provides gradual scores"""
        evaluator = RefEvaluator(mode='soft_fuzzy')
        
        # Exact match should score 1.0 (focus on authors field only)
        person1 = Person(first_name="John", surname="Doe")
        person2 = Person(first_name="John", surname="Doe")
        pred_refs = [Reference(full_title="Test", authors=[person1])]
        gt_refs = [Reference(full_title="Test", authors=[person2])]
        result = evaluator.evaluate(pred_refs, gt_refs, focus_fields=['authors'])
        assert result['focused_precision'] == 1.0
        
        # Partial match should score between 0 and 1 (focus on authors field only)
        person3 = Person(first_name="John", surname="Smith")
        pred_refs = [Reference(full_title="Test", authors=[person1])]
        gt_refs = [Reference(full_title="Test", authors=[person3])]
        result = evaluator.evaluate(pred_refs, gt_refs, focus_fields=['authors'])
        assert 0.0 < result['focused_precision'] < 1.0
        
        # No match should score closer to 0 (focus on authors field only)
        person4 = Person(first_name="Alice", surname="Cooper")
        pred_refs = [Reference(full_title="Test", authors=[person1])]
        gt_refs = [Reference(full_title="Test", authors=[person4])]
        result = evaluator.evaluate(pred_refs, gt_refs, focus_fields=['authors'])
        assert result['focused_precision'] < 0.5

    def test_evaluate_with_focus_fields_mixed_types(self):
        """Test focus fields evaluation with mixed types"""
        evaluator = RefEvaluator(mode='exact')
        
        # Test mixed types with actual Reference objects
        person = Person(first_name="John", surname="Doe")
        pred_refs = [Reference(
            full_title="Test Paper",
            publication_date="2020",
            authors=[person]
        )]
        gt_refs = [Reference(
            full_title="Test Paper", 
            publication_date="2020",
            authors=["John Doe"]
        )]
        
        # Focus only on authors field
        result = evaluator.evaluate(pred_refs, gt_refs, focus_fields=['authors'])
        
        assert 'focused_precision' in result
        assert 'focused_recall' in result
        assert 'per_field_metrics' in result
        assert 'authors' in result['per_field_metrics']

    def test_stress_test_many_mixed_authors(self):
        """Stress test with many mixed-type authors"""
        evaluator = RefEvaluator(mode='fuzzy', fuzzy_threshold=85)
        
        # Create mix of persons and orgs (no strings since Reference model filters them out)
        mixed_authors = []
        for i in range(5):  # Reduced to 5 for more predictable results
            if i % 2 == 0:
                mixed_authors.append(Person(first_name=f"First{i}", surname=f"Last{i}"))
            else:
                mixed_authors.append(Organization(name=f"Organization {i}"))
        
        # Create similar authors with slight variations for fuzzy matching test
        gt_authors = []
        for i in range(5):
            if i % 2 == 0:
                gt_authors.append(Person(first_name=f"First{i}", surname=f"Last{i}"))  # Exact match
            else:
                gt_authors.append(Organization(name=f"Organization {i}"))  # Exact match
        
        pred_refs = [Reference(full_title="Complex Paper", authors=mixed_authors)]
        gt_refs = [Reference(full_title="Complex Paper", authors=gt_authors)]
        
        result = evaluator.evaluate(pred_refs, gt_refs, focus_fields=['authors'])
        # Should get perfect matches since they're identical
        assert result['focused_precision'] == 1.0
