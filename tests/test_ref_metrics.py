"""
Test module for citation_index.evaluation.ref_metrics

This module tests all the evaluation functionality including:
- RefEvaluator with exact, fuzzy, and soft_fuzzy modes
- String-based reference evaluation
- Mixed type handling for author/editor/translator fields
"""

import pytest

from citation_index.core.models import Reference, References
from citation_index.core.models.person import Person
from citation_index.evaluation.ref_metrics import (
    RefEvaluator,
    string_reference_eval,
)


class TestRefEvaluator:
    """Test cases for RefEvaluator class"""

    def test_init_invalid_mode(self):
        """Test RefEvaluator initialization with invalid mode"""
        with pytest.raises(
            ValueError, match="mode must be 'exact', 'fuzzy', or 'soft_fuzzy'"
        ):
            RefEvaluator(mode="invalid")

    def test_exact_mode_evaluation(self):
        """Test exact mode evaluation"""
        evaluator = RefEvaluator(mode="exact")

        # Perfect match case
        pred_refs = [Reference(full_title="Deep Learning", publication_date="2020")]
        gt_refs = [Reference(full_title="Deep Learning", publication_date="2020")]

        result = evaluator.evaluate(pred_refs, gt_refs)

        assert result["overall_precision"] == 1.0
        assert result["overall_recall"] == 1.0
        assert result["overall_micro_f1"] == 1.0
        assert result["overall_macro_f1"] == 1.0

    def test_fuzzy_mode_evaluation(self):
        """Test fuzzy mode evaluation"""
        evaluator = RefEvaluator(mode="fuzzy", fuzzy_threshold=80)

        # Similar titles should match in fuzzy mode
        pred_refs = [Reference(full_title="Deep Learning Basics")]
        gt_refs = [Reference(full_title="Deep Learning Basic")]

        result = evaluator.evaluate(pred_refs, gt_refs)

        # Should have high scores due to fuzzy matching
        assert result["overall_precision"] > 0.5
        assert result["overall_recall"] > 0.5

    def test_soft_fuzzy_mode_evaluation(self):
        """Test soft_fuzzy mode evaluation"""
        evaluator = RefEvaluator(mode="soft_fuzzy")

        pred_refs = [Reference(full_title="Deep Learning")]
        gt_refs = [Reference(full_title="Deep Learning Fundamentals")]

        result = evaluator.evaluate(pred_refs, gt_refs)

        # Should have intermediate scores due to partial similarity
        assert 0.0 < result["overall_precision"] < 1.0
        assert 0.0 < result["overall_recall"] < 1.0

    def test_focus_fields_evaluation(self):
        """Test evaluation with focus_fields parameter"""
        evaluator = RefEvaluator(mode="exact")

        pred_refs = [
            Reference(
                full_title="Deep Learning",
                publication_date="2020",
                publication_place="New York",
            )
        ]
        gt_refs = [
            Reference(
                full_title="Deep Learning",  # matches
                publication_date="2021",  # doesn't match
                publication_place="New York",  # matches
            )
        ]

        # Test with focus on specific fields
        result = evaluator.evaluate(
            pred_refs, gt_refs, focus_fields=["full_title", "publication_place"]
        )

        # Should have focused metrics
        assert "focused_precision" in result
        assert "focused_recall" in result
        assert "focused_micro_f1" in result
        assert "per_field_metrics" in result

        # Should have perfect scores for focused fields (2/2 match)
        assert result["focused_precision"] == 1.0
        assert result["focused_recall"] == 1.0

        # Per-field metrics should be available
        assert "full_title" in result["per_field_metrics"]
        assert "publication_place" in result["per_field_metrics"]
        assert result["per_field_metrics"]["full_title"]["f1"] == 1.0

    def test_mismatched_lengths_error(self):
        """Test error when prediction and label batch counts don't match"""
        evaluator = RefEvaluator(mode="exact")

        # Test mismatched batch counts, not individual reference counts
        pred_batches = [References(references=[Reference(full_title="Test1")])]
        gt_batches = [
            References(references=[Reference(full_title="Test1")]),
            References(references=[Reference(full_title="Test2")]),
        ]

        with pytest.raises(
            ValueError, match="Predictions and labels must have the same length"
        ):
            evaluator.evaluate(pred_batches, gt_batches)

    def test_new_metrics_structure(self):
        """Test that the new metrics structure works correctly"""
        evaluator = RefEvaluator(mode="exact")

        pred_refs = [Reference(full_title="Test")]
        gt_refs = [Reference(full_title="Test")]

        # Without focus_fields - should only have overall metrics
        result = evaluator.evaluate(pred_refs, gt_refs)
        assert "overall_precision" in result
        assert "overall_recall" in result
        assert "overall_micro_f1" in result
        assert "overall_macro_f1" in result
        assert result["overall_precision"] == 1.0

        # With focus_fields - should have both overall and focused metrics
        result = evaluator.evaluate(pred_refs, gt_refs, focus_fields=["full_title"])
        assert "overall_precision" in result
        assert "focused_precision" in result
        assert "per_field_metrics" in result
        assert result["focused_precision"] == 1.0


class TestStringReferenceEval:
    """Test cases for string-based reference evaluation"""

    def test_perfect_match(self):
        """Test perfect string matching"""
        gt = ["Smith J., 2020", "Doe J., 2019"]
        pred = ["Smith J., 2020", "Doe J., 2019"]

        result = string_reference_eval(gt, pred, similarity_mode="fuzzy")

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1_score"] == 1.0
        assert result["avg_similarity"] == 1.0

    def test_fuzzy_mode(self):
        """Test fuzzy string matching"""
        gt = ["Smith J., 2020"]
        pred = ["Smith J, 2020"]  # Missing period

        result = string_reference_eval(
            gt, pred, similarity_mode="fuzzy", similarity_threshold=0.8
        )

        assert result["precision"] > 0.8
        assert result["recall"] > 0.8


class TestMixedTypeHandling:
    """Test cases for mixed string/object type handling in field matching"""

    def test_person_to_person_fuzzy_matching(self):
        """Test fuzzy matching between Person objects"""
        evaluator = RefEvaluator(mode="fuzzy", fuzzy_threshold=80)

        # Similar names should match
        person1 = Person(first_name="John", surname="Doe")
        person2 = Person(first_name="Jon", surname="Doe")  # Slight difference

        pred_refs = [Reference(full_title="Test", authors=[person1])]
        gt_refs = [Reference(full_title="Test", authors=[person2])]

        result = evaluator.evaluate(pred_refs, gt_refs)
        # Should still match due to high similarity
        assert result["overall_precision"] > 0.5

    def test_mixed_type_conversion(self):
        """Test that mixed types are converted and work"""
        evaluator = RefEvaluator(mode="exact")

        # Test Person object vs. string using actual Reference objects
        person = Person(first_name="John", surname="Doe")
        pred_refs = [Reference(full_title="Test", authors=[person])]
        gt_refs = [Reference(full_title="Test", authors=["John Doe"])]

        # Should work by converting Person to string
        result = evaluator.evaluate(pred_refs, gt_refs, focus_fields=["authors"])
        assert result["focused_precision"] >= 0.0  # At least runs without error

    def test_hungarian_matching_different_lengths(self):
        """Test Hungarian matching when list lengths differ"""
        evaluator = RefEvaluator(mode="exact")

        # Different lengths - should use Hungarian matching (focus on authors field only)
        person1 = Person(first_name="John", surname="Doe")
        person2 = Person(first_name="Jane", surname="Smith")
        person3 = Person(first_name="Bob", surname="Wilson")
        pred_refs = [Reference(full_title="Test", authors=[person1, person2, person3])]
        gt_refs = [Reference(full_title="Test", authors=[person1, person2])]

        result = evaluator.evaluate(pred_refs, gt_refs, focus_fields=["authors"])
        # Should match 2 out of 3 predictions with 2 labels
        expected_precision = 2 / 3  # 2 matches out of 3 predictions
        expected_recall = 2 / 2  # 2 matches out of 2 labels

        assert abs(result["focused_precision"] - expected_precision) < 0.01
        assert abs(result["focused_recall"] - expected_recall) < 0.01
