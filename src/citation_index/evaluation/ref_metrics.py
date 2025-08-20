import numpy as np
import logging
from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np
from pydantic import BaseModel
from rapidfuzz.fuzz import ratio
from rapidfuzz.distance import Levenshtein
from citation_index.core.models import Reference, References

class RefEvaluator:
    """
    Evaluate reference extraction with exact or fuzzy matching.

    Args:
        mode: 'exact' for strict equality, 'fuzzy' for fuzzy string matching.
        fuzzy_threshold: Minimum similarity (0-100) for fuzzy match (only used if mode='fuzzy').
    """
    def __init__(self, mode: str = 'exact', fuzzy_threshold: float = 90):
        # Accept three evaluation modes:
        # 1. 'exact'       – strict equality
        # 2. 'fuzzy'       – boolean match based on fuzzy_threshold
        # 3. 'soft_fuzzy'  – use raw fuzzy similarity (0‒1) as the match weight
        if mode not in ('exact', 'fuzzy', 'soft_fuzzy'):
            raise ValueError("mode must be 'exact', 'fuzzy', or 'soft_fuzzy'")
        self.mode = mode
        # Allow users to specify threshold either in [0,1] or [0,100].
        if mode == 'fuzzy' and 0 < fuzzy_threshold <= 1:
            fuzzy_threshold *= 100  # convert proportion -> percentage expected by RapidFuzz ratio
        self.fuzzy_threshold = fuzzy_threshold

    def evaluate(
        self,
        predictions: Union[List[Reference], References, List[References]],
        labels: Union[List[Reference], References, List[References]],
        focus_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate predictions against labels and return metrics.
        
        Always calculates overall metrics using all fields.
        If focus_fields is provided, also calculates focused metrics and per-field metrics.
        
        Returns: dict with overall metrics, and optionally focused/per-field metrics
        """
        # Normalize input
        pred_list = self._normalize(predictions)
        label_list = self._normalize(labels)
        if len(pred_list) != len(label_list):
            raise ValueError(f"Predictions and labels must have the same length: {len(pred_list)} vs {len(label_list)}")

        # Always evaluate ALL fields for overall metrics
        overall_macro_f1s = []
        overall_field_stats = {}

        for pred_refs, label_refs in zip(pred_list, label_list):
            f1s, stats = self._evaluate_single(pred_refs, label_refs, focus_fields=None)  # All fields
            overall_macro_f1s.append(np.mean(f1s) if f1s else 1.0)
            self._update_stats(overall_field_stats, stats)

        # Calculate overall metrics
        overall_total = {k: 0 for k in ['matches', 'predictions', 'labels']}
        for v in overall_field_stats.values():
            for k in overall_total:
                overall_total[k] += v.get(k, 0)
        overall_micro = self._prf(overall_total['matches'], overall_total['predictions'], overall_total['labels'])

        # Build result with overall metrics
        result = {
            'overall_precision': round(overall_micro['precision'], 4),
            'overall_recall': round(overall_micro['recall'], 4),
            'overall_micro_f1': round(overall_micro['f1'], 4),
            'overall_macro_f1': round(float(np.mean(overall_macro_f1s)), 4),
        }

        # If focus_fields provided, also calculate focused metrics
        if focus_fields is not None:
            focused_macro_f1s = []
            focused_field_stats = {}

            for pred_refs, label_refs in zip(pred_list, label_list):
                f1s, stats = self._evaluate_single(pred_refs, label_refs, focus_fields=focus_fields)
                focused_macro_f1s.append(np.mean(f1s) if f1s else 1.0)
                self._update_stats(focused_field_stats, stats)

            # Calculate focused metrics
            focused_total = {k: 0 for k in ['matches', 'predictions', 'labels']}
            for v in focused_field_stats.values():
                for k in focused_total:
                    focused_total[k] += v.get(k, 0)
            focused_micro = self._prf(focused_total['matches'], focused_total['predictions'], focused_total['labels'])

            # Add focused metrics to result
            result.update({
                'focused_precision': round(focused_micro['precision'], 4),
                'focused_recall': round(focused_micro['recall'], 4),
                'focused_micro_f1': round(focused_micro['f1'], 4),
                'focused_macro_f1': round(float(np.mean(focused_macro_f1s)), 4),
            })

            # Per-field metrics for focused fields only
            per_field_metrics = {}
            for field in focus_fields:
                if field in focused_field_stats:
                    field_prf = self._prf(
                        focused_field_stats[field]['matches'],
                        focused_field_stats[field]['predictions'],
                        focused_field_stats[field]['labels']
                    )
                    per_field_metrics[field] = {
                        'precision': round(field_prf['precision'], 4),
                        'recall': round(field_prf['recall'], 4),
                        'f1': round(field_prf['f1'], 4),
                    }
                else:
                    # Field not found in any predictions/labels
                    per_field_metrics[field] = {
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                    }
            
            result['per_field_metrics'] = per_field_metrics
        
        return result

    def _evaluate_single(self, preds: List[Reference], labels: List[Reference], focus_fields: Optional[List[str]] = None):
        # Hungarian matching for best F1 assignment
        n_pred, n_label = len(preds), len(labels)
        f1_matrix = np.zeros((n_pred, n_label))
        stats_matrix = [[None for _ in range(n_label)] for _ in range(n_pred)]
        for i, p in enumerate(preds):
            for j, l in enumerate(labels):
                f1, stats = self._reference_f1_and_stats(p, l, focus_fields=focus_fields)
                f1_matrix[i, j] = f1
                stats_matrix[i][j] = stats
        from scipy.optimize import linear_sum_assignment
        if n_pred and n_label:
            row_ind, col_ind = linear_sum_assignment(-f1_matrix)
        else:
            row_ind, col_ind = [], []
        f1s = [f1_matrix[i, j] for i, j in zip(row_ind, col_ind)]
        # Add zeros for unmatched
        f1s += [0.0] * (abs(n_pred - n_label))
        # Collect stats
        stats = {}
        for i, j in zip(row_ind, col_ind):
            self._update_stats(stats, stats_matrix[i][j])
        # Hallucinated
        for i in set(range(n_pred)) - set(row_ind):
            self._update_stats(stats, self._reference_stats(preds[i], None, focus_fields=focus_fields))
        # Missed
        for j in set(range(n_label)) - set(col_ind):
            self._update_stats(stats, self._reference_stats(None, labels[j], focus_fields=focus_fields))
        return f1s, stats

    def _reference_f1_and_stats(self, pred: Optional[Reference], label: Optional[Reference], focus_fields: Optional[List[str]] = None):
        if pred is None and label is None:
            return 1.0, {}
        if pred is None or label is None:
            return 0.0, self._reference_stats(pred, label, focus_fields=focus_fields)
        matches, n_pred, n_label, field_stats = 0, 0, 0, {}
        fields = pred.model_fields
        if focus_fields is not None:
            fields = [f for f in fields if f in focus_fields]
        for field in fields:
            if getattr(pred, field) is None and getattr(label, field) is None:
                continue
            pred_val = getattr(pred, field)
            label_val = getattr(label, field)
            m, p, l = self._field_match(pred_val, label_val)
            matches += m
            n_pred += p
            n_label += l
            field_stats[field] = {'matches': m, 'predictions': p, 'labels': l}
        f1 = self._prf(matches, n_pred, n_label)['f1']
        return f1, field_stats

    def _reference_stats(self, pred: Optional[Reference], label: Optional[Reference], focus_fields: Optional[List[str]] = None):
        stats = {}
        fields = set()
        if pred is not None:
            fields.update(pred.model_fields)
        if label is not None:
            fields.update(label.model_fields)
        if focus_fields is not None:
            fields = {f for f in fields if f in focus_fields}
        for field in fields:
            pred_val = getattr(pred, field, None) if pred is not None else None
            label_val = getattr(label, field, None) if label is not None else None
            m, p, l = self._field_match(pred_val, label_val)
            stats[field] = {'matches': m, 'predictions': p, 'labels': l}
        return stats

    def _field_match(self, pred, label):
        # Handles str, list, or None
        if pred is None and label is None:
            return 0, 0, 0
        if isinstance(pred, list) and isinstance(label, list):
            # Best matching for lists - use order-preserving matching for same-length lists
            n_pred, n_label = len(pred), len(label)
            if n_pred == 0 or n_label == 0:
                return 0, n_pred, n_label
            
            # If lists are the same length, assume order should be preserved
            if n_pred == n_label:
                match_sum = 0
                for p, l in zip(pred, label):
                    match_sum += self._match_mixed_types(p, l)
                if self.mode == 'soft_fuzzy':
                    matches = float(match_sum)
                else:
                    matches = int(match_sum)
                return matches, n_pred, n_label
            else:
                # Different lengths - use Hungarian matching
                sim_matrix = np.zeros((n_pred, n_label))
                for i, p in enumerate(pred):
                    for j, l in enumerate(label):
                        sim_matrix[i, j] = self._match_mixed_types(p, l)
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(-sim_matrix)
                match_sum = sum(sim_matrix[i, j] for i, j in zip(row_ind, col_ind))
                if self.mode == 'soft_fuzzy':
                    matches = float(match_sum)
                else:
                    matches = int(match_sum)
                return matches, n_pred, n_label
        if isinstance(pred, list):
            return 0, len(pred), 0
        if isinstance(label, list):
            return 0, 0, len(label)
        if pred is None:
            return 0, 0, 1
        if label is None:
            return 0, 1, 0
        # Convert the similarity / match value appropriately for the chosen mode
        match_val = self._match_mixed_types(pred, label)
        if self.mode == 'soft_fuzzy':
            matches = float(match_val)
        else:
            matches = int(match_val)
        return (matches, 1, 1)

    def _match_mixed_types(self, a, b):
        """Handle matching between mixed string/object types.
        
        Supports Person, Organization, and str types.
        Returns match indicator or similarity score depending on mode.
        """
        from citation_index.core.models.person import Person
        from citation_index.core.models.organization import Organization
        
        if a is None or b is None:
            return 0.0 if self.mode == 'soft_fuzzy' else False
        
        # Check if both are the same type
        a_is_person = isinstance(a, Person)
        b_is_person = isinstance(b, Person)
        a_is_org = isinstance(a, Organization)
        b_is_org = isinstance(b, Organization)
        a_is_str = isinstance(a, str)
        b_is_str = isinstance(b, str)
        
        # Both are Person objects - compare fields
        if a_is_person and b_is_person:
            return self._match_person_objects(a, b)
        
        # Both are Organization objects - compare names
        elif a_is_org and b_is_org:
            return self._is_match(getattr(a, 'name', ''), getattr(b, 'name', ''))
        
        # Both are strings - direct comparison
        elif a_is_str and b_is_str:
            return self._is_match(a, b)
        
        # Mixed types - convert to string and log warning
        else:
            a_str = self._convert_to_string(a)
            b_str = self._convert_to_string(b)
            
            # Log warning about type mismatch
            type_a = type(a).__name__ if not a_is_str else 'str'
            type_b = type(b).__name__ if not b_is_str else 'str'
            logging.warning(f"Type mismatch in field comparison: {type_a} vs {type_b}. Converting to strings for comparison.")
            
            return self._is_match(a_str, b_str)
    
    def _match_person_objects(self, person_a, person_b):
        """Match two Person objects by comparing their fields."""
        if person_a is None or person_b is None:
            return 0.0 if self.mode == 'soft_fuzzy' else False
        
        # Get all person fields to compare
        person_fields = ['first_name', 'middle_name', 'surname', 'role_name']
        matches = 0
        total_fields = 0
        
        for field in person_fields:
            val_a = getattr(person_a, field, None)
            val_b = getattr(person_b, field, None)
            
            # Skip if both are None
            if val_a is None and val_b is None:
                continue
                
            total_fields += 1
            if self._is_match(val_a, val_b):
                if self.mode == 'soft_fuzzy':
                    matches += self._is_match(val_a, val_b)
                else:
                    matches += 1
        
        if total_fields == 0:
            return 1.0 if self.mode == 'soft_fuzzy' else True
        
        if self.mode == 'soft_fuzzy':
            return matches / total_fields
        else:
            return matches == total_fields
    
    def _convert_to_string(self, obj):
        """Convert Person/Organization/str to string representation."""
        from citation_index.core.models.person import Person
        from citation_index.core.models.organization import Organization
        
        if isinstance(obj, Person):
            parts = []
            if hasattr(obj, 'first_name') and obj.first_name:
                parts.append(str(obj.first_name))
            if hasattr(obj, 'middle_name') and obj.middle_name:
                parts.append(str(obj.middle_name))
            if hasattr(obj, 'surname') and obj.surname:
                parts.append(str(obj.surname))
            if hasattr(obj, 'role_name') and obj.role_name and not parts:
                parts.append(str(obj.role_name))
            return ' '.join(parts) if parts else ''
        
        elif isinstance(obj, Organization):
            return str(getattr(obj, 'name', ''))
        
        elif isinstance(obj, str):
            return obj
        
        else:
            return str(obj) if obj is not None else ''

    def _is_match(self, a, b):
        """Return match indicator or similarity score depending on mode."""
        if a is None or b is None:
            return 0.0 if self.mode == 'soft_fuzzy' else False

        # Normalize strings for case-insensitive comparison
        str_a = str(a).strip().lower() if a is not None else ""
        str_b = str(b).strip().lower() if b is not None else ""

        if self.mode == 'exact':
            return str_a == str_b
        elif self.mode == 'fuzzy':
            return ratio(str_a, str_b) >= self.fuzzy_threshold
        elif self.mode == 'soft_fuzzy':
            # Scale RapidFuzz ratio (0‒100) to 0‒1 float for weighted matching
            return ratio(str_a, str_b) / 100.0
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    @staticmethod
    def _prf(matches, predictions, labels):
        precision = matches / predictions if predictions else 0.0
        recall = matches / labels if labels else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {'precision': precision, 'recall': recall, 'f1': f1}

    @staticmethod
    def _normalize(x):
        # Accepts References, list of Reference, or list of References
        if isinstance(x, References):
            return [list(x)]
        if isinstance(x, list):
            if len(x) == 0:
                return [[]]
            if isinstance(x[0], Reference):
                return [x]
            if isinstance(x[0], References):
                return [list(xx) for xx in x]
            if isinstance(x[0], list):
                # List of lists - check if inner lists contain References
                if len(x[0]) == 0 or isinstance(x[0][0], Reference):
                    return x
        raise ValueError(f"Unsupported input type: {type(x)}")

    @staticmethod
    def _update_stats(stats: Dict[str, Dict[str, int]], new_stats: Dict[str, Dict[str, int]]):
        for field, v in new_stats.items():
            if field not in stats:
                stats[field] = {'matches': 0, 'predictions': 0, 'labels': 0}
            for k in ['matches', 'predictions', 'labels']:
                stats[field][k] += v.get(k, 0)
        return stats


def string_reference_eval(
    references_data: List[str],
    response_list: List[str],
    similarity_mode: str = 'fuzzy',
    similarity_threshold: float = 0.8
) -> Dict[str, float]:
    """
    Evaluate string-based reference extraction using Levenshtein or fuzzy ratio.
    Returns: dict with precision, recall, f1_score, avg_similarity
    """
    def normalize_text(text):
        if not isinstance(text, str):
            text = str(text)
        return text.strip()

    gt_refs = [normalize_text(ref) for ref in references_data if normalize_text(ref)]
    pred_refs = [normalize_text(ref) for ref in response_list if normalize_text(ref)]
    n_gt = len(gt_refs)
    n_pred = len(pred_refs)

    if n_gt == 0 and n_pred == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 'avg_similarity': 1.0}
    if n_gt == 0 or n_pred == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'avg_similarity': 0.0}

    similarity_matrix = np.zeros((n_gt, n_pred))
    for i, gt_ref in enumerate(gt_refs):
        for j, pred_ref in enumerate(pred_refs):
            if similarity_mode == 'levenshtein':
                # Normalized Levenshtein ratio
                max_len = max(len(gt_ref), len(pred_ref))
                if max_len == 0:
                    similarity = 1.0
                else:
                    similarity = 1 - Levenshtein.distance(gt_ref, pred_ref) / max_len
            elif similarity_mode == 'fuzzy':
                similarity = ratio(gt_ref, pred_ref) / 100.0
            else:
                raise ValueError(f"Unknown similarity_mode: {similarity_mode}")
            similarity_matrix[i, j] = similarity

    cost_matrix = 1.0 - similarity_matrix
    from scipy.optimize import linear_sum_assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matched_pairs = 0
    total_similarity = 0
    for i, j in zip(row_indices, col_indices):
        similarity = similarity_matrix[i, j]
        if similarity >= similarity_threshold:
            matched_pairs += 1
            total_similarity += similarity

    precision = matched_pairs / n_pred
    recall = matched_pairs / n_gt
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_similarity = total_similarity / matched_pairs if matched_pairs > 0 else 0.0

    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1_score, 4),
        'avg_similarity': round(avg_similarity, 4)
    }


if __name__ == "__main__":
    # String-based test
    gt = ["Smith J., 2020", "Doe J., 2019", "Brown S., 2018"]
    pred = ["Smith J., 2020", "Doe J., 2019", "Browne S., 2018"]
    print("String eval (levenshtein):", string_reference_eval(gt, pred, similarity_mode='levenshtein'))
    print("String eval (fuzzy):", string_reference_eval(gt, pred, similarity_mode='fuzzy'))

    # Reference-based test
    from citation_index.core.models import Reference
    gt_refs = [Reference(full_title="NLP approaches in natural language processing"), Reference(full_title="Deep learning fundamentals")]
    pred_refs = [Reference(full_title="NLP approaches in natural language processing"), Reference(full_title="Machine learning fundamentals")]
    evaluator = RefEvaluator(mode='exact')
    print("Reference eval (exact):", evaluator.evaluate(pred_refs, gt_refs))
    evaluator_fuzzy = RefEvaluator(mode='fuzzy', fuzzy_threshold=70)
    print("Reference eval (fuzzy 70%):", evaluator_fuzzy.evaluate(pred_refs, gt_refs))
    evaluator_fuzzy = RefEvaluator(mode='fuzzy', fuzzy_threshold=0.95)
    print("Reference eval (fuzzy 95%):", evaluator_fuzzy.evaluate(pred_refs, gt_refs))
    # Soft-fuzzy evaluation (uses raw similarity instead of thresholding)
    evaluator_soft = RefEvaluator(mode='soft_fuzzy')
    print("Reference eval (soft_fuzzy):", evaluator_soft.evaluate(pred_refs, gt_refs))