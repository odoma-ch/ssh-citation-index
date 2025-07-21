import numpy as np
from typing import List, Dict, Any, Union, Optional
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
        If focus_fields is provided, only those fields are evaluated.
        Returns: dict with precision, recall, micro_f1, macro_f1, per_class_f1
        """
        # Normalize input
        pred_list = self._normalize(predictions)
        label_list = self._normalize(labels)
        if len(pred_list) != len(label_list):
            raise ValueError(f"Predictions and labels must have the same length: {len(pred_list)} vs {len(label_list)}")

        # Per-reference F1s for macro
        macro_f1s = []
        # Per-field stats for micro and per-class
        field_stats = {}

        for pred_refs, label_refs in zip(pred_list, label_list):
            f1s, stats = self._evaluate_single(pred_refs, label_refs, focus_fields=focus_fields)
            macro_f1s.append(np.mean(f1s) if f1s else 1.0)
            self._update_stats(field_stats, stats)

        # Micro metrics
        total = {k: 0 for k in ['matches', 'predictions', 'labels']}
        for v in field_stats.values():
            for k in total:
                total[k] += v.get(k, 0)
        micro = self._prf(total['matches'], total['predictions'], total['labels'])

        # Per-class F1
        per_class_f1 = {field: self._prf(v['matches'], v['predictions'], v['labels'])['f1'] for field, v in field_stats.items()}

        # Round results to 4 decimal places for readability and consistency
        rounded_per_class = {field: round(f, 4) for field, f in per_class_f1.items()}
        return {
            'precision': round(micro['precision'], 4),
            'recall': round(micro['recall'], 4),
            'micro_f1': round(micro['f1'], 4),
            'macro_f1': round(float(np.mean(macro_f1s)), 4),
            'per_class_f1': rounded_per_class,
        }

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
            # Best matching for lists
            n_pred, n_label = len(pred), len(label)
            if n_pred == 0 or n_label == 0:
                return 0, n_pred, n_label
            sim_matrix = np.zeros((n_pred, n_label))
            for i, p in enumerate(pred):
                for j, l in enumerate(label):
                    sim_matrix[i, j] = self._is_match(p, l)
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-sim_matrix)
            match_sum = sum(sim_matrix[i, j] for i, j in zip(row_ind, col_ind))
            # In soft_fuzzy mode we keep the fractional similarity; otherwise we cast to int.
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
        match_val = self._is_match(pred, label)
        if self.mode == 'soft_fuzzy':
            matches = float(match_val)
        else:
            matches = int(match_val)
        return (matches, 1, 1)

    def _is_match(self, a, b):
        """Return match indicator or similarity score depending on mode."""
        if a is None or b is None:
            return 0.0 if self.mode == 'soft_fuzzy' else False

        if self.mode == 'exact':
            return a == b
        elif self.mode == 'fuzzy':
            return ratio(str(a), str(b)) >= self.fuzzy_threshold
        elif self.mode == 'soft_fuzzy':
            # Scale RapidFuzz ratio (0‒100) to 0‒1 float for weighted matching
            return ratio(str(a), str(b)) / 100.0
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