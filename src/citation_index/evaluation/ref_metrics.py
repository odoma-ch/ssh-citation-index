import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
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

        # Per-class metrics (P, R, F1)
        per_class_metrics = {field: self._prf(v['matches'], v['predictions'], v['labels']) for field, v in field_stats.items()}
        per_class_f1 = {field: metrics['f1'] for field, metrics in per_class_metrics.items()}
        per_class_precision = {field: metrics['precision'] for field, metrics in per_class_metrics.items()}
        per_class_recall = {field: metrics['recall'] for field, metrics in per_class_metrics.items()}

        # Round results to 4 decimal places for readability and consistency
        rounded_per_class_f1 = {field: round(f, 4) for field, f in per_class_f1.items()}
        rounded_per_class_precision = {field: round(p, 4) for field, p in per_class_precision.items()}
        rounded_per_class_recall = {field: round(r, 4) for field, r in per_class_recall.items()}
        
        return {
            'precision': round(micro['precision'], 4),
            'recall': round(micro['recall'], 4),
            'micro_f1': round(micro['f1'], 4),
            'macro_f1': round(float(np.mean(macro_f1s)), 4),
            'per_class_f1': rounded_per_class_f1,
            'per_class_precision': rounded_per_class_precision,
            'per_class_recall': rounded_per_class_recall,
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


def _canonicalize_text(text: Optional[str]) -> str:
    """Normalize free text for robust fuzzy matching."""
    if not text:
        return ""
    import re
    t = str(text).strip().upper()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\.+$", "", t)  # trailing periods
    return t


def _person_to_canonical(person: Any) -> str:
    """Convert a Person or Organization to a canonical author string.

    - Person: SURNAME INITIALS (e.g., "DOE J S") if available
    - Organization: normalized organization name
    - Fallback: empty string
    """
    from citation_index.core.models.person import Person
    from citation_index.core.models.organization import Organization
    if isinstance(person, Person):
        parts: List[str] = []
        surname = _canonicalize_text(getattr(person, "surname", None))
        if surname:
            parts.append(surname)
        initials: List[str] = []
        for field_name in ("first_name", "middle_name"):
            value = getattr(person, field_name, None)
            if value:
                for token in str(value).split():
                    token = token.strip()
                    if token:
                        initials.append(token[0].upper())
        if initials:
            parts.append(" ".join(initials))
        # If surname is missing, try using role_name or first_name as fallback
        if not parts:
            for fallback in ("first_name", "role_name"):
                val = _canonicalize_text(getattr(person, fallback, None))
                if val:
                    parts.append(val)
                    break
        return " ".join(parts).strip()
    if isinstance(person, Organization):
        name = _canonicalize_text(getattr(person, "name", None))
        return name
    # Unknown type or None
    return ""


def _split_gt_authors(gt_author_text: str) -> List[str]:
    """Split ground-truth author string into individual author tokens.

    Improved heuristics: split on commas, semicolons, and common conjunctions (and/e/&),
    as well as hyphen-separated lists seen in the dataset (" - ").
    Also handles various spacing and punctuation patterns.
    """
    import re
    if not gt_author_text:
        return []
    text = gt_author_text.strip()
    # Normalize unusual spaces and multiple spaces
    text = re.sub(r"\s+", " ", text)
    
    # Enhanced splitting patterns to handle more cases
    # - Standard separators: comma, semicolon
    # - Conjunctions: and, e (Italian), & 
    # - Hyphens with surrounding spaces
    # - "et al." patterns
    split_pattern = r"\s*(?:,|;|\band\b|\be\b|&|\s+-\s+|\s-\s|\bet\s+al\.?)\s*"
    parts = re.split(split_pattern, text, flags=re.IGNORECASE)
    
    # Clean up parts: remove empty strings, extra whitespace, trailing punctuation
    cleaned_parts = []
    for part in parts:
        part = part.strip()
        if part:
            # Remove trailing punctuation but preserve important punctuation like initials
            part = re.sub(r"[,;]+$", "", part)
            if part:  # Check again after cleaning
                cleaned_parts.append(part)
    
    return cleaned_parts


def _canonicalize_author_token(token: str) -> str:
    """Canonicalize a single author token into SURNAME INITIALS-like form.

    Improved canonicalization:
    - Uppercase and normalize spacing
    - Remove trailing punctuation but preserve initials
    - Handle common name patterns and abbreviations
    """
    import re
    if not token:
        return ""
    
    # Basic canonicalization from _canonicalize_text
    t = str(token).strip().upper()
    t = re.sub(r"\s+", " ", t)
    
    # Remove trailing periods and commas, but preserve initials like "J." within names
    t = re.sub(r"[,;]+$", "", t)  # Remove trailing commas/semicolons
    t = re.sub(r"\.+$", "", t)    # Remove trailing periods only at the end
    
    # Handle some common patterns
    # Normalize "AA. VV." (Authors Various) pattern
    t = re.sub(r"^AA\.\s*VV\.?$", "AA VV", t)
    
    # Handle bracket patterns like "[Domenico Malipiero]"
    t = re.sub(r"^\[(.*)\]$", r"\1", t)
    
    # Final cleanup
    t = t.strip()
    return t


def _author_list_similarity(pred_authors: List[str], gt_authors: List[str], threshold: float = 85.0) -> Tuple[float, int, int, int]:
    """Compute matching between predicted and ground-truth author lists.

    Returns a tuple: (avg_similarity, matched_pairs, n_pred, n_gt).
    A pair counts as matched if fuzzy ratio >= threshold.
    """
    from rapidfuzz.fuzz import ratio as fuzz_ratio
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    n_pred, n_gt = len(pred_authors), len(gt_authors)
    if n_pred == 0 and n_gt == 0:
        return 1.0, 0, 0, 0
    if n_pred == 0 or n_gt == 0:
        return 0.0, 0, n_pred, n_gt

    sim = np.zeros((n_pred, n_gt), dtype=float)
    for i, p in enumerate(pred_authors):
        for j, g in enumerate(gt_authors):
            sim[i, j] = fuzz_ratio(p, g)
    # Hungarian on cost = 100 - similarity
    cost = 100.0 - sim
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_pairs = 0
    similarity_sum = 0.0
    for i, j in zip(row_ind, col_ind):
        s = sim[i, j]
        if s >= threshold:
            matched_pairs += 1
            similarity_sum += s
    avg_sim = similarity_sum / matched_pairs if matched_pairs else 0.0
    return avg_sim, matched_pairs, n_pred, n_gt


def _extract_year(text: Optional[str]) -> Optional[str]:
    """Extract a 4-digit year when present; return None otherwise.

    Looks for 4-digit numbers in a broad historical range.
    """
    if not text:
        return None
    import re
    m = re.search(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b", str(text))
    if m:
        return m.group(1)
    return None


def map_linkedbook_item_to_reference_and_authors(item: Dict[str, Any]) -> Tuple[Reference, List[str]]:
    """Map a LinkedBook JSONL item (with 'tags') to a Reference and canonical GT authors list.

    Mapping policy:
    - title            -> full_title
    - publicationplace -> publication_place
    - year or publicationnumber-year -> publication_date (best 4-digit extraction)
    - archival fields (archivalreference, archive_lib, box, pagination, conjunction) are ignored for fields
    - authors taken from tags.author and split/canonicalized for author evaluation
    """
    tags = (item.get('tags') or {}) if isinstance(item, dict) else {}

    title = tags.get('title')
    place = tags.get('publicationplace')
    year = tags.get('year')
    pubnum_year = tags.get('publicationnumber-year')

    year_val = _extract_year(year) or _extract_year(pubnum_year)

    ref = Reference(
        full_title=title,
        publication_place=place,
        publication_date=year_val,
    )

    gt_author_text = tags.get('author', '')
    gt_authors = [_canonicalize_author_token(a) for a in _split_gt_authors(gt_author_text)]

    return ref, gt_authors


def evaluate_linkedbook_fields_and_authors(
    pred_batches: List[References],
    gt_batches: List[List[Dict[str, Any]]],
    focus_fields: Optional[List[str]] = None,
    author_threshold: float = 85.0,
    mode: str = 'soft_fuzzy',
) -> Dict[str, Any]:
    """Special evaluation for LinkedBook:

    - Field-focused evaluation on title/place/year using the existing RefEvaluator
      with Hungarian matching.
    - Per-author list matching with canonicalization and Hungarian assignment
      at the author level.

    Args:
        pred_batches: list of References objects, each representing a batch of predicted refs
        gt_batches: list of batches; each batch is a list of dicts read from JSONL
                    with at least keys 'reference' and 'tags'.
        focus_fields: fields to evaluate on; defaults to ['full_title', 'publication_place', 'publication_date']
        author_threshold: fuzzy ratio threshold (0–100) to consider an author match
        mode: evaluation mode for fields ('exact' | 'fuzzy' | 'soft_fuzzy')

    Returns:
        dict containing 'field_metrics', 'author_metrics', and basic counts.
    """
    # Input validation
    if not pred_batches or not gt_batches:
        raise ValueError("pred_batches and gt_batches cannot be empty")
    
    if len(pred_batches) != len(gt_batches):
        raise ValueError(f"Mismatch in batch counts: {len(pred_batches)} pred vs {len(gt_batches)} gt batches")
    
    if focus_fields is None:
        focus_fields = ["full_title", "publication_place", "publication_date"]
    
    # Validate threshold
    if not (0 <= author_threshold <= 100):
        raise ValueError(f"author_threshold must be between 0 and 100, got {author_threshold}")

    # Build ground-truth References (fields only) and store GT author tokens per batch
    gt_refs_per_batch: List[References] = []
    gt_author_tokens_per_batch: List[List[List[str]]] = []
    for batch in gt_batches:
        refs_list: List[Reference] = []
        authors_list: List[List[str]] = []
        for item in batch:
            ref, gt_authors = map_linkedbook_item_to_reference_and_authors(item)
            refs_list.append(ref)
            authors_list.append(gt_authors)
        gt_list = refs_list
        gt_refs_per_batch.append(References(references=gt_list))
        gt_author_tokens_per_batch.append(authors_list)

    # Field-focused metrics via existing evaluator
    evaluator = RefEvaluator(mode=mode, fuzzy_threshold=90)
    field_metrics = evaluator.evaluate(
        predictions=pred_batches,
        labels=gt_refs_per_batch,
        focus_fields=focus_fields,
    )

    # Author-level metrics
    # We must align predicted refs to GT refs per batch using field similarity to get pairs.
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    macro_f1s: List[float] = []
    total_matches = 0
    total_pred = 0
    total_gt = 0

    for batch_idx, (pred_refs, gt_refs, raw_gt_batch) in enumerate(zip(pred_batches, gt_refs_per_batch, gt_batches)):
        pred_list = list(pred_refs)
        gt_list = list(gt_refs)
        n_pred, n_gt = len(pred_list), len(gt_list)
        
        # Validation: check batch size consistency
        if len(raw_gt_batch) != n_gt:
            raise ValueError(f"Batch {batch_idx}: GT references count mismatch: {len(raw_gt_batch)} raw vs {n_gt} parsed")
        
        if batch_idx >= len(gt_author_tokens_per_batch):
            raise ValueError(f"Batch {batch_idx}: Missing author tokens for this batch")
        
        if len(gt_author_tokens_per_batch[batch_idx]) != n_gt:
            raise ValueError(f"Batch {batch_idx}: Author tokens count mismatch: {len(gt_author_tokens_per_batch[batch_idx])} vs {n_gt} refs")

        # Build similarity matrix based on field F1 between refs
        f1_matrix = np.zeros((n_pred, n_gt), dtype=float)
        for i, p in enumerate(pred_list):
            for j, g in enumerate(gt_list):
                f1, _ = evaluator._reference_f1_and_stats(p, g, focus_fields=focus_fields)  # type: ignore[attr-defined]
                f1_matrix[i, j] = f1
        if n_pred and n_gt:
            row_ind, col_ind = linear_sum_assignment(-f1_matrix)
        else:
            row_ind, col_ind = [], []

        # Compute author list matching per aligned pair
        per_ref_f1: List[float] = []
        for i, j in zip(row_ind, col_ind):
            pred_ref = pred_list[i]
            # Use direct batch indexing instead of searching
            gt_authors = gt_author_tokens_per_batch[batch_idx][j] if j < len(gt_author_tokens_per_batch[batch_idx]) else []

            pred_authors_field = getattr(pred_ref, 'authors', None)
            pred_authors_tokens = []
            if isinstance(pred_authors_field, list):
                pred_authors_tokens = [_canonicalize_author_token(_person_to_canonical(a)) for a in pred_authors_field]

            avg_sim, matched_pairs, n_p, n_g = _author_list_similarity(pred_authors_tokens, gt_authors, threshold=author_threshold)
            total_matches += matched_pairs
            total_pred += n_p
            total_gt += n_g

            # Per-reference F1 for macro averaging
            precision = matched_pairs / n_p if n_p else 0.0
            recall = matched_pairs / n_g if n_g else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            per_ref_f1.append(f1)

        # Account for unpaired predictions or labels (all unmatched)
        if n_pred > n_gt:
            # Extra predicted refs: zero F1 each, and count their authors as unmatched predictions
            per_ref_f1.extend([0.0] * (n_pred - n_gt))
            for i in range(n_gt, n_pred):
                if i < len(pred_list):
                    pred_authors_field = getattr(pred_list[i], 'authors', None)
                    if isinstance(pred_authors_field, list):
                        total_pred += len(pred_authors_field)
        elif n_gt > n_pred:
            # Extra GT refs: zero F1 each, and count their authors as unmatched labels
            per_ref_f1.extend([0.0] * (n_gt - n_pred))
            for j in range(n_pred, n_gt):
                if j < len(gt_author_tokens_per_batch[batch_idx]):
                    total_gt += len(gt_author_tokens_per_batch[batch_idx][j])

        if per_ref_f1:
            macro_f1s.append(float(np.mean(per_ref_f1)))
        else:
            macro_f1s.append(1.0)

    author_precision = total_matches / total_pred if total_pred else 0.0
    author_recall = total_matches / total_gt if total_gt else 0.0
    author_micro_f1 = 2 * author_precision * author_recall / (author_precision + author_recall) if (author_precision + author_recall) else 0.0
    author_macro_f1 = float(np.mean(macro_f1s)) if macro_f1s else 1.0

    author_metrics = {
        'precision': round(author_precision, 4),
        'recall': round(author_recall, 4),
        'micro_f1': round(author_micro_f1, 4),
        'macro_f1': round(author_macro_f1, 4),
        'threshold': author_threshold,
    }

    return {
        'field_metrics': field_metrics,
        'author_metrics': author_metrics,
        'counts': {
            'num_batches': len(pred_batches),
        },
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