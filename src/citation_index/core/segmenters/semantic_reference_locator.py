"""Semantic reference section locator using embeddings and lexical cues."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

from citation_index.pipelines.text_extraction import split_pages


# Reference section queries for semantic matching
REF_QUERIES = [
    "### References", "References", "Bibliography", "Works Cited", "Literature Cited",
    "Bibliographie", "Literaturverzeichnis", "Referencias", "参考文献",

    # APA-like
    "Smith, J., & Doe, A. (2018). Title of the paper. Journal of X, 12(3), 123–145. https://doi.org/10.1016/j.x.2018.01.001",
    "Lee, K. (2020). Book Title. Oxford University Press.",

    # IEEE-like
    "[12] J. Smith and A. Doe, 'Title of the paper' in Proc. ICASSP, 2019, pp. 123–128.",

    # Chicago-like
    "Miller, Thomas. 2015. The Example Study. Chicago: UChicago Press.",

    # Non-English
    "Klein, T.; Müller, A. (2020): Ein Beitrag. Zeitschrift für Y 45(2): 77–89. doi:10.1000/xyz",
    "王小明，李雷. 2021. 论文标题[J]. 某某学报, 45(2): 77–89. https://doi.org/10.1000/xyz",

    # Strong tokens
    "doi: 10.1000/xyz", "https://doi.org/10.1000/xyz", "arXiv:2401.01234", "ISSN 1234-5678"
]

# Lexical cues for reference sections
CUE_RX = re.compile(
    r"(doi:|10\.\d{4,}/|arxiv:|journal of|vol\.|no\.|pp\.|\bet al\.|\(\d{4}\)|\b19\d{2}\b|\b20\d{2}\b|issn)",
    re.IGNORECASE
)


def _get_embeddings(texts: List[str], model: str, endpoint: str) -> np.ndarray:
    """Get embeddings for texts using the specified model and endpoint."""
    r = requests.post(endpoint, json={"model": model, "input": texts})
    r.raise_for_status()
    data = r.json()["data"]
    if all("index" in d for d in data):
        data = sorted(data, key=lambda x: x["index"])
    return np.array([d["embedding"] for d in data], dtype=float)


def _lexical_bonus(chunk_texts: List[str], bonus_per_hit: float = 0.02, max_bonus: float = 0.1) -> np.ndarray:
    """Calculate lexical bonus scores based on reference-specific cues."""
    bonuses = []
    for t in chunk_texts:
        hits = len(CUE_RX.findall(t))
        bonuses.append(min(max_bonus, hits * bonus_per_hit))
    return np.array(bonuses, dtype=float)


def locate_reference_sections_semantic(
    text_or_path: Union[str, Path],
    chunker,
    embedding_model: str = "intfloat/multilingual-e5-large-instruct",
    embedding_endpoint: str = "http://0.0.0.0:7997/embeddings",
    top_k: int = 5,
    top_percentile: float = 0.9,
    fast_path: bool = False
) -> str:
    """Locate reference sections using semantic similarity and lexical cues.
    
    This function uses embedding-based semantic search combined with lexical cues
    to identify reference sections in academic documents. It's more robust than
    rule-based approaches and works across different document formats and languages.
    
    Args:
        text_or_path: Input text string or path to markdown file
        chunker: Text chunker object with chunk() method
        embedding_model: Model name for embeddings
        embedding_endpoint: API endpoint for embedding service
        top_k: Minimum number of top chunks to consider
        top_percentile: Percentile threshold for chunk selection
        fast_path: If True, try simple regex matching first
        
    Returns:
        String containing the identified reference sections
    """
    # 1) Read content
    if isinstance(text_or_path, (str, Path)) and str(text_or_path).endswith(".md"):
        txt = open(text_or_path, "r", encoding="utf-8").read()
    else:
        txt = str(text_or_path)

    # 2) Fast path: try simple regex matching first
    if fast_path:
        pattern = r"(?im)^(#+\s*(references|bibliography|works cited|literature cited|bibliographie|literaturverzeichnis)\b.*)$"
        m = re.search(pattern, txt)
        if m:
            lvl = len(m.group(1).split()[0])  # number of '#'
            end_pattern = rf"(?m)^#{{1,{lvl}}}\s+\S"
            end = re.search(end_pattern, txt[m.end():])
            return txt[m.start(): m.end()+end.start()] if end else txt[m.start():]

    # 3) Chunk and embed
    if chunker is None:
        from chonkie import LateChunker
        chunker = LateChunker.from_recipe("markdown", lang="en")
    chunks = chunker.chunk(txt)
    chunk_texts = [c.text for c in chunks]
    
    if not chunk_texts:
        return ""
        
    doc_emb = _get_embeddings(chunk_texts, embedding_model, embedding_endpoint)

    # 4) Create task-specific queries and get embeddings
    task_prefix = 'Find in the document that follows the pattern of the query'
    queries = [task_prefix + ': ' + q for q in REF_QUERIES]
    q_emb = _get_embeddings(queries, embedding_model, embedding_endpoint)

    # 5) Calculate similarities and add lexical bonuses
    try:
        # Normalize embeddings to prevent overflow
        q_emb_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)
        doc_emb_norm = doc_emb / (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-8)
        sims = cosine_similarity(q_emb_norm, doc_emb_norm).max(axis=0)  # [n_chunks]
    except (ValueError, RuntimeWarning) as e:
        # Fallback: use simple dot product similarity
        sims = np.array([np.max(np.dot(q_emb, doc_chunk)) for doc_chunk in doc_emb])
        sims = np.clip(sims, -1, 1)  # Ensure valid similarity range
    
    sims = sims + _lexical_bonus(chunk_texts)  # small lexical boost

    # 6) Pick candidates using gap detection with adaptive fallback
    sorted_indices = np.argsort(sims)[::-1]  # Descending order
    sorted_scores = sims[sorted_indices]
    
    min_candidates = max(1, min(1, len(chunks)))
    max_candidates = len(chunks) // 2
    
    # Try gap detection first
    cand = []
    if len(sorted_scores) > 1:
        gaps = np.diff(sorted_scores)  # Negative values (scores decreasing)
        if len(gaps) > 0:
            # Find the biggest drop (most negative gap)
            biggest_gap_idx = np.argmin(gaps)
            gap_size = abs(gaps[biggest_gap_idx])
            
            # Only use gap if it's significant (> 0.05)
            if gap_size > 0.05:
                threshold = sorted_scores[biggest_gap_idx + 1]
                cand = np.where(sims >= threshold)[0]
    
    # Fallback to adaptive threshold if gap detection fails or gives bad results
    if len(cand) == 0 or len(cand) > max_candidates:
        max_score = sims.max()
        threshold = max_score * 0.7
        cand = np.where(sims >= threshold)[0]
        
        # Adjust if still outside bounds
        if len(cand) < min_candidates:
            threshold = max_score * 0.5
            cand = np.where(sims >= threshold)[0]
        elif len(cand) > max_candidates:
            threshold = max_score * 0.85
            cand = np.where(sims >= threshold)[0]
    
    # Final safety: cap at max_candidates
    if len(cand) > max_candidates:
        # Take the highest scoring candidates
        top_indices = np.argsort(sims)[::-1][:max_candidates]
        cand = top_indices
    
    if len(cand) == 0:
        return ""

    # 7) Contiguity growth: expand around best index while scores don't drop too much
    best = cand[np.argmax(sims[cand])]
    picked = {int(best)}
    drop_tolerance = 0.06  # tolerance for score drop
    
    # Expand left
    i = best - 1
    while i >= 0 and sims[i] >= sims[best] - drop_tolerance:
        picked.add(int(i))
        i -= 1
    
    # Expand right
    i = best + 1
    while i < len(sims) and sims[i] >= sims[best] - drop_tolerance:
        picked.add(int(i))
        i += 1

    # 8) Join selected chunks
    selection = sorted(picked)
    return "\n\n".join(chunks[i].text for i in selection)


