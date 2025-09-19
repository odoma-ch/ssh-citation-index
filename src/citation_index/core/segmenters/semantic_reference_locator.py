"""Semantic reference section locator using embeddings and lexical cues."""

from __future__ import annotations

import re
import logging
import threading
import time
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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
    "doi: 10.1000/xyz", "https://doi.org/10.1000/xyz", "arXiv:2401.01234", "ISSN 1234-5678", "ISBN 978-3-16-148410-0"
]

# Lexical cues for reference sections
CUE_RX = re.compile(
    r"(doi:|10\.\d{4,}/|arxiv:|journal of|vol\.|no\.|pp\.|\bet al\.|\(\d{4}\)|\b19\d{2}\b|\b20\d{2}\b|issn|isbn|edds|ed|transl)",
    re.IGNORECASE
)

# Thread-local storage for HTTP sessions
_thread_local = threading.local()

def _get_session():
    """Get a thread-local HTTP session for embedding requests with retry logic."""
    if not hasattr(_thread_local, 'session'):
        _thread_local.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,  # Total number of retries
            backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
            status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
            allowed_methods=["POST"]  # Only retry POST requests
        )
        
        # Mount adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        _thread_local.session.mount("http://", adapter)
        _thread_local.session.mount("https://", adapter)
        
        # Configure session for better performance
        _thread_local.session.headers.update({'Content-Type': 'application/json'})
    return _thread_local.session


def _get_embeddings(texts: List[str], model: str, endpoint: str, batch_size: int = 50, max_timeout: int = 120) -> np.ndarray:
    """Get embeddings for texts using the specified model and endpoint with batching and adaptive timeout."""
    if not texts:
        return np.array([])
    
    # Calculate adaptive timeout based on number of texts
    # Base timeout of 30s + 2s per text, capped at max_timeout
    timeout = min(max_timeout, max(30, 30 + len(texts) * 2))
    logging.debug(f"Using timeout of {timeout}s for {len(texts)} texts")
    
    try:
        # For large batches, process in smaller chunks to avoid timeouts
        if len(texts) > batch_size:
            logging.info(f"Processing {len(texts)} texts in batches of {batch_size}")
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                logging.debug(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                batch_embeddings = _get_embeddings_single_batch(
                    batch_texts, model, endpoint, timeout
                )
                all_embeddings.append(batch_embeddings)
                
                # Small delay between batches to avoid overwhelming the service
                if i + batch_size < len(texts):
                    time.sleep(0.1)
            
            return np.vstack(all_embeddings)
        else:
            return _get_embeddings_single_batch(texts, model, endpoint, timeout)
            
    except Exception as e:
        logging.error(f"Error getting embeddings for {len(texts)} texts: {e}")
        raise


def _get_embeddings_single_batch(texts: List[str], model: str, endpoint: str, timeout: int) -> np.ndarray:
    """Get embeddings for a single batch of texts."""
    session = _get_session()
    
    # Retry logic with exponential backoff
    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries + 1):
        try:
            r = session.post(
                endpoint, 
                json={"model": model, "input": texts}, 
                timeout=timeout
            )
            r.raise_for_status()
            data = r.json()["data"]
            
            if all("index" in d for d in data):
                data = sorted(data, key=lambda x: x["index"])
            
            embeddings = np.array([d["embedding"] for d in data], dtype=np.float64)
            
            # Check for invalid values and log warnings
            if np.any(np.isnan(embeddings)):
                logging.warning(f"NaN values found in embeddings for {len(texts)} texts")
            if np.any(np.isinf(embeddings)):
                logging.warning(f"Infinite values found in embeddings for {len(texts)} texts")
                
            return embeddings
            
        except requests.exceptions.Timeout as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logging.warning(f"Timeout on attempt {attempt + 1}/{max_retries + 1}, retrying in {delay}s...")
                time.sleep(delay)
                continue
            else:
                logging.error(f"Final timeout after {max_retries + 1} attempts for {len(texts)} texts")
                raise
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logging.warning(f"Request error on attempt {attempt + 1}/{max_retries + 1}: {e}, retrying in {delay}s...")
                time.sleep(delay)
                continue
            else:
                logging.error(f"Final request error after {max_retries + 1} attempts: {e}")
                raise
                
        except Exception as e:
            logging.error(f"Unexpected error getting embeddings: {e}")
            raise


def _lexical_bonus(chunk_texts: List[str], bonus_per_hit: float = 0.02, max_bonus: float = 0.1) -> np.ndarray:
    """Calculate lexical bonus scores based on reference-specific cues."""
    bonuses = []
    for t in chunk_texts:
        hits = len(CUE_RX.findall(t))
        bonuses.append(min(max_bonus, hits * bonus_per_hit))
    return np.array(bonuses, dtype=float)

# best
# gap_size_threshold: float = 0.05,   
# drop_tolerance: float = 0.03,
def locate_reference_sections_semantic(
    text_or_path: Union[str, Path],
    chunker=None,
    chunks=None,
    embedding_model: str = "intfloat/multilingual-e5-large-instruct",
    embedding_endpoint: str = "http://0.0.0.0:7997/embeddings",
    fast_path: bool = False,
    gap_size_threshold: float = 0.1,   
    drop_tolerance: float = 0.03,
    greedy: bool = True,
    batch_size: int = 50,
    max_timeout: int = 120
) -> str:
    """Locate reference sections using semantic similarity and lexical cues.
    
    This function uses embedding-based semantic search combined with lexical cues
    to identify reference sections in academic documents. It's more robust than
    rule-based approaches and works across different document formats and languages.
    
    Args:
        text_or_path: Input text string or path to markdown file
        chunker: Text chunker object with chunk() method. Ignored if chunks parameter is provided.
        chunks: Pre-computed chunks from the text. If provided, chunker is ignored.
                Each chunk should have a .text attribute.
        embedding_model: Model name for embeddings
        embedding_endpoint: API endpoint for embedding service
        fast_path: If True, try simple regex matching first
        gap_size_threshold: Minimum gap size to trigger gap-based candidate selection (default: 0.1)
        drop_tolerance: Maximum score drop allowed during contiguous expansion (default: 0.03)
        greedy: if true, use contiguous expansion to pick the best chunks
        batch_size: Maximum number of texts to process in a single embedding request (default: 50)
        max_timeout: Maximum timeout in seconds for embedding requests (default: 120)
        
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
    if chunks is not None:
        # Use pre-computed chunks
        chunk_texts = [c.text for c in chunks]
    else:
        # Fallback: create chunker and chunk on-the-fly
        if chunker is None:
            from chonkie import LateChunker
            chunker = LateChunker.from_recipe("markdown", lang="en")
        chunks = chunker.chunk(txt)
        chunk_texts = [c.text for c in chunks]
    
    if not chunk_texts:
        return ""
    
    # Log chunk information
    logging.info(f"Total chunks created: {len(chunk_texts)}")
    for i, chunk_text in enumerate(chunk_texts):
        chunk_preview = chunk_text[:100].replace('\n', ' ') + ('...' if len(chunk_text) > 100 else '')
        logging.debug(f"Chunk {i}: {chunk_preview}")
    
    # Get embeddings for document chunks
    logging.debug(f"Getting embeddings for {len(chunk_texts)} chunks...")
    try:
        doc_emb = _get_embeddings(chunk_texts, embedding_model, embedding_endpoint, batch_size, max_timeout)
        logging.debug(f"Document embeddings shape: {doc_emb.shape}")

        # 4) Create task-specific queries and get embeddings
        task_prefix = 'Find in the document that follows the pattern of the query'
        queries = [task_prefix + ': ' + q for q in REF_QUERIES]
        q_emb = _get_embeddings(queries, embedding_model, embedding_endpoint, batch_size, max_timeout)
        
    except Exception as e:
        logging.error(f"Failed to get embeddings after all retries: {e}")
        logging.info("Falling back to lexical-only matching")
        
        # Fallback: use only lexical cues without embeddings
        lexical_scores = _lexical_bonus(chunk_texts)
        if lexical_scores.max() == 0:
            # No lexical cues found, return empty
            logging.warning("No reference sections found using lexical fallback")
            return ""
            
        # Use lexical scores to find reference sections
        threshold = lexical_scores.max() * 0.5
        cand = np.where(lexical_scores >= threshold)[0]
        
        if len(cand) == 0:
            return ""
            
        cand = sorted(cand)
        logging.info(f"Selected chunks using lexical fallback: {cand} out of {len(chunks)} total chunks")
        return "\n\n".join(chunks[i].text for i in cand)

    # 5) Calculate similarities and add lexical bonuses
    logging.debug(f"Calculating similarities between {q_emb.shape[0]} queries and {doc_emb.shape[0]} chunks...")
    
    try:
        # Check for invalid values in embeddings
        if np.any(np.isnan(q_emb)) or np.any(np.isnan(doc_emb)):
            raise ValueError("NaN values detected in embeddings")
        if np.any(np.isinf(q_emb)) or np.any(np.isinf(doc_emb)):
            raise ValueError("Infinite values detected in embeddings")
        
        # Normalize embeddings with better numerical stability
        q_norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
        doc_norms = np.linalg.norm(doc_emb, axis=1, keepdims=True)
        
        # Prevent division by zero
        q_norms = np.maximum(q_norms, 1e-12)
        doc_norms = np.maximum(doc_norms, 1e-12)
        
        q_emb_norm = q_emb / q_norms
        doc_emb_norm = doc_emb / doc_norms
        
        # Use numpy's safer matrix multiplication to avoid sklearn warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            similarity_matrix = np.dot(q_emb_norm, doc_emb_norm.T)
            sims = np.nanmax(similarity_matrix, axis=0)  # [n_chunks]
            
        # Handle any remaining invalid values
        sims = np.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0)
        sims = np.clip(sims, -1, 1)  # Ensure valid similarity range
        
        logging.debug(f"Similarity calculation successful, range: [{sims.min():.3f}, {sims.max():.3f}]")
        
    except (ValueError, RuntimeWarning, np.linalg.LinAlgError) as e:
        logging.warning(f"Similarity calculation failed ({e}), using fallback method")
        # Fallback: use simple normalized dot product similarity
        with np.errstate(all='ignore'):
            sims = np.array([
                np.max([
                    np.dot(q_vec / (np.linalg.norm(q_vec) + 1e-12), 
                          doc_vec / (np.linalg.norm(doc_vec) + 1e-12))
                    for q_vec in q_emb
                ])
                for doc_vec in doc_emb
            ])
        sims = np.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0)
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
            
            # Only use gap if it's significant
            if gap_size > gap_size_threshold:
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
       # sort by id
    cand = sorted(cand)
    cand = [int(c) for c in cand]
    logging.info(f"Selected chunks: {cand} out of {len(chunks)} total chunks, best match: {cand[np.argmax(sims[cand])]}")
    if not greedy:
        return "\n\n".join(chunks[i].text for i in cand)
    else:
        # 7) Contiguity growth: expand around best index while scores don't drop too much
        best = cand[np.argmax(sims[cand])]
        picked = {int(best)}
    
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
        picked = picked.union(cand)
        selection = sorted(picked)
        logging.info(f"Selected chunks after greedy expansion: {selection} out of {len(chunks)} total chunks, best match: {selection[np.argmax(sims[selection])]}")
    
        return "\n\n".join(chunks[i].text for i in selection)
    
 

    


def pre_chunk_text(text: str, chunker=None) -> List:
    """
    Pre-chunk text using the provided chunker for later use in parallel processing.
    
    This function allows chunking to be done sequentially (avoiding threading issues)
    before parallel LLM processing begins.
    
    Args:
        text: Input text to chunk
        chunker: Text chunker object with chunk() method. If None, creates a LateChunker.
        
    Returns:
        List of chunks, each with a .text attribute
    """
    if chunker is None:
        from chonkie import LateChunker
        chunker = LateChunker.from_recipe("markdown", lang="en")
    
    return chunker.chunk(text)


