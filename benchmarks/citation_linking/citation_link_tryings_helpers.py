"""Candidate caching and exact-ID evaluation for citation-linking experiments."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.citation_linking.analyze_argilla_annotations import metrics, safe_div
from benchmarks.citation_linking.build_annotation_dataset import normalize_id
from citation_index.core.models import Reference

logger = logging.getLogger(__name__)
KEY = ["ref_id", "index"]


def search_reference(row: dict[str, Any]) -> tuple[Reference, str]:
    """Use supplied parsing, falling back to the reference text when title is absent."""
    reference = Reference.model_validate(row.get("parsed_reference") or {})
    mode = "parsed_title"
    if not (reference.full_title or "").strip():
        reference.full_title = row["reference"].strip()
        mode = "raw_reference"
    if not reference.full_title:
        raise ValueError("A nonempty reference or parsed title is required")
    # Identifiers and gold labels must not influence metadata-only retrieval.
    reference.identifiers = []
    return reference, mode


def _check_keys(frame: pd.DataFrame) -> None:
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError("Expected unique, non-null (ref_id, index) keys")


def retrieve_candidates(
    tasks: pd.DataFrame,
    connectors: dict[str, Any],
    cache_dir: Path,
    *,
    top_k: int = 10,
    retrieval_version: str = "metadata-v1",
    pause_seconds: float = 0.1,
) -> pd.DataFrame:
    """Checkpoint each successful search atomically; failed searches remain retryable."""
    _check_keys(tasks)
    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50 for all three connectors")
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for position, row in enumerate(tasks.to_dict("records"), start=1):
        reference, query_mode = search_reference(row)
        query = reference.model_dump(mode="json", exclude_none=True)
        request = {
            "ref_id": row["ref_id"],
            "index": row["index"],
            "query": query,
            "query_mode": query_mode,
            "top_k": top_k,
            "retrieval_version": retrieval_version,
        }
        digest = hashlib.sha256(
            json.dumps(request, sort_keys=True).encode()
        ).hexdigest()
        path = cache_dir / f"{digest}.json"
        if path.exists():
            result = json.loads(path.read_text())
        else:
            try:
                raw = connectors[row["index"]].search(reference, top_k=top_k)
                if not isinstance(raw, list):
                    raise ValueError("Connector search must return a list")
                candidates = [
                    {
                        "rank": rank,
                        "candidate_id": normalize_id(candidate["id"], row["index"]),
                        "metadata": candidate,
                    }
                    for rank, candidate in enumerate(raw[:top_k], start=1)
                ]
                result = {
                    **request,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "status": "ok",
                    "n_candidates": len(candidates),
                    "candidates": candidates,
                }
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(result, ensure_ascii=False))
                temporary.replace(path)
            except Exception as exc:
                logger.exception(
                    "Candidate retrieval failed; ref_id=%s index=%s",
                    row["ref_id"],
                    row["index"],
                )
                raise RuntimeError(
                    f"Retrieval failed for {row['ref_id']} / {row['index']}; "
                    f"{type(exc).__name__}. Successful tasks are checkpointed."
                ) from None
            time.sleep(pause_seconds)
        records.append(result)
        if position % 50 == 0 or position == len(tasks):
            print(f"Candidates ready: {position}/{len(tasks)}", flush=True)
    return pd.DataFrame(records)


def _optional_id(value: Any, index: str) -> str | None:
    return None if pd.isna(value) else normalize_id(str(value), index)


def score_predictions(tasks: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Score exactly one predicted_id (or null abstention) per task, joined by key."""
    _check_keys(tasks)
    _check_keys(predictions)
    joined = tasks.merge(
        predictions[KEY + ["predicted_id"]],
        on=KEY,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Predictions must cover exactly the evaluated task keys")
    outcomes = []
    for row in joined.to_dict("records"):
        gold = _optional_id(row["gold_id"], row["index"])
        predicted = _optional_id(row["predicted_id"], row["index"])
        if gold is None:
            outcomes.append("correct_abstain" if predicted is None else "spurious_link")
        elif predicted is None:
            outcomes.append("missed_link")
        else:
            outcomes.append("correct_link" if predicted == gold else "wrong_link")
    return joined.drop(columns="_merge").assign(outcome=outcomes)


def summarize_scores(
    scored: pd.DataFrame, by: tuple[str, ...] = ("split", "index")
) -> pd.DataFrame:
    """Reuse annotation metrics; count wrong links as both FP and FN."""
    groups = scored.groupby(list(by), dropna=False) if by else [((), scored)]
    records = []
    for key, group in groups:
        key = key if isinstance(key, tuple) else (key,)
        result = metrics(group["outcome"].value_counts().to_dict())
        # Count-based F1 is zero when errors exist but no links are correct.
        result["f1"] = safe_div(
            2 * result["TP"], 2 * result["TP"] + result["FP"] + result["FN"]
        )
        records.append({**dict(zip(by, key)), **result})
    return pd.DataFrame(records)


def retrieval_scores(tasks: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Return per-task gold rank; null gold and absent candidates remain distinct."""
    _check_keys(tasks)
    _check_keys(candidates)
    joined = tasks.merge(
        candidates[KEY + ["status", "candidates", "n_candidates"]],
        on=KEY,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all() or not joined["status"].eq("ok").all():
        raise ValueError("Retrieval scores require successful searches for every task")
    ranks = []
    for row in joined.to_dict("records"):
        gold = _optional_id(row["gold_id"], row["index"])
        ranks.append(
            next(
                (c["rank"] for c in row["candidates"] if c["candidate_id"] == gold),
                None,
            )
            if gold is not None
            else None
        )
    return joined.drop(columns="_merge").assign(gold_rank=ranks)
