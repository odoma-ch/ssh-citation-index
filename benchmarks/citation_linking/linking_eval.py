"""Candidate caching, exact-ID scoring, offline reranking inputs and evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import time
from typing import Any, Callable

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


logger = logging.getLogger(__name__)


class InferenceError(RuntimeError):
    """Adapter-generated diagnostic containing no request or response content."""


UNRESOLVED_OPENALEX = {
    "10.1111_1467-6478.00057_instance-38",
    "excite_32211_33",
    "linkedbook_26",
    "linkedbook_5",
    "linkedbook_73",
}
FIELDS = {
    "full_title": "Title",
    "authors": "Authors",
    "editors": "Editors",
    "translator": "Translators",
    "publication_date_raw": "Publication date",
    "publication_year": "Year",
    "ref_type": "Publication type",
    "journal_title": "Container",
    "publisher": "Publisher",
    "publication_place": "Publication place",
    "volume": "Volume",
    "issue": "Issue",
    "pages": "Pages",
    "cited_range": "Cited pages",
}
PAIR_COLUMNS = KEY + [
    "candidate_id",
    "rank",
    "tie_order",
    "citation_text",
    "candidate_text",
    "clipped_fields",
]


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _text(value: Any) -> str:
    """Render known name/text shapes without serializing arbitrary metadata."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(dict.fromkeys(filter(None, map(_text, value))))
    if isinstance(value, dict):
        for key in (
            "name",
            "displayName",
            "display_name",
            "fullName",
            "literal",
            "text",
            "value",
        ):
            if value.get(key):
                return _text(value[key])
        return " ".join(
            filter(
                None,
                [
                    _text(
                        value.get("given")
                        or value.get("firstName")
                        or value.get("first_name")
                    ),
                    _text(value.get("middle_name")),
                    _text(value.get("name_link")),
                    _text(
                        value.get("family")
                        or value.get("lastName")
                        or value.get("surname")
                    ),
                ],
            )
        )
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return " ".join(str(value).split())
    return ""


def _claim_text(claims: dict, prop: str) -> list[str]:
    values = []
    for statement in claims.get(prop, []):
        if statement.get("rank") == "deprecated":
            continue
        value = statement.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, dict):
            if value.get("text"):
                values.append(value["text"])
            elif value.get("time"):
                values.append(value["time"].lstrip("+")[:4])
            else:
                # Entity IDs without cached name qualifiers are not bibliographic text.
                for qualifier in ("P1932", "P1810"):
                    names = statement.get("qualifiers", {}).get(qualifier, [])
                    values.extend(
                        n.get("datavalue", {}).get("value", "") for n in names
                    )
    return values


def candidate_fields(index: str, metadata: dict) -> dict:
    """Read bibliographic fields from frozen provider records without network lookups."""
    if index == "openalex":
        source = (metadata.get("primary_location") or {}).get("source") or {}
        biblio = metadata.get("biblio") or {}
        return {
            "full_title": metadata.get("title") or metadata.get("display_name"),
            "authors": [
                a.get("author", {}).get("display_name")
                for a in metadata.get("authorships", []) or []
            ],
            "publication_year": metadata.get("publication_year"),
            "ref_type": metadata.get("type"),
            "journal_title": (
                source.get("display_name")
                if source.get("type") in {"journal", "book series", "conference"}
                else None
            ),
            "volume": biblio.get("volume"),
            "issue": biblio.get("issue"),
            "pages": "-".join(
                str(biblio[k]) for k in ("first_page", "last_page") if biblio.get(k)
            ),
        }
    if index == "matilda":
        texts = metadata.get("texts") or []
        record = next((t for t in texts if isinstance(t, dict) and t.get("title")), {})
        support = record.get("support") or []
        return {
            "full_title": record.get("title"),
            "authors": record.get("author") or record.get("authors"),
            "editors": record.get("editor") or record.get("editors"),
            "translator": record.get("translator"),
            "publication_date_raw": record.get("publicationDate")
            or record.get("publishedDate"),
            "ref_type": record.get("type"),
            "publisher": record.get("publisher"),
            "journal_title": [
                s.get("value")
                for s in support
                if isinstance(s, dict) and s.get("type") == "title"
            ],
            "volume": record.get("volume"),
            "issue": record.get("issue"),
            "pages": record.get("pages"),
        }
    if index == "wikidata":
        claims = (
            metadata.get("claims")
            or (metadata.get("entity_data") or {}).get("claims")
            or {}
        )
        return {
            "full_title": _claim_text(claims, "P1476") or metadata.get("label"),
            "authors": _claim_text(claims, "P2093") or _claim_text(claims, "P50"),
            "editors": _claim_text(claims, "P98"),
            "translator": _claim_text(claims, "P655"),
            "publication_date_raw": _claim_text(claims, "P577"),
            "journal_title": _claim_text(claims, "P1433"),
            "publisher": _claim_text(claims, "P123"),
            "volume": _claim_text(claims, "P478"),
            "issue": _claim_text(claims, "P433"),
            "pages": _claim_text(claims, "P304"),
            "description": metadata.get("description"),
        }
    raise ValueError(f"Unsupported index: {index}")


def format_record(
    fields: dict, *, raw: str = "", field_limit: int = 1000
) -> tuple[str, int]:
    if field_limit < 1:
        raise ValueError("field_limit must be positive")
    lines, clipped = [], 0
    allowed = {"reference": "Citation", **FIELDS, "description": "Description"}
    for key, label in allowed.items():
        value = _text(raw if key == "reference" else fields.get(key))
        if not value:
            continue
        if len(value) > field_limit:
            value = value[:field_limit] + " [truncated]"
            clipped += 1
        lines.append(f"{label}: {value}")
    return "\n".join(lines), clipped


def prepare_pool(
    path: Path, *, seed: int = 42, field_limit: int = 1000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate supervision from model text and shuffle candidates independently of gold."""
    with path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    tasks = pd.DataFrame(rows)
    _check_keys(tasks)
    pairs = []
    for row in rows:
        if row["status"] != "ok" or row["n_candidates"] != len(row["candidates"]):
            raise ValueError("Pool requires complete successful retrieval rows")
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unexpected split")
        citation, clipped = format_record(
            row.get("parsed_reference") or {},
            raw=row["reference"],
            field_limit=field_limit,
        )
        if not citation:
            raise ValueError("Empty citation input")
        candidates = sorted(row["candidates"], key=lambda c: c["candidate_id"])
        random.Random(digest([seed, row["ref_id"], row["index"]])).shuffle(candidates)
        ids, ranks = set(), set()
        for order, candidate in enumerate(candidates):
            identifier = normalize_id(candidate["candidate_id"], row["index"])
            if (
                not identifier
                or identifier in ids
                or candidate["rank"] in ranks
                or candidate["rank"] < 1
            ):
                raise ValueError("Invalid or duplicate candidate ID/rank")
            ids.add(identifier)
            ranks.add(candidate["rank"])
            document, n_clipped = format_record(
                candidate_fields(row["index"], candidate["metadata"]),
                field_limit=field_limit,
            )
            if not document:
                raise ValueError("Empty candidate bibliographic metadata")
            pairs.append(
                {
                    **{k: row[k] for k in KEY},
                    "candidate_id": identifier,
                    "rank": candidate["rank"],
                    "tie_order": order,
                    "citation_text": citation,
                    "candidate_text": document,
                    "clipped_fields": clipped + n_clipped,
                }
            )
    return tasks, pd.DataFrame(pairs, columns=PAIR_COLUMNS)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    )
    temporary.replace(path)


def _validate_results(results: list[dict], expected: int) -> None:
    if not isinstance(results, list) or len(results) != expected:
        raise ValueError("Scorer must return exactly one result per pair")
    for result in results:
        score = result.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError("Scorer returned a non-finite or invalid score")


def run_reranking(
    tasks: pd.DataFrame,
    pairs: pd.DataFrame,
    scorer: Callable,
    output: Path,
    cache_dir: Path,
    *,
    model_info: dict,
    pool_path: Path,
    threshold: float | None = None,
    min_margin: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """Checkpoint successful task batches; publish metrics only after complete inference."""
    _check_keys(tasks)
    output.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "model": model_info,
        "pool_sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
        "input_sha256": digest(pairs.to_dict("records")),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
            + Path(__file__).with_name("rerankers.py").read_bytes()
        ).hexdigest(),
        "threshold": threshold,
        "min_margin": min_margin,
        "tasks": len(tasks),
        "pairs": len(pairs),
        "clipped_fields": int(pairs.clipped_fields.sum()),
        "new_task_batches": 0,
        "cached_task_batches": 0,
        "new_inference_seconds": 0.0,
    }
    _write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    scored = []
    key = None
    stage = "inference"
    try:
        for number, (key, group) in enumerate(pairs.groupby(KEY, sort=False), 1):
            inputs = list(zip(group.citation_text, group.candidate_text))
            cache_key = digest([model_info, manifest["implementation_sha256"], inputs])
            cache_path = cache_dir / f"{cache_key}.json"
            if cache_path.exists():
                results = json.loads(cache_path.read_text())["results"]
                manifest["cached_task_batches"] += 1
            else:
                batch_started = time.perf_counter()
                results = scorer(inputs)
                manifest["new_inference_seconds"] += time.perf_counter() - batch_started
                _validate_results(results, len(group))
                _write_json(cache_path, {"results": results})
                manifest["new_task_batches"] += 1
            _validate_results(results, len(group))
            for row, result in zip(group.to_dict("records"), results):
                scored.append(
                    {
                        **{
                            k: row[k]
                            for k in KEY + ["candidate_id", "rank", "tie_order"]
                        },
                        **result,
                    }
                )
            if number % 100 == 0:
                print(f"Scored {number} task batches", flush=True)
        scores = pd.DataFrame(
            scored,
            columns=KEY
            + ["candidate_id", "rank", "tie_order", "score"]
            + sorted(
                {k for r in scored for k in r}
                - set(KEY + ["candidate_id", "rank", "tie_order", "score"])
            ),
        )
        stage = "evaluation"
        key = None
        tables = evaluate_reranking(
            tasks, scores, threshold=threshold, min_margin=min_margin
        )
        for name, table in tables.items():
            if name in {"candidate_scores", "predictions"}:
                table.to_json(
                    output / f"{name}.jsonl",
                    orient="records",
                    lines=True,
                    force_ascii=False,
                )
            else:
                table.to_csv(output / f"{name}.csv", index=False)
        truncation = [r.get("truncated") for r in scored]
        manifest.update(
            status="complete",
            elapsed_seconds=time.perf_counter() - started,
            known_truncated_pairs=sum(v is True for v in truncation),
            unknown_truncation_pairs=sum(v is None for v in truncation),
        )
        _write_json(output / "manifest.json", manifest)
        return tables
    except Exception as exc:
        detail = str(exc) if isinstance(exc, InferenceError) else type(exc).__name__
        manifest.update(
            status="failed",
            error=type(exc).__name__,
            error_detail=detail,
            stage=stage,
            task=list(key) if key is not None else None,
            elapsed_seconds=time.perf_counter() - started,
        )
        _write_json(output / "manifest.json", manifest)
        logger.exception(
            "Reranking failed; model=%s stage=%s task=%s",
            model_info.get("model_id"),
            stage,
            key,
            exc_info=(
                RuntimeError,
                RuntimeError(detail),
                exc.__traceback__,
            ),
        )
        raise RuntimeError(
            f"Reranking failed ({detail}); see manifest and checkpoints"
        ) from None


def evaluate_reranking(
    tasks: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    threshold: float | None = None,
    min_margin: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """Rank every candidate; compute all-data/dev metrics without treating misses as negatives."""
    _check_keys(tasks)
    if (
        min_margin < 0
        or not math.isfinite(min_margin)
        or (threshold is not None and not math.isfinite(threshold))
    ):
        raise ValueError(
            "Threshold and margin must be finite; margin must be nonnegative"
        )
    expected = {
        (r.ref_id, r.index, normalize_id(c["candidate_id"], r.index))
        for r in tasks.itertuples()
        for c in r.candidates
    }
    actual = list(scores[KEY + ["candidate_id"]].itertuples(index=False, name=None))
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Scores must cover exactly the pool's candidate keys")
    _validate_results(scores.to_dict("records"), len(scores))
    listwise = "listwise_selected" in scores.columns
    if listwise:
        for _, group in scores.groupby(KEY):
            flags = group.listwise_selected.tolist()
            none_values = group.none_score.tolist()
            if (
                any(type(flag) is not bool for flag in flags)
                or sum(flags) > 1
                or len(set(none_values)) != 1
                or not math.isfinite(none_values[0])
                or not 0 <= none_values[0] <= 1
                or not group.score.between(0, 1).all()
                or not math.isclose(
                    group.score.sum() + none_values[0], 1.0, abs_tol=1e-3
                )
            ):
                raise ValueError("Invalid listwise choice distribution")
            chosen = group.loc[group.listwise_selected, "score"]
            chosen_score = float(chosen.iloc[0]) if len(chosen) else none_values[0]
            if chosen_score < max(group.score.max(), none_values[0]):
                raise ValueError(
                    "Listwise selected option must have maximum probability"
                )
    tie_columns = ["listwise_selected", "tie_order"] if listwise else ["tie_order"]
    ranked = scores.sort_values(
        KEY + ["score"] + tie_columns,
        ascending=[True, True, False] + ([False, True] if listwise else [True]),
    ).copy()
    ranked["reranked_position"] = ranked.groupby(KEY).cumcount() + 1
    groups = {key: group for key, group in ranked.groupby(KEY)}
    predictions, diagnostics = [], []
    for row in tasks.itertuples():
        group = groups.get((row.ref_id, row.index), ranked.iloc[:0])
        best = group.iloc[0] if len(group) else None
        margin = (
            float(group.iloc[0].score - group.iloc[1].score) if len(group) > 1 else None
        )
        accepted = (
            best is not None
            and (not listwise or bool(best.listwise_selected))
            and (threshold is None or best.score >= threshold)
            and (margin is None or margin >= min_margin)
        )
        predictions.append(
            {
                "ref_id": row.ref_id,
                "index": row.index,
                "predicted_id": best.candidate_id if accepted else None,
                "top_score": float(best.score) if best is not None else None,
                "margin": margin,
                **(
                    {"none_score": float(best.none_score) if best is not None else None}
                    if listwise
                    else {}
                ),
            }
        )
        gold = normalize_id(row.gold_id, row.index) if pd.notna(row.gold_id) else None
        hit = group.loc[group.candidate_id.eq(gold)]
        gold_rank = int(hit.iloc[0].reranked_position) if len(hit) else None
        original_rank = next(
            (
                c["rank"]
                for c in row.candidates
                if normalize_id(c["candidate_id"], row.index) == gold
            ),
            None,
        )
        diagnostics.append(
            {
                "ref_id": row.ref_id,
                "index": row.index,
                "split": row.split,
                "has_gold": gold is not None,
                "gold_retrieved": bool(len(hit)),
                "top1": int(gold_rank == 1),
                "rr": 1 / gold_rank if gold_rank else 0,
                "retrieval_top1": int(original_rank == 1),
                "retrieval_rr": 1 / original_rank if original_rank else 0,
            }
        )
    predictions = pd.DataFrame(predictions)
    diagnostics = pd.DataFrame(diagnostics)
    excluded = (
        tasks["index"].eq("openalex")
        & tasks.ref_id.isin(UNRESOLVED_OPENALEX)
        & tasks.gold_id.isna()
    )
    eligible = tasks.loc[~excluded]
    scored_tasks = score_predictions(
        eligible, predictions.merge(eligible[KEY], on=KEY, validate="one_to_one")
    )
    tables = {"candidate_scores": ranked, "predictions": predictions}
    for scope, split in (("all", None), ("dev", "validation")):
        selected = (
            scored_tasks
            if split is None
            else scored_tasks.loc[scored_tasks.split.eq(split)]
        )
        selected_tasks = tasks if split is None else tasks.loc[tasks.split.eq(split)]
        summary = pd.concat(
            [
                summarize_scores(selected, by=()).assign(index="all"),
                summarize_scores(selected, by=("index",)),
            ],
            ignore_index=True,
        )
        excluded_counts = (
            tasks.loc[excluded & tasks.index.isin(selected_tasks.index)]
            .groupby("index")
            .size()
        )
        summary["excluded_unresolved"] = [
            (
                int(excluded_counts.sum())
                if idx == "all"
                else int(excluded_counts.get(idx, 0))
            )
            for idx in summary["index"]
        ]
        tables[f"scores_{scope}"] = summary
        positive = diagnostics.loc[
            diagnostics.has_gold & (diagnostics.split.eq(split) if split else True)
        ]
        ranking_rows = []
        for index in ["all", *sorted(selected_tasks["index"].unique())]:
            group = (
                positive
                if index == "all"
                else positive.loc[positive["index"].eq(index)]
            )
            for subset in ("all_positive", "gold_retrieved"):
                g = (
                    group
                    if subset == "all_positive"
                    else group.loc[group.gold_retrieved]
                )
                ranking_rows.append(
                    {
                        "index": index,
                        "subset": subset,
                        "n_eval": len(g),
                        "top1": g.top1.mean(),
                        "mrr": g.rr.mean(),
                        "retrieval_top1": g.retrieval_top1.mean(),
                        "retrieval_mrr": g.retrieval_rr.mean(),
                    }
                )
        tables[f"ranking_{scope}"] = pd.DataFrame(ranking_rows)
    return tables
