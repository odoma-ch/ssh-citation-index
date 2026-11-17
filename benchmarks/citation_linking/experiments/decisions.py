"""Search-first and API-matching controls, and rejection thresholds, on frozen pools."""

import hashlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import requests

from benchmarks.citation_linking.experiments.linking_eval import (
    _check_keys,
    digest,
    evaluate_reranking,
    KEY,
    normalize_id,
    score_predictions,
    summarize_scores,
    unresolved,
)
from benchmarks.citation_linking.experiments.rerankers import (
    LABELS,
    validate_probabilities,
)
from citation_index.core.models import Reference
from citation_index.pipelines import citation_linking
from citation_index.utils import reference_matching


def _wikidata_ids(tasks: pd.DataFrame) -> set[str]:
    required = set()
    for row in tasks.loc[tasks["index"].eq("wikidata")].itertuples():
        for candidate in row.candidates:
            claims = candidate["metadata"].get("claims") or {}
            for prop in ("P50", "P1433"):
                for statement in claims.get(prop, []):
                    value = (
                        statement.get("mainsnak", {}).get("datavalue", {}).get("value")
                    )
                    if isinstance(value, dict) and value.get("id"):
                        required.add(value["id"])
    return required


def wikidata_labels(tasks: pd.DataFrame, cache_path: Path) -> dict:
    """Cache English entity labels needed by the API's Wikidata converter."""
    labels = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    required = _wikidata_ids(tasks)
    missing = sorted(required - labels.keys())
    with requests.Session() as session:
        session.headers["User-Agent"] = (
            "CitationIndex/1.0 (https://github.com/citation-index) wikidata-connector"
        )
        for start in range(0, len(missing), 50):
            batch = missing[start : start + 50]
            try:
                response = session.get(
                    "https://www.wikidata.org/w/api.php",
                    params={
                        "action": "wbgetentities",
                        "ids": "|".join(batch),
                        "props": "labels",
                        "languages": "en",
                        "format": "json",
                    },
                    timeout=30,
                )
                response.raise_for_status()
                entities = response.json()["entities"]
                if not set(batch).issubset(entities):
                    raise ValueError("Incomplete entity response")
                resolved = {
                    qid: entities[qid].get("labels", {}).get("en", {}).get("value")
                    for qid in batch
                }
            except Exception as exc:
                raise RuntimeError(
                    f"Wikidata label lookup failed ({type(exc).__name__})"
                ) from None
            labels.update(resolved)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(labels, ensure_ascii=False, indent=2))
            temporary.replace(cache_path)
    return labels


def baseline_predictions(tasks: pd.DataFrame, labels: dict) -> dict[str, pd.DataFrame]:
    """Reuse API conversion and selection, replacing search with the frozen pool."""
    _check_keys(tasks)
    if _wikidata_ids(tasks) - labels.keys():
        raise ValueError("Required Wikidata labels have not been cached")
    # Conversion methods need no authenticated sessions; search and label IO are replaced.
    mappers = {
        name: cls.__new__(cls)
        for name, cls in citation_linking.CONNECTOR_TYPES.items()
        if name in set(tasks["index"])
    }
    if "wikidata" in mappers:

        def cached_labels(qids, language="en"):
            if language != "en" or any(qid not in labels for qid in qids):
                raise ValueError("Required Wikidata labels have not been cached")
            return {qid: labels[qid] for qid in qids if labels[qid]}

        mappers["wikidata"]._get_labels_for_qids = cached_labels
    results = {"search_first": [], "api_match": []}
    for row in tasks.itertuples():
        candidates = [
            dict(c["metadata"], id=c["candidate_id"])
            for c in sorted(row.candidates, key=lambda c: c["rank"])
        ]
        connector = SimpleNamespace(
            search=lambda reference, top_k: candidates,
            _result_to_reference=mappers[row.index]._result_to_reference,
        )
        selected = None
        if row.parsed_reference is not None:
            reference = Reference.model_validate(row.parsed_reference)
            selected = citation_linking._link_one(
                reference, row.index, connector, len(candidates)
            )["id"]
        for method, identifier in (
            ("search_first", candidates[0]["id"] if candidates else None),
            ("api_match", selected),
        ):
            results[method].append(
                {
                    "ref_id": row.ref_id,
                    "index": row.index,
                    "predicted_id": (
                        normalize_id(identifier, row.index) if identifier else None
                    ),
                }
            )
    return {
        method: pd.DataFrame(rows, columns=KEY + ["predicted_id"])
        for method, rows in results.items()
    }


def evaluate_baselines(
    tasks: pd.DataFrame, labels: dict, pool_path: Path, run_dir: Path
) -> None:
    """Save all/dev decision metrics with the same unresolved-gold exclusions as GTE."""
    eligible = tasks.loc[~(unresolved(tasks) & tasks.gold_id.isna())]
    source_paths = [
        Path(__file__),
        Path(citation_linking.__file__),
        Path(reference_matching.__file__),
    ]
    source_paths += [
        Path(inspect.getfile(citation_linking.CONNECTOR_TYPES[name]))
        for name in sorted(tasks["index"].unique())
    ]
    for method, predictions in baseline_predictions(tasks, labels).items():
        output = run_dir / method / pool_path.stem
        output.mkdir(parents=True, exist_ok=True)
        (output / "manifest.json").write_text(json.dumps({"status": "running"}))
        scored = score_predictions(
            eligible, predictions.merge(eligible[KEY], on=KEY, validate="one_to_one")
        )
        predictions.to_json(output / "predictions.jsonl", orient="records", lines=True)
        for scope in ("all", "dev"):
            selected = (
                scored if scope == "all" else scored.loc[scored.split.eq("validation")]
            )
            summary = pd.concat(
                [
                    summarize_scores(selected, by=()).assign(index="all"),
                    summarize_scores(selected, by=("index",)),
                ],
                ignore_index=True,
            )
            summary.to_csv(output / f"scores_{scope}.csv", index=False)
        manifest = {
            "status": "complete",
            "model": {"family": "baseline", "mode": method},
            "pool_sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
            "implementation_sha256": hashlib.sha256(
                b"".join(p.read_bytes() for p in source_paths)
            ).hexdigest(),
            "wikidata_labels_sha256": hashlib.sha256(
                json.dumps(labels, sort_keys=True).encode()
            ).hexdigest(),
            "tasks": len(tasks),
            "missing_parsed_reference": int(tasks.parsed_reference.isna().sum()),
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


def compact_comparison(
    run_dir: Path, methods: list[str], *, index: str = "all"
) -> pd.DataFrame:
    """Show one bounded decision table; retain detailed per-index metrics on disk."""
    rows = []
    for pool in ("search_top10", "search_perfect"):
        for scope in ("all", "dev"):
            for method in methods:
                directory = run_dir / method / pool
                manifest_path = directory / "manifest.json"
                if not manifest_path.exists():
                    continue
                manifest = json.loads(manifest_path.read_text())
                if manifest["status"] != "complete":
                    continue
                pool_path = run_dir.parent.parent / f"{pool}.jsonl"
                if (
                    manifest["pool_sha256"]
                    != hashlib.sha256(pool_path.read_bytes()).hexdigest()
                ):
                    raise ValueError("Comparison pool differs from saved run")
                table = pd.read_csv(directory / f"scores_{scope}.csv")
                for row in table.loc[table["index"].eq(index)].itertuples():
                    rows.append(
                        {
                            "pool": "normal" if pool == "search_top10" else "oracle",
                            "scope": scope,
                            "method": method,
                            "correct": row.correct_link,
                            "precision": row.precision,
                            "recall": row.recall,
                            "f1": row.f1,
                        }
                    )
    return pd.DataFrame(
        rows,
        columns=["pool", "scope", "method", "correct", "precision", "recall", "f1"],
    )


def evaluate_rejection(
    tasks: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    threshold: float | None = None,
    require_match_label: bool = False,
) -> dict[str, pd.DataFrame]:
    """Gate the top-ranked candidate; a failed gate never promotes a lower candidate."""
    tables = evaluate_reranking(tasks, scores, threshold=threshold)
    if not require_match_label:
        return tables
    if "label_scores" not in scores or "listwise_selected" in scores:
        raise ValueError("Match-label rejection requires pairwise label probabilities")
    for row in scores.itertuples():
        validate_probabilities(row.label_scores, set(LABELS))
        if not math.isclose(row.score, row.label_scores["match"], abs_tol=1e-8):
            raise ValueError("Candidate score must equal its match probability")
    best = tables["candidate_scores"].drop_duplicates(KEY).copy()
    best["match_label_wins"] = best.label_scores.map(
        lambda p: p["match"] > max(p["different"], p["uncertain"])
    )
    predictions = tables["predictions"].merge(
        best[KEY + ["match_label_wins"]], on=KEY, how="left", validate="one_to_one"
    )
    predictions["match_label_wins"] = predictions.match_label_wins.eq(True)
    predictions.loc[~predictions.match_label_wins, "predicted_id"] = None
    tables["predictions"] = predictions
    eligible = tasks.loc[~(unresolved(tasks) & tasks.gold_id.isna())]
    scored = score_predictions(
        eligible, predictions.merge(eligible[KEY], on=KEY, validate="one_to_one")
    )
    for scope in ("all", "dev"):
        selected = (
            scored if scope == "all" else scored.loc[scored.split.eq("validation")]
        )
        summary = pd.concat(
            [
                summarize_scores(selected, by=()).assign(index="all"),
                summarize_scores(selected, by=("index",)),
            ],
            ignore_index=True,
        )
        tables[f"scores_{scope}"] = summary.merge(
            tables[f"scores_{scope}"][["index", "excluded_unresolved"]],
            on="index",
            validate="one_to_one",
        )
    return tables


def apply_saved_decisions(
    run_dir: Path,
    model_name: str,
    pools: dict,
    pool_paths: dict,
    *,
    threshold: float | None = None,
    require_match_label: bool = False,
) -> None:
    """Apply model-cell parameters to its cached scores and canonical result files."""
    prepared = {}
    for pool_name, (tasks, pairs) in pools.items():
        source = run_dir / model_name / pool_name
        manifest = json.loads((source / "manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError(f"{model_name}/{pool_name}: inference is incomplete")
        if (
            manifest["pool_sha256"]
            != hashlib.sha256(pool_paths[pool_name].read_bytes()).hexdigest()
        ):
            raise ValueError("The pool differs from the saved inference run")
        if manifest["input_sha256"] != digest(pairs.to_dict("records")):
            raise ValueError("The formatted inputs differ from the saved inference run")
        score_path = source / "candidate_scores.jsonl"
        scores = pd.read_json(score_path, lines=True, convert_dates=False)
        tables = evaluate_rejection(
            tasks, scores, threshold=threshold, require_match_label=require_match_label
        )
        manifest.update(
            threshold=threshold,
            min_margin=0.0,
            decision_policy={
                "threshold": threshold,
                "require_match_label": require_match_label,
                "label_ties": "reject" if require_match_label else "not_applicable",
                "parameter_source": "model_cell",
                "scores_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
                "implementation_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
        )
        prepared[source] = (manifest, tables)
    for source, (manifest, tables) in prepared.items():
        manifest["status"] = "running"
        (source / "manifest.json").write_text(json.dumps(manifest, indent=2))
        for name, table in tables.items():
            if name == "candidate_scores":
                continue
            if name == "predictions":
                table.to_json(
                    source / "predictions.jsonl", orient="records", lines=True
                )
            else:
                table.to_csv(source / f"{name}.csv", index=False)
        manifest["status"] = "complete"
        (source / "manifest.json").write_text(json.dumps(manifest, indent=2))


def select_threshold(
    tasks: pd.DataFrame, scores: pd.DataFrame, *, require_match_label: bool = False
) -> tuple[float, pd.DataFrame]:
    """Maximize dev F1; break ties by precision, then the stricter threshold."""
    if "listwise_selected" in scores:
        raise ValueError("This policy selects thresholds for pairwise scorers only")
    dev = tasks.loc[
        tasks.split.eq("validation") & ~(unresolved(tasks) & tasks.gold_id.isna())
    ]
    if dev.empty or not dev.gold_id.notna().any():
        raise ValueError(
            "Threshold selection requires dev tasks with positive gold targets"
        )
    dev_scores = scores.merge(dev[KEY], on=KEY, validate="many_to_one")
    predictions = evaluate_rejection(
        dev, dev_scores, require_match_label=require_match_label
    )["predictions"]
    scored = score_predictions(dev, predictions).merge(
        predictions[KEY + ["top_score"]], on=KEY, validate="one_to_one"
    )
    values = sorted(scored.top_score.dropna().unique())
    if not values:
        raise ValueError("Threshold selection requires scored dev candidates")
    reject_all = math.nextafter(float(values[-1]), math.inf)
    if not math.isfinite(reject_all):
        raise ValueError("Cannot represent a finite reject-all threshold")
    n_gold = int(dev.gold_id.notna().sum())
    rows = []
    for threshold in [*values, reject_all]:
        accepted = scored.predicted_id.notna() & scored.top_score.ge(threshold)
        n_links = int(accepted.sum())
        tp = int((accepted & scored.outcome.eq("correct_link")).sum())
        rows.append(
            {
                "threshold": float(threshold),
                "accepted": n_links,
                "TP": tp,
                "FP": n_links - tp,
                "FN": n_gold - tp,
                "precision": tp / n_links if n_links else 0.0,
                "recall": tp / n_gold,
                "f1": 2 * tp / (n_links + n_gold),
            }
        )
    sweep = pd.DataFrame(rows)
    best = sweep.sort_values(["f1", "precision", "threshold"], ascending=False).iloc[0]
    sweep["selected"] = sweep.threshold.eq(best.threshold)
    return float(best.threshold), sweep


def evaluate_saved_rejection(
    run_dir: Path,
    model_name: str,
    pools: dict,
    pool_paths: dict,
    *,
    require_match_label: bool = False,
) -> dict:
    """Freeze a normal-dev threshold and export separate *_reject all/dev runs."""
    sources = {}
    for pool_name, (tasks, pairs) in pools.items():
        source = run_dir / model_name / pool_name
        manifest = json.loads((source / "manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError(f"{model_name}/{pool_name}: inference is incomplete")
        if (
            manifest["pool_sha256"]
            != hashlib.sha256(pool_paths[pool_name].read_bytes()).hexdigest()
        ):
            raise ValueError("The pool differs from the saved inference run")
        if manifest["input_sha256"] != digest(pairs.to_dict("records")):
            raise ValueError("The formatted inputs differ from the saved inference run")
        score_path = source / "candidate_scores.jsonl"
        scores = pd.read_json(score_path, lines=True, convert_dates=False)
        # Validate every pool before writing any policy results.
        evaluate_rejection(tasks, scores, require_match_label=require_match_label)
        sources[pool_name] = (manifest, scores, score_path)
    normal_tasks = pools["search_top10"][0]
    threshold, sweep = select_threshold(
        normal_tasks,
        sources["search_top10"][1],
        require_match_label=require_match_label,
    )
    selected = sweep.loc[sweep.selected].iloc[0]
    policy = {
        "source_method": model_name,
        "result_method": (
            f"{model_name}_match_reject"
            if require_match_label
            else f"{model_name}_reject"
        ),
        "require_match_label": require_match_label,
        "label_ties": "reject" if require_match_label else "not_applicable",
        "selection_status": "selected" if selected.TP > 0 else "no_correct_dev_links",
        "dev_accepted": int(selected.accepted),
        "dev_correct_links": int(selected.TP),
        "threshold": threshold,
        "min_margin": 0.0,
        "selection_pool": "search_top10",
        "selection_split": "validation",
        "objective": "max_f1_then_precision_then_threshold",
        "selection_scores_sha256": hashlib.sha256(
            sources["search_top10"][2].read_bytes()
        ).hexdigest(),
        "policy_implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "dev_is_tuning_data": True,
    }
    destination = run_dir / policy["result_method"]
    destination.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(destination / "dev_threshold_sweep.csv", index=False)
    (destination / "policy.json").write_text(json.dumps(policy, indent=2))
    for pool_name, (tasks, _) in pools.items():
        source_manifest, scores, score_path = sources[pool_name]
        output = destination / pool_name
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "model": source_manifest["model"],
            "pool_sha256": source_manifest["pool_sha256"],
            "input_sha256": source_manifest["input_sha256"],
            "source_inference_manifest": source_manifest,
            "status": "running",
            "threshold": threshold,
            "min_margin": 0.0,
            "policy": policy,
            "source_scores_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
            "evaluation_only": True,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        tables = evaluate_rejection(
            tasks, scores, threshold=threshold, require_match_label=require_match_label
        )
        for name, table in tables.items():
            if name == "candidate_scores":
                continue
            if name == "predictions":
                table.to_json(
                    output / "predictions.jsonl", orient="records", lines=True
                )
            else:
                table.to_csv(output / f"{name}.csv", index=False)
        manifest["status"] = "complete"
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return policy
