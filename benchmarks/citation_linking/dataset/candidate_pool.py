"""Search candidates of the citation-linking benchmark.

Helper of ``build_annotation_dataset.py``, not a script. One store file holds, for every
``(ref_id, index)`` task, the candidates a metadata-only search returned and, where
the search missed the gold record, that record's metadata (the oracle candidate).

The published searches are frozen and are normally only read back from the store:
OpenAlex was queried on 16-17 October 2026, Matilda and Wikidata on 19 October and
OpenCitations on 13-16 November. ``search`` runs the same metadata-only search again
for the indexes it is given and replaces their rows, which moves that cut-off date
and, because the indexes change, the candidates. The October pools are labelled
``combined_v1`` and were assembled in the notebook; a new search uses the connectors
as they are now and will not reproduce them exactly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from citation_index.core.connectors import (
    MatildaConnector,
    OpenAlexConnector,
    WikidataConnector,
)
from citation_index.core.models import Reference
from citation_index.core.connectors.opencitations import (
    OpenCitationsConnector,
    OpenCitationsError,
    Query,
)

logger = logging.getLogger("candidate_pool")

ROOT = Path(__file__).resolve().parents[3]
LOGS = ROOT / "logs/citation_link_tryings"
STORE = LOGS / "candidates.jsonl"
CHECKPOINTS = LOGS / "candidate_search/checkpoints"
# Local pools the notebook and the GLiNER export read; written at every build.
POOL_DIR = LOGS / "four_index"
OPENCITATIONS = "opencitations"
TOP_K = 10  # also the OpenCitations connector shortlist
Key = tuple[str, str]


class StrictConnector(OpenCitationsConnector):
    """A frozen pool must not depend on which query happened to time out.

    The API connector keeps the hits of the strategies that answered; here any failed
    SPARQL strategy fails the task, so it is retried on the next run.
    """

    def sparql_candidates(self, query: Query) -> tuple[list[dict], str, list[str]]:
        raw, strategy, errors = super().sparql_candidates(query)
        if errors:
            raise OpenCitationsError("; ".join(errors))
        return raw, strategy, errors


def read_store(path: Path = STORE) -> dict[Key, dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    store = {(row["ref_id"], row["index"]): row for row in rows}
    if len(store) != len(rows):
        raise ValueError(f"Duplicate tasks in {path}")
    return store


def write_store(store: dict[Key, dict[str, Any]], path: Path = STORE) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in store.values():
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


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
    from benchmarks.citation_linking.dataset.build_annotation_dataset import (
        normalize_id,
    )

    if tasks[["ref_id", "index"]].duplicated().any():
        raise ValueError("Expected unique (ref_id, index) keys")
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


# index -> (retrieval version, connector factory)
SEARCHES = {
    "openalex": ("metadata-v1", OpenAlexConnector),
    "matilda": ("metadata-v1", MatildaConnector),
    "wikidata": ("metadata-v1", WikidataConnector),
    # The endpoint answers a prefix query in ~0.2s or (about one request in four,
    # Nov 2026) never: give up on a hung request fast and let the retries redo it.
    OPENCITATIONS: (
        "opencitations-sparql-v1",
        lambda: StrictConnector(sparql_timeout=5, author_timeout=45),
    ),
}


def search(tasks: list[dict[str, Any]], store: dict[Key, dict], index: str) -> None:
    """Live metadata-only search of one index; identifiers and gold never enter a query.

    Resumable: finished searches are read back from their checkpoint, so only missing
    ones reach the index. ``store`` is updated in place once every task has an answer.
    """
    version, connector = SEARCHES[index]
    frame = pd.DataFrame([task for task in tasks if task["index"] == index])[
        ["ref_id", "source", "reference", "parsed_reference", "index"]
    ]
    done = {
        saved["ref_id"]
        for saved in map(
            json.loads, (p.read_text() for p in CHECKPOINTS.glob("*.json"))
        )
        if saved["index"] == index and saved["retrieval_version"] == version
    }
    # Cached references first (read back instantly), so the bar then moves with the loop.
    frame = frame.sort_values("ref_id", key=lambda ids: ~ids.isin(done), kind="stable")
    connectors = {index: connector()}
    results, failed = [], []
    bar = tqdm(
        total=len(frame),
        initial=int(frame.ref_id.isin(done).sum()),
        desc=f"{index} search",
        unit="ref",
    )
    with logging_redirect_tqdm():  # log lines print above the bar
        for position in range(len(frame)):
            row = frame.iloc[[position]]
            try:  # one task per call: a failed search must not stop the others
                results.append(
                    retrieve_candidates(
                        row,
                        connectors,
                        CHECKPOINTS,
                        top_k=TOP_K,
                        retrieval_version=version,
                    )
                )
            except RuntimeError as exc:
                logger.error("%s", exc)
                failed.append(row.ref_id.iloc[0])
            else:  # the bar counts finished searches only
                bar.update(row.ref_id.iloc[0] not in done)
            bar.set_postfix(failed=len(failed))
    bar.close()
    if failed:  # unknown is not "no candidates": never store a partial snapshot
        raise SystemExit(f"{len(failed)} searches failed; rerun to retry: {failed}")
    for found in pd.concat(results, ignore_index=True).to_dict("records"):
        key = (found["ref_id"], index)
        day = found["retrieved_at"][:10]
        store[key] = {
            "ref_id": found["ref_id"],
            "index": index,
            "retrieval_method": version,
            "retrieval_source": f"{index}_search_{day.replace('-', '')}",
            "retrieved_on": day,
            "candidates": list(found["candidates"]),
            "oracle": store.get(key, {}).get("oracle"),
        }


def opencitations_record(gold_id: str, connector: OpenCitationsConnector) -> dict:
    """Exact gold record for the oracle pool, in the shape the search returns."""
    from benchmarks.citation_linking.dataset.build_annotation_dataset import (
        normalize_id,
    )

    records = connector.fetch_metadata([f"omid:{gold_id}"])
    if len(records) != 1:
        raise ValueError(f"Expected one Meta record for {gold_id}, got {len(records)}")
    scored = {"score": None, "title_similarity": None}
    candidate = connector._candidate(records[0], scored, "gold_injection")
    if normalize_id(candidate["id"], OPENCITATIONS) != gold_id:
        raise ValueError(
            f"Meta returned {candidate['id']} for {gold_id} (merged record?)"
        )
    return candidate


def attach(
    rows: list[dict[str, Any]], store: dict[Key, dict[str, Any]]
) -> list[dict[str, Any]]:
    """One candidate row per release row, with the oracle record its gold needs.

    The oracle record is kept only while it is the gold and the search missed it. A
    missing OpenCitations record is fetched from Meta and cached in ``store``; for the
    other indexes it must already be stored.
    """
    connector = None
    output = []
    for row in rows:
        key = (row["ref_id"], row["index"])
        if key not in store:
            raise ValueError(f"No stored search for {key}; run with --search")
        found = store[key]
        gold = row["gold_id"]
        oracle = None
        if gold and gold not in {c["candidate_id"] for c in found["candidates"]}:
            oracle = found.get("oracle")
            if not oracle or oracle["candidate_id"] != gold:
                if row["index"] != OPENCITATIONS:
                    raise ValueError(f"No stored metadata for gold {gold} of {key}")
                connector = connector or OpenCitationsConnector()
                oracle = {
                    "candidate_id": gold,
                    "metadata": opencitations_record(gold, connector),
                    "metadata_source": "opencitations_meta_rest",
                }
                found["oracle"] = oracle
        output.append(
            {**{k: v for k, v in found.items() if k != "oracle"}, "oracle": oracle}
        )
    return output


def candidates_table(candidates: list[dict[str, Any]]) -> Any:
    """Published form: provider metadata as JSON text, since its fields differ by index."""
    import pyarrow as pa

    text = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
    record = [("candidate_id", pa.string()), ("metadata", pa.string())]
    schema = pa.schema(
        [
            ("ref_id", pa.string()),
            ("index", pa.string()),
            ("retrieval_method", pa.string()),
            ("retrieved_on", pa.string()),
            ("n_candidates", pa.int32()),
            ("candidates", pa.list_(pa.struct([("rank", pa.int32()), *record]))),
            (
                "oracle_candidate",
                pa.struct([*record, ("metadata_source", pa.string())]),
            ),
        ]
    )
    rows = [
        {
            **row,
            "n_candidates": len(row["candidates"]),
            "candidates": [
                {**c, "metadata": text(c["metadata"])} for c in row["candidates"]
            ],
            "oracle_candidate": row["oracle"]
            and {**row["oracle"], "metadata": text(row["oracle"]["metadata"])},
        }
        for row in candidates
    ]
    return pa.Table.from_pylist(rows, schema=schema)


def write_pools(
    tasks: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    revision: str,
    pool_dir: Path = POOL_DIR,
) -> None:
    """Local ``search_top10`` (real search) and ``search_perfect`` (plus oracle) pools.

    ``tasks`` are release rows with their ``split``, aligned with ``candidates``.
    """
    pool_dir.mkdir(parents=True, exist_ok=True)
    for name in ("search_top10", "search_perfect"):
        lines = []
        for task, found in zip(tasks, candidates):
            ranked = list(found["candidates"])
            oracle = found["oracle"] if name == "search_perfect" else None
            if oracle:
                ranked.append(
                    {
                        "rank": len(ranked) + 1,
                        "candidate_id": oracle["candidate_id"],
                        "metadata": oracle["metadata"],
                    }
                )
            row = {
                **task,
                "dataset_revision": revision,
                "retrieval_method": found["retrieval_method"],
                "retrieval_source": found["retrieval_source"],
                "status": "ok",
                "n_candidates": len(ranked),
                "candidates": ranked,
                "injected_gold_id": oracle and oracle["candidate_id"],
                "injected_metadata_source": oracle and oracle["metadata_source"],
            }
            lines.append(json.dumps(row, ensure_ascii=False) + "\n")
        (pool_dir / f"{name}.jsonl").write_text("".join(lines), encoding="utf-8")
        logger.info("%d tasks -> %s", len(lines), pool_dir / f"{name}.jsonl")
