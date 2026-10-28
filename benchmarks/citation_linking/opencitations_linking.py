"""OpenCitations baseline for the citation-linking benchmark.

Search logic lives in ``citation_index.core.connectors.opencitations`` (shared with the API).
This script adds the benchmark routing (``analysis/opencitations_prep.csv`` decides which
refs have a confirmed ID) and the OpenAlex-ID lookup the API connector deliberately skips:

* **lookup**  - reference has a confirmed DOI / OpenAlex ID from the other indexes
  -> OpenCitations Meta REST API (``/metadata/doi:..__openalex:..``, batched).
* **search**  - everything else, or a lookup miss -> SPARQL against the QLever-backed
  Meta endpoint, then REST ``omid:`` fetch for the shortlisted candidates.

SPARQL design (see ``oc_sparql/AGENTS.md`` and the QLever wiki; probed 2026-10-21):
``REGEX`` / ``CONTAINS(LCASE())`` over ~100M titles time out (>120s) while a
case-sensitive ``STRSTARTS(?title, "...")`` prefix filter is a vocabulary range scan
(~40ms). OC titles are stored title-cased, so queries use title-cased prefixes.
Queries return only ``?br ?title ?date`` (no OPTIONALs: QLever evaluates them over the
whole dataset); metadata is fetched through REST afterwards. Fuzzy scoring is Python-side.

Timeouts / HTTP failures are recorded as ``status="error"``, never as "not found".

    python opencitations_linking.py --self-check
    python opencitations_linking.py --limit 5            # smoke test
    python opencitations_linking.py                      # full run, resumable
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from citation_index.core.connectors.opencitations import (  # noqa: E402
    REST_BASE,
    OpenCitationsConnector,
    OpenCitationsError,
    Query,
    lookup_keys,
    normalize_doi,
    normalize_record,
    record_tokens,
    score_candidate,
    title_prefix_query,
    title_prefix_variants,
)

logger = logging.getLogger("opencitations_linking")

PREP_CSV = HERE / "argilla_annotations/analysis/opencitations_prep.csv"
SEARCH_JSONL = ROOT / "logs/citation_link_tryings/search_top10.jsonl"
OUT_DIR = HERE / "opencitations_baseline"

REST_BATCH = 20
TOP_N = 5  # candidates kept per ref


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def chunks(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def lookup_ids(row: pd.Series) -> list[str]:
    """Confirmed IDs usable against the Meta REST API (wikidata/matilda are not)."""
    ids = [f"doi:{d}" for d in dict.fromkeys(filter(None, map(normalize_doi, (row.doi, row.resolved_doi))))]
    if pd.notna(row.openalex_id) and row.openalex_id_invalid != True:  # noqa: E712 - NaN/True/False
        ids.append(f"openalex:{str(row.openalex_id).upper()}")
    return ids


def run_lookup(client: OpenCitationsConnector, wanted: dict[str, list[str]]) -> tuple[dict[str, dict], dict[str, str]]:
    """Return ``({ref_id: raw record}, {ref_id: error})`` for refs with confirmed IDs."""
    flat = sorted({i for ids in wanted.values() for i in ids})
    index: dict[str, dict] = {}
    failed: dict[str, str] = {}
    for batch in chunks(flat, REST_BATCH):
        try:
            for record in client.fetch_metadata(batch):
                for key in lookup_keys(record_tokens(record)):
                    index.setdefault(key, record)
        except OpenCitationsError as exc:
            logger.exception("REST lookup batch failed")
            failed.update({i: str(exc) for i in batch})
    found, errs = {}, {}
    for ref_id, ids in wanted.items():
        hit = next((index[i] for i in ids if i in index), None)
        if hit:
            found[ref_id] = hit
        elif err := next((failed[i] for i in ids if i in failed), None):
            errs[ref_id] = err
    return found, errs


# ---------------------------------------------------------------------------
# Results -> rows
# ---------------------------------------------------------------------------
def result_row(inputs: dict, route: str, status: str, candidates: list[dict], **extra: Any) -> dict:
    return {
        "ref_id": inputs["ref_id"],
        "source": inputs["source"],
        "original_ref_string": inputs["reference"],
        "oc_route": route,
        "oc_status": status,
        "candidates": candidates[:TOP_N],
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }


def to_dataset_row(result: dict, min_title_sim: float) -> dict:
    """Same columns as the other ``index`` rows of ``yurui983/citation_linking_baseline``."""
    top = (result["candidates"] or [None])[0]
    usable = top and (result["oc_route"].endswith("lookup") or top["title_similarity"] >= min_title_sim)
    matched_id = top["omid"] if usable else None
    return {
        "ref_id": result["ref_id"],
        "source": result["source"],
        "original_ref_string": result["original_ref_string"],
        "index": "opencitations",
        "matched_id": matched_id or "Not Found",
        "matched_doi": (top["doi"] or "") if usable else "",
        "matched_result": {
            "first_author": top["first_author"], "journal": top["journal"],
            "title": top["title"], "year": top["year"],
        } if usable else {"first_author": None, "journal": None, "title": None, "year": None},
        "is_match_by_similarity": bool(usable and top["is_match_by_similarity"]),
        "matched_link": f"{REST_BASE}/metadata/omid:{matched_id}" if matched_id else "https://sparql.opencitations.net/",
        "oc_route": result["oc_route"],
        "oc_status": result["oc_status"],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_inputs(prep_csv: Path, search_jsonl: Path) -> tuple[pd.DataFrame, dict[str, dict]]:
    prep = pd.read_csv(prep_csv)
    prep = prep[~prep["excluded_multi_ref"].astype(bool)].reset_index(drop=True)
    inputs: dict[str, dict] = {}
    with search_jsonl.open() as handle:
        for line in handle:
            rec = json.loads(line)
            inputs.setdefault(rec["ref_id"], {k: rec[k] for k in ("ref_id", "source", "reference", "parsed_reference")})
    missing = set(prep["ref_id"]) - set(inputs)
    if missing:
        raise ValueError(f"{len(missing)} prep refs missing from {search_jsonl.name}: {sorted(missing)[:3]}")
    return prep, inputs


def load_done(path: Path) -> dict[str, dict]:
    """Resume: keep the last row per ref; errored refs are retried."""
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        rows[row["ref_id"]] = row
    return {k: v for k, v in rows.items() if v["oc_status"] != "error"}


def run(args: argparse.Namespace) -> None:
    prep, inputs = load_inputs(args.prep_csv, args.search_jsonl)
    if args.refs:
        prep = prep[prep["ref_id"].isin(args.refs.split(","))]
    if args.limit:
        prep = prep.groupby("oc_route", group_keys=False).head(args.limit)  # sample every route
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "results.jsonl"
    done = load_done(results_path)
    todo = prep[~prep["ref_id"].isin(done)]
    logger.info("%d refs total, %d already done, %d to run", len(prep), len(done), len(todo))
    client = OpenCitationsConnector()
    sink = results_path.open("a")

    def emit(result: dict) -> None:
        sink.write(json.dumps(result, ensure_ascii=False) + "\n")
        sink.flush()
        done[result["ref_id"]] = result

    # --- phase 1: REST lookup by confirmed IDs ---
    wanted = {r.ref_id: ids for _, r in todo.iterrows() if (ids := lookup_ids(r))}
    found, lookup_errors = run_lookup(client, wanted)
    for ref_id, record in found.items():
        query = Query.from_parsed(inputs[ref_id]["parsed_reference"])
        candidate = normalize_record(record)
        emit(result_row(inputs[ref_id], "lookup", "ok", [{**candidate, **score_candidate(query, candidate)}],
                        lookup_ids=wanted[ref_id]))
    logger.info("lookup: %d/%d found, %d errored", len(found), len(wanted), len(lookup_errors))

    # --- phase 2: SPARQL search for the rest (and lookup misses) ---
    for n, ref_id in enumerate([r for r in todo.ref_id if r not in found], start=1):
        route = "search_after_lookup_miss" if ref_id in wanted else "search"
        query = Query.from_parsed(inputs[ref_id]["parsed_reference"])
        if ref_id in lookup_errors:
            emit(result_row(inputs[ref_id], route, "error", [], error=lookup_errors[ref_id]))
            continue
        if not query.title:  # ponytail: no raw-reference fallback, prefix search needs a parsed title
            emit(result_row(inputs[ref_id], route, "no_query", []))
            continue
        raw, strategy, errors = client.sparql_candidates(query)
        try:
            ranked = client.rank_candidates(query, raw) if raw else []
        except OpenCitationsError as exc:
            errors.append(f"metadata: {exc}")
            ranked = []
        # a failed strategy with no hit means "unknown", not "not found"
        status = "error" if errors and not ranked else ("ok" if ranked else "not_found")
        emit(result_row(inputs[ref_id], route, status, ranked, query_strategy=strategy,
                        n_raw_candidates=len(raw), errors=errors or None))
        if n % 10 == 0:
            logger.info("search: %d done", n)
    sink.close()

    rows = [to_dataset_row(done[r], args.min_title_sim) for r in prep.ref_id if r in done]
    (args.out_dir / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    summary = pd.DataFrame(rows).groupby(["oc_route", "oc_status"]).size()
    print(summary.to_string())
    print(f"linked: {sum(r['matched_id'] != 'Not Found' for r in rows)}/{len(rows)}  -> {args.out_dir}")


def self_check() -> None:
    assert normalize_doi("https://doi.org/10.1371/Journal.Pone.1") == "10.1371/journal.pone.1"
    assert normalize_doi(float("nan")) is None
    variants = title_prefix_variants("3-deazaneplanocin A (DZNep), an inhibitor of EZH2", 6)
    assert "3-Deazaneplanocin A (DZNep), An Inhibitor Of" in variants, variants
    assert any(v.startswith("3-Deazaneplanocin A (Dznep)") for v in title_prefix_variants("3-deazaneplanocin A (DZNep), an inhibitor of EZH2", 6))
    assert title_prefix_variants("THE HISTORY OF EZH2 RESEARCH", 4)[1] == "The History Of Ezh2"
    assert title_prefix_variants("Hi", 5) == []
    assert 'STRSTARTS(?title, "A \\"q\\" B")' in title_prefix_query('A "q" B')
    # batch mapping: response order and extra records must not matter
    records = [
        {"id": "doi:10.1182/BLOOD-2009 openalex:W2 omid:br/061"},
        {"id": "doi:10.1371/journal.pone.1 openalex:W1 pmid:9 omid:br/062"},
    ]
    index = {k: r for r in records for k in lookup_keys(record_tokens(r))}
    assert index["doi:10.1371/journal.pone.1"]["id"].endswith("br/062")
    assert index["openalex:W2"]["id"].endswith("br/061") and index["doi:10.1182/blood-2009"]
    cand = normalize_record({"id": "doi:10.1/x omid:br/1", "title": "Citizenship And Feminist Politics",
                             "author": "Phillips, Anne [omid:ra/1]", "pub_date": "1991-01-01", "venue": "Foo [issn:1]"})
    query = Query.from_parsed({"full_title": "Citizenship and Feminist Politics", "authors": ["A. Phillips"], "publication_year": 1991})
    assert query.family == "Phillips" and cand["first_author"] == "Phillips" and cand["journal"] == "Foo"
    assert score_candidate(query, cand)["is_match_by_similarity"]
    print("self-check ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prep-csv", type=Path, default=PREP_CSV)
    parser.add_argument("--search-jsonl", type=Path, default=SEARCH_JSONL)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--limit", type=int, help="smoke test: first N refs of every route")
    parser.add_argument("--refs", help="comma-separated ref_ids")
    parser.add_argument("--min-title-sim", type=float, default=70.0,
                        help="search hits below this title similarity are exported as Not Found")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    self_check() if args.self_check else run(args)


if __name__ == "__main__":
    main()
