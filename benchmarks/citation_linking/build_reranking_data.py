"""Export normal and gold-completed candidates from the frozen search experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from benchmarks.citation_linking.build_annotation_dataset import normalize_id
from benchmarks.citation_linking.citation_link_tryings_helpers import search_reference
from citation_index.core.connectors import WikidataConnector

OUTPUT = ROOT / "logs/citation_link_tryings"
REVISION = "598fb258f3cfed7781167b7f595e3536c3e2f660"


def read_jsonl(path: Path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def complete_candidates(
    row: dict[str, Any], metadata: dict[str, Any] | None, source: str | None
) -> dict[str, Any]:
    """Append a verified missing gold record without removing any retrieved item."""
    candidates = row["candidates"]
    gold = row["gold_id"]
    result = {**row, "candidates": list(candidates)}
    if gold is None or any(c["candidate_id"] == gold for c in candidates):
        return result
    if not metadata or normalize_id(metadata.get("id"), row["index"]) != gold:
        raise ValueError(
            f"Missing or mismatched gold metadata: {row['ref_id']} / {row['index']}"
        )
    result["candidates"].append(
        {
            "rank": len(candidates) + 1,
            "candidate_id": gold,
            "metadata": metadata,
        }
    )
    result.update(
        n_candidates=len(result["candidates"]),
        injected_gold_id=gold,
        injected_metadata_source=source,
    )
    return result


def build(source_root: Path, output: Path = OUTPUT) -> None:
    final = source_root / "final_report_20261020"
    manifest = json.loads((final / "manifest.json").read_text())
    assert manifest["dataset"]["revision"] == REVISION
    tasks = list(read_jsonl(final / "tasks.jsonl"))
    searches = {
        (r["ref_id"], r["index"]): r
        for r in read_jsonl(final / "searches.jsonl")
        if r["method"] == "combined"
    }
    assert len(tasks) == len(searches) == 1443
    assert len({(r["ref_id"], r["index"]) for r in tasks}) == len(tasks)
    assert {(r["ref_id"], r["index"]) for r in tasks} == set(searches)
    diagnostics = {}
    for path in (source_root / "full_search_20261016/diagnostics").glob("*.json"):
        diagnostic = json.loads(path.read_text())
        diagnostics[diagnostic["ref_id"], diagnostic["index"]] = diagnostic
    matilda = {
        r["gold_id"]: r
        for r in read_jsonl(source_root / "reranking_gold_metadata.jsonl")
    }
    wiki = WikidataConnector()
    normal, perfect = [], []
    for task in tasks:
        key = task["ref_id"], task["index"]
        search = searches[key]
        candidates = search["candidates"]
        assert search["status"] == "ok"
        ids = [c["candidate_id"] for c in candidates]
        assert len(ids) == len(set(ids)) <= 10
        assert [c["rank"] for c in candidates] == list(range(1, len(ids) + 1))
        assert all(
            normalize_id(c["metadata"]["id"], task["index"]) == c["candidate_id"]
            for c in candidates
        )
        row = {
            **task,
            "dataset_revision": REVISION,
            "retrieval_method": "combined_v1",
            "retrieval_source": search["source_run"],
            "status": "ok",
            "n_candidates": len(candidates),
            "candidates": candidates,
            "injected_gold_id": None,
            "injected_metadata_source": None,
        }
        metadata, source = None, None
        gold = task["gold_id"]
        if gold is not None and gold not in ids:
            if task["index"] == "matilda":
                record = matilda[gold]
                metadata, source = record["metadata"], record["source"]
            else:
                diagnostic = diagnostics[key]
                assert (
                    diagnostic["gold_id"] == gold
                    and diagnostic["gold_lookup_status"] == 200
                )
                metadata, source = diagnostic["gold_metadata"], "diagnostic_id_lookup"
                if task["index"] == "wikidata":
                    reference, _ = search_reference(task)
                    language = wiki._detect_language(reference.full_title)
                    assert "missing" not in metadata
                    metadata = {
                        "id": gold,
                        "label": metadata.get("labels", {})
                        .get(language, {})
                        .get("value", gold),
                        "description": metadata.get("descriptions", {})
                        .get(language, {})
                        .get("value", ""),
                        "match": {"type": "identifier", "language": language},
                        "claims": metadata.get("claims", {}),
                        "entity_data": metadata,
                    }
        normal.append(row)
        perfect.append(complete_candidates(row, metadata, source))
    assert sum(r["injected_gold_id"] is not None for r in perfect) == 31
    assert sum(r["gold_id"] is not None for r in normal) == 454
    assert all(
        r["gold_id"] is None
        or r["gold_id"] in [c["candidate_id"] for c in r["candidates"]]
        for r in perfect
    )
    assert all(
        b["candidates"][: len(a["candidates"])] == a["candidates"]
        for a, b in zip(normal, perfect)
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("search_top10.jsonl", normal),
        ("search_perfect.jsonl", perfect),
    ):
        path = output / name
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            for row in rows:
                stream.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
        temporary.replace(path)
        print(f"{path}: {len(rows)} tasks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=OUTPUT / "archive")
    args = parser.parse_args()
    build(args.source_root)
