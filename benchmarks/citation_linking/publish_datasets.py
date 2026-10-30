"""Dataset preparation and publishing as one CLI.

Subcommands:
  opencitations-prep    build the OpenCitations routing table from annotations
  opencitations-upload  publish the OpenCitations baseline to Hugging Face and/or Argilla
  dasi-ebda-upload      add the DASI/EBDA annotations as a split of the HF annotations dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

from huggingface_hub import CommitOperationAdd, hf_hub_download, HfApi
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

ID_COLUMNS = {
    "openalex": "openalex_id",
    "wikidata": "wikidata_id",
    "matilda": "matilda_id",
}


DOI_PRIORITY = ["openalex", "matilda", "wikidata"]  # same order as the first prep round


COLUMNS = [
    "ref_id",
    "source",
    "excluded_multi_ref",
    "oc_route",
    "doi",
    "doi_from",
    "openalex_id",
    "openalex_id_from",
    "wikidata_id",
    "wikidata_id_from",
    "matilda_id",
    "matilda_id_from",
    "n_confirmed_ids",
    "resolvable_to_doi",
    "original_ref_string",
    "resolved_doi",
    "resolved_doi_from",
    "id_resolves",
    "openalex_id_invalid",
]


def clean_doi(value: object) -> str | None:
    doi = str(value or "").strip().lower().removeprefix("https://doi.org/")
    return doi or None


def confirmed_id(row: dict) -> tuple[str | None, str | None]:
    """(id, how) the annotator confirmed for one (ref, index) row."""
    if row["label"] == "correct_match":
        return row["matched_id"], "confirmed_match"
    if row["label"] == "wrong_match" and row["gold_id"]:
        return row["gold_id"], "gold_correction"
    return None, None


def resolve_openalex(
    work_id: str, session: requests.Session
) -> tuple[str | None, bool]:
    """(doi, resolves). A 404 means the id does not exist (OpenAlex ids >= W7... are often dead)."""
    response = session.get(
        f"https://api.openalex.org/works/{work_id}",
        params={"select": "id,doi"},
        timeout=30,
    )
    if response.status_code == 404:
        return None, False
    response.raise_for_status()
    return clean_doi(response.json().get("doi")), True


def resolve_wikidata(qid: str, session: requests.Session) -> tuple[str | None, bool]:
    response = session.get(
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbgetentities",
            "ids": qid,
            "props": "claims",
            "format": "json",
        },
        timeout=30,
    )
    response.raise_for_status()
    entity = response.json().get("entities", {}).get(qid, {})
    if "missing" in entity:
        return None, False
    claims = entity.get("claims", {}).get("P356", [])
    return (
        clean_doi(claims[0]["mainsnak"].get("datavalue", {}).get("value"))
        if claims
        else None
    ), True


def route(prep: dict) -> str:
    if prep["doi"]:
        return "doi_lookup"
    if prep["resolved_doi"]:
        return "doi_lookup_after_resolve"
    if prep["openalex_id"] or prep["wikidata_id"]:
        return "id_only_no_doi"
    return "matilda_id_only" if prep["matilda_id"] else "manual_search"


def build_prep(rows: list[dict], session: requests.Session) -> pd.DataFrame:
    by_ref: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_ref.setdefault(row["ref_id"], {})[row["index"]] = row
    out = []
    for ref_id, per_index in by_ref.items():
        first = next(iter(per_index.values()))
        prep: dict = {c: None for c in COLUMNS}
        prep.update(
            ref_id=ref_id,
            source=first["source"],
            original_ref_string=first["original_ref_string"],
            excluded_multi_ref=any(
                r["label"] == "multi_ref_context" for r in per_index.values()
            ),
        )
        for index, row in per_index.items():
            prep[ID_COLUMNS[index]], prep[f"{ID_COLUMNS[index]}_from"] = confirmed_id(
                row
            )
        for index in DOI_PRIORITY:  # DOI only from a candidate the annotator confirmed
            row = per_index.get(index)
            if (
                row
                and row["label"] == "correct_match"
                and clean_doi(row["matched_doi"])
            ):
                prep["doi"], prep["doi_from"] = clean_doi(row["matched_doi"]), index
                break
        prep["n_confirmed_ids"] = sum(bool(prep[c]) for c in ID_COLUMNS.values())
        prep["resolvable_to_doi"] = bool(
            prep["doi"] or prep["openalex_id"] or prep["wikidata_id"]
        )
        if not prep["doi"] and not prep["excluded_multi_ref"]:
            for index, resolver in (
                ("openalex", resolve_openalex),
                ("wikidata", resolve_wikidata),
            ):
                if prep[ID_COLUMNS[index]]:
                    doi, resolves = resolver(prep[ID_COLUMNS[index]], session)
                    prep["id_resolves"] = bool(prep["id_resolves"]) or resolves
                    if index == "openalex" and not resolves:
                        prep["openalex_id_invalid"] = True
                    if doi:
                        prep["resolved_doi"], prep["resolved_doi_from"] = doi, index
                        break
        prep["oc_route"] = route(prep)
        out.append(prep)
    return pd.DataFrame(out, columns=COLUMNS)


def self_check() -> None:
    assert clean_doi("https://doi.org/10.1/ABC") == "10.1/abc" and clean_doi("") is None
    assert confirmed_id(
        {"label": "wrong_match", "gold_id": "W1", "matched_id": "W2"}
    ) == ("W1", "gold_correction")
    assert confirmed_id(
        {"label": "no_record_in_index", "gold_id": None, "matched_id": "W2"}
    ) == (None, None)
    assert (
        route(
            {
                "doi": None,
                "resolved_doi": None,
                "openalex_id": None,
                "wikidata_id": None,
                "matilda_id": "u",
            }
        )
        == "matilda_id_only"
    )
    print("self-check ok")


def opencitations_prep_main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        return self_check()
    rows = [
        json.loads(line)
        for line in args.annotations.read_text().splitlines()
        if line.strip()
    ]
    session = requests.Session()
    session.headers["User-Agent"] = "citation-index/1.0 (mailto:yurui.zhu@odoma.ch)"
    if key := os.environ.get("OPENALEX_API_KEY") or os.environ.get("OPENALEX_API"):
        session.headers["Authorization"] = f"Bearer {key}"
    prep = build_prep(rows, session)
    prep.to_csv(args.out, index=False)
    print(prep.oc_route.value_counts().to_string())


HERE = Path(__file__).resolve().parent


ROWS = HERE / "opencitations_baseline/rows.jsonl"


HF_REPO = "yurui983/citation_linking_baseline"


HF_PATH = "data/opencitations_search-00000-of-00001.parquet"


ARGILLA_API_URL = os.environ.get("ARGILLA_API_URL") or "https://argilla.graphia-ssh.eu/"


TEMPLATE_DATASET = "citation_linking_openalex"


DATASET_NAME = "citation_linking_opencitations"


SEARCH_METHOD = {
    "lookup": "ID lookup (REST API, DOI/OpenAlex ID)",
    "search": "SPARQL search (title / author / year)",
    "search_after_lookup_miss": "SPARQL search (ID lookup found nothing in OpenCitations)",
}


logger = logging.getLogger("upload_opencitations")


def load_rows(path: Path, routes: set[str] | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if routes:
        rows = [r for r in rows if r["oc_route"] in routes]
    if len({r["ref_id"] for r in rows}) != len(rows):
        raise ValueError("ref_id must be unique in rows.jsonl")
    return rows


def parquet_table(rows: list[dict[str, Any]]):
    import pyarrow as pa

    schema = pa.schema(
        [
            ("ref_id", pa.string()),
            ("source", pa.string()),
            ("original_ref_string", pa.string()),
            ("index", pa.string()),
            ("matched_id", pa.string()),
            ("matched_doi", pa.string()),
            (
                "matched_result",
                pa.struct(
                    [
                        ("first_author", pa.string()),
                        ("journal", pa.string()),
                        ("title", pa.string()),
                        ("year", pa.int64()),
                    ]
                ),
            ),
            ("is_match_by_similarity", pa.bool_()),
            ("matched_link", pa.string()),
            ("oc_route", pa.string()),
            ("oc_status", pa.string()),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def push_hf(
    rows: list[dict[str, Any]], repo: str, path_in_repo: str, push: bool
) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    local = HERE / "opencitations_baseline/opencitations.parquet"
    pq.write_table(parquet_table(rows), local)
    api = HfApi(token=os.environ["HF_TOKEN"])
    exists = api.file_exists(repo, path_in_repo, repo_type="dataset")
    print(
        f"HF {repo}:{path_in_repo} ({len(rows)} rows) - {'EXISTS, would replace' if exists else 'new file'}"
    )
    if push:
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=path_in_repo,
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"add OpenCitations baseline ({len(rows)} rows)",
        )
        print("uploaded")


def argilla_record(row: dict[str, Any]) -> dict[str, Any]:
    matched = row["matched_result"]
    matched_md = (
        "```json\n"
        + json.dumps({k: v for k, v in matched.items() if v is not None}, indent=2)
        + "\n```"
        if row["matched_id"] != "Not Found"
        else ""
    )
    link = row["matched_link"]
    return {
        "ref_id": row["ref_id"],
        "original_ref_string": row["original_ref_string"],
        "source": row["source"],
        "matched_id": row["matched_id"],
        "matched_doi": row["matched_doi"],
        "matched_result": matched_md,
        "is_match_by_similarity": "true" if row["is_match_by_similarity"] else "false",
        "matched_link": f"[{link}]({link})",
        "search_method": SEARCH_METHOD[row["oc_route"]],
        "metadata": {
            "ref_id": row["ref_id"],
            "source": row["source"],
            "index": "opencitations",
            "oc_route": row["oc_route"],
        },
    }


def push_argilla(rows: list[dict[str, Any]], name: str, push: bool) -> None:
    import argilla as rg

    client = rg.Argilla(
        api_key=os.environ["ARGILLA_API_TOKEN"], api_url=ARGILLA_API_URL
    )
    if client.datasets(name=name) is not None:
        raise SystemExit(
            f"Argilla dataset {name!r} already exists; refusing to touch it"
        )
    template = client.datasets(name=TEMPLATE_DATASET)
    if template is None:
        raise SystemExit(f"template dataset {TEMPLATE_DATASET!r} not found")
    settings = template.settings
    settings.fields.add(rg.TextField(name="search_method", title="Search Method"))
    records = [argilla_record(r) for r in rows]
    print(
        f"Argilla {name}: {len(records)} records, fields {[f.name for f in settings.fields]}"
    )
    if push:
        dataset = rg.Dataset(name=name, settings=settings, client=client)
        dataset.create()
        dataset.records.log(records)
        print("created and logged")


def append_argilla(rows: list[dict[str, Any]], name: str, push: bool) -> None:
    """Add new records to an existing dataset; never updates or deletes existing ones."""
    import argilla as rg

    client = rg.Argilla(
        api_key=os.environ["ARGILLA_API_TOKEN"], api_url=ARGILLA_API_URL
    )
    dataset = client.datasets(name=name)
    if dataset is None:
        raise SystemExit(f"Argilla dataset {name!r} not found")
    existing = [r for r in dataset.records(with_responses=True)]
    seen = {r.fields["ref_id"] for r in existing}
    clash = seen & {r["ref_id"] for r in rows}
    if clash:
        raise SystemExit(
            f"{len(clash)} ref_ids already in {name!r}, e.g. {sorted(clash)[:3]}; refusing to touch them"
        )
    field_names = {f.name for f in dataset.settings.fields}
    records = [argilla_record(r) for r in rows]
    missing = {k for rec in records for k in rec if k != "metadata"} - field_names
    if missing:
        raise SystemExit(f"dataset {name!r} has no fields {sorted(missing)}")
    print(f"Argilla {name}: {len(existing)} existing records, appending {len(records)}")
    if push:
        dataset.records.log(
            [{k: v for k, v in rec.items() if k != "metadata"} for rec in records]
        )
        after = len(list(dataset.records))
        print(
            f"appended; now {after} records (expected {len(existing) + len(records)})"
        )


def opencitations_upload_main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", type=Path, default=ROWS)
    parser.add_argument(
        "--routes", help="comma-separated oc_route filter (default: all)"
    )
    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--hf-path", default=HF_PATH)
    parser.add_argument("--argilla-name", default=DATASET_NAME)
    parser.add_argument("--push-hf", action="store_true")
    parser.add_argument("--push-argilla", action="store_true")
    parser.add_argument(
        "--append",
        action="store_true",
        help="append --rows to the existing Argilla dataset (no HF)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    routes = set(args.routes.split(",")) if args.routes else None
    rows = load_rows(args.rows, routes)
    print(pd.DataFrame(rows).groupby(["oc_route", "oc_status"]).size().to_string())
    if args.append:
        append_argilla(rows, args.argilla_name, args.push_argilla)
        return
    push_hf(rows, args.hf_repo, args.hf_path, args.push_hf)
    if os.environ.get("ARGILLA_API_TOKEN"):
        push_argilla(rows, args.argilla_name, args.push_argilla)
    elif args.push_argilla:
        raise SystemExit("--push-argilla needs ARGILLA_API_TOKEN")
    else:
        print("ARGILLA_API_TOKEN unset: skipping Argilla")


SRC = HERE / "argilla_annotations/dasi_ebda"


REPO = "yurui983/citation_linking_annotations"


SPLIT = "dasi_ebda"


PARQUET = f"data/{SPLIT}-00000-of-00001.parquet"


ANALYSIS_FILES = [
    "overall",
    "metrics_by_index",
    "metrics_by_source",
    "metrics_by_index_source",
    "outcome_counts",
]


def build_parquet(out: Path) -> int:
    old_schema = pq.read_schema(
        hf_hub_download(
            REPO, "data/annotations-00000-of-00001.parquet", repo_type="dataset"
        )
    )
    rows = [
        json.loads(line)
        for line in (SRC / "citation_linking_annotations.jsonl")
        .read_text()
        .splitlines()
        if line
    ]
    table = pa.Table.from_pylist(
        [{k: r.get(k) for k in old_schema.names} for r in rows], schema=old_schema
    )
    pq.write_table(table, out)
    return len(rows)


def patched_readme(n_rows: int, n_bytes: int) -> str:
    readme = Path(hf_hub_download(REPO, "README.md", repo_type="dataset")).read_text()
    if f"name: {SPLIT}" in readme:
        return readme
    readme = readme.replace(
        "    num_examples: 1500\n",
        f"    num_examples: 1500\n  - name: {SPLIT}\n    num_bytes: {n_bytes}\n    num_examples: {n_rows}\n",
    )
    readme = readme.replace(
        "    path: data/annotations-*\n",
        f"    path: data/annotations-*\n  - split: {SPLIT}\n    path: data/{SPLIT}-*\n",
    )
    return readme + (
        f"\n## Splits\n\n- `annotations`: 500 references x 3 indexes (openalex, matilda, wikidata) from cex, excite, "
        f"linkedbook, brill, legal_study_mpilhlt (1500 rows).\n- `{SPLIT}`: 200 references x 3 indexes from the DASI and "
        f"EBDA bibliographies (600 rows), same schema. Argilla datasets `citation_linking_dasi_ebda_{{index}}`. "
        f"One row carries a `manual_fix` note.\n"
    )


def dasi_ebda_upload_main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    local = SRC / "dasi_ebda.parquet"
    n_rows = build_parquet(local)
    readme = patched_readme(n_rows, local.stat().st_size)
    ops = [
        CommitOperationAdd(PARQUET, str(local)),
        CommitOperationAdd("README.md", readme.encode()),
        *[
            CommitOperationAdd(
                f"analysis/{SPLIT}/{n}.csv", str(SRC / f"analysis/{n}.csv")
            )
            for n in ANALYSIS_FILES
        ],
    ]
    print(f"{n_rows} rows -> {REPO}:{PARQUET}; files: {[o.path_in_repo for o in ops]}")
    if args.push:
        HfApi(token=os.environ["HF_TOKEN"]).create_commit(
            REPO,
            ops,
            repo_type="dataset",
            commit_message=f"add {SPLIT} split ({n_rows} rows) + analysis",
        )
        print("pushed")


COMMANDS = {
    "opencitations-prep": opencitations_prep_main,
    "opencitations-upload": opencitations_upload_main,
    "dasi-ebda-upload": dasi_ebda_upload_main,
}


def main() -> None:
    """Dispatch ``publish_datasets.py <command> ...`` to the command's own parser."""
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(
            f"usage: publish_datasets.py {{{','.join(COMMANDS)}}} [options]"
        )
    command = sys.argv[1]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    sys.argv = [f"publish_datasets.py {command}", *sys.argv[2:]]
    COMMANDS[command]()


if __name__ == "__main__":
    main()
