#!/usr/bin/env python3
"""Prepare BibTeX bibliography entries, run DOI-first linking, and publish results."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from citation_index.core.connectors import (
    MatildaConnector,
    OpenAlexConnector,
    WikidataConnector,
)
from citation_index.core.models.reference import Reference
from citation_index.utils.reference_matching import custom_match, reference_match_fields

LOGGER = logging.getLogger(__name__)
INDEXES = ("openalex", "matilda", "wikidata")
DEFAULT_OUTPUT = ROOT / "logs/citation_linking/dasi_ebda"
DEFAULT_INPUT = Path(__file__).parent / "DASI_EBDA"
HF_REPO = "yurui983/citation_linking_baseline"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    value = unquote(value.strip())
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value, flags=re.I)
    return value.lower() if re.fullmatch(r"10\.\d{4,9}/\S+", value) else None


def plain(value: str) -> str:
    from pybtex.richtext import Text

    value = re.sub(r"\\(?:textit|textrm)\b", "", value)
    for command, replacement in (
        ("textless", "<"),
        ("textgreater", ">"),
        ("textbackslash", "\\"),
    ):
        value = value.replace("\\" + command + "{}", replacement)
    return " ".join(Text.from_latex(value).render_as("text").split())


def displayed_person(person: Any, harvard: bool) -> tuple[str, dict[str, str]]:
    surname = plain(
        " ".join(person.prelast_names + person.last_names + person.lineage_names)
    )
    given = plain(" ".join(person.first_names + person.middle_names))
    if harvard:
        given = " ".join(
            (
                word
                if re.fullmatch(r"(?:[^\W\d_]\.)+", word)
                else "-".join(part[0] + "." for part in word.split("-") if part)
            )
            for word in given.split()
        )
    parsed = {k: v for k, v in (("surname", surname), ("first_name", given)) if v}
    return ", ".join(filter(None, (surname, given))), parsed


def join_names(names: list[str]) -> str:
    if len(names) < 2:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def names_for_style(
    people: list[tuple[str, dict[str, str]]], harvard: bool, editors: bool = False
) -> str:
    names = []
    for position, (inverted, person) in enumerate(people):
        if harvard or (position == 0 and not editors):
            names.append(inverted)
        else:
            names.append(
                " ".join(
                    filter(None, (person.get("first_name"), person.get("surname")))
                )
            )
    return join_names(names)


def sentence(text: str) -> str:
    return text if text.endswith((".", "?", "!")) else text + "."


def render_entry(entry: Any, style: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    harvard = style == "harvard"
    fields = {
        key: plain(value)
        for key, value in entry.fields.items()
        if key not in {"abstract", "note", "keywords"}
    }
    title, year = fields["title"], fields["year"]
    parsed: dict[str, Any] = {"full_title": title, "publication_date_raw": year}
    if year.isdigit():
        parsed["publication_year"] = int(year)
    raw: dict[str, str] = {}
    authors = [
        displayed_person(person, harvard) for person in entry.persons.get("author", [])
    ]
    if authors:
        parsed["authors"] = [person for _, person in authors]
    names = names_for_style(authors, harvard)
    pieces = (
        [f"{names} ({year})." if harvard and names else f"{sentence(names)} {year}."]
        if names
        else []
    )
    article = entry.type == "article"
    container = fields.get("booktitle") or (
        fields.get("journal") if not article else None
    )
    pieces.append(
        f"‘{title}’."
        if harvard and (article or container)
        else f"“{sentence(title)}”" if article or container else sentence(title)
    )
    if not names:
        pieces.append(f"({year})." if harvard else year + ".")
    if article and fields.get("journal"):
        parsed["journal_title"] = fields["journal"]
        venue = fields["journal"]
        for key, target in (("volume", "volume"), ("number", "issue")):
            if fields.get(key):
                parsed[target] = fields[key]
        if "volume" in parsed:
            venue += " " + parsed["volume"]
        if "issue" in parsed:
            venue += " (" + parsed["issue"] + ")"
        if fields.get("pages"):
            parsed["pages"] = fields["pages"].replace("--", "–")
            venue += (", pp. " if harvard else ": ") + parsed["pages"]
        pieces.append(sentence(venue))
    else:
        if container:
            raw["booktitle"] = container
            text = "In " + container
            editors = [
                displayed_person(person, harvard)
                for person in entry.persons.get("editor", [])
            ]
            if editors:
                parsed["editors"] = [person for _, person in editors]
                text += ", edited by " + names_for_style(editors, harvard, editors=True)
            pieces.append(sentence(text))
        for key, label in (
            ("edition", "Edition"),
            ("series", "Series"),
            ("volume", "Volume"),
            ("number", "Number"),
            ("type", "Format"),
        ):
            if fields.get(key):
                raw[key] = fields[key]
                pieces.append(sentence(f"{label}: {fields[key]}"))
        place, publisher = fields.get("address"), fields.get("publisher")
        if place:
            parsed["publication_place"] = place
        if publisher:
            parsed["publisher"] = publisher
        if place or publisher:
            pieces.append(sentence(": ".join(filter(None, (place, publisher)))))
        if fields.get("pages"):
            parsed["pages"] = fields["pages"].replace("--", "–")
            pieces.append("pp. " + parsed["pages"] + ".")
    if raw:
        parsed["raw"] = raw

    supplied_doi = fields.get("doi")
    doi = normalize_doi(supplied_doi)
    url = fields.get("url", "")
    origin = "doi_field" if supplied_doi else None
    if not supplied_doi:
        doi = normalize_doi(url)
        if not doi:
            match = re.fullmatch(
                r"https?://[^/]+/document/doi/(10\.\d{4,9}/.+?)/(?:html|pdf)", url
            )
            doi = normalize_doi(match.group(1)) if match else None
        if doi:
            origin = "url_field"
    identifiers = []
    if doi:
        link = "https://doi.org/" + doi
        pieces.append(link)
        identifiers.append({"scheme": "doi", "value": link, "normalized": doi})
    elif url:
        pieces.append(("Available from: " if harvard else "") + url)
        identifiers.append({"scheme": "url", "value": url, "normalized": url})
    if harvard and (doi or url):
        if fields.get("urldate"):
            accessed = fields["urldate"]
            pieces.append(f"[Accessed {accessed}].")
            parsed.setdefault("raw", {})["access_date"] = accessed
    if identifiers:
        parsed["identifiers"] = identifiers
    reference = " ".join(pieces)
    audit = {"supplied_doi": supplied_doi, "normalized_doi": doi, "origin": origin}
    if supplied_doi and not doi:
        audit["warning"] = "Invalid supplied DOI; excluded from lookup"
    elif not doi and re.search(r"10\.\d{4,9}/", url):
        audit["warning"] = "Ambiguous DOI-like URL; preserved as URL only"
    if not authors:
        audit["missing_author"] = True
    Reference.model_validate(parsed)
    return reference, parsed, audit


def prepare(input_dir: Path, output: Path) -> list[dict[str, Any]]:
    from pybtex.database import parse_file

    rows, audits, hashes = [], [], {}
    for filename, source in (("DASI.bib", "dasi"), ("EbDA.bib", "ebda")):
        path = input_dir / filename
        entries = parse_file(path).entries
        if len(entries) != 100:
            raise ValueError(f"{filename}: expected 100 entries, found {len(entries)}")
        hashes[filename] = digest(path)
        for position, key in enumerate(sorted(entries)):
            style = "chicago" if position % 2 == 0 else "harvard"
            text, parsed, audit = render_entry(entries[key], style)
            ref_id = f"{source}_{key}"
            rows.append(
                {
                    "ref_id": ref_id,
                    "source": source,
                    "citation_style": style,
                    "original_string": text,
                    "parsed": parsed,
                }
            )
            audits.append({"ref_id": ref_id, **audit})
    manifest = {
        "created_at": utc_now(),
        "input_sha256": hashes,
        "script_sha256": digest(Path(__file__)),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "top_k": 10,
        "method": "doi_first_metadata_fallback",
        "references": len(rows),
    }
    prepared_path = output / "references.json"
    if (output / "search_results.json").exists() and prepared_path.exists():
        if json.loads(prepared_path.read_text()) != rows:
            raise ValueError(
                "Prepared references conflict with existing search checkpoint; use a new output directory"
            )
    write_json(prepared_path, rows)
    write_json(output / "doi_audit.json", audits)
    if not (output / "search_results.json").exists():
        write_json(output / "manifest.json", manifest)
    return rows


def candidate_dois(candidate: dict[str, Any], connector: Any, index: str) -> set[str]:
    values = []
    if index == "openalex":
        values.extend([candidate.get("doi"), (candidate.get("ids") or {}).get("doi")])
    elif index == "wikidata":
        values.extend(
            item.get("mainsnak", {}).get("datavalue", {}).get("value")
            for item in candidate.get("claims", {}).get("P356", [])
        )
    else:
        for text in candidate.get("texts", []) or []:
            for identifier in text.get("identifier", []) or []:
                value = identifier.get("doi")
                values.extend(value if isinstance(value, list) else [value])
    values.extend(
        identifier.normalized or identifier.value
        for identifier in connector._result_to_reference(candidate).identifiers
        if identifier.scheme == "doi"
    )
    return {
        doi
        for value in values
        if isinstance(value, str) and (doi := normalize_doi(value))
    }


def simplify(
    candidate: dict[str, Any], connector: Any, index: str, reference: Reference
) -> dict[str, Any]:
    if not candidate.get("id"):
        raise ValueError(f"{index} candidate is missing its provider ID")
    mapped = connector._result_to_reference(candidate)
    fields = reference_match_fields(mapped)
    is_match, details = custom_match(reference, fields)
    dois = sorted(candidate_dois(candidate, connector, index))
    return {
        **fields,
        "journal": mapped.journal_title,
        "ids": {f"{index}_id": candidate.get("id"), "doi": dois[0] if dois else None},
        "is_match": is_match,
        "match_details": details,
        "doi_identifiers": dois,
    }


def search_pair(
    row: dict[str, Any], connector: Any, index: str, state: dict[str, Any] | None = None
) -> dict[str, Any]:
    state = state if state is not None else {}
    reference = Reference.model_validate(row["parsed"])
    doi = next(
        (
            identifier.normalized or normalize_doi(identifier.value)
            for identifier in reference.identifiers
            if identifier.scheme == "doi"
        ),
        None,
    )
    selected = None
    if doi:
        if not state.get("doi_search", {}).get("success"):
            raw = connector.search_by_id(doi, identifier_type="doi", top_k=10)
            candidates = [
                simplify(candidate, connector, index, reference) for candidate in raw
            ]
            state["doi_search"] = {
                "success": True,
                "queried_doi": doi,
                "num_results": len(raw),
                "candidates": candidates,
                "searched_at": utc_now(),
            }
        selected = next(
            (
                candidate
                for candidate in state["doi_search"]["candidates"]
                if doi in candidate["doi_identifiers"]
            ),
            None,
        )
    if selected is None:
        search_reference = reference.model_copy(update={"identifiers": []})
        raw = connector.search(search_reference, top_k=10)
        selected = simplify(raw[0], connector, index, reference) if raw else None
        state["metadata_search"] = {
            "success": True,
            "num_results": len(raw),
            "top_result": selected,
            "searched_at": utc_now(),
        }
        state["retrieval_method"] = "metadata"
    else:
        state["retrieval_method"] = "doi"
        selected = {**selected, "ids": {**selected["ids"], "doi": doi}}
    state.update(
        {
            "success": True,
            "selected_result": selected,
            "doi_exact_match": bool(
                doi and selected and doi in selected["doi_identifiers"]
            ),
        }
    )
    state.pop("error", None)
    return state


def search(
    output: Path, connectors: dict[str, Any] | None = None, delay: float = 0.3
) -> list[dict[str, Any]]:
    prepared = json.loads((output / "references.json").read_text())
    path = output / "search_results.json"
    results = (
        json.loads(path.read_text())
        if path.exists()
        else [{**row, "search_results": {}} for row in prepared]
    )
    if [{k: row[k] for k in prepared[0]} for row in results] != prepared:
        raise ValueError("Search checkpoint does not match prepared references")
    manifest = json.loads((output / "manifest.json").read_text())
    if (
        manifest["method"] != "doi_first_metadata_fallback"
        or manifest["top_k"] != 10
        or manifest["script_sha256"] != digest(Path(__file__))
    ):
        raise ValueError(
            "Checkpoint script or search parameters differ; use a new output directory"
        )
    if connectors is None:
        openalex = OpenAlexConnector()
        if not openalex.api_key:
            raise RuntimeError(
                "Export OPENALEX_API before running authenticated searches"
            )
        openalex._get(f"{openalex.base_url}/rate-limit")
        connectors = {
            "openalex": openalex,
            "matilda": MatildaConnector(),
            "wikidata": WikidataConnector(),
        }
    for row in results:
        for index in INDEXES:
            state = row["search_results"].setdefault(index, {})
            if state.get("success"):
                continue
            try:
                for attempt in range(3):
                    try:
                        search_pair(row, connectors[index], index, state)
                        break
                    except Exception:
                        if index == "openalex" or attempt == 2:
                            raise
                        time.sleep(2**attempt)
            except Exception as exc:
                state.update(
                    success=False,
                    error={
                        "type": type(exc).__name__,
                        "stage": (
                            "metadata"
                            if state.get("doi_search", {}).get("success")
                            else (
                                "doi"
                                if any(
                                    i.get("scheme") == "doi"
                                    for i in row["parsed"].get("identifiers", [])
                                )
                                else "metadata"
                            )
                        ),
                    },
                )
                write_json(path, results)
                try:
                    raise RuntimeError(
                        f"Search failed; ref_id={row['ref_id']}; index={index}; error_type={type(exc).__name__}; checkpoint saved"
                    ) from None
                except RuntimeError:
                    LOGGER.exception("Benchmark search failed")
                    raise
            write_json(path, results)
            LOGGER.info(
                "Completed %s / %s using %s",
                row["ref_id"],
                index,
                state["retrieval_method"],
            )
            time.sleep(delay)
    manifest["search_completed_at"] = utc_now()
    write_json(output / "manifest.json", manifest)
    write_json(output / "publication_rows.json", publication_rows(results))
    return results


def publication_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(results) != 200 or len({row["ref_id"] for row in results}) != 200:
        raise ValueError("Publication requires 200 unique references")
    rows = []
    for row in results:
        for index in INDEXES:
            state = row["search_results"].get(index, {})
            if not state.get("success"):
                raise ValueError(f"Incomplete search: {row['ref_id']} / {index}")
            top = state.get("selected_result") or {}
            identifier = (top.get("ids") or {}).get(f"{index}_id")
            if identifier:
                identifier = str(identifier).rstrip("/").rsplit("/", 1)[-1]
            link = {
                "openalex": "https://openalex.org/",
                "matilda": "https://matilda.science/?l=en",
                "wikidata": "https://www.wikidata.org/wiki/Wikidata:Main_Page",
            }[index]
            if identifier:
                link = {
                    "openalex": f"https://openalex.org/works/{identifier.lower()}",
                    "matilda": f"https://matilda.science/work/{identifier}",
                    "wikidata": f"https://www.wikidata.org/wiki/{identifier}",
                }[index]
            rows.append(
                {
                    "ref_id": row["ref_id"],
                    "source": row["source"],
                    "original_ref_string": row["original_string"],
                    "index": index,
                    "matched_id": identifier or "Not Found",
                    "matched_doi": (top.get("ids") or {}).get("doi") or "",
                    "matched_result": {
                        key: top.get(key)
                        for key in ("first_author", "journal", "title", "year")
                    },
                    "is_match_by_similarity": bool(top.get("is_match")),
                    "matched_link": link,
                }
            )
    return rows


def merge_rows(
    existing: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged = list(existing)
    by_key = {(row["ref_id"], row["index"]): row for row in existing}
    if len(by_key) != len(existing):
        raise ValueError("Existing split has duplicate reference/index rows")
    for row in incoming:
        key = (row["ref_id"], row["index"])
        if key in by_key:
            if by_key[key] != row:
                raise ValueError(f"Conflicting published row: {key}")
        else:
            merged.append(row)
            by_key[key] = row
    return merged


def argilla_fields(row: dict[str, Any]) -> dict[str, str]:
    fields = {key: value for key, value in row.items() if key != "index"}
    fields["matched_result"] = (
        "```json\n"
        + json.dumps(row["matched_result"], ensure_ascii=False, indent=2)
        + "\n```"
    )
    fields["is_match_by_similarity"] = str(row["is_match_by_similarity"]).lower()
    fields["matched_link"] = f"[{row['matched_link']}]({row['matched_link']})"
    return fields


def settings_snapshot(settings: Any) -> dict[str, Any]:
    """Compare annotation configuration without server-assigned resource IDs."""
    snapshot = settings.serialize()
    for group in ("fields", "questions", "vectors", "metadata"):
        for item in snapshot.get(group, []):
            for key in ("id", "dataset_id", "inserted_at", "updated_at"):
                item.pop(key, None)
    return snapshot


def publish(output: Path, repo: str = HF_REPO) -> dict[str, Any]:
    import argilla as rg
    from datasets import Dataset, load_dataset
    from huggingface_hub import (
        CommitOperationAdd,
        CommitOperationDelete,
        DatasetCard,
        HfApi,
        hf_hub_download,
    )

    results = json.loads((output / "search_results.json").read_text())
    prepared = json.loads((output / "references.json").read_text())
    if [{key: row[key] for key in prepared[0]} for row in results] != prepared:
        raise ValueError("Search results differ from prepared references")
    rows = publication_rows(results)
    write_json(output / "publication_rows.json", rows)
    token = os.environ.get("ARGILLA_API_TOKEN")
    if not token or not token.strip():
        raise RuntimeError("Export a valid ARGILLA_API_TOKEN before publishing")
    client = rg.Argilla(
        api_url=os.getenv("ARGILLA_API_URL") or "https://argilla.graphia-ssh.eu/",
        api_key=token,
    )
    sources, targets = {}, {}
    for index in INDEXES:
        source = client.datasets(name=f"citation_linking_{index}")
        if source is None:
            raise ValueError(
                f"Source Argilla dataset missing: citation_linking_{index}"
            )
        write_json(
            output / "argilla_settings" / f"{index}.json",
            settings_snapshot(source.settings),
        )
        target = client.datasets(
            name=f"citation_linking_dasi_ebda_{index}", workspace=source.workspace.name
        )
        expected = {
            row["ref_id"]: argilla_fields(row) for row in rows if row["index"] == index
        }
        if target is not None:
            if settings_snapshot(target.settings) != settings_snapshot(source.settings):
                raise ValueError(f"Existing Argilla settings conflict: {index}")
            for record in target.records:
                if (
                    record.id not in expected
                    or record.fields.to_dict() != expected[record.id]
                ):
                    raise ValueError(f"Existing Argilla record conflicts: {index}")
        sources[index], targets[index] = source, target

    api = HfApi()
    info = api.dataset_info(repo)
    if not info.private:
        raise ValueError("The baseline HF repository must remain private")
    revision = info.sha
    cache_dir = str(output / "hf_cache")
    existing = load_dataset(repo, revision=revision, token=True, cache_dir=cache_dir)
    files = api.list_repo_files(repo, repo_type="dataset", revision=revision)
    staging = output / "hf_upload"
    operations, expected_splits, statistics = [], {}, []
    for split in ("full", "annotated"):
        dataset = existing[split]
        if set(dataset.column_names) != set(rows[0]):
            raise ValueError(f"Unexpected HF schema in {split}")
        merged = merge_rows(dataset.to_list(), rows)
        expected_splits[split] = merged
        updated = Dataset.from_list(merged, features=dataset.features)
        relative = f"data/{split}-00000-of-00001.parquet"
        local = staging / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        updated.to_parquet(str(local))
        operations.append(
            CommitOperationAdd(path_in_repo=relative, path_or_fileobj=str(local))
        )
        for old_file in files:
            if (
                re.fullmatch(rf"data/{split}-\d+-of-\d+\.parquet", old_file)
                and old_file != relative
            ):
                operations.append(CommitOperationDelete(path_in_repo=old_file))
        statistics.append(
            {
                "name": split,
                "num_examples": len(updated),
                "num_bytes": updated.data.nbytes,
            }
        )

    card_path = hf_hub_download(
        repo,
        "README.md",
        repo_type="dataset",
        revision=revision,
        cache_dir=str(output / "hf_cache/hub"),
    )
    card = DatasetCard(Path(card_path).read_text())
    card_info = card.data.dataset_info
    if not isinstance(card_info, dict):
        raise ValueError("Expected the existing single-configuration dataset card")
    card_info["splits"] = statistics
    card_info["dataset_size"] = sum(item["num_bytes"] for item in statistics)
    card_info["download_size"] = sum(
        (staging / f"data/{split}-00000-of-00001.parquet").stat().st_size
        for split in ("full", "annotated")
    )
    marker = "<!-- dasi-ebda-extension -->"
    end_marker = "<!-- /dasi-ebda-extension -->"
    section = f"{marker}\n## DASI/EbDA extension\n\n200 references (100 DASI and 100 EbDA), each searched against OpenAlex, Matilda, and Wikidata. Reference strings are complete bibliography entries generated from BibTeX, evenly mixing Chicago author–date and Leeds Harvard per corpus. Parsed objects contain only displayed values. Supplied DOIs are unverified.\n\nThis extension uses DOI-first lookup with metadata fallback; historical rows use metadata-only retrieval. Exact DOI hits retain the existing title/author/year similarity flag and require human verification. The annotated split denotes annotation selection, not completed labels. Detailed retrieval provenance and the DOI audit are in `extensions/dasi_ebda/`.\n\nFull: {len(expected_splits['full'])} rows. Annotated: {len(expected_splits['annotated'])} rows.\n{end_marker}"
    if marker in card.text:
        if end_marker not in card.text:
            raise ValueError("Existing extension card section has no closing marker")
        card.text = re.sub(
            re.escape(marker) + r".*?" + re.escape(end_marker),
            lambda _: section,
            card.text,
            flags=re.S,
        )
    else:
        card.text = card.text.rstrip() + "\n\n" + section + "\n"
    readme = staging / "README.md"
    readme.write_text(str(card), encoding="utf-8")
    operations.append(
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(readme))
    )
    for filename in (
        "references.json",
        "search_results.json",
        "publication_rows.json",
        "doi_audit.json",
        "manifest.json",
    ):
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"extensions/dasi_ebda/{filename}",
                path_or_fileobj=str(output / filename),
            )
        )
    commit = api.create_commit(
        repo,
        repo_type="dataset",
        operations=operations,
        parent_commit=revision,
        commit_message="Add DASI and EbDA DOI-first linking extension",
    )
    reloaded = load_dataset(repo, revision=commit.oid, token=True, cache_dir=cache_dir)
    for split, expected in expected_splits.items():
        if reloaded[split].to_list() != expected:
            raise RuntimeError(f"HF readback differs: {split}")
    report: dict[str, Any] = {
        "hf_repo": repo,
        "hf_commit": commit.oid,
        "splits": statistics,
        "argilla": {},
    }
    write_json(output / "publication_report.json", report)
    for index in INDEXES:
        source, target = sources[index], targets[index]
        if target is None:
            target = rg.Dataset(
                name=f"citation_linking_dasi_ebda_{index}",
                workspace=source.workspace,
                settings=rg.Settings.from_json(
                    output / "argilla_settings" / f"{index}.json"
                ),
                client=client,
            )
            target.create()
        expected = {
            row["ref_id"]: argilla_fields(row) for row in rows if row["index"] == index
        }
        existing_ids = {record.id for record in target.records}
        records = [
            rg.Record(id=ref_id, fields=fields)
            for ref_id, fields in expected.items()
            if ref_id not in existing_ids
        ]
        if records:
            target.records.log(records)
        check = client.datasets(name=target.name, workspace=source.workspace.name)
        actual = {record.id: record.fields.to_dict() for record in check.records}
        if actual != expected or settings_snapshot(check.settings) != settings_snapshot(
            source.settings
        ):
            raise RuntimeError(f"Argilla readback differs: {index}")
        report["argilla"][index] = {
            "name": target.name,
            "workspace": source.workspace.name,
            "records": len(actual),
        }
        write_json(output / "publication_report.json", report)
    return report


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "search", "publish"))
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hf-repo", default=HF_REPO)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.stage == "prepare":
        rows = prepare(args.input_dir, args.output_dir)
        LOGGER.info(
            "Prepared %s full bibliography entries in %s", len(rows), args.output_dir
        )
    elif args.stage == "search":
        search(args.output_dir)
    else:
        publish(args.output_dir, args.hf_repo)


if __name__ == "__main__":
    main()
