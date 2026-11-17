"""Raw annotations -> reviewed gold dataset with candidates -> Hugging Face ``citation_linking``.

Second of the two dataset scripts (``annotation/export_argilla_annotations.py`` is the first). It
reads the raw Argilla export, applies every reviewed correction, attaches the frozen
search candidates and writes the release. Run from the repository root:

    python benchmarks/citation_linking/dataset/build_annotation_dataset.py --output-dir <new dir>
    python benchmarks/citation_linking/dataset/build_annotation_dataset.py --output-dir <new dir> --push

    # search an index again first (live, resumable; moves its cut-off date)
    python benchmarks/citation_linking/dataset/build_annotation_dataset.py --search opencitations ...

All corrections live in this file: ``ANNOTATION_FIXES`` (contradictory answers) and
``REVIEW_OVERRIDES`` (gold targets and DOIs). ``CORRECTIONS.md`` explains each one.
Splits are frozen to the published release unless ``--new-splits`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from citation_index.core.models import Reference  # noqa: E402
from citation_index.utils.identifier_parser import parse_identifier  # noqa: E402
from benchmarks.citation_linking.dataset import candidate_pool  # noqa: E402
from benchmarks.citation_linking.dataset.release_splits import (  # noqa: E402
    CONFIGS,
    INDEXES,
    RATIOS,
    SPLITS,
    preserve_splits,
    published_splits,
    split_rows,
    split_statistics,
)
from benchmarks.citation_linking.annotation.export_argilla_annotations import (  # noqa: E402
    RAW,
    align_schema,
    flatten_to_row,
)

SOURCE_FILES = (
    "api_search_test_set.jsonl",
    "brillkg_sample_200_parsed.jsonl",
    "legal_study_data_mpilhlt/legal_study_parsed.json",
    "DASI_EBDA/references.json",
)
HF_REPO = "yurui983/citation_linking"
# Contradictory or incomplete answers, corrected after review. Each entry replaces one
# answer of one (ref_id, index) annotation; the label is derived again from the answers.
# (ref_id, index): (question, corrected answer, what changed and why)
ANNOTATION_FIXES = {
    ("brill_a15e3650-8e13-4550-8b67-2158fc2df786_634", "openalex"): (
        "no_match",
        "true",
        "no_match: false -> true (no candidate returned and no_match was left false; the matilda and wikidata rows for this reference both read correctly-not-found, so this is a correct abstention)",
    ),
    ("10.1515_zfrs-1980-0104_instance-63", "openalex"): (
        "no_match",
        "true",
        "no_match: false -> true (annotator marked the record as present in OpenAlex but gave no ID; OpenAlex holds 91 Marshall Clinard works, none on victimization, and no title match for 'Comparative Crime Victimization Surveys' 1978 -- the work is absent, so the returned candidate is a spurious link)",
    ),
    ("cex_COM-SCI_28_20", "matilda"): (
        "is_match_correct",
        "false",
        "is_match_correct: true -> false (candidate 'Pixeldust Studios Reptopia Magic Leap Experience' is a different work from the reference 'Magic Leap. 2018'; matches how openalex was answered)",
    ),
    ("10.1111_1467-6478.00057_instance-71", "matilda"): (
        "correct_id",
        "[SKIP] multiple references packed in one single context (Harvard Law Review eds. 1990; Dean v. District of Columbia 1995; Overington, The Age 1997)",
        "correct_id -> [SKIP] (reference text bundles three publications, the same class as the 18 references the openalex annotator marked; excluded from the analysis)",
    ),
    ("10.1515_zfrs-1980-0104_instance-7", "matilda"): (
        "is_match_correct",
        "false",
        "is_match_correct: true -> false (candidate is an unrelated PSSRI volume on the China-Pakistan Economic Corridor)",
    ),
    ("cex_COM-SCI_28_20", "wikidata"): (
        "is_match_correct",
        "false",
        "is_match_correct: true -> false (candidate Q18351488 is Magic Leap the company, not a publication record)",
    ),
    ("brill_7b98252c-c219-491b-8a44-3e4286c9ca47_2", "opencitations"): (
        "is_match_correct",
        "false",
        "is_match_correct: true -> false (candidate br/06020233318 is Barco, a 2011 catalogue of the BnF Hebrew manuscripts 1-32, not the cited Latin manuscript 'ms lat. 8589'; annotator also marked no_match=true and the openalex/matilda/wikidata rows for this reference read correctly-not-found, so this is a spurious link)",
    ),
    ("dasi_Brunner2001", "openalex"): (
        "is_match_correct",
        "false",
        "is_match_correct: true -> false (candidate W2750627499 is Linke, 'Gender - Bilder - Sanaa' (2017), an unrelated work; annotator also marked no_match=true and the matilda/wikidata rows for this reference read correctly-not-found, so this is a spurious link)",
    ),
}
# Reviewed corrections (2026-10, 2026-11) that override raw annotation evidence.
# gold_id entries are keyed by target index. cex_ENV-SCI_54_63 needs no entry:
# both DOIs are valid, so its gold_doi stays null and the conflict stays listed.
# References removed from the release after review, with the reason.
EXCLUDED_REFERENCES = {
    # Authors and year are Harvey & Rauch 1997 (Health & Social Work 22); title,
    # journal and volume are Harvey & Hill 2004 (Social Work 49). The cited work
    # cannot be told, and both papers are indexed.
    "cex_SOC-SCI_102_13": "reference_blends_two_works",
}
# Cleared gold targets searched for again (2026-11): the index holds no record of
# the cited work, so these null golds are confirmed negatives, not unknowns.
VERIFIED_ABSENT = frozenset(
    {
        ("10.1111_1467-6478.00057_instance-38", "openalex"),
        ("excite_32211_33", "openalex"),
        ("linkedbook_26", "openalex"),
        ("linkedbook_73", "openalex"),
        ("brill_20f2caaa-677a-42c1-8857-19e196f8c74e_331", "matilda"),
        ("10.1515_zfrs-1980-0104_instance-43", "opencitations"),
        ("10.1515_zfrs-1980-0104_instance-43", "wikidata"),
        ("dasi_Wellsted1837", "opencitations"),
        ("excite_9543_36", "opencitations"),
        ("excite_9543_36", "wikidata"),
        ("linkedbook_85", "opencitations"),
    }
)
REVIEW_OVERRIDES = {
    "10.1111_1467-6478.00057_instance-38": {"gold_id": {"openalex": None}},
    "excite_32211_33": {"gold_id": {"openalex": None}},
    "linkedbook_26": {"gold_id": {"openalex": None}},
    # Dead ID; the Italian translation the reference cites was found again (2026-11).
    "linkedbook_5": {"gold_id": {"openalex": "W618320456"}},
    "linkedbook_73": {"gold_id": {"openalex": None}},
    "linkedbook_94": {"gold_id": {"openalex": "W7019729045"}},
    "cex_PHY-AST_96_10": {
        "gold_id": {"openalex": "W1968542656"},
    },
    "excite_9543_52": {
        "gold_id": {"openalex": "W4233971661"},
    },
    # Same book under two DOIs; keep the older record (original 1989 date).
    "10.1111_1467-6478.00057_instance-63": {
        "gold_doi": "10.1515/9780691215983",
        "gold_id": {"matilda": "d6313d68-4fc7-4307-9b92-48b25363cdda"},
    },
    # OpenAlex is right; the Matilda record is a review of the book by another author.
    "brill_20f2caaa-677a-42c1-8857-19e196f8c74e_331": {
        "gold_doi": "10.1017/cbo9780511583599",
        "gold_id": {"matilda": None},
    },
    # The cited work is the 1996 book chapter (Matilda, OpenAlex W4299787999). The
    # OpenCitations and Wikidata records are the 1994 journal version of the same text.
    "excite_9543_36": {
        "gold_doi": "10.1515/9781685853952-010",
        "gold_id": {
            "openalex": "W4299787999",
            "opencitations": None,
            "wikidata": None,
        },
    },
    # The cited work is a book chapter; OpenAlex had matched the whole book.
    "excite_9543_10": {
        "gold_doi": "10.7560/720466-008",
        "gold_id": {"openalex": "W4408028697"},
    },
    # Cross-index review after the OpenCitations round (2026-11): references where
    # annotators judged the same DOI differently. A gold record is the cited
    # publication itself, not a review of it or another edition of the same text.
    # Rejected candidates that the annotator's index does hold:
    "excite_9543_33": {"gold_id": {"openalex": "W58313853"}},
    "10.1515_zfrs-1980-0103_instance-30": {"gold_id": {"openalex": "W181794323"}},
    # OpenAlex had accepted a different paper (Korhonen et al., Ecological Economics).
    "cex_ENV-SCI_54_70": {
        "gold_doi": "10.1016/j.resconrec.2017.08.027",
        "gold_id": {"openalex": "W2769483102"},
    },
    # Book records that carry the DOI of a book review: the gold IDs stand, the DOI
    # does not identify the cited book.
    "excite_9543_34": {"gold_doi": None},
    "10.1111_1467-6478.00057_instance-40": {"gold_doi": None},
    # 1924 four-page article of the same title, not the 1926 book (Wikidata has it).
    "linkedbook_85": {"gold_doi": None, "gold_id": {"opencitations": None}},
    # Reviews of the cited book (by Drinan and by Foster), not the book.
    "10.1515_zfrs-1980-0104_instance-43": {
        "gold_doi": None,
        "gold_id": {"opencitations": None, "wikidata": None},
    },
    # Harvey & Hill 2004: same title and journal, different co-author and year.
    # DASI/EBDA review (2026-11): these references print their DOI, which was looked
    # up in each index. OpenAlex merged the annotated record into another one.
    "dasi_Yule2002": {"gold_id": {"openalex": "W3112109909"}},
    # The accepted OpenCitations record is another article by Wellsted in the same
    # volume (p. 102, DOI 10.2307/1797516); the cited DOI is not in OpenCitations.
    "dasi_Wellsted1837": {
        "gold_doi": "10.2307/1797510",
        "gold_id": {"opencitations": None},
    },
    # Matilda holds the article (found by its printed DOI); no candidate was shown.
    "ebda_DalleyTeissier1992": {
        "gold_id": {"matilda": "7847d2f3-385d-497b-a716-8e030d4d3a56"}
    },
}
PARSED_TEXT_FIELDS = (
    "full_title",
    "journal_title",
    "publisher",
    "publication_place",
    "publication_date_raw",
    "volume",
    "issue",
    "pages",
    "cited_range",
    "footnote_number",
    "ref_type",
)


def read_sources(directory: Path) -> dict[str, dict[str, Any]]:
    """Read the original reference inputs, excluding file-level metadata records."""
    sources = {}
    for name in SOURCE_FILES:
        path = directory / name
        text = path.read_text(encoding="utf-8")
        records = (
            json.loads(text)
            if path.suffix == ".json"
            else [json.loads(line) for line in text.splitlines() if line.strip()]
        )
        for row in records:
            if "_metadata" in row:
                continue
            ref_id = row["ref_id"]
            if ref_id in sources:
                raise ValueError(f"Duplicate source reference: {ref_id}")
            sources[ref_id] = row
    return sources


def normalize_id(value: str, index: str) -> str:
    """Validate a gold target and return the provider's short identifier."""
    value = value.strip()
    if index == "opencitations":
        # OMID of a bibliographic resource; also accepts omid:/URL forms ending in br/<digits>.
        match = re.fullmatch(r"(?:.*[/:])?(br/0[1-9]\d*)", value.lower())
        if not match:
            raise ValueError(f"Invalid opencitations gold ID: {value!r}")
        return match.group(1)
    if "://" in value:
        url = urlparse(value)
        domain = {
            "openalex": "openalex.org",
            "wikidata": "wikidata.org",
            "matilda": "matilda.science",
        }[index]
        if url.hostname not in (domain, "www." + domain):
            raise ValueError(f"Unexpected {index} identifier host")
        value = url.path.rstrip("/").rsplit("/", 1)[-1]
    if index == "matilda":
        return str(UUID(value))
    value = value.upper()
    pattern = r"W[1-9]\d*" if index == "openalex" else r"Q[1-9]\d*"
    if not re.fullmatch(pattern, value):
        raise ValueError(f"Invalid {index} gold ID: {value!r}")
    return value


def normalize_doi(value: str) -> str:
    identifier = parse_identifier(value, "doi")
    doi = identifier.normalized.lower() if identifier else ""
    if not re.fullmatch(r"10\.\d{4,9}/\S+", doi):
        raise ValueError(f"Invalid DOI: {value!r}")
    return doi


def normalize_parsed(raw: dict[str, Any] | None, source: str) -> dict[str, Any] | None:
    """Reuse the runtime model's legacy-field and LinkedBook tag conversion."""
    if not raw:
        return None
    if source == "linkedbook":
        ref, authors = Reference.from_linkedbook({"tags": raw})
        ref.authors = authors or None
    else:
        ref = Reference.model_validate(raw)
    parsed = {key: getattr(ref, key) for key in PARSED_TEXT_FIELDS}
    parsed["publication_year"] = ref.publication_year
    for field in ("authors", "editors", "translator"):
        names = []
        for person in getattr(ref, field) or []:
            if isinstance(person, str):
                name = person
            elif hasattr(person, "name"):
                name = person.name
            else:
                name = " ".join(
                    getattr(person, part)
                    for part in (
                        "role_name",
                        "first_name",
                        "middle_name",
                        "name_link",
                        "surname",
                    )
                    if getattr(person, part)
                )
            if name:
                names.append(name)
        parsed[field] = names or None
    parsed["identifiers"] = [item.model_dump() for item in ref.identifiers]
    return parsed


def read_raw_file(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def corrected_annotations(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Raw export rows with ``ANNOTATION_FIXES`` applied and labels derived again."""
    rows = []
    for row in raw:
        fix = ANNOTATION_FIXES.get((row["ref_id"], row["index"]))
        if fix:
            question, answer, reason = fix
            row = {**row, question: answer, f"{question}.responses": [answer]}
        rows.append({**flatten_to_row(row, row["index"]), "manual_fix": fix and fix[2]})
    return align_schema(rows)


def prepare_rows(
    annotations: list[dict[str, Any]], sources: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project finalized annotations onto public fields; withhold conflicting DOIs."""
    by_ref = defaultdict(list)
    keys = set()
    for row in annotations:
        key = (row["ref_id"], row["index"])
        if key in keys:
            raise ValueError(f"Duplicate annotation: {key}")
        if row["index"] not in INDEXES:
            raise ValueError(f"Unsupported target index: {row['index']}")
        keys.add(key)
        by_ref[row["ref_id"]].append(row)

    output, excluded, conflicts = [], [], []
    # Corrections made to the working copy of the raw annotations after review.
    manual_fixes = [
        {"ref_id": row["ref_id"], "index": row["index"], "fix": row["manual_fix"]}
        for row in annotations
        if row.get("manual_fix")
    ]
    for ref_id, group in sorted(by_ref.items()):
        if ref_id in EXCLUDED_REFERENCES:
            excluded.append({"ref_id": ref_id, "reason": EXCLUDED_REFERENCES[ref_id]})
            continue
        if any(row["label"] == "multi_ref_context" for row in group):
            excluded.append({"ref_id": ref_id, "reason": "multi_ref_context"})
            continue
        reasons = set()
        if {row["index"] for row in group} != set(INDEXES):
            reasons.add("incomplete_index_annotations")
        for row in group:
            if row["label"] not in {
                "correct_match",
                "wrong_match",
                "no_record_in_index",
            }:
                reasons.add(row["label"])
            if row["needs_review"]:
                reasons.add(row["review_reason"])
            for question in ("is_match_correct", "no_match", "correct_id"):
                submitted = [
                    value
                    for value, status in zip(
                        row.get(f"{question}.responses") or [],
                        row.get(f"{question}.responses.status") or [],
                    )
                    if status == "submitted"
                ]
                if len(set(submitted)) > 1:
                    reasons.add("conflicting_submitted_responses")
                if row.get(question) is not None and not submitted:
                    reasons.add("unsubmitted_response")
        if reasons:
            excluded.append({"ref_id": ref_id, "reason": ", ".join(sorted(reasons))})
            continue

        original = sources[ref_id]
        source = group[0]["source"]
        reference = group[0]["original_ref_string"]
        if not reference.strip() or any(
            row["source"] != source or row["original_ref_string"] != reference
            for row in group
        ):
            raise ValueError(f"Inconsistent reference metadata: {ref_id}")
        original_source = original["source"]
        if source == "legal_study_mpilhlt":
            expected_doi = ref_id.rsplit("_instance-", 1)[0].replace("_", "/", 1)
            if normalize_doi(original_source) != expected_doi:
                raise ValueError(f"Citing-document mismatch: {ref_id}")
            original_source = source
        if original_source != source or " ".join(
            original["original_string"].split()
        ) != " ".join(reference.split()):
            raise ValueError(f"Source text or corpus mismatch: {ref_id}")
        document_id = original.get("file_id") or original.get("publication_id")
        if source == "legal_study_mpilhlt":
            if "_instance-" not in ref_id:
                raise ValueError(f"Missing legal citing-document identifier: {ref_id}")
            document_id = ref_id.rsplit("_instance-", 1)[0]
        # LinkedBook's grouped file contains synthetic batches, not citing documents.
        if source == "linkedbook":
            document_id = None
        parsed = normalize_parsed(original.get("parsed"), source)
        evidence = [
            {"index": row["index"], "doi": normalize_doi(row["matched_doi"])}
            for row in group
            if row["label"] == "correct_match" and row.get("matched_doi")
        ]
        dois = {item["doi"] for item in evidence}
        gold_doi = next(iter(dois)) if len(dois) == 1 else None
        fix = REVIEW_OVERRIDES.get(ref_id, {})
        if "gold_doi" in fix:
            gold_doi = fix["gold_doi"] and normalize_doi(fix["gold_doi"])
        elif len(dois) > 1:
            conflicts.append({"ref_id": ref_id, "evidence": evidence})
        for row in sorted(group, key=lambda item: item["index"]):
            target = None
            if row["label"] != "no_record_in_index":
                target = normalize_id(
                    (
                        row["matched_id"]
                        if row["label"] == "correct_match"
                        else row["gold_id"]
                    ),
                    row["index"],
                )
            if row["index"] in fix.get("gold_id", {}):
                corrected = fix["gold_id"][row["index"]]
                target = (
                    normalize_id(corrected, row["index"])
                    if corrected is not None
                    else None
                )
            output.append(
                {
                    "ref_id": ref_id,
                    "source": source,
                    "document_id": (
                        str(document_id) if document_id is not None else None
                    ),
                    "reference": reference,
                    "parsed_reference": parsed,
                    "index": row["index"],
                    "gold_id": target,
                    "gold_doi": gold_doi,
                }
            )
    if not output:
        raise ValueError("No finalized annotations remain after exclusions")
    return output, {
        "excluded_references": excluded,
        "doi_conflicts": conflicts,
        "manual_fixes": manual_fixes,
    }


def arrow_table(rows: list[dict[str, Any]]) -> Any:
    """Use explicit nullable types so all split files share the same schema."""
    import pyarrow as pa

    parsed = [(field, pa.string()) for field in PARSED_TEXT_FIELDS]
    parsed += [("publication_year", pa.int64())]
    parsed += [
        (field, pa.list_(pa.string())) for field in ("authors", "editors", "translator")
    ]
    parsed += [
        (
            "identifiers",
            pa.list_(
                pa.struct(
                    [
                        (field, pa.string())
                        for field in ("scheme", "value", "normalized")
                    ]
                )
            ),
        )
    ]
    schema = pa.schema(
        [
            ("ref_id", pa.string()),
            ("source", pa.string()),
            ("document_id", pa.string()),
            ("reference", pa.string()),
            ("parsed_reference", pa.struct(parsed)),
            ("index", pa.string()),
            ("gold_id", pa.string()),
            ("gold_doi", pa.string()),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def write_release(
    rows: list[dict[str, Any]],
    review: dict[str, Any],
    output: Path,
    seed: int,
    inputs: list[Path],
    card_path: Path,
    split_from: Path | None = None,
    candidates: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    import pyarrow
    import pyarrow.parquet as pq
    import sklearn

    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    table = arrow_table(rows)
    assignments = {
        config: (
            preserve_splits(
                rows, config, split_from, seed, frozenset(EXCLUDED_REFERENCES)
            )
            if split_from is not None
            else split_rows(rows, config, seed)
        )
        for config in CONFIGS
    }
    card = card_path.read_text(encoding="utf-8")
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "rows": len(rows),
        "references": len({r["ref_id"] for r in rows}),
        "target_ratios": dict(zip(SPLITS, RATIOS)),
        "versions": {
            "scikit-learn": sklearn.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "inputs": [
            {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in inputs
        ],
        "review": review,
        "annotation_fixes": [
            {"ref_id": ref_id, "index": index, "question": question, "answer": answer}
            for (ref_id, index), (question, answer, _) in ANNOTATION_FIXES.items()
        ],
        "target_overrides": REVIEW_OVERRIDES,
        "configurations": {},
    }
    yaml = ["---", "pretty_name: Citation Linking", "configs:"]
    stats_lines = [
        "\n### Build statistics\n",
        f"{summary['references']} references; {len(rows)} reference–index rows.\n",
        "| Configuration | Split | Rows | References | Gold IDs | Gold DOIs |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for config, (partitions, split_info) in assignments.items():
        yaml.append(f"- config_name: {config}")
        if config == CONFIGS[0]:
            yaml.append("  default: true")
        yaml.append("  data_files:")
        stats = split_statistics(rows, partitions)
        summary["configurations"][config] = {**split_info, "splits": stats}
        (output / config).mkdir()
        for split, positions in partitions.items():
            path = output / config / f"{split}.parquet"
            pq.write_table(table.take(positions), path, compression="zstd")
            if pq.read_table(path).to_pylist() != [rows[i] for i in positions]:
                raise ValueError(f"Parquet round-trip failed: {config}/{split}")
            yaml += [f"  - split: {split}", f"    path: {config}/{split}.parquet"]
            s = stats[split]
            stats_lines.append(
                f"| {config} | {split} | {s['rows']} | {s['references']} | {s['gold_ids']} | {s['gold_dois']} |"
            )
    stats_lines += [
        "",
        f"Seed: {seed}. Exact assignments, source hashes, library versions, and per-stratum counts are in `build_report.json`.",
        f"Excluded references: {len(review['excluded_references'])}. References with two valid DOIs for the same work: {len(review['doi_conflicts'])}; their `gold_doi` is null.",
        f"Annotation answers corrected after review: {len(review['manual_fixes'])}; references with a reviewed gold override: {len(REVIEW_OVERRIDES)}. `CORRECTIONS.md` lists every one with its reason.",
        "Gold DOIs come from metadata of human-confirmed matches, reconciled across indexes for each reference. DOI fields have not been independently adjudicated. Unknown values remain null; no live index lookups were performed.",
    ]
    if candidates is not None:
        (output / "candidates").mkdir()
        pq.write_table(
            candidate_pool.candidates_table(candidates),
            output / "candidates/search.parquet",
            compression="zstd",
        )
        yaml += [
            "- config_name: candidates",
            "  data_files:",
            "  - split: search",
            "    path: candidates/search.parquet",
        ]
        days: dict[str, list[str]] = defaultdict(list)
        for row in candidates:
            days[row["index"]].append(row["retrieved_on"])
        summary["candidates"] = {
            index: {
                "tasks": len(found),
                "searched_from": min(found),
                "searched_until": max(found),
                "candidates": sum(
                    len(c["candidates"]) for c in candidates if c["index"] == index
                ),
                "oracle_candidates": sum(
                    bool(c["oracle"]) for c in candidates if c["index"] == index
                ),
            }
            for index, found in sorted(days.items())
        }
        stats_lines += [
            "",
            "| Index | Searched (cut-off) | Tasks | Candidates | Gold found by search | Oracle records |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for index, info in summary["candidates"].items():
            gold = sum(r["index"] == index and r["gold_id"] is not None for r in rows)
            when = " to ".join(
                dict.fromkeys((info["searched_from"], info["searched_until"]))
            )
            stats_lines.append(
                f"| {index} | {when} | {info['tasks']} | {info['candidates']} | "
                f"{gold - info['oracle_candidates']} of {gold} | {info['oracle_candidates']} |"
            )
    (output / "CORRECTIONS.md").write_text(
        (HERE / "CORRECTIONS.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    yaml += ["---", ""]
    (output / "README.md").write_text(
        "\n".join(yaml) + card + "\n".join(stats_lines) + "\n", encoding="utf-8"
    )
    (output / "build_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary, assignments[CONFIGS[1]][0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        nargs="+",
        default=[RAW / "annotations.jsonl", RAW / "dasi_ebda.jsonl"],
        help="raw exports",
    )
    parser.add_argument("--source-dir", type=Path, default=DATA / "sources")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "logs/citation_linking_release"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-from",
        type=Path,
        help="Release directory whose row order and splits are kept (default: published)",
    )
    parser.add_argument(
        "--new-splits", action="store_true", help="Assign splits from scratch"
    )
    parser.add_argument(
        "--search",
        nargs="+",
        choices=list(candidate_pool.SEARCHES),
        default=[],
        help="Search these indexes again before building (live, resumable)",
    )
    parser.add_argument(
        "--search-only", action="store_true", help="Stop after --search, build nothing"
    )
    parser.add_argument("--candidates", type=Path, default=candidate_pool.STORE)
    parser.add_argument("--pool-dir", type=Path, default=candidate_pool.POOL_DIR)
    parser.add_argument("--repo", default=HF_REPO)
    parser.add_argument("--push", action="store_true", help="Publish to Hugging Face")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Corrected copies of the raw exports: input of the analysis and the training export.
    annotations = []
    for path in args.annotations:
        corrected = corrected_annotations(read_raw_file(path))
        (DATA / "annotations/corrected" / path.name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in corrected),
            encoding="utf-8",
        )
        annotations += corrected
    unknown = set(ANNOTATION_FIXES) - {(r["ref_id"], r["index"]) for r in annotations}
    if unknown:
        raise ValueError(f"Fixes for unknown annotations: {sorted(unknown)}")
    annotations = align_schema(annotations)
    rows, review = prepare_rows(annotations, read_sources(args.source_dir))

    store = candidate_pool.read_store(args.candidates)
    for index in args.search:
        candidate_pool.search(rows, store, index)
        candidate_pool.write_store(store, args.candidates)  # keep finished indexes
    if args.search_only:
        return
    candidates = candidate_pool.attach(rows, store)
    candidate_pool.write_store(store, args.candidates)

    split_from = args.split_from
    if split_from is None and not args.new_splits:
        split_from = published_splits(args.repo)
    inputs = [
        *args.annotations,
        *(args.source_dir / name for name in SOURCE_FILES),
        Path(__file__),
        HERE / "candidate_pool.py",
        HERE / "release_splits.py",
        HERE / "DATASET_CARD.md",
        HERE / "CORRECTIONS.md",
    ]
    summary, partitions = write_release(
        rows,
        review,
        args.output_dir,
        args.seed,
        inputs,
        HERE / "DATASET_CARD.md",
        split_from,
        candidates,
    )
    print(
        f"Built {summary['references']} references / {summary['rows']} rows in {args.output_dir}"
    )
    for config, info in summary["configurations"].items():
        print(
            f"{config}: "
            + ", ".join(
                f"{split}={stats['rows']}" for split, stats in info["splits"].items()
            )
        )
    print(
        f"Excluded references: {len(review['excluded_references'])}; unresolved DOI conflicts: {len(review['doi_conflicts'])}"
    )

    revision = "local"
    if args.push:
        from huggingface_hub import HfApi

        commit = HfApi().upload_folder(
            folder_path=args.output_dir,
            repo_id=args.repo,
            repo_type="dataset",
            commit_message=f"reviewed gold with search candidates ({summary['rows']} tasks)",
        )
        revision = commit.oid
        print(f"Pushed {args.repo} at {revision}")
    order = [(split, i) for split in SPLITS for i in partitions[split]]
    candidate_pool.write_pools(
        [{**rows[i], "split": split} for split, i in order],
        [candidates[i] for _, i in order],
        revision,
        args.pool_dir,
    )


if __name__ == "__main__":
    main()
