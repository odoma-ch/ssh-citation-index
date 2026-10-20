"""Build local Hugging Face release files from reviewed citation-linking annotations.

Run from the repository root with the benchmark environment (pyarrow and
scikit-learn installed):

    python benchmarks/citation_linking/build_annotation_dataset.py

Outputs default to logs/citation_linking_release. No network calls or uploads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import warnings
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from citation_index.core.models import Reference  # noqa: E402
from citation_index.utils.identifier_parser import parse_identifier  # noqa: E402
from export_argilla_annotations import load_local  # noqa: E402

INDEXES = ("matilda", "openalex", "wikidata")
SPLITS = ("train", "validation", "test")
RATIOS = (0.6, 0.2, 0.2)
CONFIGS = ("reference_disjoint", "reference_index_disjoint")
SOURCE_FILES = (
    "api_search_test_set.jsonl",
    "brillkg_sample_200_parsed.jsonl",
    "legal_study_data_mpilhlt/legal_study_parsed.json",
)
# Reviewed corrections (2026-10) that override raw annotation evidence.
# gold_id entries are keyed by target index. cex_ENV-SCI_54_63 needs no entry:
# both DOIs are valid, so its gold_doi stays null and the conflict stays listed.
REVIEW_OVERRIDES = {
    "10.1111_1467-6478.00057_instance-38": {"gold_id": {"openalex": None}},
    "excite_32211_33": {"gold_id": {"openalex": None}},
    "linkedbook_26": {"gold_id": {"openalex": None}},
    "linkedbook_5": {"gold_id": {"openalex": None}},
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
    # OpenAlex is right; the Matilda DOI belongs to a work with a different author.
    "brill_20f2caaa-677a-42c1-8857-19e196f8c74e_331": {
        "gold_doi": "10.1017/cbo9780511583599",
    },
    # The cited work is a book chapter; OpenAlex had matched the whole book.
    "excite_9543_10": {
        "gold_doi": "10.7560/720466-008",
        "gold_id": {"openalex": "W4408028697"},
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
    for ref_id, group in sorted(by_ref.items()):
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
            gold_doi = normalize_doi(fix["gold_doi"])
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
    return output, {"excluded_references": excluded, "doi_conflicts": conflicts}


def reference_groups(rows: list[dict[str, Any]]) -> list[str]:
    """Join references transitively by normalized text, gold DOI, or provider ID."""
    parents = {row["ref_id"]: row["ref_id"] for row in rows}

    def root(ref_id: str) -> str:
        while parents[ref_id] != ref_id:
            parents[ref_id] = parents[parents[ref_id]]
            ref_id = parents[ref_id]
        return ref_id

    owners = {}
    for row in rows:
        ref_id = row["ref_id"]
        identities = [("text", " ".join(row["reference"].casefold().split()))]
        if row["gold_id"]:
            identities.append((row["index"], row["gold_id"]))
        if row["gold_doi"]:
            identities.append(("doi", row["gold_doi"]))
        for identity in identities:
            other = owners.setdefault(identity, ref_id)
            left, right = sorted((root(ref_id), root(other)))
            parents[right] = left
    return [root(row["ref_id"]) for row in rows]


def split_rows(
    rows: list[dict[str, Any]], config: str, seed: int
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Choose a reproducible grouped, stratified 60/20/20 assignment."""
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    if config not in CONFIGS:
        raise ValueError(f"Unknown configuration: {config}")
    groups = reference_groups(rows) if config == CONFIGS[0] else list(range(len(rows)))
    if len(set(groups)) < 5:
        raise ValueError("At least five assignment groups are required")
    strata = [
        f"{row['source']}/{row['index']}/{row['gold_id'] is not None}" for row in rows
    ]
    split_info = {"seed": seed, "assignment_groups": len(set(groups))}
    if config == "reference_index_disjoint":
        rng = random.Random(seed)
        buckets = defaultdict(list)
        for position, stratum in enumerate(strata):
            buckets[stratum].append(position)
        partitions = {split: [] for split in SPLITS}
        for _, bucket in sorted(buckets.items()):
            rng.shuffle(bucket)
            quotas = [divmod(len(bucket) * weight, 5) for weight in (3, 1, 1)]
            sizes = [whole for whole, remainder in quotas]
            order = list(range(3))
            rng.shuffle(order)
            order.sort(key=lambda i: quotas[i][1], reverse=True)
            for i in order[: len(bucket) - sum(sizes)]:
                sizes[i] += 1
            offset = 0
            for split, size in zip(SPLITS, sizes):
                partitions[split].extend(bucket[offset : offset + size])
                offset += size
        partitions = {split: sorted(values) for split, values in partitions.items()}
        split_info["method"] = (
            "Seeded shuffle within source/index/gold-presence strata; largest-remainder 60/20/20 allocation with seeded tie breaking."
        )
    else:
        _, labels = np.unique(strata, return_inverse=True)
        totals = np.bincount(labels)
        targets = np.asarray(RATIOS)[:, None] * totals
        best = None
        # ponytail: bounded heuristic search; use an exact solver if strict quotas are required.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The least populated class in y")
            for trial in range(16):
                splitter = StratifiedGroupKFold(
                    n_splits=5, shuffle=True, random_state=seed + trial
                )
                folds = [
                    heldout
                    for _, heldout in splitter.split(
                        np.zeros(len(rows)), labels, groups
                    )
                ]
                counts = np.asarray(
                    [np.bincount(labels[fold], minlength=len(totals)) for fold in folds]
                )
                for dev, test in permutations(range(5), 2):
                    actual = np.asarray(
                        [totals - counts[dev] - counts[test], counts[dev], counts[test]]
                    )
                    score = float(
                        np.sum((actual - targets) ** 2 / np.maximum(targets, 1))
                    )
                    if best is None or score < best[0]:
                        best = (score, seed + trial, dev, test, folds)
        score, chosen_seed, dev, test, folds = best
        partitions = {
            "train": sorted(
                i for fold in range(5) if fold not in (dev, test) for i in folds[fold]
            ),
            "validation": sorted(folds[dev].tolist()),
            "test": sorted(folds[test].tolist()),
        }
        split_info.update(
            method="Best of 16 seeded StratifiedGroupKFold assignments and all validation/test fold pairs; other three folds form training. Minimize normalized squared deviation from source/index/gold-presence stratum targets.",
            selected_seed=chosen_seed,
            validation_fold=dev,
            test_fold=test,
            stratification_score=score,
        )
    assigned = [i for partition in partitions.values() for i in partition]
    if sorted(assigned) != list(range(len(rows))):
        raise ValueError("Split partitions do not cover each row exactly once")
    group_sets = [{groups[i] for i in partitions[split]} for split in SPLITS]
    if any(group_sets[a] & group_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Assignment group crosses split boundaries")
    return partitions, split_info


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


def split_statistics(
    rows: list[dict[str, Any]], partitions: dict[str, list[int]]
) -> dict[str, Any]:
    training_refs = {rows[i]["ref_id"] for i in partitions["train"]}
    result = {}
    for split, positions in partitions.items():
        subset = [rows[i] for i in positions]
        counts = Counter(
            (r["source"], r["index"], r["gold_id"] is not None) for r in subset
        )
        result[split] = {
            "rows": len(subset),
            "references": len({r["ref_id"] for r in subset}),
            "gold_ids": sum(r["gold_id"] is not None for r in subset),
            "gold_dois": sum(r["gold_doi"] is not None for r in subset),
            "missing_parsed_rows": sum(r["parsed_reference"] is None for r in subset),
            "missing_document_rows": sum(r["document_id"] is None for r in subset),
            "references_seen_in_train": len(
                {r["ref_id"] for r in subset} & training_refs
            ),
            "strata": [
                {
                    "source": source,
                    "index": index,
                    "has_gold": present,
                    "rows": counts[source, index, present],
                }
                for source in sorted({r["source"] for r in rows})
                for index in INDEXES
                for present in (False, True)
            ],
        }
    return result


def preserve_splits(
    rows: list[dict[str, Any]], config: str, source: Path
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Keep existing row order and assignments when correcting gold labels."""
    import pyarrow.parquet as pq

    positions = {(r["ref_id"], r["index"]): i for i, r in enumerate(rows)}
    if len(positions) != len(rows):
        raise ValueError("Duplicate current task keys")
    partitions, seen, files = {}, set(), []
    for split in SPLITS:
        path = source / config / f"{split}.parquet"
        keys = [(r["ref_id"], r["index"]) for r in pq.read_table(path).to_pylist()]
        if len(set(keys)) != len(keys) or seen.intersection(keys):
            raise ValueError("Duplicate parent task keys")
        if any(key not in positions for key in keys):
            raise ValueError("Parent and current task keys differ")
        seen.update(keys)
        partitions[split] = [positions[key] for key in keys]
        files.append(
            {
                "file": f"{config}/{split}.parquet",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if seen != set(positions):
        raise ValueError("Parent and current task keys differ")
    groups = reference_groups(rows) if config == CONFIGS[0] else list(range(len(rows)))
    group_sets = [{groups[i] for i in partitions[s]} for s in SPLITS]
    if any(group_sets[a] & group_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Corrected identity crosses preserved split boundaries")
    return partitions, {
        "method": "Preserved parent release row order and split assignments",
        "parent_files": files,
    }


def write_release(
    rows: list[dict[str, Any]],
    review: dict[str, Any],
    output: Path,
    seed: int,
    inputs: list[Path],
    card_path: Path,
    split_from: Path | None = None,
) -> dict[str, Any]:
    import pyarrow
    import pyarrow.parquet as pq
    import sklearn

    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    table = arrow_table(rows)
    assignments = {
        config: (
            preserve_splits(rows, config, split_from)
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
        f"Excluded references: {len(review['excluded_references'])}. References with conflicting DOI evidence: {len(review['doi_conflicts'])}; their `gold_doi` values are withheld pending review.",
        "Gold DOIs come from metadata of human-confirmed matches, reconciled across indexes for each reference. DOI fields have not been independently adjudicated. Unknown values remain null; no live index lookups were performed.",
    ]
    yaml += ["---", ""]
    (output / "README.md").write_text(
        "\n".join(yaml) + card + "\n".join(stats_lines) + "\n", encoding="utf-8"
    )
    (output / "build_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=HERE / "argilla_annotations/citation_linking_annotations.jsonl",
    )
    parser.add_argument("--source-dir", type=Path, default=HERE)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "logs/citation_linking_release"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-from",
        type=Path,
        help="Parent release directory whose row order and splits must be preserved",
    )
    args = parser.parse_args()
    rows, review = prepare_rows(
        load_local(args.annotations), read_sources(args.source_dir)
    )
    inputs = [
        args.annotations,
        *(args.source_dir / name for name in SOURCE_FILES),
        Path(__file__),
        HERE / "DATASET_CARD.md",
    ]
    summary = write_release(
        rows,
        review,
        args.output_dir,
        args.seed,
        inputs,
        HERE / "DATASET_CARD.md",
        args.split_from,
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


if __name__ == "__main__":
    main()
