"""Train / validation / test assignment of the citation-linking release.

Helper of ``build_annotation_dataset.py``, not a script. Two configurations: whole
references (``reference_disjoint``) or single reference-index tasks
(``reference_index_disjoint``). Published assignments are frozen: ``preserve_splits``
keeps them and only places tasks the parent release lacks.
"""

from __future__ import annotations

import hashlib
import random
import warnings
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path
from typing import Any

INDEXES = ("matilda", "openalex", "opencitations", "wikidata")
SPLITS = ("train", "validation", "test")
RATIOS = (0.6, 0.2, 0.2)
CONFIGS = ("reference_disjoint", "reference_index_disjoint")


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
    rows: list[dict[str, Any]], config: str, source: Path, seed: int
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Keep existing row order and assignments; append tasks the parent lacks.

    New tasks of a known reference (a newly annotated index) follow their
    reference's split in ``reference_disjoint``. Everything else, whole new
    references included, gets the usual seeded stratified allocation among the new
    rows only, so the parent's assignments never move.
    """
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
    added = [i for key, i in positions.items() if key not in seen]
    fresh = added
    if config == CONFIGS[0]:
        split_of = {rows[i]["ref_id"]: s for s in SPLITS for i in partitions[s]}
        for i in added:
            if rows[i]["ref_id"] in split_of:
                partitions[split_of[rows[i]["ref_id"]]].append(i)
        fresh = [i for i in added if rows[i]["ref_id"] not in split_of]
    if fresh:
        extra, _ = split_rows([rows[i] for i in fresh], config, seed)
        for split, chosen in extra.items():
            partitions[split] += [fresh[i] for i in chosen]
    groups = reference_groups(rows) if config == CONFIGS[0] else list(range(len(rows)))
    group_sets = [{groups[i] for i in partitions[s]} for s in SPLITS]
    # Parent assignments are frozen, so a new gold ID/DOI can reveal that two
    # references in different splits cite the same work. Report it; never reshuffle.
    crossing = sorted(
        {
            rows[i]["ref_id"]
            for a, b in ((0, 1), (0, 2), (1, 2))
            for s in (SPLITS[a], SPLITS[b])
            for i in partitions[s]
            if groups[i] in group_sets[a] & group_sets[b]
        }
    )
    if crossing and not added:
        raise ValueError("Corrected identity crosses preserved split boundaries")
    return partitions, {
        "method": "Preserved parent release row order and split assignments"
        + ("; appended new tasks" if added else ""),
        "parent_files": files,
        "added_rows": len(added),
        "same_work_across_splits": crossing,
    }


def published_splits(repo: str) -> Path:
    """Split files of the published release, the parent whose assignments are kept."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo,
            repo_type="dataset",
            allow_patterns=[f"{c}/*.parquet" for c in CONFIGS],
        )
    )
