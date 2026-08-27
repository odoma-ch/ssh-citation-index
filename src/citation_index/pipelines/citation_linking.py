"""Citation linking through external scholarly indexes."""

import re
from typing import Any, Dict, Iterable, List, Type

from ..core.connectors import MatildaConnector, OpenAlexConnector, WikidataConnector
from ..core.connectors.base import BaseConnector
from ..core.models import Reference
from ..utils.reference_matching import calculate_title_similarity

SUPPORTED_TARGETS = ("openalex", "matilda", "wikidata")
CONNECTOR_TYPES: Dict[str, Type[BaseConnector]] = {
    "openalex": OpenAlexConnector,
    "matilda": MatildaConnector,
    "wikidata": WikidataConnector,
}
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def parse_targets(value: str) -> List[str]:
    """Parse a comma-separated target list, expanding ``all``."""
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not requested:
        raise ValueError("At least one linking target is required")

    invalid = sorted(set(requested) - set(SUPPORTED_TARGETS) - {"all"})
    if invalid:
        raise ValueError(
            f"Unsupported linking target(s): {', '.join(invalid)}. "
            f"Choose from {', '.join(SUPPORTED_TARGETS)}, or all"
        )
    if "all" in requested:
        return list(SUPPORTED_TARGETS)
    return list(dict.fromkeys(requested))


def split_references(value: str, batched: bool) -> List[str]:
    """Return one reference or non-empty newline-delimited batch entries."""
    references = (
        [line.strip() for line in value.splitlines() if line.strip()]
        if batched
        else [value.strip()]
    )
    if not references or not references[0]:
        raise ValueError("At least one non-empty reference is required")
    return references


def link_references(
    reference_strings: Iterable[str], targets: Iterable[str], top_k: int = 10
) -> List[Dict[str, Any]]:
    """Link raw reference strings and return IDs from each requested target."""
    target_list = list(targets)
    connectors = {name: CONNECTOR_TYPES[name]() for name in target_list}
    results = []

    for reference_string in reference_strings:
        links = {
            name: _link_one(reference_string, name, connector, top_k)
            for name, connector in connectors.items()
        }
        results.append({"reference": reference_string, "links": links})

    return results


def _link_one(
    reference_string: str,
    target: str,
    connector: BaseConnector,
    top_k: int,
) -> Dict[str, Any]:
    doi = _extract_doi(reference_string)
    raw_results = connector.search_by_id(doi, "doi", top_k=top_k) if doi else []
    matched_by_doi = bool(raw_results)

    if not raw_results:
        raw_results = connector.search(
            Reference(full_title=reference_string), top_k=top_k
        )

    if not raw_results:
        return {"id": None, "doi": None}

    candidate = max(
        raw_results,
        key=lambda item: calculate_title_similarity(
            reference_string, _candidate_title(target, item)
        ),
    )
    if (
        not matched_by_doi
        and calculate_title_similarity(
            reference_string, _candidate_title(target, candidate)
        )
        < 90
    ):
        return {"id": None, "doi": None}

    return {
        "id": _candidate_id(candidate),
        "doi": _candidate_doi(target, candidate),
    }


def _extract_doi(value: str) -> str | None:
    match = DOI_PATTERN.search(value)
    return match.group(0).rstrip(".,;)") if match else None


def _candidate_id(candidate: Dict[str, Any]) -> str | None:
    value = candidate.get("id")
    return str(value) if value else None


def _candidate_title(target: str, candidate: Dict[str, Any]) -> str:
    if target == "openalex":
        return candidate.get("title") or ""
    if target == "wikidata":
        return candidate.get("label") or ""

    for text in candidate.get("texts", []) or []:
        title = text.get("title") if isinstance(text, dict) else None
        if isinstance(title, list) and title:
            return str(title[0])
        if isinstance(title, str):
            return title
    return ""


def _candidate_doi(target: str, candidate: Dict[str, Any]) -> str | None:
    if target == "openalex":
        value = candidate.get("doi") or (candidate.get("ids") or {}).get("doi")
    elif target == "wikidata":
        value = _wikidata_claim(candidate.get("claims") or {}, "P356")
    else:
        value = None
        for text in candidate.get("texts", []) or []:
            identifiers = text.get("identifier", []) if isinstance(text, dict) else []
            for identifier in identifiers:
                dois = identifier.get("doi") if isinstance(identifier, dict) else None
                if isinstance(dois, list) and dois:
                    value = dois[0]
                    break
                if isinstance(dois, str):
                    value = dois
                    break
            if value:
                break

    if not value:
        return None
    return re.sub(
        r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", str(value), flags=re.I
    )


def _wikidata_claim(claims: Dict[str, Any], property_id: str) -> str | None:
    statements = claims.get(property_id) or []
    if not statements:
        return None
    value = statements[0].get("mainsnak", {}).get("datavalue", {}).get("value")
    return value if isinstance(value, str) else None
