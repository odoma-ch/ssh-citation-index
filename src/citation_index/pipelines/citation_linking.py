"""Citation linking through external scholarly indexes."""

import re
from typing import Any, Dict, Iterable, List, Type

from ..core.connectors import MatildaConnector, OpenAlexConnector, WikidataConnector
from ..core.connectors.base import BaseConnector
from ..core.models import Reference
from ..utils.reference_matching import custom_match, reference_match_fields

SUPPORTED_TARGETS = ("openalex", "matilda", "wikidata")
CONNECTOR_TYPES: Dict[str, Type[BaseConnector]] = {
    "openalex": OpenAlexConnector,
    "matilda": MatildaConnector,
    "wikidata": WikidataConnector,
}


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


def link_references(
    references: Iterable[Dict[str, Any]], targets: Iterable[str], top_k: int = 10
) -> List[Dict[str, Any]]:
    """Search parsed references and filter candidates using bibliographic fields."""
    connectors = {name: CONNECTOR_TYPES[name]() for name in targets}
    results = []
    for parsed_reference in references:
        reference = Reference.model_validate(parsed_reference)
        links = {
            name: _link_one(reference, name, connector, top_k)
            for name, connector in connectors.items()
        }
        results.append({"reference": parsed_reference, "links": links})
    return results


def _link_one(
    reference: Reference,
    target: str,
    connector: BaseConnector,
    top_k: int,
) -> Dict[str, Any]:
    matches = []
    for candidate in connector.search(reference, top_k=top_k):
        fields = reference_match_fields(connector._result_to_reference(candidate))
        is_match, details = custom_match(reference, fields)
        if is_match:
            matches.append((details["title_similarity"], candidate))
    if not matches:
        return {"id": None, "doi": None}
    candidate = max(matches, key=lambda item: item[0])[1]
    return {"id": _candidate_id(candidate), "doi": _candidate_doi(target, candidate)}


def _candidate_id(candidate: Dict[str, Any]) -> str | None:
    value = candidate.get("id")
    return str(value) if value else None


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
