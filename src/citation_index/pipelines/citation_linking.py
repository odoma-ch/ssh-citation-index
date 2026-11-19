"""Citation linking through external scholarly indexes."""

import logging
import re
from typing import Any, Dict, Iterable, List, Type

from ..core.connectors import (
    MatildaConnector,
    OpenAlexConnector,
    OpenCitationsConnector,
    WikidataConnector,
)
from ..core.connectors.base import BaseConnector
from ..core.models import Reference
from ..llm.linking_decider_client import LinkingDeciderError, decide_link
from ..utils.reference_matching import custom_match, reference_match_fields

logger = logging.getLogger(__name__)

SUPPORTED_TARGETS = ("openalex", "matilda", "wikidata", "opencitations")
# match: title/author/year rules; model: fine-tuned GLiNER2 decider service
LINKING_METHODS = ("match", "model")
# DOIs that only ever name a book review. Providers (OpenCitations most) attach them to
# records whose title, author and year are the reviewed book's, so neither the rules
# nor the decider can tell them apart. ponytail: Choice Reviews Online only; JSTOR
# (10.2307) and most publishers mix reviews and articles under one prefix.
REVIEW_DOI = re.compile(r"^10\.5860/choice\.", re.I)
CONNECTOR_TYPES: Dict[str, Type[BaseConnector]] = {
    "openalex": OpenAlexConnector,
    "matilda": MatildaConnector,
    "wikidata": WikidataConnector,
    "opencitations": OpenCitationsConnector,
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
    references: Iterable[Dict[str, Any]],
    targets: Iterable[str],
    top_k: int = 10,
    method: str = "match",
    decider_endpoint: str | None = None,
    decider_timeout: float = 60.0,
    filter_reviews_by_doi: bool = False,
) -> List[Dict[str, Any]]:
    """Search parsed references and pick the matching candidate per index.

    ``method="match"`` filters candidates with title/author/year rules.
    ``method="model"`` sends each (reference, index) pool to the decider
    service at ``decider_endpoint``; references need ``raw_reference``.
    ``filter_reviews_by_doi`` drops candidates with a review-only DOI
    (``REVIEW_DOI``) before either method chooses.

    A failing source is reported per link as ``{"id": None, "doi": None,
    "error": ...}`` so the other sources' links survive. If every lookup
    fails, the first error is re-raised instead of returning an empty success.
    """
    if method not in LINKING_METHODS:
        raise ValueError(f"method must be one of {LINKING_METHODS}, got {method!r}")
    if method == "model" and not decider_endpoint:
        raise ValueError("method='model' requires a decider endpoint")
    connectors = {name: CONNECTOR_TYPES[name]() for name in targets}
    results = []
    errors: List[Exception] = []
    for parsed_reference in references:
        reference = Reference.model_validate(parsed_reference)
        links = {}
        for name, connector in connectors.items():
            try:
                if method == "model":
                    links[name] = _decide_one(
                        reference,
                        parsed_reference,
                        name,
                        connector,
                        top_k,
                        decider_endpoint,
                        decider_timeout,
                        filter_reviews_by_doi,
                    )
                else:
                    links[name] = _link_one(
                        reference, name, connector, top_k, filter_reviews_by_doi
                    )
            except LinkingDeciderError:
                # The decider is shared by every index, not one source among
                # several: an outage must fail the job, not read as "no match".
                raise
            except Exception as exc:
                logger.exception("Linking via %s failed", name)
                errors.append(exc)
                links[name] = {"id": None, "doi": None, "error": str(exc)}
        results.append({"reference": parsed_reference, "links": links})
    if errors and len(errors) == sum(len(result["links"]) for result in results):
        raise errors[0]
    return results


def _link_one(
    reference: Reference,
    target: str,
    connector: BaseConnector,
    top_k: int,
    filter_reviews: bool = False,
) -> Dict[str, Any]:
    matches = []
    candidates = connector.search(reference, top_k=top_k)
    for candidate in (
        _drop_reviews(target, candidates) if filter_reviews else candidates
    ):
        fields = reference_match_fields(connector._result_to_reference(candidate))
        is_match, details = custom_match(reference, fields)
        if is_match:
            matches.append((details["title_similarity"], candidate))
    if not matches:
        return {"id": None, "doi": None}
    candidate = max(matches, key=lambda item: item[0])[1]
    return {"id": _candidate_id(candidate), "doi": _candidate_doi(target, candidate)}


def _decide_one(
    reference: Reference,
    parsed_reference: Dict[str, Any],
    target: str,
    connector: BaseConnector,
    top_k: int,
    endpoint: str,
    timeout: float,
    filter_reviews: bool = False,
) -> Dict[str, Any]:
    # The decider was trained on raw search results, so send them unconverted.
    candidates = connector.search(reference, top_k=top_k)[:top_k]
    if filter_reviews:
        candidates = _drop_reviews(target, candidates)
    if not candidates:
        return {"id": None, "doi": None}
    result = decide_link(endpoint, parsed_reference, target, candidates, timeout)
    if result["choice"] is None:
        return {"id": None, "doi": None, "score": result["score"]}
    candidate = candidates[result["choice"]]
    return {
        "id": _candidate_id(candidate),
        "doi": _candidate_doi(target, candidate),
        "score": result["score"],
    }


def _drop_reviews(
    target: str, candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    return [
        candidate
        for candidate in candidates
        if not REVIEW_DOI.match(_candidate_doi(target, candidate) or "")
    ]


def _candidate_id(candidate: Dict[str, Any]) -> str | None:
    value = candidate.get("id")
    return str(value) if value else None


def _candidate_doi(target: str, candidate: Dict[str, Any]) -> str | None:
    if target == "openalex":
        value = candidate.get("doi") or (candidate.get("ids") or {}).get("doi")
    elif target == "wikidata":
        value = _wikidata_claim(candidate.get("claims") or {}, "P356")
    elif target == "opencitations":
        value = candidate.get("doi")
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
