"""Client for the citation-linking decider service (services/linking_decider)."""

import logging
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)


class LinkingDeciderError(Exception):
    """Raised when the decider fails or returns an unusable response."""


def decide_link(
    endpoint: str,
    reference: Dict[str, Any],
    index: str,
    candidates: List[Dict[str, Any]],
    timeout: float,
) -> Dict[str, Any]:
    """Ask the decider which raw search result, if any, the reference cites.

    ``reference`` must carry ``raw_reference``: the model was trained on the
    original citation string together with the parsed fields. Returns
    ``{"choice": int | None, "score": float, ...}`` where ``choice`` indexes
    ``candidates``.
    """
    raw_reference = (reference.get("raw_reference") or "").strip()
    if not raw_reference:
        raise ValueError("Model linking requires raw_reference on every reference")

    url = f"{endpoint.rstrip('/')}/v1/decide"
    try:
        resp = requests.post(
            url,
            json={
                "reference": reference,
                "raw_reference": raw_reference,
                "index": index,
                "candidates": candidates,
            },
            timeout=timeout,
        )
        if resp.status_code >= 400:
            raise LinkingDeciderError(
                f"Linking decider returned HTTP {resp.status_code}: {resp.text[:300]}"
            )
        result = resp.json()
        choice = result["choice"]
        float(result["score"])
    except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
        raise LinkingDeciderError(f"Linking decider request failed: {exc}") from exc

    if choice is not None and (
        not isinstance(choice, int) or not 0 <= choice < len(candidates)
    ):
        raise LinkingDeciderError(
            f"Decider chose {choice!r} among {len(candidates)} candidates"
        )
    return result
