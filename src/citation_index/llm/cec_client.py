"""Client for the OpenCitations CEC classifier (see deploy-llms/cec-classifier/API.md)."""

import logging
from typing import Any, Dict, List, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)

CEC_MODES = ("WS", "WoS", "M")


class CECError(Exception):
    """Raised when the CEC classifier fails or returns an unusable response."""


def classify_citations(
    endpoint: str,
    items: Sequence[Tuple[str, str]],
    mode: str,
    timeout: float,
) -> List[Dict[str, Any]]:
    """Classify ``(section, citation)`` pairs; results keep the input order.

    One request only: the pod rebuilds its predictor per call (~10s fixed cost)
    and serves one request at a time, so callers should batch, and a retry
    would just queue behind the failed call.
    """
    if mode not in CEC_MODES:
        raise ValueError(f"mode must be one of {CEC_MODES}, got {mode!r}")
    if not items:
        raise ValueError("At least one citation is required")

    url = f"{endpoint.rstrip('/')}/cic/api/classify"
    try:
        # Any X-Request-Source other than "web-interface" keeps errors in the manifest
        resp = requests.post(
            url,
            json={"mode": mode, "data": [list(pair) for pair in items]},
            headers={"X-Request-Source": "cli"},
            timeout=timeout,
        )
        resp.raise_for_status()
        result = resp.json()["result"]
    except (requests.RequestException, KeyError, ValueError) as exc:
        raise CECError(f"CEC classifier request failed: {exc}") from exc

    if len(result) != len(items):
        raise CECError(f"CEC returned {len(result)} results for {len(items)} citations")
    # Keyed by position as strings ("0", "1", ...); sort numerically, not lexically
    return [result[str(i)] for i in range(len(items))]
