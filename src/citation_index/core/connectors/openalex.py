"""OpenAlex API connector for reference search and disambiguation."""

import logging
import os
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

import requests

from .base import BaseConnector
from ..models.reference import Reference
from ..models.references import References
from ...utils.reference_matching import extract_family_name, normalize_title

logger = logging.getLogger(__name__)


class OpenAlexConnector(BaseConnector):
    """Connector for the OpenAlex API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        email: Optional[str] = None,
    ) -> None:
        key = next(
            (
                value.strip()
                for value in (
                    api_key,
                    os.getenv("OPENALEX_API"),
                    os.getenv("OPENALEX_API_KEY"),
                )
                if value and value.strip()
            ),
            None,
        )
        super().__init__(key, base_url)
        self.base_url = base_url or "https://api.openalex.org"
        self.session = requests.Session()

        # Use provided email or default
        user_agent_email = email or "your-email@domain.com"
        headers = {
            "User-Agent": f"citation-index/1.0 (mailto:{user_agent_email})",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self.session.headers.update(headers)

    def _get(self, url: str, **kwargs: Any) -> requests.Response:
        """Make a bounded GET request, preserving budget and transport failures."""
        for attempt in range(3):
            response = None
            try:
                response = self.session.get(url, timeout=30, **kwargs)
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 2:
                    try:
                        raise requests.ConnectionError(
                            "OpenAlex transport failed after 3 attempts"
                        ) from None
                    except requests.RequestException:
                        logger.exception("OpenAlex transport failed; attempts=3")
                        raise
            except requests.RequestException:
                raise requests.RequestException("OpenAlex request failed") from None
            else:
                status = response.status_code
                remaining = response.headers.get("X-RateLimit-Remaining")
                reset = response.headers.get("X-RateLimit-Reset")
                exhausted = False
                if status == 429:
                    try:
                        exhausted = (
                            float(remaining) <= 0 if remaining is not None else False
                        )
                    except (ValueError, TypeError):
                        pass
                    try:
                        exhausted |= (
                            response.json().get("error") == "not_enough_credits"
                        )
                    except (ValueError, AttributeError):
                        pass
                if exhausted or (status != 429 and not 500 <= status < 600):
                    if status >= 400:
                        message = (
                            f"OpenAlex daily budget exhausted; remaining={remaining}; reset_seconds={reset}"
                            if exhausted
                            else f"OpenAlex request failed; HTTP {status}"
                        )
                        try:
                            raise requests.HTTPError(
                                message, response=response
                            ) from None
                        except requests.HTTPError:
                            logger.exception(
                                "OpenAlex request failed; status=%s; budget_exhausted=%s",
                                status,
                                exhausted,
                            )
                            raise
                    return response
                if attempt == 2:
                    try:
                        raise requests.HTTPError(
                            f"OpenAlex request failed after 3 attempts; HTTP {status}",
                            response=response,
                        ) from None
                    except requests.HTTPError:
                        logger.exception(
                            "OpenAlex retries exhausted; status=%s", status
                        )
                        raise

            delay = 2**attempt
            if response is not None:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        try:
                            retry_at = parsedate_to_datetime(retry_after)
                            delay = max(
                                delay,
                                (retry_at - datetime.now(timezone.utc)).total_seconds(),
                            )
                        except (ValueError, TypeError, OverflowError):
                            pass
                if delay > 60:
                    raise requests.HTTPError(
                        f"OpenAlex retry deferred; retry_after_seconds={delay}",
                        response=response,
                    ) from None
            logger.warning(
                "OpenAlex transient request failure; attempt=%s; retry_seconds=%s",
                attempt + 1,
                delay,
            )
            time.sleep(delay)
        raise AssertionError("Unreachable retry state")

    def search(
        self, reference: Reference, top_k: int = 10, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        self._validate_reference(reference)

        title = reference.full_title.strip()
        title = normalize_title(title)
        url = f"{self.base_url}/works"

        # Use title only for search to maximize recall
        # Author and year constraints can filter out valid results due to name variations
        params: Dict[str, Any] = {
            "search": title,
            "per_page": max(1, min(top_k or 100, 100)),
            "sort": "relevance_score:desc",
        }

        response = self._get(url, params=params)

        payload = response.json()
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise ValueError("Unexpected OpenAlex search response structure")

        return results[: top_k or len(results)]

    def search_by_id(
        self,
        identifier: str,
        identifier_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        if not identifier:
            return []

        id_type = (identifier_type or self._infer_identifier_type(identifier)).lower()
        if id_type == "doi":
            url = f"{self.base_url}/works/doi:{self._normalize_doi(identifier)}"
        elif id_type in {"openalex", "openalex_id"}:
            normalized = self._normalize_openalex_id(identifier)
            if not normalized:
                return []
            url = f"{self.base_url}/works/{normalized}"
        else:
            logger.warning("Unsupported identifier type for OpenAlex: %s", id_type)
            return []

        try:
            response = self._get(url)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 404:
                return []
            raise

        record = response.json()
        if not isinstance(record, dict):
            raise ValueError("Unexpected OpenAlex identifier response structure")
        return [record]

    def map_to_references(self, raw_results: List[Dict[str, Any]]) -> References:
        references = References()

        for result in raw_results:
            try:
                ref = self._map_single_result(result)
            except Exception as exc:
                logger.warning("Could not map OpenAlex result: %s", exc)
                continue

            if ref:
                references.append(ref)

        return references

    @staticmethod
    def _first_author(reference: Reference) -> Optional[str]:
        """Extract first author's last name for search.

        Uses the extract_family_name utility to handle various name formats.
        """
        if not reference.authors:
            return None

        first = reference.authors[0]

        # Handle string author names
        if isinstance(first, str):
            return extract_family_name(first)

        # Handle object with display_name attribute
        display_name = getattr(first, "display_name", None)
        if display_name:
            return extract_family_name(str(display_name))

        # Try to get surname directly
        surname = getattr(first, "surname", None)
        if surname:
            return str(surname).strip() or None

        return None

    @staticmethod
    def _normalize_doi(doi: str) -> str:
        clean = doi.strip()
        prefixes = [
            "https://doi.org/",
            "http://doi.org/",
            "http://dx.doi.org/",
            "doi:",
        ]
        for prefix in prefixes:
            if clean.lower().startswith(prefix):
                clean = clean[len(prefix) :]
                break
        return clean

    @staticmethod
    def _normalize_openalex_id(identifier: str) -> Optional[str]:
        token = identifier.rsplit("/", 1)[-1].upper()
        if token.startswith("W"):
            return token
        return None

    @staticmethod
    def _infer_identifier_type(identifier: str) -> str:
        trimmed = identifier.strip()
        if trimmed.lower().startswith("10.") or "doi" in trimmed.lower():
            return "doi"
        if trimmed.upper().startswith("W") or "openalex" in trimmed.lower():
            return "openalex"
        return "doi"

    def _result_to_reference(self, result: Dict[str, Any]) -> Reference:
        """Convert OpenAlex API result to Reference object."""
        ref = self._map_single_result(result)
        if ref is None:
            # Return an empty Reference if mapping fails
            return Reference(full_title="")
        return ref

    @staticmethod
    def _map_single_result(result: Dict[str, Any]) -> Optional[Reference]:
        from ...utils.identifier_parser import Identifier

        title = result.get("title")
        if not title:
            return None

        publication_year = result.get("publication_year")

        authors: List[str] = []
        for authorship in result.get("authorships", []) or []:
            author_info = authorship.get("author", {})
            display_name = author_info.get("display_name")
            if display_name:
                authors.append(display_name)

        primary_location = result.get("primary_location") or {}
        source = primary_location.get("source") or {}

        journal_title = source.get("display_name")
        publisher = source.get("host_organization_name")

        biblio = result.get("biblio") or {}
        volume = biblio.get("volume")
        issue = biblio.get("issue")

        pages = None
        first_page = biblio.get("first_page")
        last_page = biblio.get("last_page")
        if first_page and last_page:
            pages = f"{first_page}-{last_page}"
        elif first_page:
            pages = first_page

        # Extract identifiers
        identifiers = []
        ids = result.get("ids") or {}

        # DOI
        doi = ids.get("doi")
        if doi:
            doi_normalized = OpenAlexConnector._normalize_doi(doi)
            identifiers.append(
                Identifier(scheme="doi", value=doi, normalized=doi_normalized)
            )

        # OpenAlex ID
        openalex_id = ids.get("openalex")
        if openalex_id:
            identifiers.append(
                Identifier(scheme="openalex", value=openalex_id, normalized=openalex_id)
            )

        # PMID
        pmid = ids.get("pmid")
        if pmid:
            identifiers.append(
                Identifier(
                    scheme="pmid",
                    value=str(pmid),
                    normalized=str(pmid).replace(
                        "https://pubmed.ncbi.nlm.nih.gov/", ""
                    ),
                )
            )

        # Get reference type from OpenAlex
        ref_type = result.get(
            "type"
        )  # OpenAlex uses types like "article", "book", etc.

        # Get publication date if available, otherwise fall back to year
        publication_date_raw = result.get("publication_date") or (
            str(publication_year) if publication_year else None
        )

        return Reference(
            full_title=title,
            authors=authors or None,
            journal_title=journal_title,
            publisher=publisher,
            publication_year=publication_year,
            publication_date_raw=publication_date_raw,
            volume=volume,
            issue=issue,
            pages=pages,
            identifiers=identifiers,
            ref_type=ref_type,
        )


# Example usage:
# connector = OpenAlexConnector()
# query = Reference(full_title="Your Paper Title")
# raw_results = connector.search(query, top_k=5)
# mapped = connector.map_to_references(raw_results)
# for ref in mapped.references:
#     print(ref.full_title)
