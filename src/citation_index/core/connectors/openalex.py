"""OpenAlex API connector for reference search and disambiguation."""

import logging
from typing import Any, Dict, List, Optional

import requests

from .base import BaseConnector
from ..models.reference import Reference
from ..models.references import References

logger = logging.getLogger(__name__)


class OpenAlexConnector(BaseConnector):
    """Connector for the OpenAlex API."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        super().__init__(api_key, base_url)
        self.base_url = base_url or "https://api.openalex.org"
        self.session = requests.Session()

        headers = {
            "User-Agent": "citation-index/1.0 (mailto:your-email@domain.com)",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self.session.headers.update(headers)

    def search(self, reference: Reference, top_k: int = 10, **kwargs: Any) -> List[Dict[str, Any]]:
        self._validate_reference(reference)

        title = reference.full_title.strip()
        url = f"{self.base_url}/works"
        params: Dict[str, Any] = {
            "filter": f"title.search:{title}",
            "per-page": min(top_k, 200),
            "sort": "relevance_score:desc",
        }

        search_terms = [title]
        author_name = self._first_author(reference)
        if author_name:
            search_terms.append(author_name)

        params["search"] = " ".join(search_terms)

        year = None
        if reference.publication_date:
            year = Reference._extract_year(reference.publication_date)  # type: ignore[attr-defined]
        if year:
            params["filter"] += f",publication_year:{year}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("OpenAlex search failed: %s", exc)
            return []

        payload = response.json()
        results = payload.get("results", [])
        if not isinstance(results, list):
            logger.warning("Unexpected OpenAlex response structure")
            return []

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
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 404:
                return []
            logger.error("OpenAlex identifier lookup failed: %s", exc)
            return []
        except requests.RequestException as exc:
            logger.error("OpenAlex identifier lookup failed: %s", exc)
            return []

        record = response.json()
        if not isinstance(record, dict):
            return []
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
        if not reference.authors:
            return None

        first = reference.authors[0]
        if isinstance(first, str):
            return first.strip() or None

        display_name = getattr(first, "display_name", None)
        if display_name:
            return str(display_name).strip() or None

        name_parts = [
            getattr(first, "first_name", "") or "",
            getattr(first, "surname", "") or "",
        ]
        author_name = " ".join(part for part in name_parts if part).strip()
        return author_name or None

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

    @staticmethod
    def _map_single_result(result: Dict[str, Any]) -> Optional[Reference]:
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

        ids = result.get("ids") or {}
        doi = ids.get("doi")

        return Reference(
            full_title=title,
            authors=authors or None,
            journal_title=journal_title,
            publisher=publisher,
            publication_date=str(publication_year) if publication_year else None,
            volume=volume,
            issue=issue,
            pages=pages,
            doi=doi,
        )


# Example usage:
# connector = OpenAlexConnector()
# query = Reference(full_title="Your Paper Title")
# raw_results = connector.search(query, top_k=5)
# mapped = connector.map_to_references(raw_results)
# for ref in mapped.references:
#     print(ref.full_title)
