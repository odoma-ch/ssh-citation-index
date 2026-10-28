"""OpenCitations Meta connector: DOI lookup first, then SPARQL title/author search.

``search`` tries two routes in order:

1. **ID lookup** - a DOI on the reference goes to the Meta REST API
   (``/metadata/doi:...``). No OpenAlex IDs are used: OpenAlex is its own target and runs
   alongside this one.
2. **SPARQL search** - on a miss (or without a DOI), query the QLever-backed Meta endpoint.
   ``REGEX`` / ``CONTAINS(LCASE())`` over ~100M titles time out, while a case-sensitive
   ``STRSTARTS(?title, "...")`` prefix filter is a vocabulary range scan (~40ms). OC stores
   titles title-cased, so queries use title-cased prefixes; family name + year is the
   fallback. Queries return only ``?br ?title ?date`` (an ``OPTIONAL`` makes QLever scan
   the whole dataset), metadata is fetched afterwards through REST ``omid:`` lookups, and
   fuzzy scoring happens in Python.

Failures raise ``OpenCitationsError``; they are never reported as "no result".
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import requests

from .base import BaseConnector
from ..models.reference import Reference
from ..models.references import References
from ...utils.reference_matching import calculate_title_similarity, extract_family_name

logger = logging.getLogger(__name__)

REST_BASE = "https://api.opencitations.net/meta/v1"
SPARQL_ENDPOINT = "https://sparql.opencitations.net/meta"
TOKEN_ENV = "OPENCITATIONS_ACCESS_TOKEN"

PREFIX_WORDS = (10, 6, 4)  # longest, most selective prefix first
MIN_PREFIX_CHARS = 12
GOOD_TITLE_SIM = 90.0
MIN_SHORTLIST_SIM = 50.0
DOI_MIN_TITLE_SIM = 80.0  # a DOI hit below this is a wrongly parsed DOI, not a match
SHORTLIST = 5

SPARQL_PREFIXES = """\
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX pro: <http://purl.org/spar/pro/>
"""


class OpenCitationsError(RuntimeError):
    """A request to an OpenCitations service failed after retries."""


# ---------------------------------------------------------------------------
# Pure helpers (also imported by benchmarks/citation_linking/opencitations_linking.py)
# ---------------------------------------------------------------------------
def normalize_doi(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and value != value):
        return None
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", str(value).strip(), flags=re.I)
    return doi.lower() or None


def extract_year(*values: Any) -> Optional[int]:
    for value in values:
        match = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", str(value)) if value else None
        if match:
            return int(match.group(1))
    return None


def _capitalise(token: str) -> str:
    """Upper-case the first letter of a token and of each hyphen/bracket-led part."""
    return re.sub(r"(^|[-(\[\"“‘])([^\W\d_])", lambda m: m.group(1) + m.group(2).upper(), token)


def title_prefix_variants(title: str, n_words: int) -> List[str]:
    """OC-cased prefixes of ``title``: rest-kept (``EZH2``), rest-lowered, and raw."""
    words = title.split()[:n_words]
    if not words:
        return []

    def strip(text: str) -> str:
        return text.rstrip(".,:;")  # a shortened prefix stays a valid prefix

    variants = (
        " ".join(_capitalise(w) for w in words),
        " ".join(_capitalise(w.lower()) for w in words),
        " ".join(words),
    )
    return list(dict.fromkeys(strip(v) for v in variants if len(strip(v)) >= MIN_PREFIX_CHARS))


def sparql_literal(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def title_prefix_query(prefix: str, limit: int = 100) -> str:
    return (
        f"{SPARQL_PREFIXES}SELECT ?br ?title ?date WHERE {{\n"
        f"  ?br dcterms:title ?title .\n"
        f"  FILTER(STRSTARTS(?title, {sparql_literal(prefix)}))\n"
        f"  OPTIONAL {{ ?br prism:publicationDate ?date }}\n}} LIMIT {limit}"
    )


def author_year_query(family: str, years: Iterable[int], limit: int = 300) -> str:
    # ?date is an xsd:date literal, so STRSTARTS needs STR()
    year_filter = " || ".join(f'STRSTARTS(STR(?date), "{y}")' for y in years)
    return (
        f"{SPARQL_PREFIXES}SELECT DISTINCT ?br ?title ?date WHERE {{\n"
        f"  VALUES ?family {{ {sparql_literal(family)} }}\n"
        f"  ?ra foaf:familyName ?family .\n"
        f"  ?role pro:withRole pro:author ;\n"
        f"      pro:isHeldBy ?ra .\n"
        f"  ?br pro:isDocumentContextFor ?role ;\n"
        f"      dcterms:title ?title ;\n"
        f"      prism:publicationDate ?date .\n"
        f"  FILTER({year_filter})\n}} LIMIT {limit}"
    )


def strip_brackets(value: str) -> str:
    return re.sub(r"\s*\[[^\]]*\]", "", value).strip()


def record_tokens(record: Dict[str, Any]) -> Dict[str, str]:
    """``{"doi": "10.x/y", "openalex": "W1", "omid": "br/06..."}`` from a Meta ``id`` string."""
    tokens: Dict[str, str] = {}
    for item in (record.get("id") or "").split():
        scheme, _, value = item.partition(":")
        if scheme in ("doi", "openalex", "omid") and scheme not in tokens:
            tokens[scheme] = value.lower() if scheme == "doi" else value
    return tokens


def lookup_keys(tokens: Dict[str, str]) -> set:
    """Normalised ``scheme:value`` keys a record can be found under."""
    keys = {f"{k}:{v}" for k, v in tokens.items() if k != "openalex"}
    if "openalex" in tokens:
        keys.add(f"openalex:{tokens['openalex'].upper()}")
    return keys


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a Meta REST record into the fields used for scoring and benchmark rows."""
    tokens = record_tokens(record)
    authors = [strip_brackets(a) for a in (record.get("author") or "").split(";") if a.strip()]
    return {
        "omid": tokens.get("omid"),
        "doi": tokens.get("doi"),
        "title": record.get("title") or "",
        "authors": authors,
        "first_author": authors[0].split(",")[0].strip() if authors else None,
        "year": extract_year(record.get("pub_date")),
        "journal": strip_brackets(record["venue"]) if record.get("venue") else None,
        "volume": record.get("volume") or None,
        "type": record.get("type") or None,
    }


@dataclass(frozen=True)
class Query:
    """Reference fields the OC search uses (metadata only, never gold labels)."""

    title: str
    family: Optional[str]
    year: Optional[int]
    doi: Optional[str] = None

    @classmethod
    def from_reference(cls, reference: Reference) -> "Query":
        first = (reference.authors or [None])[0]
        if isinstance(first, str):
            family = extract_family_name(first)
        else:
            family = getattr(first, "surname", None) or getattr(first, "name", None)
        doi = next(
            (
                normalize_doi(i.normalized or i.value)
                for i in reference.identifiers or []
                if i.scheme.lower() == "doi"
            ),
            None,
        )
        return cls(
            title=(reference.full_title or "").strip(),
            family=family,
            year=reference.publication_year or extract_year(reference.publication_date_raw),
            doi=doi,
        )

    @classmethod
    def from_parsed(cls, parsed: Optional[Dict[str, Any]]) -> "Query":
        """Same as ``from_reference`` for a parsed-reference dict, ignoring identifiers."""
        return cls.from_reference(
            Reference.model_validate({**(parsed or {}), "identifiers": []})
        )


def score_candidate(query: Query, candidate: Dict[str, Any]) -> Dict[str, Any]:
    title_sim = calculate_title_similarity(query.title, candidate["title"]) if query.title else 0.0
    year_ok = bool(query.year and candidate["year"] and abs(query.year - candidate["year"]) <= 1)
    family = (query.family or "").lower()
    author_ok = bool(family and any(family in a.lower() for a in candidate["authors"]))
    return {
        "title_similarity": round(title_sim, 1),
        "year_match": year_ok,
        "author_match": author_ok,
        "score": round(title_sim + 10 * year_ok + 10 * author_ok, 1),
        "is_match_by_similarity": title_sim >= GOOD_TITLE_SIM and (year_ok or author_ok),
    }


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------
class OpenCitationsConnector(BaseConnector):
    """Connector for OpenCitations Meta (REST API + SPARQL)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        sparql_endpoint: Optional[str] = None,
        max_retries: int = 5,
        sparql_timeout: int = 30,
        author_timeout: int = 90,
        request_pause: float = 0.3,
    ):
        """
        Args:
            api_key: OpenCitations access token (defaults to ``OPENCITATIONS_ACCESS_TOKEN``).
            sparql_timeout: Per-request timeout for title-prefix queries (they answer in <1s).
            author_timeout: Per-request timeout for family-name + year queries, which take
                0.4-35s depending on how common the surname is.
            request_pause: Seconds to sleep after each request, to stay polite.
        """
        super().__init__(api_key or os.environ.get(TOKEN_ENV), base_url or REST_BASE)
        self.sparql_endpoint = sparql_endpoint or SPARQL_ENDPOINT
        self.max_retries = max_retries
        self.sparql_timeout = sparql_timeout
        self.author_timeout = author_timeout
        self.request_pause = request_pause
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "citation-index (odoma.ch)"
        if self.api_key:
            self.session.headers["authorization"] = self.api_key

    # -- HTTP ---------------------------------------------------------------
    def _get(self, url: str, retries: Optional[int] = None, **kwargs: Any) -> requests.Response:
        last: Optional[Exception] = None
        for attempt in range(retries or self.max_retries):
            try:
                response = self.session.get(url, **kwargs)
                if response.status_code in (429, 500, 502, 503, 504):
                    raise OpenCitationsError(f"HTTP {response.status_code}")
                response.raise_for_status()
                time.sleep(self.request_pause)
                return response
            except (requests.RequestException, OpenCitationsError) as exc:
                last = exc
                time.sleep(min(2**attempt, 30))
        raise OpenCitationsError(f"{url[:80]}: {type(last).__name__}: {last}") from last

    def fetch_metadata(self, ids: List[str]) -> List[Dict[str, Any]]:
        """REST batch lookup (``doi:..``, ``omid:br/..``, ``openalex:W..``).

        Unknown IDs are simply absent from the response; the order is not the input order.
        """
        if not ids:
            return []
        path = "__".join(quote(i, safe=":/_") for i in ids)
        data = self._get(f"{self.base_url}/metadata/{path}", timeout=60).json()
        return data if isinstance(data, list) else []

    def _sparql(self, query: str, timeout: int) -> List[Dict[str, Any]]:
        response = self._get(
            self.sparql_endpoint,
            params={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=timeout,
            retries=2,
        )
        try:
            return response.json()["results"]["bindings"]
        except (KeyError, ValueError) as exc:
            raise OpenCitationsError(f"Unexpected SPARQL response: {exc}") from exc

    # -- search -------------------------------------------------------------
    def search(self, reference: Reference, top_k: int = 10, **kwargs: Any) -> List[Dict[str, Any]]:
        """DOI lookup, then SPARQL title / author search.

        Candidates are Meta REST records with ``id`` rewritten to the bare OMID
        (``br/0623...``), a bare ``doi`` and the original identifier string under
        ``identifiers``. Extra kwargs (e.g. the old ``threshold``) are accepted and ignored.
        """
        query = Query.from_reference(reference)
        if query.doi:
            hit = self._lookup_doi(query)
            if hit:
                return [hit]
        self._validate_reference(reference)
        raw, _, errors = self.sparql_candidates(query)
        ranked = self._rank_records(query, raw)
        if not ranked and errors:
            raise OpenCitationsError("; ".join(errors))
        if errors:
            logger.warning("OpenCitations SPARQL partial failure: %s", "; ".join(errors))
        return [self._candidate(record, scored, "sparql") for record, scored in ranked[: top_k or None]]

    def _lookup_doi(self, query: Query) -> Optional[Dict[str, Any]]:
        for record in self.fetch_metadata([f"doi:{query.doi}"]):
            flat = normalize_record(record)
            scored = score_candidate(query, flat)
            # a DOI hit with an unrelated title means the parsed DOI was wrong
            if not query.title or scored["title_similarity"] >= DOI_MIN_TITLE_SIM:
                return self._candidate(record, scored, "doi")
            logger.info("OpenCitations DOI %s rejected: title similarity %.0f", query.doi, scored["title_similarity"])
        return None

    @staticmethod
    def _candidate(record: Dict[str, Any], scored: Dict[str, Any], route: str) -> Dict[str, Any]:
        tokens = record_tokens(record)
        return {
            **record,
            "id": tokens.get("omid"),
            "doi": tokens.get("doi"),
            "identifiers": record.get("id"),
            "match_score": scored["score"],
            "title_similarity": scored["title_similarity"],
            "search_route": route,
        }

    def sparql_candidates(self, query: Query) -> Tuple[List[Dict[str, Any]], str, List[str]]:
        """Run the SPARQL strategies -> (``{omid, title, date}`` candidates, strategy, errors)."""
        seen: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = []
        strategy = "none"

        def run(sparql: str, name: str, timeout: int) -> None:
            nonlocal strategy
            try:
                for binding in self._sparql(sparql, timeout):
                    br = binding["br"]["value"].rsplit("/meta/", 1)[-1]
                    seen.setdefault(
                        br,
                        {
                            "omid": br,
                            "title": binding["title"]["value"],
                            "date": binding.get("date", {}).get("value"),  # absent for undated records
                        },
                    )
                    strategy = name if strategy == "none" else strategy
            except (OpenCitationsError, KeyError) as exc:
                logger.warning("%s failed: %s", name, exc)
                errors.append(f"{name}: {exc}")

        def best_title_sim() -> float:
            return max((calculate_title_similarity(query.title, c["title"]) for c in seen.values()), default=0.0)

        if not query.title:
            return [], strategy, errors
        for n_words in PREFIX_WORDS:
            for prefix in title_prefix_variants(query.title, n_words):
                run(title_prefix_query(prefix), f"title_prefix_{n_words}w", self.sparql_timeout)
            if best_title_sim() >= GOOD_TITLE_SIM:
                return list(seen.values()), strategy, errors

        if query.family and query.year:  # ponytail: LIMIT 300 truncates very common surnames
            years = (query.year - 1, query.year, query.year + 1)
            for family in dict.fromkeys((query.family, query.family.title())):
                run(author_year_query(family, years), "author_year", self.author_timeout)
        return list(seen.values()), strategy, errors

    def _rank_records(
        self, query: Query, raw: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Shortlist by title similarity, fetch metadata via REST, rescore, sort by score."""
        sims = {c["omid"]: calculate_title_similarity(query.title, c["title"]) for c in raw}
        shortlist = sorted(
            (c for c in raw if sims[c["omid"]] >= MIN_SHORTLIST_SIM), key=lambda c: -sims[c["omid"]]
        )[:SHORTLIST]
        ranked = []
        for record in self.fetch_metadata([f"omid:{c['omid']}" for c in shortlist]):
            flat = normalize_record(record)
            ranked.append((record, {**flat, **score_candidate(query, flat)}))
        return sorted(ranked, key=lambda pair: pair[1]["score"], reverse=True)

    def rank_candidates(self, query: Query, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Scored, flattened candidates (used by the benchmark script)."""
        return [scored for _, scored in self._rank_records(query, raw)]

    def search_by_id(
        self,
        identifier: str,
        identifier_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Fetch raw Meta records by DOI, ISBN, ISSN or OMID. Unknown IDs give ``[]``."""
        if not identifier:
            return []
        if identifier_type is None:
            if identifier.startswith(("10.", "DOI:", "doi:", "https://doi.org/")):
                identifier_type = "doi"
            elif identifier.startswith(("978", "979")) and len(identifier.replace("-", "")) in (10, 13):
                identifier_type = "isbn"
            elif len(identifier.replace("-", "")) == 8:
                identifier_type = "issn"
            elif identifier.startswith(("br/", "omid:")):
                identifier_type = "omid"
            else:
                logger.warning("Could not infer identifier type for %s", identifier)
                return []
        if identifier_type == "doi":
            clean = f"doi:{normalize_doi(identifier)}"
        else:
            clean = identifier if identifier.startswith(f"{identifier_type}:") else f"{identifier_type}:{identifier}"
        return self.fetch_metadata([clean])

    # -- mapping ------------------------------------------------------------
    def map_to_references(self, raw_results: List[Dict[str, Any]]) -> References:
        references = References()
        for result in raw_results:
            ref = self._map_metadata_record(result)
            if ref:
                references.append(ref)
        return references

    def _result_to_reference(self, result: Dict[str, Any]) -> Reference:
        return self._map_metadata_record(result) or Reference(full_title="")

    def _map_metadata_record(self, record: Dict[str, Any]) -> Optional[Reference]:
        title = record.get("title") or record.get("title_str")
        if not title:
            return None
        pub_date = record.get("pub_date") or record.get("year") or record.get("publication_date")
        venue = record.get("venue") or record.get("journal")
        publisher = record.get("publisher")
        doi = record.get("doi") or record_tokens(record).get("doi")
        return Reference(
            full_title=title,
            identifiers=[{"scheme": "doi", "value": doi}] if doi else [],
            authors=self._normalise_people(record.get("author")) or None,
            editors=self._normalise_people(record.get("editor")) or None,
            journal_title=self._strip_bracketed(venue) if isinstance(venue, str) else venue,
            publisher=self._strip_bracketed(publisher) if isinstance(publisher, str) else publisher,
            publication_date_raw=str(pub_date) if pub_date else None,
            volume=record.get("volume") or None,
            issue=record.get("issue") or None,
            pages=record.get("page") or record.get("pages") or None,
        )

    @staticmethod
    def _split_semi_colon(value: Optional[str]) -> List[str]:
        return [
            OpenCitationsConnector._strip_bracketed(chunk.strip())
            for chunk in (value or "").split(";")
            if chunk.strip()
        ]

    @staticmethod
    def _strip_bracketed(value: str) -> str:
        match = re.match(r"^([^[]+)", value)
        return match.group(1).strip() if match else value.strip()

    @staticmethod
    def _normalise_people(raw: Any) -> List[str]:
        if isinstance(raw, str):
            return OpenCitationsConnector._split_semi_colon(raw)
        people: List[str] = []
        for entry in raw if isinstance(raw, list) else []:
            if isinstance(entry, dict):
                name = entry.get("name") or " ".join(
                    part for part in (entry.get("given"), entry.get("family")) if part
                ).strip()
            else:
                name = entry if isinstance(entry, str) else None
            if name:
                cleaned = OpenCitationsConnector._strip_bracketed(name)
                if cleaned:
                    people.append(cleaned)
        return people
