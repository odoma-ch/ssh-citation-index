"""Wikidata elastic search connector for reference search and disambiguation."""

import logging
import re
from typing import Any, Dict, List, Optional

import requests

from .base import BaseConnector
from ..models.reference import Reference
from ..models.references import References
from ...utils.reference_matching import extract_family_name, normalize_title


logger = logging.getLogger(__name__)

# Language detection
try:
    from lingua import Language, LanguageDetectorBuilder
    LINGUA_AVAILABLE = True
except ImportError:
    LINGUA_AVAILABLE = False
    logger.warning("lingua library not available, will default to English for Wikidata searches")


# Wikidata property mappings
PROPERTY_MAP = {
    "doi": "P356",
    "isbn": "P212",
    "issn": "P236",
    "pmid": "P698",
    "pmcid": "P932",
    "arxiv": "P818",
}


class WikidataConnector(BaseConnector):
    """Connector for Wikidata elastic search API."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: int = 30):
        """Initialize Wikidata connector."""
        super().__init__(api_key, base_url)
        self.api_url = base_url or "https://www.wikidata.org/w/api.php"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CitationIndex/1.0 (https://github.com/citation-index) wikidata-connector",
            "Accept": "application/json"
        })
        
        # Initialize language detector if available
        if LINGUA_AVAILABLE:
            languages = [
                Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH,
                Language.ITALIAN, Language.PORTUGUESE, Language.DUTCH, Language.RUSSIAN,
                Language.CHINESE, Language.JAPANESE, Language.KOREAN, Language.ARABIC
            ]
            self.language_detector = LanguageDetectorBuilder.from_languages(*languages).build()
        else:
            self.language_detector = None
    
    def search(self, reference: Reference, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Search for books and publications on Wikidata using elastic search."""
        self._validate_reference(reference)
        return self._search_elastic(reference, top_k, **kwargs)

    def search_by_id(
        self,
        identifier: str,
        identifier_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Search Wikidata for items by identifier using haswbstatement with CirrusSearch."""
        if not identifier:
            return []

        inferred = identifier_type or self._guess_identifier_type(identifier)
        if not inferred or inferred not in PROPERTY_MAP:
            logger.warning(f"Unsupported identifier type for Wikidata: {inferred}")
            return []
        
        limit = kwargs.get("top_k") or kwargs.get("limit") or 10
        language = kwargs.get("language", "en")
        
        property_id = PROPERTY_MAP[inferred.lower()]
        cleaned_id = self._clean_identifier(inferred.lower(), identifier)
        if not cleaned_id:
            logger.warning(f"Invalid identifier value: {identifier}")
            return []
        
        # Use haswbstatement search with MediaWiki query API
        params = {
            "action": "query",
            "list": "search",
            "srsearch": f"haswbstatement:{property_id}={cleaned_id}",
            "srnamespace": "0",
            "format": "json",
            "srlimit": min(max(1, limit), 50),
        }
        
        try:
            response = self.session.get(self.api_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            search_results = data.get("query", {}).get("search", [])
            if not search_results:
                return []
            
            # Extract Q IDs and get entity details
            qids = [r.get("title") for r in search_results if r.get("title", "").startswith("Q")]
            return self._get_entity_details(qids, language) if qids else []
            
        except requests.RequestException as exc:
            logger.warning(f"Wikidata identifier search failed: {exc}")
            return []
    
    def map_to_references(self, raw_results: List[Dict[str, Any]]) -> References:
        """Transform Wikidata elastic search results to Reference objects."""
        references = References()
        
        # Pre-fetch all journal labels in batch for efficiency
        journal_qids = []
        for hit in raw_results:
            claims = hit.get("claims", {})
            if claims and "P1433" in claims:
                journal_qid = self._extract_claim_value(claims, "P1433", "item")
                if journal_qid and journal_qid not in journal_qids:
                    journal_qids.append(journal_qid)
        
        journal_labels = self._get_labels_for_qids(journal_qids) if journal_qids else {}
        
        for hit in raw_results:
            try:
                qid = hit.get("id", "")
                if not qid or not qid.startswith("Q"):
                    continue
                
                title = hit.get("label", "")
                if not title:
                    continue
                
                claims = hit.get("claims", {})
                metadata = self._extract_metadata(claims, journal_labels)
                
                # Fallback: extract year from description if not in claims
                if not metadata["pub_date"]:
                    description = hit.get("description", "")
                    year = self._extract_year(description)
                    if year:
                        metadata["pub_date"] = str(year)
                
                reference = Reference(
                    full_title=title,
                    authors=metadata["authors"] if metadata["authors"] else None,
                    publication_date=metadata["pub_date"],
                    journal_title=metadata["journal"],
                    volume=metadata["volume"],
                    issue=metadata["issue"],
                    pages=metadata["pages"]
                )
                
                references.append(reference)
                
            except Exception as exc:
                logger.warning(f"Could not map Wikidata result: {exc}")
                continue
        
        return references
    
    def _result_to_reference(self, result: Dict[str, Any]) -> Reference:
        """Convert Wikidata elastic search result to Reference object."""
        try:
            qid = result.get("id", "")
            if not qid or not qid.startswith("Q"):
                return Reference(full_title="")
            
            title = result.get("label", "")
            if not title:
                return Reference(full_title="")
            
            claims = result.get("claims", {})
            metadata = self._extract_metadata(claims, {})
            
            # Fallback: extract year from description if not in claims
            if not metadata["pub_date"]:
                description = result.get("description", "")
                year = self._extract_year(description)
                if year:
                    metadata["pub_date"] = str(year)
            
            return Reference(
                full_title=title,
                authors=metadata["authors"] if metadata["authors"] else None,
                publication_date=metadata["pub_date"],
                journal_title=metadata["journal"],
                volume=metadata["volume"],
                issue=metadata["issue"],
                pages=metadata["pages"]
            )
            
        except Exception as e:
            logger.error(f"_result_to_reference failed for {result.get('id', 'unknown')}: {e}")
            return Reference(full_title="")
    
    # ========== Entity and Label Fetching ==========
    
    def _get_entity_details(self, qids: List[str], language: str = "en") -> List[Dict[str, Any]]:
        """Fetch entity details for a list of QIDs using wbgetentities."""
        if not qids:
            return []
        
        params = {
            "action": "wbgetentities",
            "ids": "|".join(qids[:50]),
            "props": "labels|descriptions|claims",
            "languages": language,
            "format": "json",
        }
        
        try:
            response = self.session.get(self.api_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            entities = data.get("entities", {})
            results = []
            
            for qid in qids:
                if qid in entities:
                    entity = entities[qid]
                    labels = entity.get("labels", {})
                    descriptions = entity.get("descriptions", {})
                    
                    results.append({
                        "id": qid,
                        "label": labels.get(language, {}).get("value", qid),
                        "description": descriptions.get(language, {}).get("value", ""),
                        "match": {"type": "identifier", "language": language},
                        "claims": entity.get("claims", {}),
                        "entity_data": entity
                    })
            
            return results
            
        except requests.RequestException as exc:
            logger.warning(f"Failed to fetch entity details: {exc}")
            return []
    
    def _get_labels_for_qids(self, qids: List[str], language: str = "en") -> Dict[str, str]:
        """Fetch labels for multiple QIDs in batch."""
        if not qids:
            return {}
        
        labels_map = {}
        
        # Batch request up to 50 entities at once (Wikidata API limit)
        for i in range(0, len(qids), 50):
            batch = qids[i:i+50]
            
            params = {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "labels",
                "languages": language,
                "format": "json",
            }
            
            try:
                response = self.session.get(self.api_url, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                
                entities = data.get("entities", {})
                for qid in batch:
                    if qid in entities:
                        labels = entities[qid].get("labels", {})
                        label = labels.get(language, {}).get("value")
                        if label:
                            labels_map[qid] = label
                
            except requests.RequestException as exc:
                logger.debug(f"Failed to fetch labels for batch: {exc}")
                continue
        
        return labels_map
    
    def _get_label_for_qid(self, qid: str, language: str = "en") -> Optional[str]:
        """Fetch the label for a single QID."""
        labels = self._get_labels_for_qids([qid], language)
        return labels.get(qid)
    
    # ========== Search ==========
    
    def _search_elastic(self, reference: Reference, top_k: int, **kwargs) -> List[Dict[str, Any]]:
        """Use wbsearchentities to search for entities by title."""
        if not reference.full_title:
            return []
        
        title = normalize_title(reference.full_title or "")
        language = kwargs.get("language") or self._detect_language(reference.full_title)
        
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": language,
            "type": "item",
            "limit": max(1, min(top_k, 50)),
            "search": title,
        }
        
        try:
            response = self.session.get(self.api_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            hits = data.get("search", [])
            
            # Score and sort results
            scored_hits = self._score_elastic_results(hits, reference)
            top_hits = scored_hits[:top_k]
            
            # Fetch full entity details including claims for top results
            qids = [hit.get("id") for hit in top_hits if hit.get("id", "").startswith("Q")]
            if qids:
                return self._get_entity_details(qids, language)
            
            return top_hits
            
        except requests.RequestException as exc:
            logger.warning(f"Wikidata elastic search failed: {exc}")
            return []
    
    def _score_elastic_results(self, hits: List[Dict[str, Any]], reference: Reference) -> List[Dict[str, Any]]:
        """Score and sort elastic search results based on relevance."""
        want_year = self._extract_year(reference.publication_date) if reference.publication_date else None
        want_surnames = self._extract_surnames(reference.authors) if reference.authors else set()
        
        for hit in hits:
            score = 0.0
            
            # Title similarity
            label = hit.get("label", "")
            if label and reference.full_title:
                query_tokens = set(self._normalize(reference.full_title).lower().split())
                item_tokens = set(self._normalize(label).lower().split())
                if query_tokens and item_tokens:
                    overlap = len(query_tokens & item_tokens)
                    score += min(1.0, overlap / max(3, len(query_tokens))) * 0.55
            
            # Year match in description
            if want_year:
                description = hit.get("description", "")
                if description:
                    year_match = self._extract_year(description)
                    if year_match:
                        dy = abs(want_year - year_match)
                        score += (1.0 if dy == 0 else 0.6 if dy == 1 else 0.0) * 0.15
            
            # Author match in description
            if want_surnames:
                description = hit.get("description", "").lower()
                surname_hits = sum(1 for s in want_surnames if s in description)
                if surname_hits > 0:
                    score += min(1.0, surname_hits / max(1, len(want_surnames))) * 0.2
            
            hit["score_hint"] = round(score, 3)
        
        # Sort by score descending
        hits.sort(key=lambda x: -x.get("score_hint", 0))
        return hits
    
    # ========== Metadata Extraction ==========
    
    def _extract_metadata(self, claims: Dict[str, Any], journal_labels: Dict[str, str]) -> Dict[str, Any]:
        """Extract all metadata from Wikidata claims."""
        metadata = {
            "authors": [],
            "pub_date": None,
            "journal": None,
            "volume": None,
            "issue": None,
            "pages": None,
        }
        
        if not claims:
            return metadata
        
        # Extract authors
        metadata["authors"] = self._get_author_names(claims)
        
        # Extract publication date (P577)
        pub_time = self._extract_claim_value(claims, "P577", "time")
        if pub_time:
            year = self._extract_year_from_time(pub_time)
            if year:
                metadata["pub_date"] = str(year)
        
        # Extract journal/publication (P1433)
        journal_qid = self._extract_claim_value(claims, "P1433", "item")
        if journal_qid:
            # Use pre-fetched label from batch, or fetch individually
            metadata["journal"] = journal_labels.get(journal_qid) or self._get_label_for_qid(journal_qid) or journal_qid
        
        # Extract volume (P478)
        metadata["volume"] = self._extract_claim_value(claims, "P478", "string")
        
        # Extract issue (P433)
        metadata["issue"] = self._extract_claim_value(claims, "P433", "string")
        
        # Extract pages (P304)
        metadata["pages"] = self._extract_claim_value(claims, "P304", "string")
        
        return metadata
    
    def _extract_claim_value(self, claims: Dict[str, Any], property_id: str, value_type: str = "string") -> Optional[Any]:
        """Extract a single value from Wikidata claims."""
        if not claims or property_id not in claims:
            return None
        
        statements = claims[property_id]
        if not statements:
            return None
        
        statement = statements[0]
        mainsnak = statement.get("mainsnak", {})
        
        if mainsnak.get("snaktype") != "value":
            return None
        
        datavalue = mainsnak.get("datavalue", {})
        value = datavalue.get("value")
        
        if not value:
            return None
        
        # Extract based on type
        if value_type == "string":
            return value if isinstance(value, str) else None
        elif value_type == "time":
            return value.get("time", "") if isinstance(value, dict) else None
        elif value_type == "item":
            return value.get("id") if isinstance(value, dict) else None
        elif value_type == "quantity":
            return value.get("amount") if isinstance(value, dict) else None
        
        return None
    
    def _get_author_names(self, claims: Dict[str, Any]) -> List[str]:
        """Extract author names from claims (P2093 or P50 with qualifiers)."""
        author_list = []
        
        # Try P2093 (author name string) first
        if "P2093" in claims:
            for statement in claims["P2093"]:
                mainsnak = statement.get("mainsnak", {})
                if mainsnak.get("snaktype") != "value":
                    continue
                
                datavalue = mainsnak.get("datavalue", {})
                value = datavalue.get("value")
                
                if not value or not isinstance(value, str):
                    continue
                
                # Get series ordinal from qualifiers
                series_ordinal = self._get_series_ordinal(statement)
                author_list.append((series_ordinal, value))
        
        # Try P50 (author item references) if no P2093
        if not author_list and "P50" in claims:
            author_qids_to_fetch = []
            
            for statement in claims["P50"]:
                mainsnak = statement.get("mainsnak", {})
                if mainsnak.get("snaktype") != "value":
                    continue
                
                datavalue = mainsnak.get("datavalue", {})
                value = datavalue.get("value")
                
                if not isinstance(value, dict):
                    continue
                
                author_qid = value.get("id")
                if not author_qid:
                    continue
                
                series_ordinal = self._get_series_ordinal(statement)
                
                # Check for name qualifiers (P1932 or P1810)
                author_name = self._get_name_qualifier(statement)
                
                if author_name:
                    author_list.append((series_ordinal, author_name))
                else:
                    author_qids_to_fetch.append((series_ordinal, author_qid))
            
            # Fetch labels for QIDs that don't have name qualifiers
            if author_qids_to_fetch:
                qids = [qid for _, qid in author_qids_to_fetch]
                labels = self._get_labels_for_qids(qids)
                
                for series_ordinal, qid in author_qids_to_fetch:
                    label = labels.get(qid)
                    if label:
                        author_list.append((series_ordinal, label))
        
        # Sort by series ordinal and extract names
        author_list.sort(key=lambda x: x[0])
        return [name for _, name in author_list]
    
    def _get_series_ordinal(self, statement: Dict[str, Any]) -> int:
        """Extract series ordinal (P1545) from statement qualifiers."""
        qualifiers = statement.get("qualifiers", {})
        if "P1545" in qualifiers:
            ordinal_snak = qualifiers["P1545"][0]
            ordinal_datavalue = ordinal_snak.get("datavalue", {})
            ordinal_value = ordinal_datavalue.get("value")
            if ordinal_value:
                try:
                    return int(ordinal_value)
                except (ValueError, TypeError):
                    pass
        return 999  # Default for items without ordinal
    
    def _get_name_qualifier(self, statement: Dict[str, Any]) -> Optional[str]:
        """Extract name from P1932 (object stated as) or P1810 (subject named as) qualifiers."""
        qualifiers = statement.get("qualifiers", {})
        
        # Try P1932 first (object stated as - preferred for author names)
        if "P1932" in qualifiers:
            name_snak = qualifiers["P1932"][0]
            name_datavalue = name_snak.get("datavalue", {})
            name_value = name_datavalue.get("value")
            if name_value and isinstance(name_value, str):
                return name_value
        
        # Try P1810 (subject named as)
        if "P1810" in qualifiers:
            name_snak = qualifiers["P1810"][0]
            name_datavalue = name_snak.get("datavalue", {})
            name_value = name_datavalue.get("value")
            if name_value and isinstance(name_value, str):
                return name_value
        
        return None
    
    # ========== Identifier Handling ==========
    
    @staticmethod
    def _guess_identifier_type(identifier: str) -> Optional[str]:
        """Guess the type of identifier from its format."""
        token = identifier.strip()
        lower = token.lower()
        digits = re.sub(r"[^0-9xX]", "", token)

        if lower.startswith("10.") or lower.startswith("doi:") or "doi.org" in lower:
            return "doi"
        if token.upper().startswith("Q") and token[1:].isdigit():
            return "wikidata"
        if re.match(r"^\d{4}\.[0-9]{4,5}$", lower) or "arxiv" in lower:
            return "arxiv"
        if lower.startswith("pmcid") or lower.startswith("pmc"):
            return "pmcid"
        if lower.startswith("pmid"):
            return "pmid"
        if digits.isdigit() and len(digits) in {13, 10}:
            return "isbn"
        if digits and len(digits) == 8:
            return "issn"
        if digits.isdigit() and 5 <= len(digits) <= 8:
            return "pmid"
        return None
    
    def _clean_identifier(self, id_type: str, identifier: str) -> Optional[str]:
        """Clean identifier value based on type."""
        if id_type == "doi":
            return self._clean_doi(identifier)
        elif id_type == "isbn":
            return self._clean_isbn(identifier)
        elif id_type == "issn":
            return self._clean_issn(identifier)
        elif id_type in {"pmid", "pmcid", "arxiv"}:
            return identifier.strip()
        return None

    @staticmethod
    def _clean_doi(doi: str) -> str:
        """Clean DOI by removing common prefixes."""
        cleaned = doi.strip()
        prefixes = [
            "https://doi.org/",
            "http://doi.org/",
            "http://dx.doi.org/",
            "doi:",
        ]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                return cleaned[len(prefix):]
        return cleaned

    @staticmethod
    def _clean_isbn(isbn: str) -> str:
        """Clean ISBN by removing non-alphanumeric characters."""
        return re.sub(r"[^0-9Xx]", "", isbn)

    @staticmethod
    def _clean_issn(issn: str) -> str:
        """Clean ISSN by removing non-alphanumeric characters."""
        return re.sub(r"[^0-9Xx]", "", issn)
    
    # ========== Utility Methods ==========
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text using lingua library."""
        if not text or not self.language_detector:
            return "en"
        
        try:
            detected = self.language_detector.detect_language_of(text)
            if detected:
                return detected.iso_code_639_1.name.lower()
        except Exception as exc:
            logger.debug(f"Language detection failed: {exc}")
        
        return "en"
    
    @staticmethod
    def _normalize(s: str) -> str:
        """Normalize string by collapsing whitespace."""
        return re.sub(r"\s+", " ", s.strip())
    
    @staticmethod
    def _extract_year(date_str: str) -> Optional[int]:
        """Extract a 4-digit year from a date string."""
        if not date_str:
            return None
        match = re.search(r'\b(19|20)\d{2}\b', str(date_str))
        if match:
            try:
                return int(match.group(0))
            except (ValueError, TypeError):
                pass
        return None
    
    def _extract_year_from_time(self, time_str: str) -> Optional[int]:
        """Extract year from Wikidata time string (format: +YYYY-MM-DDTHH:MM:SSZ)."""
        if not time_str:
            return None
        match = re.match(r'[+-](\d{4})', time_str)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, TypeError):
                pass
        return None
    
    @staticmethod
    def _extract_surnames(authors: List[str]) -> set:
        """Extract surnames from author list for matching."""
        surnames = set()
        for author in authors:
            if isinstance(author, str):
                family_name = extract_family_name(author)
                if family_name:
                    normalized = re.sub(r"\s+", " ", family_name.strip())
                    if normalized:
                        surnames.add(normalized.lower())
        return surnames

