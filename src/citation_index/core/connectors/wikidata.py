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


class WikidataConnector(BaseConnector):
    """Connector for Wikidata elastic search API."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: int = 30):
        """
        Initialize Wikidata connector.
        
        Args:
            api_key: Not used for Wikidata (kept for interface compatibility)
            base_url: Base URL for Wikidata API (default: https://www.wikidata.org/w/api.php)
            timeout: Request timeout in seconds
        """
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
        """
        Search for books and publications on Wikidata using elastic search.
        
        Language is automatically detected from the reference title using the lingua
        library. If lingua is not available, defaults to English.
        
        Args:
            reference: Reference object with search criteria (title is mandatory)
            top_k: Maximum number of results to return
            **kwargs: Additional search parameters:
                - language: Language code override (e.g., 'en', 'fr', 'de')
                  If not provided, language is auto-detected from title
        
        Returns:
            List of raw API response data
            
        Raises:
            ValueError: If reference title is missing
        """
        self._validate_reference(reference)
        return self._search_elastic(reference, top_k, **kwargs)

    def search_by_id(
        self,
        identifier: str,
        identifier_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Search Wikidata for items by identifier using haswbstatement with CirrusSearch.
        
        Note: haswbstatement only works with the MediaWiki query API (action=query&list=search),
        not with wbsearchentities. We use the query API and then fetch entity details.
        
        Args:
            identifier: The identifier value (e.g., DOI, ISBN, PMID)
            identifier_type: Type of identifier (e.g., "doi", "isbn", "pmid")
            **kwargs: Additional parameters including:
                - top_k or limit: Maximum number of results (default: 10)
                - language: Language for labels (default: "en")
        
        Returns:
            List of entity details (similar to wbsearchentities format)
        """
        if not identifier:
            return []

        inferred = identifier_type or self._guess_identifier_type(identifier)
        if not inferred:
            logger.warning("Could not determine identifier type for %s", identifier)
            return []

        limit = kwargs.get("top_k") or kwargs.get("limit") or 10
        language = kwargs.get("language", "en")
        
        property_id = self._get_property_id(inferred.lower())
        if not property_id:
            logger.warning("Unsupported identifier type for Wikidata: %s", inferred)
            return []
        
        # Clean the identifier based on type
        cleaned_id = self._clean_identifier(inferred.lower(), identifier)
        if not cleaned_id:
            logger.warning("Invalid identifier value: %s", identifier)
            return []
        
        # Use haswbstatement search with MediaWiki query API
        search_expr = f"haswbstatement:{property_id}={cleaned_id}"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": search_expr,
            "srnamespace": "0",  # Main namespace (Wikidata items)
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
            
            # Extract Q IDs from titles and get entity details
            qids = [result.get("title") for result in search_results if result.get("title", "").startswith("Q")]
            
            if not qids:
                return []
            
            # Fetch entity details using wbgetentities or return simplified format
            return self._get_entity_details(qids, language)
            
        except requests.RequestException as exc:
            logger.warning("Wikidata identifier search failed: %s", exc)
            return []
    
    def _get_entity_details(self, qids: List[str], language: str = "en") -> List[Dict[str, Any]]:
        """
        Fetch entity details for a list of QIDs using wbgetentities.
        
        Args:
            qids: List of Wikidata QIDs (e.g., ["Q123", "Q456"])
            language: Language for labels and descriptions
            
        Returns:
            List of entity details in a format compatible with wbsearchentities
        """
        if not qids:
            return []
        
        # Batch request up to 50 entities at once (Wikidata API limit)
        params = {
            "action": "wbgetentities",
            "ids": "|".join(qids[:50]),  # Limit to 50 entities
            "props": "labels|descriptions",
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
                    
                    # Extract label and description
                    labels = entity.get("labels", {})
                    descriptions = entity.get("descriptions", {})
                    
                    label = labels.get(language, {}).get("value", qid)
                    description = descriptions.get(language, {}).get("value", "")
                    
                    # Format similar to wbsearchentities results
                    results.append({
                        "id": qid,
                        "label": label,
                        "description": description,
                        "match": {"type": "identifier", "language": language}
                    })
            
            return results
            
        except requests.RequestException as exc:
            logger.warning("Failed to fetch entity details: %s", exc)
            return []
    
    def _search_elastic(self, reference: Reference, top_k: int, **kwargs) -> List[Dict[str, Any]]:
        """
        Use wbsearchentities to search for entities by title.
        
        Returns basic entity information from the elastic search API.
        """
        if not reference.full_title:
            return []
        
        title = normalize_title(reference.full_title or "")
        # Detect language from title if not provided
        language = kwargs.get("language")
        if not language:
            language = self._detect_language(reference.full_title)
        
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
            return scored_hits[:top_k]
            
        except requests.RequestException as exc:
            logger.warning("Wikidata elastic search failed: %s", exc)
            return []
    
    def _get_property_id(self, id_type: str) -> Optional[str]:
        """Map identifier type to Wikidata property ID."""
        property_map = {
            "doi": "P356",
            "isbn": "P212",  # ISBN-13 (we'll search for both ISBN-13 and ISBN-10)
            "issn": "P236",
            "pmid": "P698",
            "pmcid": "P932",
            "arxiv": "P818",
        }
        return property_map.get(id_type)
    
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
    def _guess_identifier_type(identifier: str) -> Optional[str]:
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

    @staticmethod
    def _clean_doi(doi: str) -> str:
        cleaned = doi.strip()
        prefixes = [
            "https://doi.org/",
            "http://doi.org/",
            "http://dx.doi.org/",
            "doi:",
        ]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix) :]
                break
        return cleaned

    @staticmethod
    def _clean_isbn(isbn: str) -> str:
        return re.sub(r"[^0-9Xx]", "", isbn)

    @staticmethod
    def _clean_issn(issn: str) -> str:
        digits = re.sub(r"[^0-9Xx]", "", issn)
        return digits
    
    def _score_elastic_results(self, hits: List[Dict[str, Any]], reference: Reference) -> List[Dict[str, Any]]:
        """Score and sort elastic search results based on relevance.
        
        Elastic search results have structure:
        {
            "id": "Q...",
            "label": "Title",
            "description": "...",
            "match": {"type": "label", "language": "en", "text": "..."}
        }
        """
        want_year = None
        if reference.publication_date:
            want_year = self._extract_year(reference.publication_date)
        
        want_surnames = set()
        if reference.authors:
            want_surnames = self._extract_surnames(reference.authors)
        
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
            
            # Check description for year hints
            if want_year:
                description = hit.get("description", "")
                if description:
                    year_match = self._extract_year(description)
                    if year_match:
                        dy = abs(want_year - year_match)
                        score += (1.0 if dy == 0 else 0.6 if dy == 1 else 0.0) * 0.15
            
            # Check description for author hints
            if want_surnames:
                description = hit.get("description", "").lower()
                surname_hits = sum(1 for s in want_surnames if s in description)
                if surname_hits > 0:
                    score += min(1.0, surname_hits / max(1, len(want_surnames))) * 0.2
            
            hit["score_hint"] = round(score, 3)
        
        # Sort by score descending
        hits.sort(key=lambda x: -x.get("score_hint", 0))
        return hits
    
    def map_to_references(self, raw_results: List[Dict[str, Any]]) -> References:
        """
        Transform Wikidata elastic search results to Reference objects.
        
        Args:
            raw_results: List of elastic search results from wbsearchentities
            
        Returns:
            References object containing mapped Reference instances
        """
        references = References()
        
        for hit in raw_results:
            try:
                # Extract QID
                qid = hit.get("id", "")
                if not qid or not qid.startswith("Q"):
                    continue
                
                # Extract title from label
                title = hit.get("label", "")
                if not title:
                    continue
                
                # Extract description (may contain author/year hints)
                description = hit.get("description", "")
                
                # Try to extract year from description
                pub_date = None
                if description:
                    year = self._extract_year(description)
                    if year:
                        pub_date = str(year)
                
                # Create Reference object with basic info
                # Note: elastic search doesn't provide full metadata,
                # so we only have QID, title, and hints from description
                reference = Reference(
                    full_title=title,
                    publication_date=pub_date,
                    source="wikidata",
                    wikidata_id=qid
                )
                
                references.append(reference)
                
            except Exception as exc:
                logger.warning("Could not map Wikidata result: %s", exc)
                continue
        
        return references
    
    def _result_to_reference(self, result: Dict[str, Any]) -> Reference:
        """Convert Wikidata elastic search result to Reference object."""
        try:
            # Extract QID
            qid = result.get("id", "")
            if not qid or not qid.startswith("Q"):
                return Reference(full_title="")
            
            # Extract title
            title = result.get("label", "")
            if not title:
                return Reference(full_title="")
            
            # Extract description
            description = result.get("description", "")
            
            # Try to extract year from description
            pub_date = None
            if description:
                year = self._extract_year(description)
                if year:
                    pub_date = str(year)
            
            # Create Reference object
            return Reference(
                full_title=title,
                publication_date=pub_date,
                source="wikidata",
                wikidata_id=qid
            )
            
        except Exception:
            return Reference(full_title="")
    
    # Utility methods
    
    def _detect_language(self, text: str) -> str:
        """
        Detect language of the given text using lingua library.
        
        Args:
            text: Text to detect language from
            
        Returns:
            Two-letter ISO 639-1 language code (e.g., "en", "fr", "de")
            Defaults to "en" if detection fails or lingua is not available
        """
        if not text or not self.language_detector:
            return "en"
        
        try:
            detected = self.language_detector.detect_language_of(text)
            if detected:
                # Get ISO 639-1 code (e.g., "EN" -> "en")
                lang_code = detected.iso_code_639_1.name.lower()
                return lang_code
        except Exception as exc:
            logger.debug("Language detection failed: %s", exc)
        
        return "en"  # Default to English
    
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
    
    @staticmethod
    def _extract_surnames(authors: List[str]) -> set:
        """Extract surnames from author list for matching.
        
        Uses the extract_family_name utility to properly handle various name formats.
        """
        surnames = set()
        for author in authors:
            if isinstance(author, str):
                # Use utility function to extract family name (handles comma format properly)
                family_name = extract_family_name(author)
                if family_name:
                    # Normalize and add to set
                    normalized = WikidataConnector._normalize(family_name)
                    if normalized:
                        surnames.add(normalized.lower())
        return surnames


# Example usage:
# connector = WikidataConnector()
# 
# # Search by title using elastic search (language auto-detected)
# query = Reference(full_title="The Great Gatsby")
# raw = connector.search(query, top_k=5)
# mapped = connector.map_to_references(raw)
# 
# # Search with explicit language override
# query_fr = Reference(full_title="Le Petit Prince")
# raw_fr = connector.search(query_fr, top_k=5, language="fr")
# mapped_fr = connector.map_to_references(raw_fr)
# 
# # Identifier lookup using haswbstatement (e.g. DOI)
# doi_results = connector.search_by_id("10.1234/example", "doi")
# doi_refs = connector.map_to_references(doi_results)
