"""Model input formatting for the citation-linking decider.

Copied from the training code so serving reproduces the exact text the model was
fine-tuned on:

- ``_text``, ``_claim_text``, ``candidate_fields``, ``FIELDS``, ``format_record``:
  ``benchmarks/citation_linking/experiments/linking_eval.py``
- ``NONE_DESCRIPTION``, ``listwise_request``, ``bias_none_probabilities``,
  ``validate_probabilities``: ``benchmarks/citation_linking/experiments/rerankers.py``

``tests/test_linking_decider_formatting.py`` checks this module against exported
training examples; change it only together with a retrained model.

Torch-free on purpose: the API-side tests import it without the model stack.
"""

from __future__ import annotations

import math
import re
from typing import Any

FIELD_LIMIT = 1000  # per-field character clip used for training
MAX_CANDIDATES = 10  # training pools never held more

FIELDS = {
    "full_title": "Title",
    "authors": "Authors",
    "editors": "Editors",
    "translator": "Translators",
    "publication_date_raw": "Publication date",
    "publication_year": "Year",
    "ref_type": "Publication type",
    "journal_title": "Container",
    "publisher": "Publisher",
    "publication_place": "Publication place",
    "volume": "Volume",
    "issue": "Issue",
    "pages": "Pages",
    "cited_range": "Cited pages",
}

NONE_DESCRIPTION = (
    "None of the listed candidates identifies the publication cited by the reference."
)

INDEXES = ("openalex", "matilda", "wikidata", "opencitations")


def _text(value: Any) -> str:
    """Render known name/text shapes without serializing arbitrary metadata."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(dict.fromkeys(filter(None, map(_text, value))))
    if isinstance(value, dict):
        for key in (
            "name",
            "displayName",
            "display_name",
            "fullName",
            "literal",
            "text",
            "value",
        ):
            if value.get(key):
                return _text(value[key])
        return " ".join(
            filter(
                None,
                [
                    _text(
                        value.get("given")
                        or value.get("firstName")
                        or value.get("first_name")
                    ),
                    _text(value.get("middle_name")),
                    _text(value.get("name_link")),
                    _text(
                        value.get("family")
                        or value.get("lastName")
                        or value.get("surname")
                    ),
                ],
            )
        )
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return " ".join(str(value).split())
    return ""


def _claim_text(claims: dict, prop: str) -> list[str]:
    values = []
    for statement in claims.get(prop, []):
        if statement.get("rank") == "deprecated":
            continue
        value = statement.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, dict):
            if value.get("text"):
                values.append(value["text"])
            elif value.get("time"):
                values.append(value["time"].lstrip("+")[:4])
            else:
                # Entity IDs without cached name qualifiers are not bibliographic text.
                for qualifier in ("P1932", "P1810"):
                    names = statement.get("qualifiers", {}).get(qualifier, [])
                    values.extend(
                        n.get("datavalue", {}).get("value", "") for n in names
                    )
    return values


def candidate_fields(index: str, metadata: dict) -> dict:
    """Read bibliographic fields from a raw connector search result."""
    if index == "openalex":
        source = (metadata.get("primary_location") or {}).get("source") or {}
        biblio = metadata.get("biblio") or {}
        return {
            "full_title": metadata.get("title") or metadata.get("display_name"),
            "authors": [
                a.get("author", {}).get("display_name")
                for a in metadata.get("authorships", []) or []
            ],
            "publication_year": metadata.get("publication_year"),
            "ref_type": metadata.get("type"),
            "journal_title": (
                source.get("display_name")
                if source.get("type") in {"journal", "book series", "conference"}
                else None
            ),
            "volume": biblio.get("volume"),
            "issue": biblio.get("issue"),
            "pages": "-".join(
                str(biblio[k]) for k in ("first_page", "last_page") if biblio.get(k)
            ),
        }
    if index == "matilda":
        texts = metadata.get("texts") or []
        record = next((t for t in texts if isinstance(t, dict) and t.get("title")), {})
        support = record.get("support") or []
        return {
            "full_title": record.get("title"),
            "authors": record.get("author") or record.get("authors"),
            "editors": record.get("editor") or record.get("editors"),
            "translator": record.get("translator"),
            "publication_date_raw": record.get("publicationDate")
            or record.get("publishedDate"),
            "ref_type": record.get("type"),
            "publisher": record.get("publisher"),
            "journal_title": [
                s.get("value")
                for s in support
                if isinstance(s, dict) and s.get("type") == "title"
            ],
            "volume": record.get("volume"),
            "issue": record.get("issue"),
            "pages": record.get("pages"),
        }
    if index == "wikidata":
        claims = (
            metadata.get("claims")
            or (metadata.get("entity_data") or {}).get("claims")
            or {}
        )
        return {
            "full_title": _claim_text(claims, "P1476") or metadata.get("label"),
            "authors": _claim_text(claims, "P2093") or _claim_text(claims, "P50"),
            "editors": _claim_text(claims, "P98"),
            "translator": _claim_text(claims, "P655"),
            "publication_date_raw": _claim_text(claims, "P577"),
            "journal_title": _claim_text(claims, "P1433"),
            "publisher": _claim_text(claims, "P123"),
            "volume": _claim_text(claims, "P478"),
            "issue": _claim_text(claims, "P433"),
            "pages": _claim_text(claims, "P304"),
            "description": metadata.get("description"),
        }
    if index == "opencitations":
        # Meta REST record: people are "Family, Given [omid:ra/..]; ..." strings.
        def people(value: Any) -> list[str]:
            text = re.sub(r"\s*\[[^\]]*\]", "", str(value or ""))
            return [name for part in text.split(";") if (name := part.strip())]

        return {
            "full_title": metadata.get("title"),
            "authors": people(metadata.get("author")),
            "editors": people(metadata.get("editor")),
            "publication_date_raw": metadata.get("pub_date"),
            "ref_type": metadata.get("type"),
            "journal_title": people(metadata.get("venue")),
            "publisher": people(metadata.get("publisher")),
            "volume": metadata.get("volume"),
            "issue": metadata.get("issue"),
            "pages": metadata.get("page"),
        }
    raise ValueError(f"Unsupported index: {index}")


def format_record(
    fields: dict, *, raw: str = "", field_limit: int = FIELD_LIMIT
) -> tuple[str, int]:
    if field_limit < 1:
        raise ValueError("field_limit must be positive")
    lines, clipped = [], 0
    allowed = {"reference": "Citation", **FIELDS, "description": "Description"}
    for key, label in allowed.items():
        value = _text(raw if key == "reference" else fields.get(key))
        if not value:
            continue
        if len(value) > field_limit:
            value = value[:field_limit] + " [truncated]"
            clipped += 1
        lines.append(f"{label}: {value}")
    return "\n".join(lines), clipped


def listwise_request(citation: str, candidates: list[str]) -> tuple[str, dict]:
    """Model text and ``candidate_i``/``none`` label descriptions for one task."""
    if not candidates:
        raise ValueError("Listwise input requires at least one candidate")
    labels = {}
    for i, candidate in enumerate(candidates, 1):
        # Parentheses and schema markers have structural meaning in GLiNER labels.
        description = (
            re.sub(r"\[(?:P|L|C|E|R|DESCRIPTION|EXAMPLE|OUTPUT)\]", " ", candidate)
            .replace("(", "[")
            .replace(")", "]")
        )
        labels[f"candidate_{i}"] = " ".join(description.split()) or "-"
    labels["none"] = NONE_DESCRIPTION
    return f"Citation:\n{citation}", labels


def build_request(
    reference: dict, raw_reference: str, index: str, candidates: list[dict]
) -> tuple[str, dict]:
    """Format one (reference, index) task exactly as the training export did."""
    citation, _ = format_record(reference, raw=raw_reference)
    if not citation:
        raise ValueError("Empty citation input")
    # Training pools never held a candidate without bibliographic fields; live
    # search can. Such a candidate keeps its slot as the "-" placeholder label
    # instead of voiding the other candidates.
    documents = [
        format_record(candidate_fields(index, candidate))[0] for candidate in candidates
    ]
    return listwise_request(citation, documents)


def validate_probabilities(probabilities: dict, labels: set[str]) -> None:
    if not isinstance(probabilities, dict) or set(probabilities) != labels:
        raise ValueError("Expected probabilities for exactly the requested labels")
    for value in probabilities.values():
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("Invalid decision-label probability")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-3):
        raise ValueError("Decision-label probabilities must sum to one")


def bias_none_probabilities(probabilities: dict, bias: float) -> dict:
    """Add a constant to the none logit, preserving relative candidate scores."""
    if not math.isfinite(bias):
        raise ValueError("The none logit bias must be finite")
    if "none" not in probabilities or len(probabilities) < 2:
        raise ValueError("Expected candidates and a none option")
    validate_probabilities(probabilities, set(probabilities))
    if bias == 0:
        return dict(probabilities)
    logits = {
        label: (math.log(p) if p > 0 else -math.inf) + (bias if label == "none" else 0)
        for label, p in probabilities.items()
    }
    peak = max(logits.values())
    weights = {label: math.exp(value - peak) for label, value in logits.items()}
    total = sum(weights.values())
    return {label: weight / total for label, weight in weights.items()}
