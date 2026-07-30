"""Guided-decoding schema contract for reference parsing.

Whatever the schema leaves optional, the model drops. With nothing required it omitted
every title (0/63 references on a footnote-style PDF); with only the titles required it
omitted every author instead. Requiring `raw`/`identifiers` is equally wrong — the model
then emits a malformed string for `raw` and the response fails JSON parsing.
"""

from citation_index.core.models.reference import Reference
from citation_index.core.models.references import References


def _reference_schema() -> dict:
    schema = References.schema_without_excluded()
    items = schema["properties"]["references"]["items"]
    if "$ref" in items:
        return schema["$defs"][items["$ref"].split("/")[-1]]
    return items


def test_reference_schema_requires_every_concrete_field():
    ref_schema = _reference_schema()
    expected = set(ref_schema["properties"]) - set(Reference.OPTIONAL_SCHEMA_FIELDS)
    assert set(ref_schema["required"]) == expected
    for field in ("full_title", "journal_title", "authors", "publisher"):
        assert field in ref_schema["required"]


def test_reference_schema_keeps_free_form_fields_optional():
    required = set(_reference_schema()["required"])
    assert not required & set(Reference.OPTIONAL_SCHEMA_FIELDS)


def test_reference_schema_omits_excluded_fields():
    properties = _reference_schema()["properties"]
    excluded = {name for name, info in Reference.model_fields.items() if info.exclude}
    assert excluded, "expected some Reference fields to be marked exclude=True"
    assert not excluded & set(properties)
