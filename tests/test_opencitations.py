"""OpenCitations connector: DOI lookup, then SPARQL search (HTTP mocked)."""

import pytest

from citation_index.core.connectors import opencitations as oc
from citation_index.core.connectors.opencitations import (
    OpenCitationsConnector,
    OpenCitationsError,
)
from citation_index.core.models import Reference
from citation_index.pipelines import citation_linking

OMID = "br/062301522213"
TITLE = (
    "3-Deazaneplanocin A (DZNep), An Inhibitor Of The Histone Methyltransferase EZH2, "
    "Induces Apoptosis And Reduces Cell Migration In Chondrosarcoma Cells"
)
RECORD = {
    "id": f"doi:10.1371/journal.pone.0098176 openalex:W1983545974 pmid:24852755 omid:{OMID}",
    "title": TITLE,
    "author": "Girard, Nicolas [orcid:0000-0002-7636-6688 omid:ra/069048221]; Bazille, Céline [omid:ra/1]",
    "pub_date": "2014-05-22",
    "venue": "Plos One [issn:1932-6203 omid:br/068038897]",
    "volume": "9",
    "page": "e98176",
}
BINDING = {
    "br": {"value": f"https://w3id.org/oc/meta/{OMID}"},
    "title": {"value": TITLE},
    "date": {"value": "2014-05-22"},
}


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"HTTP {self.status_code}")


class FakeSession:
    """Routes REST ``/metadata/...`` and SPARQL calls to canned payloads."""

    def __init__(self, rest, bindings=(), sparql_status=200):
        self.headers = {}
        self.rest, self.bindings, self.sparql_status = rest, list(bindings), sparql_status
        self.sparql_calls, self.rest_calls = [], []

    def get(self, url, params=None, **kwargs):
        if url == oc.SPARQL_ENDPOINT:
            self.sparql_calls.append(params["query"])
            if self.sparql_status != 200:
                return FakeResponse({}, self.sparql_status)
            return FakeResponse({"results": {"bindings": self.bindings}})
        ids = url.split("/metadata/", 1)[1].split("__")
        self.rest_calls.append(ids)
        return FakeResponse([r for r in self.rest if any(i in r["id"] for i in ids)])


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(oc.time, "sleep", lambda *_: None)


def connector(session):
    conn = OpenCitationsConnector()
    conn.session = session
    return conn


def reference(**fields):
    return Reference(
        full_title=fields.pop("full_title", TITLE),
        authors=["Nicolas Girard"],
        publication_year=2014,
        **fields,
    )


def with_doi(doi="https://doi.org/10.1371/journal.pone.0098176", **fields):
    return reference(identifiers=[{"scheme": "doi", "value": doi}], **fields)


def test_doi_hit_returns_omid_and_skips_sparql():
    session = FakeSession([RECORD])
    result = connector(session).search(with_doi())

    assert [r["id"] for r in result] == [OMID]
    assert result[0]["doi"] == "10.1371/journal.pone.0098176"
    assert result[0]["search_route"] == "doi"
    assert session.rest_calls == [["doi:10.1371/journal.pone.0098176"]]
    assert session.sparql_calls == []


def test_doi_miss_falls_back_to_sparql():
    session = FakeSession([RECORD], bindings=[BINDING])
    # the DOI is unknown to OC (REST finds nothing under it), the title prefix is not
    result = connector(session).search(with_doi("10.9999/unknown"))

    assert [r["id"] for r in result] == [OMID]
    assert result[0]["search_route"] == "sparql"
    assert session.sparql_calls and "STRSTARTS(?title" in session.sparql_calls[0]
    assert session.rest_calls[0] == ["doi:10.9999/unknown"]
    assert session.rest_calls[1] == [f"omid:{OMID}"]


def test_doi_hit_with_unrelated_title_is_rejected():
    wrong = {**RECORD, "title": "Quantum Chromodynamics On The Lattice"}
    session = FakeSession([wrong], bindings=[BINDING])
    assert connector(session).search(with_doi())[0]["search_route"] == "sparql"


def test_no_doi_goes_straight_to_sparql():
    session = FakeSession([RECORD], bindings=[BINDING])
    assert connector(session).search(reference())[0]["id"] == OMID
    assert not any(c[0].startswith("doi:") for c in session.rest_calls)


def test_sparql_failure_without_candidates_raises():
    with pytest.raises(OpenCitationsError):
        connector(FakeSession([], sparql_status=503)).search(reference())


def test_empty_sparql_result_is_an_empty_list_not_an_error():
    assert connector(FakeSession([], bindings=[])).search(reference()) == []


def test_title_is_required_without_a_doi_hit():
    with pytest.raises(ValueError):
        connector(FakeSession([])).search(Reference(full_title=" "))


def test_pipeline_links_opencitations_candidate():
    link = citation_linking._link_one(
        reference(), "opencitations", connector(FakeSession([RECORD], bindings=[BINDING])), 5
    )
    assert link == {"id": OMID, "doi": "10.1371/journal.pone.0098176"}


def test_targets():
    assert citation_linking.parse_targets("opencitations") == ["opencitations"]
    assert "opencitations" in citation_linking.parse_targets("all")


def test_title_prefix_variants_match_oc_casing():
    variants = oc.title_prefix_variants("3-deazaneplanocin A (DZNep), an inhibitor of EZH2", 6)
    assert "3-Deazaneplanocin A (DZNep), An Inhibitor Of" in variants
    assert any(v.startswith("3-Deazaneplanocin A (Dznep)") for v in variants)
    assert oc.title_prefix_variants("THE HISTORY OF EZH2 RESEARCH", 4)[1] == "The History Of Ezh2"
    assert oc.title_prefix_variants("Hi", 5) == []


def test_queries_escape_quotes_and_use_str_for_dates():
    assert 'STRSTARTS(?title, "A \\"q\\" B")' in oc.title_prefix_query('A "q" B')
    assert 'STRSTARTS(STR(?date), "2014")' in oc.author_year_query("Girard", [2014])


def test_batch_ids_map_back_independent_of_response_order():
    other = {"id": "doi:10.1182/BLOOD-2009 openalex:W2 omid:br/061"}
    index = {k: r for r in (other, RECORD) for k in oc.lookup_keys(oc.record_tokens(r))}
    assert index["doi:10.1371/journal.pone.0098176"] is RECORD
    assert index["openalex:W2"] is other and index["doi:10.1182/blood-2009"] is other


def test_author_query_filters_on_author_role():
    query = oc.author_year_query("Girard", [2014])
    assert "pro:withRole pro:author" in query
    assert "OPTIONAL { ?br prism:publicationDate ?date }" in oc.title_prefix_query("Some Long Title")


def test_person_author_builds_author_year_query():
    ref = Reference(
        full_title=TITLE,
        authors=[{"first_name": "Nicolas", "surname": "Girard"}],
        publication_year=2014,
    )
    query = oc.Query.from_reference(ref)
    assert query.family == "Girard"

    session = FakeSession([], bindings=[])
    connector(session).search(ref)
    assert any("foaf:familyName" in q for q in session.sparql_calls)


def test_undated_sparql_binding_is_kept():
    undated = {k: v for k, v in BINDING.items() if k != "date"}
    result = connector(FakeSession([RECORD], bindings=[undated])).search(reference())
    assert [r["id"] for r in result] == [OMID]


def test_mapped_reference_keeps_doi():
    session = FakeSession([RECORD])
    conn = connector(session)
    refs = conn.map_to_references(conn.search(with_doi()))
    ident = refs[0].identifiers[0]
    assert (ident.scheme, ident.value) == ("doi", "10.1371/journal.pone.0098176")
