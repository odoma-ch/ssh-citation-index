"""Tests for GROBID reference + citing-context extraction and its endpoints."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from lxml import etree

from citation_index.core.models import Reference
from citation_index.core.parsers.tei_bibl_parser import TeiBiblParser
from citation_index.llm.grobid_client import GrobidClient
from citation_index.pipelines.grobid_references import parse_grobid_fulltext

SAMPLE = Path(__file__).parent / "data" / "grobid_fulltext_sample.tei.xml"

TEI = """<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:space="preserve">
<teiHeader><fileDesc><sourceDesc><biblStruct><analytic>
  <title level="a">The paper itself</title></analytic></biblStruct></sourceDesc></fileDesc></teiHeader>
<text><body>
  <div><head>Methods</head>
    <p><s>We follow <ref type="bibr" target="#b1">Smith (2020)</ref>
      and <ref type="bibr" target="#b2">Doe (2019)</ref>.</s>
    <s>Unlinked <ref type="bibr">(Nobody, 1999)</ref> is skipped.</s></p></div>
  <div><p>Bare paragraph cites <ref type="bibr" target="#b2">Doe</ref>
    <ref type="bibr" target="#b2">again</ref>.</p></div>
</body>
<back><div><listBibl>
  <biblStruct xml:id="b0"/>
  <biblStruct xml:id="b1"><analytic><title level="a">Smith title</title>
    <author><persName><forename>J</forename><surname>Smith</surname></persName></author></analytic>
    <monogr><imprint><date type="published" when="2020"/></imprint></monogr>
    <note type="raw_reference">Smith, J. (2020). Smith title.</note></biblStruct>
  <biblStruct xml:id="b2"><analytic><title level="a">Doe title</title></analytic>
    <monogr><imprint><date type="published" when="2019"/></imprint></monogr>
    <note type="raw_reference">Doe (2019). Doe title.</note></biblStruct>
</listBibl></div></back></text></TEI>"""


def test_ids_stay_aligned_when_an_empty_biblstruct_precedes():
    out = parse_grobid_fulltext(TEI, include_parsed=True, include_context=True)

    # b0 is empty and dropped; header biblStruct is not a reference
    assert out["references"] == ["Smith, J. (2020). Smith title.", "Doe (2019). Doe title."]
    assert out["count"] == 2
    assert [p["full_title"] for p in out["parsed_references"]] == ["Smith title", "Doe title"]
    assert [c["ref_id"] for c in out["contexts"][0]] == ["b1"]
    assert {c["ref_id"] for c in out["contexts"][1]} == {"b2"}


def test_context_shape_section_marker_and_unlinked_marker():
    out = parse_grobid_fulltext(TEI, include_context=True)
    smith = out["contexts"][0]

    assert smith == [
        {
            "section": "Methods",
            "citation": "We follow Smith (2020) and Doe (2019).",
            "marker": "Smith (2020)",
            "ref_id": "b1",
        }
    ]
    all_sentences = [c["citation"] for ctxs in out["contexts"] for c in ctxs]
    assert not any("Unlinked" in s for s in all_sentences)


def test_paragraph_fallback_without_sentences_is_deduplicated_and_sectionless():
    doe = parse_grobid_fulltext(TEI, include_context=True)["contexts"][1]

    # Methods sentence + one paragraph (two markers in it collapse to one context)
    assert [c["section"] for c in doe] == ["Methods", ""]
    assert doe[1]["citation"] == "Bare paragraph cites Doe again."


def test_optional_keys_only_when_requested():
    out = parse_grobid_fulltext(TEI)
    assert set(out) == {"references", "method", "count"}


def test_invalid_tei_raises():
    with pytest.raises(Exception):
        parse_grobid_fulltext("not xml")


def test_real_grobid_sample():
    out = parse_grobid_fulltext(SAMPLE.read_text(encoding="utf-8"), True, True)

    assert out["count"] == len(out["parsed_references"]) == len(out["contexts"]) == 20
    assert sum(len(c) for c in out["contexts"]) == 16
    beedie = out["contexts"][6][0]
    assert "Beedie" in out["references"][6]
    assert beedie["ref_id"] == "b6"
    assert beedie["section"] == "Dispositional Affect, Mood, and Emotion"


def test_real_grobid_sample_keeps_page_ranges():
    parsed = parse_grobid_fulltext(SAMPLE.read_text(encoding="utf-8"), True)[
        "parsed_references"
    ]

    # GROBID writes ranges as <biblScope unit="page" from="243" to="260"/>
    assert parsed[0]["pages"] == "243-260"
    assert sum(1 for p in parsed if p["pages"]) == 20


@pytest.mark.parametrize(
    "scope,expected",
    [
        ('<biblScope unit="page" from="221" to="234"/>', "221-234"),
        ('<biblScope unit="page" from="12" to="12"/>', "12"),
        ('<biblScope unit="page" from="7"/>', "7"),
        ('<biblScope unit="page">p. 199</biblScope>', "p. 199"),
    ],
)
def test_page_scope_text_or_from_to(scope, expected):
    tei = TEI.replace(
        '<date type="published" when="2019"/>',
        f'<date type="published" when="2019"/>{scope}',
    )
    parsed = parse_grobid_fulltext(tei, include_parsed=True)["parsed_references"]

    assert parsed[1]["pages"] == expected


REF_TYPES = Path(__file__).parent / "data" / "grobid_ref_types.tei.xml"


def _refs_by_xml_id() -> dict:
    list_bibl = etree.parse(str(REF_TYPES)).find(".//{*}listBibl")
    ids = [
        b.get("{http://www.w3.org/XML/1998/namespace}id")
        for b in list_bibl.iterfind("{*}biblStruct")
    ]
    return dict(zip(ids, TeiBiblParser().to_references(list_bibl)))


@pytest.mark.parametrize(
    "xml_id,expected",
    [
        ("journal-article", "journal-article"),
        # <meeting> holding only an <address> is GROBID's place of publication
        ("book-chapter", "book-chapter"),
        ("book", "book"),
        ("book-untitled-analytic", "book"),
        ("proceedings-paper", "proceedings-paper"),
        ("unknown", None),
    ],
)
def test_ref_type_from_tei_structure(xml_id, expected):
    assert _refs_by_xml_id()[xml_id].ref_type == expected


def test_container_title_is_the_book_or_proceedings_title():
    refs = _refs_by_xml_id()

    chapter = refs["book-chapter"]
    assert chapter.full_title.startswith("Testfall Äthiopien")
    assert chapter.container_title.startswith("the Mediterranean, Asia")
    assert chapter.raw["tei"]["monographic_title"] == chapter.container_title

    proceedings = refs["proceedings-paper"]
    assert proceedings.container_title.startswith("Proceedings of the 4th")
    assert refs["journal-article"].container_title is None
    assert refs["journal-article"].journal_title == "African Affairs"
    # A book has no container: its own title is the monographic title
    assert refs["book"].container_title is None
    assert refs["book"].full_title.startswith("The Ethiopian election")
    untitled = refs["book-untitled-analytic"]
    assert untitled.container_title is None
    assert untitled.full_title == "Kollektives Handeln und Ambiguität"


def test_tei_export_without_raw_writes_container_as_monographic_title():
    ref = Reference(full_title="Chapter title", container_title="Book title")

    xml = TeiBiblParser().to_xml(ref)
    bibl = etree.fromstring(xml.encode()).find(".//{*}biblStruct")

    assert bibl.findtext("{*}analytic/{*}title[@level='a']") == "Chapter title"
    assert bibl.findtext("{*}monogr/{*}title[@level='m']") == "Book title"
    [round_trip] = TeiBiblParser().to_references(bibl)
    assert round_trip.container_title == "Book title"
    assert round_trip.ref_type == "book-chapter"


def test_tei_export_without_raw_writes_journal_article_title_as_analytic():
    ref = Reference(full_title="Article title", journal_title="Journal")

    xml = TeiBiblParser().to_xml(ref)
    bibl = etree.fromstring(xml.encode()).find(".//{*}biblStruct")

    assert bibl.findtext("{*}analytic/{*}title[@level='a']") == "Article title"
    assert bibl.find("{*}monogr/{*}title[@level='m']") is None
    [round_trip] = TeiBiblParser().to_references(bibl)
    assert round_trip.full_title == "Article title"
    assert round_trip.ref_type == "journal-article"


def test_fulltext_request_uses_grobid_parameter_names():
    client = GrobidClient(endpoint="http://grobid", max_retries=0)
    resp = Mock(status_code=200, text="<TEI/>")
    pdf = Path(__file__)  # any existing file; GROBID is mocked
    with patch.object(client.session, "post", return_value=resp) as post:
        client.process_full_text(
            pdf, consolidate_references=False, include_raw_references=True, segment_sentences=True
        )
    data = post.call_args.kwargs["data"]
    # GROBID silently ignores unknown fields, so a wrong name disables the feature
    assert data["includeRawCitations"] == "1"
    assert data["consolidateCitations"] == "0"
    assert data["segmentSentences"] == "1"


# ---- endpoints (need python-multipart for UploadFile routes) ----

pytest.importorskip("python_multipart")
from fastapi.testclient import TestClient  # noqa: E402

from citation_index import api  # noqa: E402

PDF = ("paper.pdf", b"%PDF-1.4", "application/pdf")


@pytest.fixture
def client():
    with patch.object(api, "redis_conn"), patch.object(api, "storage") as storage, patch.object(
        api, "queue_default"
    ) as queue, patch.object(api, "queue_llm") as queue_llm, patch.object(
        api,
        "format_job_response",
        return_value=api.JobResponse(job_id="j", status="queued", created_at="t"),
    ):
        test_client = TestClient(api.app)
        test_client.storage, test_client.queue, test_client.queue_llm = (
            storage,
            queue,
            queue_llm,
        )
        yield test_client


def test_extract_references_upload_defaults_to_grobid(client):
    r = client.post(
        "/extract/references?include_parsed=true&include_context=true", files={"file": PDF}
    )
    assert r.status_code == 200
    assert client.storage.save_upload.call_args.args[1:] == (b"%PDF-1.4", "input.pdf")
    kwargs = client.queue.enqueue_call.call_args.kwargs["kwargs"]
    assert kwargs["include_parsed"] is True and kwargs["include_context"] is True
    client.queue_llm.enqueue_call.assert_not_called()


def test_extract_references_json_still_uses_llm(client):
    r = client.post("/extract/references", json={"text": "refs"})
    assert r.status_code == 200
    assert client.queue_llm.enqueue_call.call_args.kwargs["kwargs"]["method"] == "full_text"
    client.queue.enqueue_call.assert_not_called()


@pytest.mark.parametrize(
    "url, kwargs",
    [
        ("/extract/references?method=grobid", {"json": {"text": "x"}}),
        ("/extract/references?include_context=true", {"json": {"text": "x"}}),
        ("/extract/references?method=semantic", {"files": {"file": PDF}}),
        ("/extract/references", {"files": {"file": ("p.txt", b"x", "text/plain")}}),
    ],
)
def test_extract_references_rejects_mismatched_input(client, url, kwargs):
    assert client.post(url, **kwargs).status_code == 400


def test_grobid_pipeline_endpoint_always_parses(client):
    r = client.post("/process/references/grobid?include_context=true", files={"file": PDF})
    assert r.status_code == 200
    kwargs = client.queue.enqueue_call.call_args.kwargs["kwargs"]
    assert kwargs["include_parsed"] is True and kwargs["include_context"] is True
