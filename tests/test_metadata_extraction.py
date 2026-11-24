"""Document metadata extraction and the per-PDF GROBID cache."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from citation_index import api
from citation_index.llm.grobid_client import GrobidClient, GrobidError
from citation_index.pipelines.metadata_extraction import (
    MetadataExtractionError,
    metadata_from_tei,
)
from citation_index.utils.storage import StorageManager

SAMPLE = Path(__file__).parent / "data" / "grobid_fulltext_sample.tei.xml"
PDF = {"file": ("paper.pdf", b"%PDF-1.4", "application/pdf")}


def test_metadata_from_tei_header():
    ref = metadata_from_tei(SAMPLE.read_text(encoding="utf-8"))

    assert ref.full_title.startswith("Affective Antecedents and Consequences")
    assert ref.authors[0].surname == "Mohr"
    assert ref.publication_year == 2019
    ids = {i.scheme: i.normalized for i in ref.identifiers}
    # md5 is GROBID's PDF hash; the DOI loses the trailing sentence period
    assert ids == {"doi": "10.1037/apl0000399"}


def test_metadata_from_tei_without_header_raises():
    tei = '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader/></TEI>'
    with pytest.raises(MetadataExtractionError):
        metadata_from_tei(tei)


def _response(status: int, text: str = "<TEI/>") -> Mock:
    return Mock(status_code=status, text=text)


def test_grobid_cache_reuses_response_per_pdf_and_params(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 a")
    client = GrobidClient("http://grobid", cache_dir=tmp_path / "cache")
    client.session.post = Mock(return_value=_response(200))

    assert client.process_full_text(pdf) == "<TEI/>"
    copy = tmp_path / "copy.pdf"  # same bytes, other upload dir -> same entry
    copy.write_bytes(pdf.read_bytes())
    assert client.process_full_text(copy) == "<TEI/>"
    assert client.session.post.call_count == 1

    client.process_full_text(pdf, segment_sentences=True)  # other params
    assert client.session.post.call_count == 2


def test_grobid_cache_skips_failures(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 a")
    client = GrobidClient("http://grobid", cache_dir=tmp_path / "cache")
    client.session.post = Mock(return_value=_response(500))

    for _ in range(2):
        with pytest.raises(GrobidError):
            client.process_full_text(pdf)
    assert client.session.post.call_count == 2
    assert not list((tmp_path / "cache").rglob("*.xml"))


def test_cleanup_prunes_grobid_cache(tmp_path):
    storage = StorageManager(tmp_path)
    (storage.grobid_cache_dir / "abc").mkdir(parents=True)

    assert storage.cleanup_old_jobs(max_age_hours=-1) == 1
    assert not (storage.grobid_cache_dir / "abc").exists()


@pytest.mark.parametrize(
    "method, queue_name", [("grobid", "queue_default"), ("llm", "queue_llm")]
)
def test_metadata_endpoint_routes_queue(method, queue_name):
    response = api.JobResponse(job_id="j", status="queued", created_at="t")
    with patch.object(api, "storage"), patch.object(
        api, "initialize_job_metadata"
    ), patch.object(api, "format_job_response", return_value=response), patch.object(
        api, queue_name
    ) as queue:
        result = TestClient(api.app).post(
            f"/extract/metadata?method={method}", files=PDF
        )
    assert result.status_code == 200
    assert queue.enqueue_call.call_args.kwargs["kwargs"]["method"] == method


def test_metadata_endpoint_rejects_non_pdf():
    result = TestClient(api.app).post(
        "/extract/metadata", files={"file": ("a.txt", b"x", "text/plain")}
    )
    assert result.status_code == 400


def test_llm_metadata_uses_first_pages_and_metadata_prompt(tmp_path):
    import pymupdf

    from citation_index.core.models import Reference, References
    from citation_index.pipelines import metadata_extraction as me

    pdf = tmp_path / "p.pdf"
    doc = pymupdf.open()
    for text in ("Title page", "Page two", "Page three"):
        doc.new_page().insert_text((72, 72), text)
    doc.save(pdf)

    refs = References(references=[Reference(full_title="Title page")])
    with patch.object(me, "parse_reference_strings", return_value=refs) as parse:
        out = me.extract_metadata_llm(pdf, Mock())

    lines, _ = parse.call_args.args
    assert "Page two" in lines[0] and "Page three" not in lines[0]
    assert (
        parse.call_args.kwargs["prompt_name"] == "prompts/prompts.yaml:metadata.default"
    )
    assert out == {"metadata": refs[0].model_dump(), "method": "llm"}
