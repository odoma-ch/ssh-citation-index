from unittest.mock import Mock

import pytest

from citation_index.core.extractors.factory import ExtractorFactory
from citation_index.core.extractors.mineru import MineruExtractor


def test_marker_is_not_available():
    assert "marker" not in ExtractorFactory.get_available_extractors()
    with pytest.raises(ValueError, match="Unsupported extractor type"):
        ExtractorFactory.create("marker")


def test_extract_uses_mineru_api(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    output_dir = tmp_path / "output"

    response = Mock()
    response.ok = True
    response.json.return_value = {
        "backend": "vlm-auto-engine",
        "version": "3.2.1",
        "task_id": "task-1",
        "results": {"paper": {"md_content": "# Paper\n\nText"}},
    }
    extractor = MineruExtractor(endpoint="http://mineru:8000", timeout=42)
    extractor.session.post = Mock(return_value=response)

    result = extractor.extract(str(pdf_path), save_dir=str(output_dir))

    assert result.text == "# Paper\n\nText"
    assert result.metadata["task_id"] == "task-1"
    assert (output_dir / "paper_mineru.md").read_text() == result.text
    _, request = extractor.session.post.call_args
    assert request["timeout"] == 42
    assert request["data"]["backend"] == "vlm-auto-engine"
    assert request["data"]["return_md"] == "true"
    assert request["files"]["files"][0] == "paper.pdf"


def test_extract_rejects_missing_markdown(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    response = Mock()
    response.ok = True
    response.json.return_value = {"results": {"paper": {}}}
    extractor = MineruExtractor()
    extractor.session.post = Mock(return_value=response)

    with pytest.raises(RuntimeError, match="does not contain Markdown"):
        extractor.extract(str(pdf_path))


def test_extract_accepts_list_result_shape(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    response = Mock()
    response.ok = True
    response.json.return_value = {
        "results": [{"file_name": "paper", "md_content": "# Paper"}]
    }
    extractor = MineruExtractor()
    extractor.session.post = Mock(return_value=response)

    assert extractor.extract(str(pdf_path)).text == "# Paper"


def test_extract_reports_invalid_json(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    response = Mock()
    response.ok = True
    response.json.side_effect = ValueError("not JSON")
    extractor = MineruExtractor()
    extractor.session.post = Mock(return_value=response)

    with pytest.raises(RuntimeError, match="invalid JSON"):
        extractor.extract(str(pdf_path))
