"""Tests for the CEC classifier client."""

from unittest.mock import Mock, patch

import pytest
import requests

from citation_index.llm.cec_client import CECError, classify_citations

ITEMS = [("Introduction", "Based on Smith (2020)."), ("", "Prior work differs.")]


def _response(result):
    resp = Mock()
    resp.json.return_value = {"result": result, "manifest": {}}
    return resp


def test_results_follow_input_order_past_ten_items():
    items = [("", f"c{i}") for i in range(12)]
    # Lexical key order would put "10" before "2"
    result = {str(i): {"CITATION": f"c{i}"} for i in reversed(range(12))}
    with patch("requests.post", return_value=_response(result)) as post:
        out = classify_citations("http://cec:5000/", items, "M", 5)
    assert [r["CITATION"] for r in out] == [f"c{i}" for i in range(12)]
    assert post.call_args.args[0] == "http://cec:5000/cic/api/classify"
    assert post.call_args.kwargs["json"]["data"][0] == ["", "c0"]
    assert post.call_args.kwargs["headers"]["X-Request-Source"] == "cli"


def test_http_error_raises():
    resp = Mock()
    resp.raise_for_status.side_effect = requests.HTTPError("500")
    with patch("requests.post", return_value=resp):
        with pytest.raises(CECError):
            classify_citations("http://cec", ITEMS, "WS", 5)


def test_count_mismatch_raises():
    with patch("requests.post", return_value=_response({"0": {}})):
        with pytest.raises(CECError):
            classify_citations("http://cec", ITEMS, "WS", 5)


def test_invalid_mode_and_empty_rejected_before_request():
    with patch("requests.post") as post:
        with pytest.raises(ValueError):
            classify_citations("http://cec", ITEMS, "bad", 5)
        with pytest.raises(ValueError):
            classify_citations("http://cec", [], "M", 5)
    post.assert_not_called()
