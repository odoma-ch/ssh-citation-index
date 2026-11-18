"""Client for the citation-linking decider service."""

import pytest
import requests

from citation_index.llm import linking_decider_client as client
from citation_index.llm.linking_decider_client import LinkingDeciderError, decide_link

REFERENCE = {"full_title": "A linked paper", "raw_reference": "Smith. A linked paper."}
CANDIDATES = [{"id": "W1"}, {"id": "W2"}]


class Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def respond(monkeypatch, response, sent=None):
    def post(url, json, timeout):
        if sent is not None:
            sent.append((url, json, timeout))
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(client.requests, "post", post)


def test_posts_raw_reference_and_returns_choice(monkeypatch):
    sent = []
    respond(monkeypatch, Response(payload={"choice": 1, "score": 0.9}), sent)

    result = decide_link("http://decider/", REFERENCE, "openalex", CANDIDATES, 5)

    assert result["choice"] == 1
    url, body, timeout = sent[0]
    assert url == "http://decider/v1/decide" and timeout == 5
    assert body["raw_reference"] == "Smith. A linked paper."
    assert body["index"] == "openalex" and body["candidates"] == CANDIDATES


def test_requires_raw_reference():
    with pytest.raises(ValueError, match="raw_reference"):
        decide_link("http://decider", {"full_title": "x"}, "openalex", CANDIDATES, 5)


@pytest.mark.parametrize(
    "response,message",
    [
        (Response(422, text="Empty candidate bibliographic metadata"), "HTTP 422"),
        (requests.ConnectionError("refused"), "request failed"),
        (Response(payload=ValueError("not json")), "request failed"),
        (Response(payload={"score": 0.9}), "request failed"),
        (Response(payload={"choice": 2, "score": 0.9}), "among 2 candidates"),
        (Response(payload={"choice": "1", "score": 0.9}), "among 2 candidates"),
    ],
    ids=["http", "network", "json", "missing", "out_of_range", "type"],
)
def test_unusable_responses_raise(monkeypatch, response, message):
    respond(monkeypatch, response)
    with pytest.raises(LinkingDeciderError, match=message):
        decide_link("http://decider", REFERENCE, "openalex", CANDIDATES, 5)
