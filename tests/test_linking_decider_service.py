"""Linking decider service: training-format parity and HTTP contract (no torch)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.linking_decider import app as service
from services.linking_decider.formatting import (
    bias_none_probabilities,
    build_request,
)

PARITY = json.loads(
    (Path(__file__).parent / "data/linking_decider_parity.json").read_text()
)


@pytest.mark.parametrize("row", PARITY, ids=[row["index"] for row in PARITY])
def test_request_matches_exported_training_example(row):
    text, labels = build_request(
        row["reference"], row["raw_reference"], row["index"], row["candidates"]
    )
    assert text == row["expected_input"]
    assert labels == row["expected_labels"]


@pytest.mark.parametrize(
    "fields, container",
    [
        # Training put a chapter's book in journal_title; since 0.6.0 the parser
        # writes it to container_title.
        ({"container_title": "Cambridge History of Christianity"}, "Cambridge History of Christianity"),
        ({"container_title": "The Book", "journal_title": "A Series"}, "The Book"),
        ({"journal_title": "Hypatia"}, "Hypatia"),
    ],
)
def test_container_title_is_sent_as_container(fields, container):
    reference = {"full_title": "Gentile Christianity", **fields}
    text, _ = build_request(reference, "Mitchell 2006", "openalex", [{"title": "X"}])
    assert f"\nContainer: {container}\n" in text + "\n"
    assert text.count("Container:") == 1


def test_none_bias_keeps_candidate_ranking():
    biased = bias_none_probabilities(
        {"candidate_1": 0.6, "candidate_2": 0.3, "none": 0.1}, 3.0
    )
    assert max(biased, key=biased.get) == "none"
    assert biased["candidate_1"] > biased["candidate_2"]
    assert sum(biased.values()) == pytest.approx(1.0)


def fake_decider(probabilities, calls):
    def classify(text, labels):
        calls.append((text, labels))
        return probabilities(labels)

    return lambda: service.Decider(classify=classify, info={"model": "fake"})


def client_for(probabilities, monkeypatch, bias="0"):
    monkeypatch.setenv("DECIDER_NONE_LOGIT_BIAS", bias)
    calls = []
    app = service.create_app(load=fake_decider(probabilities, calls))
    return TestClient(app), calls


def body(candidates):
    return {
        "reference": {"full_title": "A linked paper", "authors": ["Smith"]},
        "raw_reference": "Smith, A linked paper (2024).",
        "index": "openalex",
        "candidates": candidates,
    }


CANDIDATES = [{"title": "Unrelated"}, {"title": "A linked paper"}]


def second_wins(labels):
    return {label: 0.8 if label == "candidate_2" else 0.1 for label in labels}


def test_decide_returns_position_of_chosen_candidate(monkeypatch):
    with client_for(second_wins, monkeypatch)[0] as client:
        response = client.post("/v1/decide", json=body(CANDIDATES))
        health = client.get("/health").json()
    assert response.status_code == 200
    result = response.json()
    assert result["choice"] == 1
    assert result["score"] == pytest.approx(0.8)
    assert result["scores"] == pytest.approx([0.1, 0.8])
    assert health["model"] == "fake" and health["none_logit_bias"] == 0


def test_none_bias_turns_weak_choice_into_no_link(monkeypatch):
    with client_for(second_wins, monkeypatch, bias="7.737")[0] as client:
        result = client.post("/v1/decide", json=body(CANDIDATES)).json()
    assert result["choice"] is None
    assert result["score"] == result["none_score"]


def test_empty_pool_skips_inference(monkeypatch):
    test_client, calls = client_for(second_wins, monkeypatch)
    with test_client as client:
        result = client.post("/v1/decide", json=body([])).json()
    assert result["choice"] is None and calls == []


@pytest.mark.parametrize(
    "change",
    [
        {"candidates": [{"title": "x"}] * 11},
        {"index": "scopus"},
        {"raw_reference": ""},
    ],
    ids=["too_many", "index", "raw"],
)
def test_rejects_inputs_outside_training_distribution(monkeypatch, change):
    with client_for(second_wins, monkeypatch)[0] as client:
        response = client.post("/v1/decide", json={**body(CANDIDATES), **change})
    assert response.status_code == 422


def test_invalid_model_output_is_an_error_not_a_decision(monkeypatch):
    def broken(labels):
        return {label: 0.0 for label in labels}

    test_client, _ = client_for(broken, monkeypatch)
    with test_client as client:
        with pytest.raises(ValueError, match="sum to one"):
            client.post("/v1/decide", json=body(CANDIDATES))


def test_candidate_without_metadata_keeps_its_slot(monkeypatch):
    test_client, calls = client_for(second_wins, monkeypatch)
    with test_client as client:
        result = client.post(
            "/v1/decide", json=body([{"unknown": "field"}, {"title": "A linked paper"}])
        ).json()
    assert result["choice"] == 1
    assert calls[0][1]["candidate_1"] == "-"
