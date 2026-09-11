from fastapi.testclient import TestClient
import pytest

from citation_index.pipelines import citation_linking
from citation_index.core.models import Reference
from citation_index.core.connectors import (
    OpenAlexConnector,
    MatildaConnector,
    WikidataConnector,
)
from citation_index.utils.storage import StorageManager


def parsed(title="A linked paper", year=2024):
    return {"full_title": title, "authors": ["Smith"], "publication_year": year}


class FakeOpenAlexConnector(OpenAlexConnector):
    def search(self, reference, top_k=10, **kwargs):
        return [
            {
                "id": "https://openalex.org/W2",
                "title": "A linked paper",
                "publication_year": 2024,
                "doi": "https://doi.org/10.1234/title-match",
            }
        ]


class FakeMatildaConnector(MatildaConnector):
    def search(self, reference, top_k=10, **kwargs):
        return [
            {
                "id": "matilda-1",
                "texts": [
                    {
                        "title": ["A linked paper"],
                        "date": ["2024"],
                        "author": ["Smith"],
                        "identifier": [{"doi": ["10.1234/matilda"]}],
                    }
                ],
            }
        ]


class FakeWikidataConnector(WikidataConnector):
    def search(self, reference, top_k=10, **kwargs):
        return [
            {
                "id": "Q1",
                "label": "A linked paper",
                "claims": {
                    "P577": [
                        {"mainsnak": {
                            "snaktype": "value",
                            "datavalue": {"value": {"time": "+2024-01-01T00:00:00Z"}},
                        }}
                    ],
                    "P356": [
                        {"mainsnak": {
                            "snaktype": "value",
                            "datavalue": {"value": "10.1234/WIKIDATA"},
                        }}
                    ],
                },
            }
        ]


def test_parse_targets_and_batch_input():
    assert citation_linking.parse_targets("all") == [
        "openalex",
        "matilda",
        "wikidata",
    ]
    assert citation_linking.parse_targets("wikidata, openalex,wikidata") == [
        "wikidata",
        "openalex",
    ]
    with pytest.raises(ValueError, match="Unsupported linking target"):
        citation_linking.parse_targets("opencitations")


def test_link_references_returns_ids_and_normalized_dois(monkeypatch):
    monkeypatch.setattr(
        citation_linking,
        "CONNECTOR_TYPES",
        {
            "openalex": FakeOpenAlexConnector,
            "matilda": FakeMatildaConnector,
            "wikidata": FakeWikidataConnector,
        },
    )

    results = citation_linking.link_references(
        [parsed()],
        ["openalex", "matilda", "wikidata"],
    )

    assert results == [
        {
            "reference": parsed(),
            "links": {
                "openalex": {
                    "id": "https://openalex.org/W2",
                    "doi": "10.1234/title-match",
                },
                "matilda": {"id": "matilda-1", "doi": "10.1234/matilda"},
                "wikidata": {"id": "Q1", "doi": "10.1234/WIKIDATA"},
            },
        }
    ]


def test_link_references_rejects_weak_title_match(monkeypatch):
    class WeakConnector(FakeOpenAlexConnector):
        def search(self, reference, top_k=10, **kwargs):
            return [{"id": "W-wrong", "title": "Completely unrelated work"}]

    monkeypatch.setattr(
        citation_linking, "CONNECTOR_TYPES", {"openalex": WeakConnector}
    )

    result = citation_linking.link_references(
        [parsed()], ["openalex"]
    )

    assert result[0]["links"]["openalex"] == {"id": None, "doi": None}


def test_link_references_propagates_connector_failures(monkeypatch):
    class FailingConnector(FakeOpenAlexConnector):
        def search(self, reference, top_k=10, **kwargs):
            raise RuntimeError("service unavailable")

    monkeypatch.setattr(
        citation_linking, "CONNECTOR_TYPES", {"openalex": FailingConnector}
    )

    with pytest.raises(RuntimeError, match="service unavailable"):
        citation_linking.link_references([parsed()], ["openalex"])


def test_linking_task_persists_result_and_completes_job(monkeypatch, tmp_path):
    from citation_index import tasks

    test_storage = StorageManager(tmp_path)
    test_storage.save_intermediate(
        "job-1",
        "citation_linking_input",
        {"references": [parsed()], "targets": ["openalex"]},
    )
    metadata_updates = []

    monkeypatch.setattr(tasks, "storage", test_storage)
    monkeypatch.setattr(
        tasks,
        "link_references",
        lambda references, targets: [
            {
                "reference": references[0],
                "links": {"openalex": {"id": "W1", "doi": "10.1234/example"}},
            }
        ],
    )
    monkeypatch.setattr(
        tasks,
        "update_job_metadata",
        lambda job_id, **fields: metadata_updates.append(fields),
    )
    monkeypatch.setattr(tasks, "log_job_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks, "get_completed_stages", lambda job_id: [])

    task_result = tasks.link_references_task("job-1")

    assert task_result["reference_count"] == 1
    assert test_storage.get_result("job-1") == {
        "results": [
            {
                "reference": parsed(),
                "links": {"openalex": {"id": "W1", "doi": "10.1234/example"}},
            }
        ],
        "count": 1,
        "targets": ["openalex"],
    }
    assert metadata_updates[-1]["status"] == "completed"


def test_api_queue_worker_and_result_end_to_end(monkeypatch, tmp_path):
    """Exercise request validation, queueing, worker execution, and result storage."""
    from citation_index import api, tasks

    class EndToEndConnector(FakeOpenAlexConnector):
        def search(self, reference, top_k=10, **kwargs):
            first = "First linked paper" in reference.full_title
            return [
                {
                    "id": f"https://openalex.org/{'W1' if first else 'W2'}",
                    "title": "First linked paper" if first else "Second linked paper",
                    "publication_year": reference.publication_year,
                    "doi": f"https://doi.org/10.1234/{'first' if first else 'second'}",
                }
            ]

    test_storage = StorageManager(tmp_path)
    metadata = {}

    def initialize_metadata(job_id, job_type, **fields):
        metadata[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "type": job_type,
            "created_at": "2026-08-27T00:00:00",
            **fields,
        }

    def update_metadata(job_id, **fields):
        metadata[job_id].update(fields)

    class ImmediateQueue:
        def enqueue_call(self, function, kwargs, **options):
            assert options["job_id"] == "job-1"
            function(**kwargs)

    monkeypatch.setattr(
        citation_linking, "CONNECTOR_TYPES", {"openalex": EndToEndConnector}
    )
    monkeypatch.setattr(api, "storage", test_storage)
    monkeypatch.setattr(tasks, "storage", test_storage)
    monkeypatch.setattr(api, "create_job_id", lambda: "job-1")
    monkeypatch.setattr(api, "initialize_job_metadata", initialize_metadata)
    monkeypatch.setattr(api, "get_job_metadata", metadata.get)
    monkeypatch.setattr(api, "queue_linking", ImmediateQueue())
    monkeypatch.setattr(tasks, "update_job_metadata", update_metadata)
    monkeypatch.setattr(tasks, "log_job_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks, "get_completed_stages", lambda job_id: [])

    client = TestClient(api.app)
    submitted = client.post(
        "/link/references?batched=true&target=openalex",
        json={
            "reference": [parsed("First linked paper"), parsed("Second linked paper", 2025)]
        },
    )

    assert submitted.status_code == 200
    assert submitted.json()["status"] == "completed"

    result = client.get("/jobs/job-1")
    assert result.status_code == 200
    assert result.json() == {
        "results": [
            {
                "reference": parsed("First linked paper"),
                "links": {
                    "openalex": {
                        "id": "https://openalex.org/W1",
                        "doi": "10.1234/first",
                    }
                },
            },
            {
                "reference": parsed("Second linked paper", 2025),
                "links": {
                    "openalex": {
                        "id": "https://openalex.org/W2",
                        "doi": "10.1234/second",
                    }
                },
            },
        ],
        "count": 2,
        "targets": ["openalex"],
    }


def test_linking_task_marks_failures_and_does_not_write_result(monkeypatch, tmp_path):
    from citation_index import tasks

    test_storage = StorageManager(tmp_path)
    test_storage.save_intermediate(
        "job-failed",
        "citation_linking_input",
        {"references": [parsed()], "targets": ["openalex"]},
    )
    metadata_updates = []

    monkeypatch.setattr(tasks, "storage", test_storage)
    monkeypatch.setattr(
        tasks,
        "link_references",
        lambda references, targets: (_ for _ in ()).throw(
            RuntimeError("OpenAlex unavailable")
        ),
    )
    monkeypatch.setattr(
        tasks,
        "update_job_metadata",
        lambda job_id, **fields: metadata_updates.append(fields),
    )
    monkeypatch.setattr(tasks, "log_job_event", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="OpenAlex unavailable"):
        tasks.link_references_task("job-failed")

    assert metadata_updates[-1]["status"] == "failed"
    assert metadata_updates[-1]["failed_stage"] == "citation_linking"
    assert metadata_updates[-1]["error"] == "OpenAlex unavailable"
    assert not test_storage.result_exists("job-failed")


def test_api_rejects_unsupported_target():
    from citation_index import api

    response = TestClient(api.app).post(
        "/link/references?target=opencitations",
        json={"reference": parsed()},
    )

    assert response.status_code == 400
    assert "Unsupported linking target" in response.json()["detail"]


@pytest.mark.parametrize("patch", [
    {"full_title": None}, {"full_title": "  "}, {"authors": []},
    {"authors": [" "]}, {"authors": [{}]}, {"publication_year": None},
])
def test_api_rejects_incomplete_parsed_reference(patch):
    from citation_index import api
    response = TestClient(api.app).post(
        "/link/references", json={"reference": {**parsed(), **patch}}
    )
    assert response.status_code == 422


@pytest.mark.parametrize("field", ["full_title", "authors", "publication_year"])
def test_api_requires_linking_fields(field):
    from citation_index import api
    reference = parsed()
    del reference[field]
    assert TestClient(api.app).post(
        "/link/references", json={"reference": reference}
    ).status_code == 422


@pytest.mark.parametrize("title,author,year,expected", [
    ("A linked paper", "Jones", 2025, True),
    ("A linked paper", "Smith", 1990, True),
    ("A linked paper", "Jones", 1990, False),
    ("Unrelated work", "Smith", 2024, False),
    ("A linked paper", None, None, False),
])
def test_custom_match(title, author, year, expected):
    from citation_index.utils.reference_matching import custom_match
    assert custom_match(Reference(**parsed()), {
        "title": title, "first_author": author, "year": year
    })[0] is expected


def test_filters_before_selecting_candidate(monkeypatch):
    class Connector(FakeOpenAlexConnector):
        def search(self, reference, **kwargs):
            assert reference.authors == ["Smith"]
            assert reference.publication_year == 2024
            return [
                {"id": "wrong", "title": reference.full_title, "publication_year": 1990},
                {"id": "correct", "title": reference.full_title, "publication_year": 2024},
            ]
    monkeypatch.setattr(citation_linking, "CONNECTOR_TYPES", {"openalex": Connector})
    assert citation_linking.link_references([parsed()], ["openalex"])[0]["links"]["openalex"]["id"] == "correct"


@pytest.mark.parametrize("reference,query,status", [
    ("Smith. A linked paper. 2024.", "", 422),
    ([], "?batched=true", 400),
    ([parsed()], "", 400),
    (parsed(), "?batched=true", 400),
])
def test_api_rejects_invalid_linking_shape(reference, query, status):
    from citation_index import api
    assert TestClient(api.app).post(
        "/link/references" + query, json={"reference": reference}
    ).status_code == status


@pytest.mark.parametrize("author", ["Smith, John", {"surname": "Smith", "first_name": "John"}])
def test_match_accepts_parsed_author_formats(author):
    from citation_index.api import LinkingReference
    from citation_index.utils.reference_matching import custom_match
    reference = LinkingReference(**{**parsed(), "authors": [author]})
    assert custom_match(reference, {
        "title": "A linked paper", "first_author": "Smith", "year": 1990
    })[0]
