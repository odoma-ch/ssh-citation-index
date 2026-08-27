from fastapi.testclient import TestClient
import pytest

from citation_index.pipelines import citation_linking
from citation_index.utils.storage import StorageManager


class FakeOpenAlexConnector:
    def search_by_id(self, identifier, identifier_type=None, **kwargs):
        if identifier:
            return [
                {
                    "id": "https://openalex.org/W1",
                    "title": "A linked paper",
                    "doi": "https://doi.org/10.1234/example",
                }
            ]
        return []

    def search(self, reference, top_k=10, **kwargs):
        return [
            {
                "id": "https://openalex.org/W2",
                "title": "A linked paper",
                "doi": "https://doi.org/10.1234/title-match",
            }
        ]


class FakeMatildaConnector:
    def search_by_id(self, identifier, identifier_type=None, **kwargs):
        return []

    def search(self, reference, top_k=10, **kwargs):
        return [
            {
                "id": "matilda-1",
                "texts": [
                    {
                        "title": ["A linked paper"],
                        "identifier": [{"doi": ["10.1234/matilda"]}],
                    }
                ],
            }
        ]


class FakeWikidataConnector:
    def search_by_id(self, identifier, identifier_type=None, **kwargs):
        return []

    def search(self, reference, top_k=10, **kwargs):
        return [
            {
                "id": "Q1",
                "label": "A linked paper",
                "claims": {
                    "P356": [{"mainsnak": {"datavalue": {"value": "10.1234/WIKIDATA"}}}]
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
    assert citation_linking.split_references("first\n\n second ", True) == [
        "first",
        "second",
    ]

    with pytest.raises(ValueError, match="Unsupported linking target"):
        citation_linking.parse_targets("opencitations")
    with pytest.raises(ValueError, match="non-empty reference"):
        citation_linking.split_references(" \n ", True)


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
        ["Authors. A linked paper. 2024."],
        ["openalex", "matilda", "wikidata"],
    )

    assert results == [
        {
            "reference": "Authors. A linked paper. 2024.",
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


def test_link_references_prefers_exact_doi(monkeypatch):
    monkeypatch.setattr(
        citation_linking, "CONNECTOR_TYPES", {"openalex": FakeOpenAlexConnector}
    )

    result = citation_linking.link_references(
        ["A citation. doi:10.1234/example."], ["openalex"]
    )

    assert result[0]["links"]["openalex"] == {
        "id": "https://openalex.org/W1",
        "doi": "10.1234/example",
    }


def test_link_references_rejects_weak_title_match(monkeypatch):
    class WeakConnector(FakeOpenAlexConnector):
        def search(self, reference, top_k=10, **kwargs):
            return [{"id": "W-wrong", "title": "Completely unrelated work"}]

    monkeypatch.setattr(
        citation_linking, "CONNECTOR_TYPES", {"openalex": WeakConnector}
    )

    result = citation_linking.link_references(
        ["Authors. A linked paper. 2024."], ["openalex"]
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
        citation_linking.link_references(["A reference"], ["openalex"])


def test_linking_task_persists_result_and_completes_job(monkeypatch, tmp_path):
    from citation_index import tasks

    test_storage = StorageManager(tmp_path)
    test_storage.save_intermediate(
        "job-1",
        "citation_linking_input",
        {"references": ["A reference"], "targets": ["openalex"]},
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
                "reference": "A reference",
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
            "reference": (
                "Authors. First linked paper. 2024.\n"
                "Authors. Second linked paper. 2025."
            )
        },
    )

    assert submitted.status_code == 200
    assert submitted.json()["status"] == "completed"

    result = client.get("/jobs/job-1")
    assert result.status_code == 200
    assert result.json() == {
        "results": [
            {
                "reference": "Authors. First linked paper. 2024.",
                "links": {
                    "openalex": {
                        "id": "https://openalex.org/W1",
                        "doi": "10.1234/first",
                    }
                },
            },
            {
                "reference": "Authors. Second linked paper. 2025.",
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
        {"references": ["A reference"], "targets": ["openalex"]},
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
        json={"reference": "A reference"},
    )

    assert response.status_code == 400
    assert "Unsupported linking target" in response.json()["detail"]
