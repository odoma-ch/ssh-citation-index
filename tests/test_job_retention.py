"""Job retention: RQ TTLs on every enqueue and status lookups after RQ expiry."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from citation_index import api, tasks
from citation_index.config import settings

client = TestClient(api.app)
PDF = {"file": ("paper.pdf", b"%PDF-1.4", "application/pdf")}


def test_task_decorators_set_failure_ttl():
    rq_tasks = [
        obj for obj in vars(tasks).values() if callable(obj) and hasattr(obj, "delay")
    ]
    assert len(rq_tasks) == 8
    for task in rq_tasks:
        with patch("rq.queue.Queue.enqueue_call") as enqueue_call:
            task.delay("job-1")
        options = enqueue_call.call_args.kwargs
        assert options["result_ttl"] == settings.worker_result_ttl, task
        assert options["failure_ttl"] == settings.worker_failure_ttl, task


def test_api_enqueue_sets_failure_ttl():
    response = api.JobResponse(job_id="j", status="queued", created_at="t")
    with patch.object(api, "storage"), patch.object(
        api, "initialize_job_metadata"
    ), patch.object(api, "format_job_response", return_value=response), patch.object(
        api, "queue_default"
    ) as queue:
        assert client.post("/extract/text", files=PDF).status_code == 200
    options = queue.enqueue_call.call_args.kwargs
    assert options["result_ttl"] == settings.worker_result_ttl
    assert options["failure_ttl"] == settings.worker_failure_ttl


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_terminal_status_skips_rq_lookup(status):
    metadata = {"job_id": "job-1", "status": status}
    with patch.object(api, "get_job_metadata", return_value=metadata), patch(
        "rq.job.Job.fetch"
    ) as fetch:
        r = client.get("/jobs/job-1/status")
    assert r.status_code == 200 and r.json()["status"] == status
    fetch.assert_not_called()
