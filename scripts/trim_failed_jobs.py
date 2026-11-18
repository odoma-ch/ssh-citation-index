"""Re-apply WORKER_FAILURE_TTL to failed RQ jobs recorded before it existed.

Failed jobs enqueued without ``failure_ttl`` keep RQ's default of one year.
For each job in the failed registries this deletes the record if it ended more
than ``worker_failure_ttl`` seconds ago, otherwise shortens its expiry to the
time left. Dry run by default; pass ``--apply`` to change Redis.

    python scripts/trim_failed_jobs.py            # report only
    python scripts/trim_failed_jobs.py --apply
"""

import argparse
from datetime import datetime, timezone

import redis
from rq.exceptions import NoSuchJobError
from rq.job import Job
from rq.registry import FailedJobRegistry

from citation_index.config import settings

QUEUES = ("default", "llm-tasks", "linking")


def seconds_left(job: Job, ttl: int, now: datetime) -> int:
    """Remaining lifetime under the new TTL; 0 when expired or end time unknown."""
    ended = job.ended_at
    if ended is None:
        return 0
    if ended.tzinfo is None:
        ended = ended.replace(tzinfo=timezone.utc)
    return max(0, int(ttl - (now - ended).total_seconds()))


def trim(conn: redis.Redis, ttl: int, apply: bool) -> dict:
    now = datetime.now(timezone.utc)
    counts = {"deleted": 0, "shortened": 0}
    for queue in QUEUES:
        registry = FailedJobRegistry(queue, connection=conn)
        for job_id in registry.get_job_ids():
            try:
                job = Job.fetch(job_id, connection=conn)
            except NoSuchJobError:
                job = None
            left = seconds_left(job, ttl, now) if job else 0
            if left == 0:
                counts["deleted"] += 1
                if apply:
                    registry.remove(job_id)
                    if job:
                        job.delete(remove_from_queue=False)
            else:
                counts["shortened"] += 1
                if apply:
                    conn.zadd(registry.key, {job_id: now.timestamp() + left})
                    job.cleanup(ttl=left)
                    conn.expire(f"rq:results:{job_id}", left)
        print(f"{queue}: {registry.count} failed jobs in registry")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write changes")
    args = parser.parse_args()
    conn = redis.from_url(settings.redis_url)
    ttl = settings.worker_failure_ttl
    counts = trim(conn, ttl, args.apply)
    mode = "applied" if args.apply else "dry run"
    print(f"{mode} (failure_ttl={ttl}s): {counts}")


if __name__ == "__main__":
    main()
