"""RQ worker entry point with queue selection and graceful shutdown.

Usage:
    # Start worker for specific queues
    python -m citation_index.worker --queues default,llm-tasks
    
    # Start worker for all queues
    python -m citation_index.worker --queues default,llm-tasks,linking
    
    # Start with specific worker name
    python -m citation_index.worker --queues default --name worker-default-1
"""

import argparse
import logging
import signal
import sys
from typing import List

import redis
from rq import Worker
from rq.logutils import setup_loghandlers

from .config import settings

logger = logging.getLogger(__name__)


class GracefulWorker(Worker):
    """RQ Worker with graceful shutdown handling.
    
    On SIGTERM/SIGINT:
    - Finish current job
    - Release semaphores
    - Exit cleanly
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shutdown_requested = False
    
    def handle_warm_shutdown_request(self, signum, frame):
        """Handle graceful shutdown signal."""
        logger.info("Warm shutdown requested, finishing current job...")
        self.request_stop(signum=signum, frame=frame)
    
    def register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        # Call parent implementation first
        super().register_signal_handlers()
        
        # Override with our graceful handlers
        signal.signal(signal.SIGTERM, lambda s, f: self.handle_warm_shutdown_request(s, f))
        signal.signal(signal.SIGINT, lambda s, f: self.handle_warm_shutdown_request(s, f))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Start RQ worker for Citation Index tasks"
    )
    
    parser.add_argument(
        "--queues",
        type=str,
        default="default",
        help="Comma-separated list of queues to process (e.g., 'default,llm-tasks')"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Worker name (defaults to auto-generated)"
    )
    
    parser.add_argument(
        "--burst",
        action="store_true",
        help="Run in burst mode (process all jobs then exit)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def start_worker(queue_names: List[str], worker_name: str = None, burst: bool = False):
    """Start an RQ worker.
    
    Args:
        queue_names: List of queue names to process
        worker_name: Optional worker name
        burst: If True, process all jobs then exit
    """
    # Connect to Redis
    try:
        conn = redis.from_url(settings.redis_url)
        conn.ping()
        logger.info(f"Connected to Redis at {settings.redis_url}")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        sys.exit(1)
    
    # Create queues
    from rq import Queue
    queues = [Queue(name, connection=conn) for name in queue_names]
    
    logger.info(f"Worker will process queues: {', '.join(queue_names)}")
    
    # Create worker with graceful shutdown
    worker = GracefulWorker(
        queues,
        connection=conn,
        name=worker_name
    )
    
    logger.info(f"Starting worker: {worker.name}")
    
    # Start processing jobs
    try:
        worker.work(burst=burst, logging_level="INFO")
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.error(f"Worker error: {e}")
        sys.exit(1)
    finally:
        logger.info("Worker stopped")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Setup RQ logging
    setup_loghandlers(level=log_level)
    
    # Parse queue names
    queue_names = [q.strip() for q in args.queues.split(",")]
    
    logger.info("=" * 60)
    logger.info("Citation Index RQ Worker")
    logger.info("=" * 60)
    logger.info(f"Queues: {', '.join(queue_names)}")
    logger.info(f"Worker name: {args.name or 'auto-generated'}")
    logger.info(f"Burst mode: {args.burst}")
    logger.info(f"Redis: {settings.redis_url}")
    logger.info(f"Storage: {settings.storage_root}")
    logger.info("=" * 60)
    
    # Start worker
    start_worker(
        queue_names=queue_names,
        worker_name=args.name,
        burst=args.burst
    )


if __name__ == "__main__":
    main()
