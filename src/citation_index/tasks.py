"""RQ task wrappers around existing pipeline functions.

All tasks follow the pattern:
1. Load input from storage
2. Call existing pipeline function
3. Save output to storage
4. Update job metadata in Redis
"""

import json
import logging
import socket
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import redis
from rq import get_current_job
from rq.decorators import job

from .config import settings
from .llm.client import LLMClient
from .llm.grobid_client import GrobidClient
from .pipelines.end_to_end_parsing import run_pdf_one_step, run_pdf_semantic_one_step
from .pipelines.reference_extraction import (
    extract_text_references,
    extract_text_references_semantic_sections,
)
from .pipelines.reference_parsing import (
    parse_reference_strings,
    parse_reference_strings_grobid,
)
from .pipelines.text_extraction import extract_text
from .utils.storage import StorageManager

logger = logging.getLogger(__name__)

# Initialize global instances
redis_conn = redis.from_url(settings.redis_url)
storage = StorageManager(settings.storage_root)


# ========================
# LLM Semaphore for Rate Limiting
# ========================

class LLMSemaphore:
    """Redis-based semaphore for limiting concurrent LLM requests."""
    
    def __init__(self, redis_conn, max_concurrent: int = 4):
        self.redis = redis_conn
        self.max_concurrent = max_concurrent
        self.semaphore_key = "llm:semaphore"
    
    @contextmanager
    def acquire(self):
        """Acquire a semaphore slot (blocks until available)."""
        acquired = False
        try:
            while True:
                current = self.redis.incr(self.semaphore_key)
                if current <= self.max_concurrent:
                    acquired = True
                    break
                self.redis.decr(self.semaphore_key)
                time.sleep(0.1)
            
            yield
            
        finally:
            if acquired:
                self.redis.decr(self.semaphore_key)


llm_semaphore = LLMSemaphore(redis_conn, max_concurrent=settings.llm_max_concurrent)


# ========================
# Job Metadata Helpers
# ========================

def update_job_metadata(job_id: str, **fields):
    """Update job metadata in Redis HASH."""
    redis_conn.hset(f"job:{job_id}", mapping=fields)
    redis_conn.expire(f"job:{job_id}", settings.job_metadata_ttl)


def log_job_event(job_id: str, event: str, **extra):
    """Log an event to Redis Stream for debugging."""
    event_data = {
        "event": event,
        "timestamp": datetime.utcnow().isoformat(),
        "worker": socket.gethostname(),
        **extra
    }
    redis_conn.xadd(f"job:{job_id}:events", event_data)


def get_completed_stages(job_id: str) -> list:
    """Get list of completed stages for a job."""
    completed = redis_conn.hget(f"job:{job_id}", "completed_stages")
    if completed:
        return json.loads(completed)
    return []


# ========================
# Text Extraction Tasks
# ========================

@job('default', connection=redis_conn, timeout=settings.timeout_text_extraction, result_ttl=settings.worker_result_ttl)
def extract_text_task(
    job_id: str,
    extractor: str = "pymupdf",
    markdown: bool = True
) -> Dict[str, Any]:
    """Extract text from uploaded PDF.
    
    Args:
        job_id: Unique job identifier
        extractor: Extractor to use (pymupdf, marker, mineru, grobid)
        markdown: Whether to extract as markdown
        
    Returns:
        Small metadata dict (actual result saved to storage)
    """
    current_job = get_current_job()
    stage = "text_extraction"
    
    update_job_metadata(
        job_id,
        status="processing",
        current_stage=stage,
        started_at=datetime.utcnow().isoformat()
    )
    log_job_event(job_id, "stage_started", stage=stage)
    
    try:
        # Check if already completed (idempotency)
        if storage.intermediate_exists(job_id, stage):
            logger.info(f"Stage {stage} already completed for job {job_id}")
            result_path = storage.intermediate_dir / job_id / stage / "output.json"
            return {"result_path": str(result_path), "cached": True, "stage": stage}
        
        # Load uploaded PDF
        pdf_path = storage.get_upload_path(job_id)
        
        # Call existing pipeline function
        extract_result = extract_text(
            pdf_path=pdf_path,
            extractor=extractor,
            markdown=markdown
        )
        
        # Prepare output
        output = {
            "text": extract_result.text,
            "metadata": extract_result.metadata,
            "extractor": extractor,
            "markdown": markdown
        }
        
        # Save to storage
        result_path = storage.save_intermediate(job_id, stage, output, atomic=True)
        storage.save_result(job_id, output)
        
        # Update metadata
        completed_stages = get_completed_stages(job_id) + [stage]
        update_job_metadata(
            job_id,
            status="completed",
            completed_stages=json.dumps(completed_stages),
            completed_at=datetime.utcnow().isoformat(),
            **{f"stage_{stage}_completed_at": datetime.utcnow().isoformat()}
        )
        log_job_event(job_id, "stage_completed", stage=stage)
        
        return {
            "result_path": str(result_path),
            "cached": False,
            "stage": stage,
            "text_length": len(extract_result.text)
        }
        
    except Exception as e:
        logger.error(f"Task {stage} failed for job {job_id}: {e}")
        update_job_metadata(
            job_id,
            status="failed",
            error=str(e),
            failed_stage=stage,
            failed_at=datetime.utcnow().isoformat()
        )
        log_job_event(job_id, "stage_failed", stage=stage, error=str(e))
        raise


# ========================
# Reference Extraction Tasks
# ========================

@job('llm-tasks', connection=redis_conn, timeout=settings.timeout_reference_extraction, result_ttl=settings.worker_result_ttl)
def extract_references_task(
    job_id: str,
    method: str = "semantic",
    prompt_name: str = "prompts/reference_extraction.md",
    temperature: float = 0.3
) -> Dict[str, Any]:
    """Extract reference strings from text using LLM.
    
    Args:
        job_id: Unique job identifier
        method: Extraction method ('semantic', 'full_text', 'page_by_page')
        prompt_name: Prompt template to use
        temperature: LLM temperature
        
    Returns:
        Small metadata dict
    """
    stage = "reference_extraction"
    
    update_job_metadata(job_id, status="processing", current_stage=stage)
    log_job_event(job_id, "stage_started", stage=stage)
    
    try:
        if storage.intermediate_exists(job_id, stage):
            logger.info(f"Stage {stage} already completed for job {job_id}")
            result_path = storage.intermediate_dir / job_id / stage / "output.json"
            return {"result_path": str(result_path), "cached": True, "stage": stage}
        
        # Load text from previous stage
        text_data = storage.load_intermediate(job_id, "text_extraction")
        text = text_data["text"]
        
        # Initialize LLM client (long FTT for extraction)
        llm_client = LLMClient(
            endpoint=settings.llm_endpoint,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
            first_token_timeout=settings.llm_first_token_timeout_reference_extraction
        )
        
        # Extract references with semaphore rate limiting
        with llm_semaphore.acquire():
            if method == "semantic":
                references = extract_text_references_semantic_sections(
                    text_or_pdf=text,
                    llm_client=llm_client,
                    embedding_model=settings.embedding_model,
                    embedding_endpoint=settings.embedding_endpoint,
                    prompt_name=prompt_name,
                    temperature=temperature
                )
            else:
                references = extract_text_references(
                    text=text,
                    llm_client=llm_client,
                    prompt_name=prompt_name,
                    temperature=temperature
                )
        
        # Save output
        output = {
            "references": references,
            "method": method,
            "count": len(references)
        }
        result_path = storage.save_intermediate(job_id, stage, output, atomic=True)
        storage.save_result(job_id, output)
        
        # Update metadata
        completed_stages = get_completed_stages(job_id) + [stage]
        update_job_metadata(
            job_id,
            status="completed",
            completed_stages=json.dumps(completed_stages),
            completed_at=datetime.utcnow().isoformat(),
            **{f"stage_{stage}_completed_at": datetime.utcnow().isoformat()}
        )
        log_job_event(job_id, "stage_completed", stage=stage, reference_count=len(references))
        
        return {
            "result_path": str(result_path),
            "cached": False,
            "stage": stage,
            "reference_count": len(references)
        }
        
    except Exception as e:
        logger.error(f"Task {stage} failed for job {job_id}: {e}")
        update_job_metadata(
            job_id,
            status="failed",
            error=str(e),
            failed_stage=stage,
            failed_at=datetime.utcnow().isoformat()
        )
        log_job_event(job_id, "stage_failed", stage=stage, error=str(e))
        raise


# ========================
# Reference Parsing Tasks
# ========================

@job('llm-tasks', connection=redis_conn, timeout=settings.timeout_reference_parsing, result_ttl=settings.worker_result_ttl)
def parse_references_task(
    job_id: str,
    parser: str = "llm",
    prompt_name: str = "prompts/reference_parsing.md",
    temperature: float = 0.0
) -> Dict[str, Any]:
    """Parse reference strings into structured data.
    
    Args:
        job_id: Unique job identifier
        parser: Parser to use ('llm' or 'grobid')
        prompt_name: Prompt template (for LLM parser)
        temperature: LLM temperature
        
    Returns:
        Small metadata dict
    """
    stage = "reference_parsing"
    
    update_job_metadata(job_id, status="processing", current_stage=stage)
    log_job_event(job_id, "stage_started", stage=stage)
    
    try:
        if storage.intermediate_exists(job_id, stage):
            logger.info(f"Stage {stage} already completed for job {job_id}")
            result_path = storage.intermediate_dir / job_id / stage / "output.json"
            return {"result_path": str(result_path), "cached": True, "stage": stage}
        
        # Load references from previous stage
        ref_data = storage.load_intermediate(job_id, "reference_extraction")
        reference_lines = ref_data["references"]
        
        if parser == "grobid":
            # Use GROBID parser
            grobid_client = GrobidClient(
                endpoint=settings.grobid_endpoint,
                timeout=settings.grobid_timeout
            )
            parsed_refs = parse_reference_strings_grobid(
                reference_lines=reference_lines,
                grobid_client=grobid_client
            )
        else:
            # Use LLM parser with semaphore (short FTT, long timeout for parsing)
            llm_client = LLMClient(
                endpoint=settings.llm_endpoint,
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                timeout=settings.llm_timeout_reference_parsing,
                max_retries=settings.llm_max_retries,
                first_token_timeout=settings.llm_first_token_timeout_reference_parsing
            )
            
            with llm_semaphore.acquire():
                parsed_refs = parse_reference_strings(
                    reference_lines=reference_lines,
                    llm_client=llm_client,
                    prompt_name=prompt_name,
                    temperature=temperature
                )
        
        # Save output
        output = {
            "references": [ref.model_dump() for ref in parsed_refs],
            "parser": parser,
            "count": len(parsed_refs)
        }
        result_path = storage.save_intermediate(job_id, stage, output, atomic=True)
        storage.save_result(job_id, output)
        
        # Update metadata
        completed_stages = get_completed_stages(job_id) + [stage]
        update_job_metadata(
            job_id,
            status="completed",
            completed_stages=json.dumps(completed_stages),
            completed_at=datetime.utcnow().isoformat(),
            **{f"stage_{stage}_completed_at": datetime.utcnow().isoformat()}
        )
        log_job_event(job_id, "stage_completed", stage=stage, reference_count=len(parsed_refs))
        
        return {
            "result_path": str(result_path),
            "cached": False,
            "stage": stage,
            "reference_count": len(parsed_refs)
        }
        
    except Exception as e:
        logger.error(f"Task {stage} failed for job {job_id}: {e}")
        update_job_metadata(
            job_id,
            status="failed",
            error=str(e),
            failed_stage=stage,
            failed_at=datetime.utcnow().isoformat()
        )
        log_job_event(job_id, "stage_failed", stage=stage, error=str(e))
        raise


# ========================
# Combined Pipeline Tasks
# ========================

@job('llm-tasks', connection=redis_conn, timeout=settings.timeout_reference_extraction, result_ttl=settings.worker_result_ttl)
def extract_and_parse_references_task(
    job_id: str,
    method: str = "one_step",
    prompt_name: str = "prompts/end_to_end_parsing.md",
    temperature: float = 0.3
) -> Dict[str, Any]:
    """Extract and parse references in one LLM call (end-to-end parsing).
    
    Args:
        job_id: Unique job identifier
        method: Method to use ('one_step' or 'semantic_one_step')
        prompt_name: Prompt template
        temperature: LLM temperature
        
    Returns:
        Small metadata dict
    """
    stage = "end_to_end_parsing"
    
    update_job_metadata(job_id, status="processing", current_stage=stage)
    log_job_event(job_id, "stage_started", stage=stage)
    
    try:
        if storage.intermediate_exists(job_id, stage):
            logger.info(f"Stage {stage} already completed for job {job_id}")
            result_path = storage.intermediate_dir / job_id / stage / "output.json"
            return {"result_path": str(result_path), "cached": True, "stage": stage}
        
        # Load text from previous stage
        text_data = storage.load_intermediate(job_id, "text_extraction")
        text = text_data["text"]
        
        # Initialize LLM client (long FTT for end-to-end parsing)
        llm_client = LLMClient(
            endpoint=settings.llm_endpoint,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
            first_token_timeout=settings.llm_first_token_timeout_end_to_end
        )
        
        # Run combined extraction + parsing with semaphore
        with llm_semaphore.acquire():
            if method == "semantic_one_step":
                parsed_refs = run_pdf_semantic_one_step(
                    text_or_pdf=text,
                    llm_client=llm_client,
                    embedding_model=settings.embedding_model,
                    embedding_endpoint=settings.embedding_endpoint,
                    prompt_name=prompt_name,
                    temperature=temperature
                )
            else:
                parsed_refs = run_pdf_one_step(
                    text_or_pdf=text,
                    llm_client=llm_client,
                    prompt_name=prompt_name,
                    temperature=temperature
                )
        
        # Save output
        output = {
            "references": [ref.model_dump() for ref in parsed_refs],
            "method": method,
            "count": len(parsed_refs)
        }
        result_path = storage.save_intermediate(job_id, stage, output, atomic=True)
        storage.save_result(job_id, output)
        
        # Update metadata
        completed_stages = get_completed_stages(job_id) + [stage]
        update_job_metadata(
            job_id,
            status="completed",
            completed_stages=json.dumps(completed_stages),
            completed_at=datetime.utcnow().isoformat(),
            **{f"stage_{stage}_completed_at": datetime.utcnow().isoformat()}
        )
        log_job_event(job_id, "stage_completed", stage=stage, reference_count=len(parsed_refs))
        
        return {
            "result_path": str(result_path),
            "cached": False,
            "stage": stage,
            "reference_count": len(parsed_refs)
        }
        
    except Exception as e:
        logger.error(f"Task {stage} failed for job {job_id}: {e}")
        update_job_metadata(
            job_id,
            status="failed",
            error=str(e),
            failed_stage=stage,
            failed_at=datetime.utcnow().isoformat()
        )
        log_job_event(job_id, "stage_failed", stage=stage, error=str(e))
        raise
