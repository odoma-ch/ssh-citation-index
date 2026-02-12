"""FastAPI application for Citation Index queue system.

Provides endpoints for:
- Job submission (enqueue tasks)
- Job status checking
- Result retrieval
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import redis
from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from rq import Queue

from .config import settings
from .tasks import (
    extract_text_task,
    extract_references_task,
    parse_references_task,
    extract_and_parse_references_task,
)
from .utils.storage import StorageManager

# ========================
# Initialize
# ========================

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Citation Index API",
    description="Extract and parse citations from academic PDFs",
    version="0.1.0"
)

redis_conn = redis.from_url(settings.redis_url)
storage = StorageManager(settings.storage_root)

# Initialize RQ queues
queue_default = Queue('default', connection=redis_conn)
queue_llm = Queue('llm-tasks', connection=redis_conn)
queue_linking = Queue('linking', connection=redis_conn)


# ========================
# Request/Response Models
# ========================

class JobResponse(BaseModel):
    """Response when a job is enqueued."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status (queued, processing, completed, failed)")
    created_at: str = Field(..., description="ISO 8601 timestamp")
    message: Optional[str] = Field(None, description="Additional information")


class JobStatus(BaseModel):
    """Job status information."""
    job_id: str
    status: str
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    current_stage: Optional[str] = None
    completed_stages: Optional[list] = None
    error: Optional[str] = None
    progress: Optional[int] = None


class TextExtractionOptions(BaseModel):
    """Options for text extraction."""
    extractor: str = Field(default="pymupdf", description="Extractor to use")
    markdown: bool = Field(default=True, description="Extract as markdown")


class ReferenceExtractionOptions(BaseModel):
    """Options for reference extraction."""
    method: str = Field(default="full_text", description="Extraction method")
    prompt_name: str = Field(default="prompts/reference_extraction.md", description="Prompt template")
    temperature: float = Field(default=0.3, description="LLM temperature")


class ReferenceExtractionRequest(BaseModel):
    """Request body for reference extraction - accepts markdown text directly."""
    text: str = Field(..., description="Markdown text to extract references from")


class ReferenceParsingRequest(BaseModel):
    """Request body for reference parsing - accepts list of reference strings."""
    references: list[str] = Field(..., description="List of reference strings to parse")


class ReferenceParsingOptions(BaseModel):
    """Options for reference parsing."""
    parser: str = Field(default="llm", description="Parser to use (llm or grobid)")
    prompt_name: str = Field(default="prompts/reference_parsing.md", description="Prompt template")
    temperature: float = Field(default=0.0, description="LLM temperature")


class CombinedPipelineOptions(BaseModel):
    """Options for combined extraction and parsing."""
    method: str = Field(default="one_step", description="Method (one_step or semantic_one_step)")
    prompt_name: str = Field(default="prompts/end_to_end_parsing.md", description="Prompt template")
    temperature: float = Field(default=0.3, description="LLM temperature")


# ========================
# Helper Functions
# ========================

def create_job_id() -> str:
    """Generate a unique job ID."""
    return str(uuid.uuid4())


def initialize_job_metadata(job_id: str, job_type: str, **extra_fields):
    """Initialize job metadata in Redis."""
    metadata = {
        "job_id": job_id,
        "status": "queued",
        "type": job_type,
        "created_at": datetime.utcnow().isoformat(),
        **extra_fields
    }
    redis_conn.hset(f"job:{job_id}", mapping=metadata)
    redis_conn.expire(f"job:{job_id}", settings.job_metadata_ttl)


def get_job_metadata(job_id: str) -> dict:
    """Retrieve job metadata from Redis."""
    metadata = redis_conn.hgetall(f"job:{job_id}")
    if not metadata:
        return None
    
    # Decode bytes to strings
    decoded = {k.decode(): v.decode() for k, v in metadata.items()}
    
    # Parse JSON fields
    if "completed_stages" in decoded:
        decoded["completed_stages"] = json.loads(decoded["completed_stages"])
    
    return decoded


def format_job_response(job_id: str, message: Optional[str] = None) -> JobResponse:
    """Format a job response."""
    metadata = get_job_metadata(job_id)
    return JobResponse(
        job_id=job_id,
        status=metadata.get("status", "queued") if metadata else "queued",
        created_at=metadata.get("created_at", datetime.utcnow().isoformat()) if metadata else datetime.utcnow().isoformat(),
        message=message
    )


# ========================
# Health & Monitoring
# ========================

@app.get("/health")
def health_check():
    """Health check endpoint."""
    try:
        # Check Redis connection
        redis_conn.ping()
        redis_ok = True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        redis_ok = False
    
    # Check storage
    storage_ok = settings.storage_root.exists()
    
    status = "healthy" if (redis_ok and storage_ok) else "unhealthy"
    
    return {
        "status": status,
        "redis": "ok" if redis_ok else "error",
        "storage": "ok" if storage_ok else "error",
        "version": "0.1.0"
    }


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "Citation Index API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


# ========================
# Job Status & Results
# ========================

@app.get("/jobs/{job_id}/status", response_model=JobStatus)
def get_job_status(job_id: str):
    """Get job status, checking both metadata and RQ job state."""
    from rq.job import Job
    from rq.registry import FailedJobRegistry, FinishedJobRegistry
    
    metadata = get_job_metadata(job_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Check actual RQ job status to catch failed/killed jobs
    try:
        rq_job = Job.fetch(job_id, connection=redis_conn)
        
        # If RQ says the job failed but our metadata doesn't reflect it, update metadata
        if rq_job.is_failed and metadata.get("status") not in ["failed", "completed"]:
            error_msg = str(rq_job.exc_info) if rq_job.exc_info else "Job was killed or failed"
            redis_conn.hset(f"job:{job_id}", mapping={
                "status": "failed",
                "error": error_msg,
                "finished_at": datetime.utcnow().isoformat()
            })
            metadata["status"] = "failed"
            metadata["error"] = error_msg
            
        # Check if job is in failed registry
        elif metadata.get("status") == "processing":
            for queue_name in ["default", "llm-tasks", "linking"]:
                failed_registry = FailedJobRegistry(queue_name, connection=redis_conn)
                if job_id in failed_registry.get_job_ids():
                    redis_conn.hset(f"job:{job_id}", mapping={
                        "status": "failed",
                        "error": "Job was moved to failed registry",
                        "finished_at": datetime.utcnow().isoformat()
                    })
                    metadata["status"] = "failed"
                    metadata["error"] = "Job was moved to failed registry"
                    break
                    
    except Exception as e:
        logger.warning(f"Could not fetch RQ job {job_id}: {e}")
        # If we can't fetch the job from RQ but have metadata, continue with metadata
    
    return JobStatus(**metadata)


@app.get("/jobs/{job_id}")
def get_job_result(
    job_id: str,
    format: str = Query(default="json", description="Output format (json or xml)")
):
    """Get job result."""
    metadata = get_job_metadata(job_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    status = metadata.get("status")
    
    if status == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Job failed: {metadata.get('error', 'Unknown error')}"
        )
    
    if status != "completed":
        # Job still processing
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": status,
                "message": "Job is still processing",
                "current_stage": metadata.get("current_stage")
            }
        )
    
    # Job completed - return result
    try:
        result = storage.get_result(job_id)
        
        if format == "xml" and "xml_output" in result:
            return result["xml_output"]
        
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Job result not found in storage"
        )


# ========================
# Text Extraction
# ========================

@app.post("/extract/text", response_model=JobResponse)
async def enqueue_text_extraction(
    file: UploadFile = File(...),
    extractor: str = Query(default="pymupdf", description="Extractor to use"),
    markdown: bool = Query(default=True, description="Extract as markdown")
):
    """Extract text from a PDF."""
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create job
    job_id = create_job_id()
    
    # Save uploaded file (as input.pdf so task can find it)
    file_content = await file.read()
    storage.save_upload(job_id, file_content, "input.pdf")
    
    # Initialize metadata
    initialize_job_metadata(
        job_id,
        job_type="text_extraction",
        filename=file.filename,
        extractor=extractor
    )
    
    # Enqueue task (timeout and result_ttl must be set explicitly on enqueue_call)
    queue_default.enqueue_call(
        extract_text_task,
        args=(job_id,),
        kwargs={"extractor": extractor, "markdown": markdown},
        job_id=job_id,
        timeout=settings.timeout_text_extraction,
        result_ttl=settings.worker_result_ttl,
    )
    
    logger.info(f"Enqueued text extraction job {job_id}")
    return format_job_response(job_id, message="Text extraction job enqueued")


# ========================
# Reference Extraction
# ========================

@app.post("/extract/references", response_model=JobResponse)
async def enqueue_reference_extraction(
    body: ReferenceExtractionRequest = Body(...),
    method: str = Query(default="full_text", description="Extraction method"),
    prompt_name: str = Query(default="prompts/reference_extraction.md", description="Prompt template"),
    temperature: float = Query(default=0.3, description="LLM temperature")
):
    """Extract references from markdown text using LLM (single stage)."""
    job_id = create_job_id()

    # Save provided text as intermediate so extract_references_task can load it
    text_data = {
        "text": body.text,
        "metadata": {},
        "extractor": "provided",
        "markdown": True,
    }
    storage.save_intermediate(job_id, "text_extraction", text_data, atomic=False)

    # Initialize metadata
    initialize_job_metadata(
        job_id,
        job_type="reference_extraction",
    )

    # Single stage: reference extraction (uses provided text)
    queue_llm.enqueue_call(
        extract_references_task,
        kwargs={"job_id": job_id, "method": method, "prompt_name": prompt_name, "temperature": temperature},
        job_id=job_id,
        timeout=settings.timeout_reference_extraction,
        result_ttl=settings.worker_result_ttl,
    )

    logger.info(f"Enqueued reference extraction job {job_id}")
    return format_job_response(job_id, message="Reference extraction job enqueued")


# ========================
# Reference Parsing
# ========================

@app.post("/parse/references", response_model=JobResponse)
def parse_reference_strings_endpoint(
    body: ReferenceParsingRequest = Body(...),
    parser: str = Query(default="llm", description="Parser (llm or grobid)"),
    prompt_name: str = Query(default="prompts/reference_parsing.md", description="Prompt template"),
    temperature: float = Query(default=0.0, description="LLM temperature")
):
    """Parse a list of reference strings into structured data."""
    job_id = create_job_id()
    
    # Save references as intermediate data
    ref_data = {"references": body.references, "count": len(body.references)}
    storage.save_intermediate(job_id, "reference_extraction", ref_data)
    
    # Initialize metadata
    initialize_job_metadata(
        job_id,
        job_type="reference_parsing",
        reference_count=len(body.references)
    )
    
    # Enqueue parsing task
    queue = queue_llm if parser == "llm" else queue_default
    timeout = settings.timeout_reference_parsing if parser == "llm" else settings.timeout_reference_parsing
    queue.enqueue_call(
        parse_references_task,
        kwargs={"job_id": job_id, "parser": parser, "prompt_name": prompt_name, "temperature": temperature},
        job_id=job_id,
        timeout=timeout,
        result_ttl=settings.worker_result_ttl,
    )
    
    logger.info(f"Enqueued reference parsing job {job_id}")
    return format_job_response(job_id, message="Reference parsing job enqueued")


# ========================
# Full Reference Pipeline
# ========================

# @app.post("/process/references", response_model=JobResponse)
# async def enqueue_end_to_end_reference_pipeline(
#     file: UploadFile = File(...),
#     extractor: str = Query(default="pymupdf", description="Text extractor"),
#     method: str = Query(default="one_step", description="Processing method"),
#     prompt_name: str = Query(default="prompts/end_to_end_parsing.md", description="Prompt template"),
#     temperature: float = Query(default=0.3, description="LLM temperature")
# ):
#     """end-to-end reference pipeline: extract text → extract and parse references."""
#     if not file.filename.endswith('.pdf'):
#         raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
#     job_id = create_job_id()
    
#     # Save upload
#     file_content = await file.read()
#     storage.save_upload(job_id, file_content, file.filename)
    
#     # Initialize metadata
#     initialize_job_metadata(
#         job_id,
#         job_type="end_to_end_reference_pipeline",
#         filename=file.filename
#     )
    
#     # Stage 1: Text extraction
#     text_job = queue_default.enqueue_call(
#         extract_text_task,
#         args=(job_id,),
#         kwargs={"extractor": extractor},
#     )
    
#     # Stage 2: end-to-end extraction + parsing (depends on stage 1)
#     combined_job = queue_llm.enqueue_call(
#         extract_and_parse_references_task,
#         kwargs={"job_id": job_id, "method": method, "prompt_name": prompt_name, "temperature": temperature},
#         depends_on=text_job,
#     )
    
#     # Store stage job IDs
#     redis_conn.hset(
#         f"job:{job_id}",
#         "stage_job_ids",
#         json.dumps([text_job.id, combined_job.id])
#     )
    
#     logger.info(f"Enqueued full reference pipeline {job_id}")
#     return format_job_response(job_id, message="Full reference pipeline enqueued")


# ========================
# Run Server
# ========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "citation_index.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )
