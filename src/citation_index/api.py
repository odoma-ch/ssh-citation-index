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
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
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
    method: str = Field(default="semantic", description="Extraction method")
    prompt_name: str = Field(default="prompts/reference_extraction.md", description="Prompt template")
    temperature: float = Field(default=0.3, description="LLM temperature")


class ReferenceParsingOptions(BaseModel):
    """Options for reference parsing."""
    parser: str = Field(default="llm", description="Parser to use (llm or grobid)")
    prompt_name: str = Field(default="prompts/reference_parsing.md", description="Prompt template")
    temperature: float = Field(default=0.0, description="LLM temperature")


class CombinedPipelineOptions(BaseModel):
    """Options for combined extraction and parsing."""
    method: str = Field(default="one_step", description="Method (one_step or semantic_one_step)")
    prompt_name: str = Field(default="prompts/reference_extraction_and_parsing.md", description="Prompt template")
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
    """Get job status."""
    metadata = get_job_metadata(job_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
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
    
    # Save uploaded file
    file_content = await file.read()
    storage.save_upload(job_id, file_content, file.filename)
    
    # Initialize metadata
    initialize_job_metadata(
        job_id,
        job_type="text_extraction",
        filename=file.filename,
        extractor=extractor
    )
    
    # Enqueue task (job_id is both the RQ job ID and first arg to task)
    queue_default.enqueue(
        extract_text_task,
        job_id,  # First positional arg to task function
        extractor=extractor,
        markdown=markdown,
        job_id=job_id,  # RQ job ID (unified with pipeline job ID)
        timeout=settings.timeout_text_extraction
    )
    
    logger.info(f"Enqueued text extraction job {job_id}")
    return format_job_response(job_id, message="Text extraction job enqueued")


# ========================
# Reference Extraction
# ========================

@app.post("/extract/references", response_model=JobResponse)
async def enqueue_reference_extraction(
    file: UploadFile = File(...),
    extractor: str = Query(default="pymupdf", description="Text extractor"),
    method: str = Query(default="semantic", description="Extraction method"),
    prompt_name: str = Query(default="prompts/reference_extraction.md", description="Prompt template"),
    temperature: float = Query(default=0.3, description="LLM temperature")
):
    """Extract references from a PDF (two-stage: text extraction → reference extraction)."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    job_id = create_job_id()
    
    # Save upload
    file_content = await file.read()
    storage.save_upload(job_id, file_content, file.filename)
    
    # Initialize metadata
    initialize_job_metadata(
        job_id,
        job_type="reference_extraction_pipeline",
        filename=file.filename
    )
    
    # Stage 1: Text extraction
    text_job = queue_default.enqueue(
        extract_text_task,
        job_id,
        extractor=extractor,
        job_id=f"{job_id}_stage1",
        timeout=settings.timeout_text_extraction
    )
    
    # Stage 2: Reference extraction (depends on stage 1)
    ref_job = queue_llm.enqueue(
        extract_references_task,
        job_id=job_id,
        method=method,
        prompt_name=prompt_name,
        temperature=temperature,
        depends_on=text_job,
        timeout=settings.timeout_reference_extraction
    )
    
    # Store stage job IDs in metadata
    redis_conn.hset(
        f"job:{job_id}",
        "stage_job_ids",
        json.dumps([text_job.id, ref_job.id])
    )
    
    logger.info(f"Enqueued reference extraction pipeline {job_id}")
    return format_job_response(job_id, message="Reference extraction pipeline enqueued")


# ========================
# Reference Parsing
# ========================

@app.post("/parse/references", response_model=JobResponse)
def parse_reference_strings_endpoint(
    references: list[str],
    parser: str = Query(default="llm", description="Parser (llm or grobid)"),
    prompt_name: str = Query(default="prompts/reference_parsing.md", description="Prompt template"),
    temperature: float = Query(default=0.0, description="LLM temperature")
):
    """Parse a list of reference strings (synchronous for now)."""
    # This endpoint is synchronous since it doesn't need file upload
    # Could be made async if needed
    
    job_id = create_job_id()
    
    # Save references as intermediate data
    ref_data = {"references": references, "count": len(references)}
    storage.save_intermediate(job_id, "reference_extraction", ref_data)
    
    # Initialize metadata
    initialize_job_metadata(
        job_id,
        job_type="reference_parsing",
        reference_count=len(references)
    )
    
    # Enqueue parsing task
    queue = queue_llm if parser == "llm" else queue_default
    queue.enqueue(
        parse_references_task,
        job_id=job_id,
        parser=parser,
        prompt_name=prompt_name,
        temperature=temperature,
        timeout=settings.timeout_reference_parsing
    )
    
    logger.info(f"Enqueued reference parsing job {job_id}")
    return format_job_response(job_id, message="Reference parsing job enqueued")


# ========================
# Full Reference Pipeline
# ========================

@app.post("/process/references", response_model=JobResponse)
async def enqueue_full_reference_pipeline(
    file: UploadFile = File(...),
    extractor: str = Query(default="pymupdf", description="Text extractor"),
    method: str = Query(default="one_step", description="Processing method"),
    prompt_name: str = Query(default="prompts/reference_extraction_and_parsing.md", description="Prompt template"),
    temperature: float = Query(default=0.3, description="LLM temperature")
):
    """Full reference pipeline: extract text → extract and parse references."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    job_id = create_job_id()
    
    # Save upload
    file_content = await file.read()
    storage.save_upload(job_id, file_content, file.filename)
    
    # Initialize metadata
    initialize_job_metadata(
        job_id,
        job_type="full_reference_pipeline",
        filename=file.filename
    )
    
    # Stage 1: Text extraction
    text_job = queue_default.enqueue(
        extract_text_task,
        job_id=job_id,
        extractor=extractor,
        timeout=settings.timeout_text_extraction
    )
    
    # Stage 2: Combined extraction + parsing (depends on stage 1)
    combined_job = queue_llm.enqueue(
        extract_and_parse_references_task,
        job_id=job_id,
        method=method,
        prompt_name=prompt_name,
        temperature=temperature,
        depends_on=text_job,
        timeout=settings.timeout_reference_extraction
    )
    
    # Store stage job IDs
    redis_conn.hset(
        f"job:{job_id}",
        "stage_job_ids",
        json.dumps([text_job.id, combined_job.id])
    )
    
    logger.info(f"Enqueued full reference pipeline {job_id}")
    return format_job_response(job_id, message="Full reference pipeline enqueued")


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
