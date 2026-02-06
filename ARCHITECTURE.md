# Citation Index Architecture

Complete architecture documentation for the Citation Index queue system.

## System Overview

```mermaid
graph TB
    Client[Client/Browser] -->|HTTP POST| API[FastAPI API Server]
    API -->|Enqueue Job| Redis[(Redis)]
    Redis -->|Fetch Job| W1[RQ Worker - Default]
    Redis -->|Fetch Job| W2[RQ Worker - LLM]
    Redis -->|Fetch Job| W3[RQ Worker - Linking]
    
    W1 -->|Read/Write Files| Storage[Filesystem Storage]
    W2 -->|Read/Write Files| Storage
    W3 -->|Read/Write Files| Storage
    
    W2 -->|Rate-Limited Calls| LLM[vLLM Server]
    W1 -->|Parse References| GROBID[GROBID Server]
    W3 -->|Link Citations| APIs[External APIs<br/>OpenAlex, Wikidata, etc.]
    
    W1 -->|Update Metadata| Redis
    W2 -->|Update Metadata| Redis
    W3 -->|Update Metadata| Redis
    
    API -->|Poll Status| Redis
    Client -->|GET /jobs/:id| API
    
    Dashboard[RQ Dashboard] -->|Monitor| Redis
```

## Component Breakdown

### 1. FastAPI Server (`api.py`)

**Role**: Accept HTTP requests, enqueue jobs, return status

```mermaid
graph LR
    Request[HTTP Request] --> Validate[Validate Input]
    Validate --> Save[Save Upload to Storage]
    Save --> Create[Create Job ID]
    Create --> Enqueue[Enqueue to RQ]
    Enqueue --> Response[Return Job ID]
```

**Key Files**:
- `src/citation_index/api.py` - FastAPI routes
- Port: 8000 (default)

**Endpoints**:
- `POST /extract/text` - Text extraction
- `POST /extract/references` - Reference extraction  
- `POST /parse/references` - Reference parsing
- `POST /process/references` - Full pipeline
- `GET /jobs/{id}/status` - Job status
- `GET /jobs/{id}` - Job result
- `GET /health` - Health check

### 2. Redis

**Role**: Message broker + job metadata store

```
Redis Keys:
├── rq:queue:default          # Queue: text extraction jobs
├── rq:queue:llm-tasks        # Queue: LLM-based jobs
├── rq:queue:linking          # Queue: citation linking jobs
├── job:{job_id}              # HASH: job metadata
├── job:{job_id}:events       # STREAM: job events (optional)
└── llm:semaphore             # Counter: LLM rate limiting
```

**Job Metadata (Redis HASH)**:
```json
{
  "job_id": "550e8400-e29b-41d4...",
  "status": "processing",
  "type": "full_reference_pipeline",
  "current_stage": "reference_parsing",
  "completed_stages": ["text_extraction"],
  "created_at": "2026-02-06T12:00:00",
  "started_at": "2026-02-06T12:00:05",
  "filename": "paper.pdf"
}
```

### 3. RQ Workers (`worker.py`)

**Role**: Process jobs from queues (one job at a time per worker)

```mermaid
graph TD
    Start[Worker Start] --> Connect[Connect to Redis]
    Connect --> Listen[Listen on Queues]
    Listen --> Fetch[Fetch Job]
    Fetch --> Execute[Execute Task]
    Execute --> Success{Success?}
    Success -->|Yes| Update[Update Metadata]
    Success -->|No| Error[Log Error]
    Update --> Listen
    Error --> Listen
```

**Worker Class**: Standard `Worker` (with process forking + heartbeats)
- Essential for long-running LLM tasks (10-15 minutes)
- Heartbeats prevent false "stuck job" detection
- Process isolation for better fault tolerance

**Worker Deployment**:

| Worker Pool | Queue | Task Duration | Replicas | Why |
|-------------|-------|---------------|----------|-----|
| worker-default | `default` | Fast (~30s-2min) | 2 | Always responsive for quick tasks |
| worker-llm | `llm-tasks` | Slow (~10-15min) | 6 | High count to drain queue despite slow tasks |
| worker-linking | `linking` | Medium (~1-3min) | 2 | External API calls |

**Key Insight**: 
- `worker-llm` has 6 replicas but semaphore limit is 4
- Workers 1-4: Call vLLM
- Workers 5-6: Wait for semaphore (have jobs ready!)
- When Worker 1 finishes → Worker 5 starts immediately (no gap!)
- This maximizes throughput while protecting vLLM

### 4. Storage Manager (`utils/storage.py`)

**Role**: Manage job files on filesystem

```
storage/
├── uploads/
│   └── {job_id}/
│       └── input.pdf
├── intermediate/
│   └── {job_id}/
│       ├── text_extraction/
│       │   └── output.json
│       ├── reference_parsing/
│       │   └── output.json
│       └── citation_linking/
│           └── output.json
└── results/
    └── {job_id}/
        └── result.json
```

**Key Features**:
- Atomic writes (temp → rename for idempotency)
- Job cleanup (delete files older than 24h)
- Shared filesystem (K8s PVC with ReadWriteMany)

### 5. Task Wrappers (`tasks.py`)

**Role**: Thin wrappers around existing pipeline functions

```python
@job('llm-tasks', timeout=900)
def extract_references_task(job_id: str, **params):
    # 1. Load input from storage
    text = storage.load_intermediate(job_id, 'text_extraction')
    
    # 2. Call existing pipeline function (no changes needed!)
    with llm_semaphore.acquire():
        refs = extract_text_references(text, llm_client, **params)
    
    # 3. Save output to storage
    storage.save_intermediate(job_id, 'reference_extraction', refs)
    
    # 4. Update metadata
    redis.hset(f"job:{job_id}", "status", "completed")
```

## Job Lifecycle

### Single-Stage Job

```mermaid
sequenceDiagram
    participant C as Client
    participant API as FastAPI
    participant R as Redis
    participant W as Worker
    participant S as Storage
    
    C->>API: POST /extract/text + PDF
    API->>S: Save upload
    API->>R: HSET job:123 (metadata)
    API->>R: LPUSH rq:queue:default
    API->>C: {job_id: "123"}
    
    W->>R: BRPOP rq:queue:default
    R->>W: Job data
    W->>S: Read upload
    W->>W: extract_text()
    W->>S: Save intermediate result
    W->>R: HSET job:123 status=completed
    
    C->>API: GET /jobs/123
    API->>R: HGET job:123
    API->>S: Read result
    API->>C: Result data
```

### Multi-Stage Pipeline

```mermaid
sequenceDiagram
    participant C as Client
    participant API as FastAPI
    participant R as Redis
    participant W1 as Worker (default)
    participant W2 as Worker (llm)
    participant S as Storage
    
    C->>API: POST /process/references + PDF
    API->>S: Save upload
    API->>R: HSET job:456 (metadata)
    
    Note over API,R: Stage 1: Text Extraction
    API->>R: LPUSH rq:queue:default (job1)
    
    Note over API,R: Stage 2: Depends on Stage 1
    API->>R: LPUSH rq:queue:llm-tasks (job2, depends_on=job1)
    API->>C: {job_id: "456"}
    
    W1->>R: Fetch job1
    W1->>W1: extract_text()
    W1->>S: Save text
    W1->>R: Mark job1 complete
    
    Note over R,W2: Job2 now eligible
    W2->>R: Fetch job2
    W2->>S: Read text
    W2->>W2: extract_and_parse_references()
    W2->>S: Save parsed refs
    W2->>R: HSET job:456 status=completed
    
    C->>API: GET /jobs/456
    API->>S: Read final result
    API->>C: Parsed references
```

## Queue Strategy

### Queue Routing

```mermaid
graph LR
    subgraph Tasks
        T1[Text Extraction]
        T2[Ref Extraction LLM]
        T3[Ref Parsing LLM]
        T4[Citation Linking]
        T5[GROBID Parsing]
    end
    
    subgraph Queues
        Q1[default]
        Q2[llm-tasks]
        Q3[linking]
    end
    
    T1 --> Q1
    T2 --> Q2
    T3 --> Q2
    T4 --> Q3
    T5 --> Q1
```

**Rationale**:
- **default**: Fast, CPU-bound tasks (text extraction)
- **llm-tasks**: LLM calls (rate-limited, GPU-bound)
- **linking**: External API calls (I/O-bound, many retries)

### Worker Count vs Semaphore (Critical!)

**Why 6 workers but semaphore=4?**

```
6 LLM Workers (worker-llm replicas=6)
│
├─ Worker 1-4: Acquired semaphore → Calling vLLM
│              (4 concurrent vLLM requests - max capacity!)
│
└─ Worker 5-6: Have jobs, waiting for semaphore slots
               When Worker 1 finishes → Worker 5 starts IMMEDIATELY!
               (No gap, maximizes throughput)
```

**Purpose**:
- **Worker count (6)**: How many jobs "in flight" (some waiting)
- **Semaphore (4)**: Max concurrent vLLM calls (protects server)
- **Result**: Fast queue draining + vLLM protection

**Rule of thumb**: `workers = semaphore × 1.5 to 2`

### LLM Semaphore Implementation

```python
# In tasks.py - all LLM tasks use this
with llm_semaphore.acquire():
    result = llm_client.call(...)  # Max 4 workers here at once
```

**Algorithm** (Redis-based):
```python
# Atomic increment
current = redis.incr("llm:semaphore")
if current <= max_concurrent:  # Got a slot!
    do_llm_call()
    redis.decr("llm:semaphore")  # Release
else:
    redis.decr("llm:semaphore")  # Give back
    time.sleep(0.1)  # Retry
```

## Data Flow: Full Pipeline Example

```mermaid
graph LR
    Upload[PDF Upload] --> Q1[Queue: default]
    Q1 --> W1[Extract Text]
    W1 --> Q2[Queue: llm-tasks]
    Q2 --> Sem{Semaphore?}
    Sem -->|Wait| Sem
    Sem -->|Acquired| W2[Extract & Parse Refs]
    W2 --> Done[Result Ready]
```

**Storage Paths**:
- Upload: `storage/uploads/{job_id}/input.pdf`
- Intermediate: `storage/intermediate/{job_id}/{stage}/output.json`
- Final: `storage/results/{job_id}/result.json`

## Configuration

**Flow**: `.env` → `config.py` (Pydantic) → `api.py` / `worker.py` / `tasks.py`

**Key Settings**:
- **Redis**: `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`
- **LLM**: `LLM_ENDPOINT`, `LLM_MODEL`, `LLM_MAX_CONCURRENT` (semaphore limit)
- **Storage**: `STORAGE_ROOT` (filesystem path, e.g., `/mnt/pvc` in K8s)
- **Timeouts**: Per-stage timeouts in `config.py` (higher than app-level retries)

## Deployment Architectures

### Development (Docker Compose)

```bash
docker-compose up -d
# Services:
# - API (port 8000) - 2 replicas
# - Redis (port 6379)
# - worker-default (2 replicas)
# - worker-llm (6 replicas) - HIGH for slow tasks!
# - worker-linking (2 replicas)
# - RQ Dashboard (port 9181)
# - Shared volume: ./storage
```

### Production (Kubernetes/OKD)

**Components**:
- **API Deployment**: 2-4 pods (horizontal scaling)
- **Worker Deployments**: 
  - `worker-default`: 2 pods
  - `worker-llm`: 6-8 pods (scale based on LLM_MAX_CONCURRENT)
  - `worker-linking`: 2 pods
- **Redis**: StatefulSet or managed service
- **Storage**: PersistentVolumeClaim (ReadWriteMany, 100Gi+)

**External Services**: vLLM, GROBID (HTTP endpoints)

## Error Handling & Retry Strategy

### Two-Layer Timeout/Retry

```mermaid
graph TD
    Task[RQ Task Starts] --> App[Application Layer]
    
    subgraph Application Layer
        App --> Try1[LLMClient.call<br/>timeout=180s]
        Try1 -->|Timeout/Error| Retry1[Wait 2s]
        Retry1 --> Try2[Retry 2/3]
        Try2 -->|Timeout/Error| Retry2[Wait 4s]
        Retry2 --> Try3[Retry 3/3]
        Try3 -->|Success| Return[Return Result]
        Try3 -->|Fail| Raise[Raise Exception]
    end
    
    Raise -->|Caught by RQ| RQRetry{RQ Retry?}
    RQRetry -->|Yes| Wait[Wait 60s]
    Wait --> Task
    RQRetry -->|No| Failed[Job Failed]
    
    Return --> Success[Job Succeeded]
```

**Layer 1 - Application (Already Exists)**:
- `LLMClient`: 180s timeout, 3 retries, exponential backoff
- `GrobidClient`: 180s timeout, 3 retries, fixed delays
- Connectors: Various timeouts with retry logic

**Layer 2 - RQ (New)**:
- Job wrapper timeout: 900s (allows app retries to complete)
- RQ retries: 1-2 (only for worker crashes, not API failures)

### Idempotency

Tasks check for existing output before executing:
1. Check `storage/intermediate/{job_id}/{stage}/output.json`
2. If exists → return cached result
3. If not → execute, write to `.tmp`, then atomic rename

**Atomic writes prevent partial results** (worker crash = no corrupt file)

## Monitoring

### Health Checks

- **API**: `GET /health` → Redis + storage status
- **Workers**: RQ Dashboard at `http://localhost:9181`

### Job Tracking

- **Metadata**: Redis HASH `job:{id}` (status, stage, timestamps)
- **Events** (optional): Redis Stream `job:{id}:events`

### Key Metrics

| Metric | Source | Purpose |
|--------|--------|---------|
| Queue length | Redis `LLEN rq:queue:*` | Backlog monitoring |
| Active workers | RQ Dashboard | Worker health |
| Job success rate | Redis job metadata | Success tracking |
| LLM semaphore | Redis `GET llm:semaphore` | Concurrency usage |
| Storage size | Filesystem `du -sh storage/` | Disk usage |

## Performance Tuning

### Bottlenecks & Solutions

| Symptom | Cause | Solution |
|---------|-------|----------|
| High API latency | Too few API pods | Scale API replicas |
| Growing queue | Not enough workers | Scale worker replicas |
| Workers idle | Dependency blocking | Check job dependencies |
| LLM timeouts | Semaphore too high | Decrease `LLM_MAX_CONCURRENT` |
| Slow I/O | Storage bottleneck | Faster disk or caching |

### Scaling Guide

| Component | Low Load | Medium Load | High Load |
|-----------|----------|-------------|-----------|
| API replicas | 1 | 2 | 4 |
| worker-llm | 2 | 6 | 12 |
| LLM_MAX_CONCURRENT | 2 | 4 | 8 |

**Rule**: `worker-llm replicas = LLM_MAX_CONCURRENT × 1.5-2`

## Security Considerations

### Production Checklist

- [ ] Redis password authentication (`REDIS_PASSWORD`)
- [ ] API rate limiting (per client IP)
- [ ] File size limits (prevent DoS via large PDFs)
- [ ] Input validation (PDF format verification)
- [ ] Network policies (restrict worker → external access)
- [ ] Storage quota limits (prevent disk exhaustion)
- [ ] Log sanitization (remove sensitive data)
- [ ] HTTPS/TLS for API (via Ingress)

## Troubleshooting Guide

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Job stuck in "queued" | Status never changes | Check worker logs, ensure workers running |
| Worker crash | Jobs fail immediately | Check RAM, LLM endpoint reachable |
| LLM timeout | Jobs fail after 15 min | Increase `LLM_MAX_CONCURRENT` or add GPU |
| Storage full | Errors saving files | Clean old jobs: `storage.cleanup_old_jobs()` |
| Redis connection lost | All jobs fail | Check Redis health, restart if needed |

### Debug Commands

```bash
# Check queue lengths
docker-compose exec redis redis-cli LLEN rq:queue:default
docker-compose exec redis redis-cli LLEN rq:queue:llm-tasks

# Check active workers
docker-compose exec redis redis-cli KEYS rq:worker:*

# Check job metadata
docker-compose exec redis redis-cli HGETALL job:YOUR_JOB_ID

# Check LLM semaphore usage
docker-compose exec redis redis-cli GET llm:semaphore

# Worker logs
docker-compose logs -f worker-llm

# Storage usage
du -sh storage/
```

## References

- **RQ Documentation**: https://python-rq.org/
- **FastAPI Documentation**: https://fastapi.tianglio.com/
- **Redis Documentation**: https://redis.io/docs/
- **Pydantic Settings**: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
