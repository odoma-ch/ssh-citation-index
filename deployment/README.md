# Kubernetes / OKD Deployment

Deploy the Citation Index queue system on a Kubernetes (OKD) PaaS.

## Architecture

```
                ┌──────────────┐
                │   OKD Route  │
                └──────┬───────┘
                       │
                ┌──────▼───────┐      ┌─────────────────────────┐
                │  API (×1)    │      │  RQ Dashboard (×1)      │
                │  FastAPI     ├─────►│  Web UI on port 9181    │
                └──────┬───────┘      └────────────┬────────────┘
                       │                           │
                ┌──────▼───────────────────────────▼──┐
                │            Redis (Helm)              │
                └──┬───────────────┬──────────────┬───┘
                   │               │              │
          ┌────────▼───┐  ┌───────▼────┐  ┌──────▼──────┐
          │ worker     │  │ worker     │  │ worker      │
          │ default ×2 │  │ llm    ×6  │  │ linking ×2  │
          └────────────┘  └────────────┘  └─────────────┘
                   │               │              │
                   └───────────────┼──────────────┘
                                   │
                          ┌────────▼────────┐
                          │  PVC (RWX)      │
                          │  /app/storage   │
                          └─────────────────┘
```

## Files

| File | Contents |
|------|----------|
| `redis.yaml` | Helm values for the Redis chart |
| `citation-index.yaml` | Everything else: ConfigMap, PVC, API, Workers, RQ Dashboard |

## Prerequisites

- `kubectl` or `oc` CLI configured and authenticated
- Helm 3
- A namespace with sufficient quota for ~12 pods
- A storage class that supports **ReadWriteMany** (CephFS, NFS, etc.)

## Deployment Steps

### 1. Select namespace

```bash
oc project <your-namespace>
# or
kubectl config set-context --current --namespace=<your-namespace>
```

### 2. Deploy Redis

Create the password secret first, then install the Helm chart:

```bash
# Create Redis auth secret
kubectl create secret generic citation-index-redis-secret \
  --from-literal=redis-password='...'

# Install Redis
helm upgrade -i citation-index-redis \
  oci://registry.paas.psnc.pl/helm/redis-simple \
  -f deployment/redis.yaml

# Verify
kubectl get pods -l app.kubernetes.io/name=redis-simple
```

> **TLS note:** `redis.yaml` has TLS enabled. If your cluster doesn't
> require internal TLS, set `tls.enabled: false` before installing.
> If TLS is required, create the `redis-certs` secret with your
> certificate/key and update the app to use `rediss://` URLs.

### 3. Create application secrets

Create the secret with your real API keys (never store these in Git):

```bash
kubectl create secret generic citation-index-secret \
  --from-literal=LLM_API_KEY='' \
  --from-literal=EMBEDDING_API_KEY='your-embedding-api-key'
```

### 4. Edit the ConfigMap and image references

Before applying, open `citation-index.yaml` and update:

- **ConfigMap** values marked `# <-- UPDATE`:
  - `LLM_ENDPOINT` – your vLLM / LiteLLM proxy URL
  - `LLM_MODEL_MEDIUM_INTELLIGENCE` / `LLM_MODEL_HIGH_INTELLIGENCE`
  - `GROBID_ENDPOINT` – if GROBID runs elsewhere in the cluster
  - `EMBEDDING_ENDPOINT`
- **image:** fields (4 occurrences) – pin the release tag, for example `registry.paas.psnc.pl/graphia/citation-index:0.2.0`
  with your actual image path

### 5. Apply everything

```bash
kubectl apply -f deployment/citation-index.yaml
```

This single command creates the ConfigMap, PVC, API Deployment + Service + Route,
all three worker Deployments, and the RQ Dashboard.

### 6. Verify

```bash
# All pods running?
kubectl get pods -l app=citation-index

# API route
kubectl get route citation-index-api          # OKD
ROUTE=$(kubectl get route citation-index-api -o jsonpath='{.spec.host}')

# Health check
curl -s https://${ROUTE}/health
# → {"status":"healthy","redis":"ok","storage":"ok","version":"0.2.0"}

# Submit a test job
curl -X POST https://${ROUTE}/extract/text -F "file=@test.pdf"

# RQ Dashboard
kubectl get route citation-index-rq-dashboard
```

Expected pods: 1 API + 2 default + 6 llm + 1 dashboard = **10 total** (linking worker is WIP, commented out).

## Scaling

| Worker | Queue | Default replicas | Guidance |
|--------|-------|-----------------|----------|
| `worker-default` | `default` | 2 | Fast tasks; scale if text extraction queues build up |
| `worker-llm` | `llm-tasks` | 6 | Slow LLM tasks; `replicas >= LLM_MAX_CONCURRENT × 1.5` |
| `worker-linking` | `linking` | 2 | WIP -- commented out in YAML for now |

```bash
kubectl scale deployment citation-index-worker-llm --replicas=8
```

## Updating configuration

After changing ConfigMap values in the YAML, re-apply and restart:

```bash
kubectl apply -f deployment/citation-index.yaml
kubectl rollout restart deployment citation-index-api
kubectl rollout restart deployment citation-index-worker-default
kubectl rollout restart deployment citation-index-worker-llm
kubectl rollout restart deployment citation-index-worker-linking
```

## Useful Commands

```bash
# Tail API logs
kubectl logs -l component=api -f --tail=50

# Tail all worker logs
kubectl logs -l app=citation-index,component!=api -f --tail=50

# Check Redis connectivity from a debug pod
export REDIS_PASSWORD=$(kubectl get secret litellm-redis-secret \
  -o jsonpath="{.data.redis-password}" | base64 -d)
kubectl run redis-test --rm -it --restart=Never \
  --image=registry.paas.psnc.pl/base/bitnami/redis:7.4.2-debian-12-r0 \
  -- redis-cli -h litellm-redis-redis-simple-master -a "$REDIS_PASSWORD" ping
```
