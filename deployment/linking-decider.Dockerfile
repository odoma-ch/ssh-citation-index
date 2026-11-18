# Citation-linking decider (services/linking_decider). Build from the repo root:
#   docker build -f deployment/linking-decider.Dockerfile -t linking-decider .
# For the cluster from an Apple Silicon Mac, add --platform linux/amd64.
FROM python:3.12-slim

WORKDIR /app

# CPU-only torch: the default PyPI wheel pulls several GB of CUDA libraries.
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu

COPY services/linking_decider/requirements.txt services/linking_decider/requirements.txt
RUN pip install --no-cache-dir -r services/linking_decider/requirements.txt

COPY services/__init__.py services/__init__.py
COPY services/linking_decider/*.py services/linking_decider/

# OpenShift runs an arbitrary UID without a home directory; the Hub snapshot
# lands in HF_HOME (an emptyDir in deployment/linking-decider.yaml).
ENV HOME=/tmp \
    HF_HOME=/tmp/hf \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "services.linking_decider.app:app", "--host", "0.0.0.0", "--port", "8000"]
