FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the application as an immutable image package.
RUN pip install --no-cache-dir --no-deps .

# Create storage directory
RUN mkdir -p /app/storage

# Expose API port
EXPOSE 8000

# Default command (can be overridden in docker-compose.yml)
CMD ["uvicorn", "citation_index.api:app", "--host", "0.0.0.0", "--port", "8000"]
