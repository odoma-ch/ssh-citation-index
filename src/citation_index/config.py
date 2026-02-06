"""Configuration settings for Citation Index API and workers."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ========================
    # Redis Configuration
    # ========================
    redis_host: str = Field(default="localhost", description="Redis server host")
    redis_port: int = Field(default=6379, description="Redis server port")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_password: Optional[str] = Field(default=None, description="Redis password (optional)")
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL from components."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    # ========================
    # Storage Configuration
    # ========================
    storage_root: Path = Field(
        default=Path("./storage"),
        description="Root directory for file storage (use PV mount in K8s)"
    )
    storage_cleanup_age_hours: int = Field(
        default=24,
        description="Delete job files older than this many hours"
    )
    
    # ========================
    # LLM Configuration
    # ========================
    llm_endpoint: str = Field(
        default="http://localhost:8000/v1",
        description="vLLM API endpoint"
    )
    llm_model: str = Field(
        default="your-model-name",
        description="LLM model name"
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        description="LLM API key (optional for local vLLM)"
    )
    llm_timeout: float = Field(
        default=180.0,
        description="LLM request timeout in seconds"
    )
    llm_max_retries: int = Field(
        default=3,
        description="Maximum LLM retry attempts"
    )
    llm_max_concurrent: int = Field(
        default=4,
        description="Maximum concurrent LLM requests (semaphore limit)"
    )
    
    # ========================
    # GROBID Configuration
    # ========================
    grobid_endpoint: str = Field(
        default="http://localhost:8070",
        description="GROBID server endpoint"
    )
    grobid_timeout: float = Field(
        default=180.0,
        description="GROBID request timeout in seconds"
    )
    
    # ========================
    # Worker Configuration
    # ========================
    worker_result_ttl: int = Field(
        default=3600,
        description="How long to keep job results in Redis (seconds)"
    )
    worker_default_timeout: int = Field(
        default=600,
        description="Default job timeout in seconds"
    )
    
    # Queue-specific timeouts
    timeout_text_extraction: int = Field(default=300, description="Text extraction timeout")
    timeout_reference_extraction: int = Field(default=900, description="Reference extraction timeout")
    timeout_reference_parsing: int = Field(default=900, description="Reference parsing timeout")
    timeout_citation_linking: int = Field(default=600, description="Citation linking timeout")
    
    # ========================
    # API Configuration
    # ========================
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_reload: bool = Field(default=False, description="Enable hot reload (dev only)")
    api_workers: int = Field(default=1, description="Number of API worker processes")
    
    # ========================
    # Job Metadata Configuration
    # ========================
    job_metadata_ttl: int = Field(
        default=7200,
        description="How long to keep job metadata in Redis (seconds)"
    )
    
    # ========================
    # Embedding Service Configuration (for semantic reference locator)
    # ========================
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-large-instruct",
        description="Embedding model for semantic search"
    )
    embedding_endpoint: str = Field(
        default="http://0.0.0.0:7997/embeddings",
        description="Embedding service endpoint"
    )


# Global settings instance
settings = Settings()
