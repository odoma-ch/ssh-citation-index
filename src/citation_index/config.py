"""Configuration settings for Citation Index API and workers."""

from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field, computed_field
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
    llm_model_medium_intelligence: str = Field(
        default="your-model-name",
        description="LLM model for medium intelligence tasks (default)",
        validation_alias=AliasChoices("llm_model_medium_intelligence", "llm_model"),
    )
    llm_model_high_intelligence: str = Field(
        default="your-model-name",
        description="LLM model for high intelligence tasks"
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
    
    @computed_field
    def llm_model(self) -> str:
        """Active model used by default; currently medium intelligence."""
        return self.llm_model_medium_intelligence
    
    # Task-specific first token timeouts
    llm_first_token_timeout_reference_parsing: float = Field(
        default=30.0,
        description="Short FTT for reference parsing (model starts quickly)"
    )
    llm_first_token_timeout_reference_extraction: float = Field(
        default=90.0,
        description="Long FTT for reference extraction"
    )
    llm_first_token_timeout_end_to_end: float = Field(
        default=90.0,
        description="Long FTT for end-to-end parsing"
    )
    llm_timeout_reference_parsing: float = Field(
        default=300.0,
        description="Longer timeout for reference parsing (vs default llm_timeout)"
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
    embedding_api_key: Optional[str] = Field(
        default=None,
        description="Embedding API key (optional for local services)"
    )


# Global settings instance
settings = Settings()
