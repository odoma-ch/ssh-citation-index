"""Storage manager for job files on filesystem (backed by K8s persistent volume)."""

import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages job files on local filesystem.
    
    In production, storage_root should be mounted from a K8s PersistentVolumeClaim
    with ReadWriteMany access mode so all API and worker pods can access it.
    
    Directory structure:
        storage_root/
        ├── uploads/{job_id}/input.pdf
        ├── intermediate/{job_id}/{stage}/output.json
        └── results/{job_id}/result.json
    """
    
    def __init__(self, storage_root: Path):
        """Initialize storage manager.
        
        Args:
            storage_root: Root directory for all job files (e.g., /app/storage)
        """
        self.root = Path(storage_root)
        self.uploads_dir = self.root / "uploads"
        self.intermediate_dir = self.root / "intermediate"
        self.results_dir = self.root / "results"
        
        # Create directories if they don't exist
        for dir_path in [self.uploads_dir, self.intermediate_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Storage directory ready: {dir_path}")
    
    # ========================
    # Upload Management
    # ========================
    
    def save_upload(self, job_id: str, file_content: bytes, filename: str) -> Path:
        """Save uploaded file for a job.
        
        Args:
            job_id: Unique job identifier
            file_content: Raw file bytes
            filename: Original filename
            
        Returns:
            Path to saved file
        """
        job_upload_dir = self.uploads_dir / job_id
        job_upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = job_upload_dir / filename
        file_path.write_bytes(file_content)
        
        logger.info(f"Saved upload for job {job_id}: {file_path}")
        return file_path
    
    def get_upload_path(self, job_id: str, filename: str = "input.pdf") -> Path:
        """Get path to uploaded file for a job.
        
        Args:
            job_id: Unique job identifier
            filename: Filename to look for (default: input.pdf)
            
        Returns:
            Path to uploaded file
            
        Raises:
            FileNotFoundError: If upload doesn't exist
        """
        file_path = self.uploads_dir / job_id / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Upload not found for job {job_id}: {file_path}")
        return file_path
    
    # ========================
    # Intermediate Stage Results
    # ========================
    
    def save_intermediate(
        self, 
        job_id: str, 
        stage: str, 
        data: Dict[str, Any],
        atomic: bool = True
    ) -> Path:
        """Save intermediate stage output with optional atomic write.
        
        Args:
            job_id: Unique job identifier
            stage: Stage name (e.g., 'text_extraction', 'reference_parsing')
            data: Stage output data (will be JSON serialized)
            atomic: Use atomic write (temp file + rename) for idempotency
            
        Returns:
            Path to saved file
        """
        stage_dir = self.intermediate_dir / job_id / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = stage_dir / "output.json"
        
        if atomic:
            # Atomic write: write to temp file, then rename
            temp_path = stage_dir / "output.json.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_path.rename(output_path)
        else:
            # Direct write
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved intermediate result for job {job_id}, stage {stage}")
        return output_path
    
    def load_intermediate(self, job_id: str, stage: str) -> Dict[str, Any]:
        """Load intermediate stage output.
        
        Args:
            job_id: Unique job identifier
            stage: Stage name
            
        Returns:
            Stage output data
            
        Raises:
            FileNotFoundError: If stage output doesn't exist
        """
        output_path = self.intermediate_dir / job_id / stage / "output.json"
        if not output_path.exists():
            raise FileNotFoundError(
                f"Intermediate output not found for job {job_id}, stage {stage}"
            )
        
        with open(output_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def intermediate_exists(self, job_id: str, stage: str) -> bool:
        """Check if intermediate stage output exists (for idempotency).
        
        Args:
            job_id: Unique job identifier
            stage: Stage name
            
        Returns:
            True if stage output exists
        """
        output_path = self.intermediate_dir / job_id / stage / "output.json"
        return output_path.exists()
    
    def get_temp_path(self, job_id: str, stage: str) -> Path:
        """Get path for temporary file (for atomic writes).
        
        Args:
            job_id: Unique job identifier
            stage: Stage name
            
        Returns:
            Path to temporary file
        """
        stage_dir = self.intermediate_dir / job_id / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir / "output.json.tmp"
    
    # ========================
    # Final Results
    # ========================
    
    def save_result(self, job_id: str, result: Dict[str, Any]) -> Path:
        """Save final job result.
        
        Args:
            job_id: Unique job identifier
            result: Final result data (will be JSON serialized)
            
        Returns:
            Path to saved file
        """
        job_result_dir = self.results_dir / job_id
        job_result_dir.mkdir(parents=True, exist_ok=True)
        
        result_path = job_result_dir / "result.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved final result for job {job_id}")
        return result_path
    
    def get_result(self, job_id: str) -> Dict[str, Any]:
        """Load final job result.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Final result data
            
        Raises:
            FileNotFoundError: If result doesn't exist
        """
        result_path = self.results_dir / job_id / "result.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Result not found for job {job_id}")
        
        with open(result_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def result_exists(self, job_id: str) -> bool:
        """Check if final result exists.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            True if result exists
        """
        result_path = self.results_dir / job_id / "result.json"
        return result_path.exists()
    
    # ========================
    # Cleanup
    # ========================
    
    def cleanup_job(self, job_id: str):
        """Delete all files for a job.
        
        Args:
            job_id: Unique job identifier
        """
        for base_dir in [self.uploads_dir, self.intermediate_dir, self.results_dir]:
            job_dir = base_dir / job_id
            if job_dir.exists():
                shutil.rmtree(job_dir)
                logger.info(f"Cleaned up {job_dir}")
    
    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Delete job files older than max_age_hours.
        
        Args:
            max_age_hours: Maximum age in hours before deletion
            
        Returns:
            Number of jobs cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        for base_dir in [self.uploads_dir, self.intermediate_dir, self.results_dir]:
            if not base_dir.exists():
                continue
                
            for job_dir in base_dir.iterdir():
                if not job_dir.is_dir():
                    continue
                
                # Check modification time
                mtime = datetime.fromtimestamp(job_dir.stat().st_mtime)
                if mtime < cutoff_time:
                    shutil.rmtree(job_dir)
                    cleaned_count += 1
                    logger.info(f"Cleaned up old job: {job_dir.name}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old jobs (older than {max_age_hours}h)")
        
        return cleaned_count
    
    # ========================
    # Utilities
    # ========================
    
    def get_job_size(self, job_id: str) -> int:
        """Calculate total disk space used by a job.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Total size in bytes
        """
        total_size = 0
        for base_dir in [self.uploads_dir, self.intermediate_dir, self.results_dir]:
            job_dir = base_dir / job_id
            if job_dir.exists():
                for file_path in job_dir.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
        return total_size
