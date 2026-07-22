"""MinerU API-based PDF content extractor."""

import logging
import time
from pathlib import Path
from typing import Any

import requests

from .base import BaseExtractor, ExtractResult

logger = logging.getLogger(__name__)


class MineruAPIError(RuntimeError):
    """Raised when the external MinerU service cannot return Markdown."""


class MineruExtractor(BaseExtractor):
    """PDF content extractor using a remote MinerU service."""

    def __init__(
        self,
        endpoint: str = "http://localhost:8000",
        timeout: float = 1200.0,
        backend: str = "vlm-auto-engine",
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.backend = backend
        self.session = requests.Session()

    def extract(
        self,
        filepath: str,
        save_dir: str = None,
        **kwargs,
    ) -> ExtractResult:
        """Upload a PDF to MinerU and return its Markdown output."""
        pdf_path = Path(filepath)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        started_at = time.monotonic()
        logger.info(
            "MinerU request started: file=%s, backend=%s, endpoint=%s",
            pdf_path.name,
            self.backend,
            self.endpoint,
        )
        try:
            with pdf_path.open("rb") as pdf_file:
                response = self.session.post(
                    f"{self.endpoint}/file_parse",
                    files={
                        "files": (pdf_path.name, pdf_file, "application/pdf"),
                    },
                    data={
                        "backend": self.backend,
                        "return_md": "true",
                        "return_images": "false",
                    },
                    timeout=self.timeout,
                )

            if not response.ok:
                raise MineruAPIError(
                    f"MinerU API returned HTTP {response.status_code}: "
                    f"{response.text[:1000]}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise MineruAPIError("MinerU API returned invalid JSON") from exc

            if not isinstance(payload, dict):
                raise MineruAPIError("MinerU API returned a non-object response")

            result = self._first_result(payload.get("results"))
            text = result.get("md_content")
            if not isinstance(text, str) or not text.strip():
                raise MineruAPIError("MinerU API response does not contain Markdown")

            if save_dir:
                output_dir = Path(save_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / f"{pdf_path.stem}_mineru.md").write_text(
                    text, encoding="utf-8"
                )

            extract_result = ExtractResult(
                text=text,
                metadata={
                    "extractor": "mineru",
                    "backend": payload.get("backend", self.backend),
                    "version": payload.get("version"),
                    "task_id": payload.get("task_id"),
                },
            )
            logger.info(
                "MinerU request completed: file=%s, chars=%d, elapsed=%.1fs",
                pdf_path.name,
                len(text),
                time.monotonic() - started_at,
            )
            return extract_result
        except (requests.RequestException, MineruAPIError) as exc:
            logger.exception(
                "MinerU request failed: file=%s, backend=%s, endpoint=%s",
                pdf_path.name,
                self.backend,
                self.endpoint,
            )
            raise MineruAPIError(f"MinerU extraction failed: {exc}") from exc

    @staticmethod
    def _first_result(results: Any) -> dict:
        """Return the first file result from supported MinerU response shapes."""
        if isinstance(results, dict) and results:
            result = next(iter(results.values()))
        elif isinstance(results, list) and results:
            result = results[0]
        else:
            raise MineruAPIError("MinerU API response does not contain results")

        if not isinstance(result, dict):
            raise MineruAPIError("MinerU API result has an invalid shape")
        return result
